# -*- coding: utf-8 -*-
"""D候选配置只换来源；真实适配层保留候选孔特征且不绕过来源校验。"""
import copy
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from assemvision.config import load_config
from assemvision.core.checks import EvidenceStamp
from assemvision.core.part_identity import FrozenPartIdentity
from validate_prestart_candidate import digest, digest_text

ROOT = Path(__file__).resolve().parents[1]


def check_digest_eol():
    """★ 配置标签与行尾无关，但来源校验必须继续按字节。

    仓库 `core.autocrlf=true` 且没有 `.gitattributes`，同一份 yaml 在 Windows
    检出是 CRLF、在 Linux/树莓派检出是 LF —— 实测
    `process_sequence_size_candidate.yaml` 工作区 746 字节 / 仓库 blob 727 字节，
    差的 19 正好是行数。字节级 sha256 因此跨机对不上，归一化后必须完全相同。
    """
    with TemporaryDirectory() as folder:
        folder = Path(folder)
        text = 'prestart:\n  min_area_px: 300\n'
        crlf, lf = folder/'crlf.yaml', folder/'lf.yaml'
        crlf.write_bytes(text.replace('\n', '\r\n').encode('utf-8'))
        lf.write_bytes(text.encode('utf-8'))
        assert digest(crlf) != digest(lf), '字节级 sha256 必须能区分行尾'
        assert digest_text(crlf) == digest_text(lf), '归一化后必须跨机一致'
        assert digest_text(lf) == digest(lf), '本来就是 LF 的文件不得被改动'
    # ★ 反向守卫：workpiece/camera 的哈希是**防篡改**用途，必须继续字节级，
    #   且必须与 prestart_check*.yaml 里登记的值一致。谁在仓库里加
    #   `.gitattributes` 让它们检出成 LF，这里会立刻红 —— 而不是等到树莓派上
    #   FrozenPartIdentity 拒绝加载、主程序起不来（模块 G 一次都没跑过）。
    prestart = load_config(str(ROOT/'config/prestart_check.yaml')).data['identity']
    for key, name in (('workpiece_sha256', 'workpiece.yaml'), ('camera_sha256', 'camera.yaml')):
        assert digest(ROOT/'config'/name) == prestart[key], key + ' 与登记值不符（行尾被改过？）'
    # ★ 真实案例回归：abcde_integration_candidate.yaml 的 sequence_sha256 是在 **LF 内容**上
    #   登记的（444c0128…），而 Windows 检出是 CRLF —— 纯字节级比较必然不符，
    #   树莓派上 test_abcde_integration 就是这样挂的。这里钉住「两种口径至少要接受一个」。
    integ = load_config(str(ROOT/'config/abcde_integration_candidate.yaml')).data
    source = ROOT/'config'/integ['sequence']
    assert integ['sequence_sha256'] in {digest(source), digest_text(source)}, \
        '集成候选冻结来源 hash 两种口径都对不上（文件真的被改了？）'
    print('配置标签：行尾无关、来源校验仍按字节、登记值一致通过')


def main():
    check_digest_eol()
    old = load_config(str(ROOT/'config/prestart_check.yaml')).data
    candidate = load_config(str(ROOT/'config/prestart_check_wheel10_candidate.yaml')).data
    changed = copy.deepcopy(candidate)
    for key in ('model', 'policy', 'model_sha256', 'policy_sha256'):
        assert candidate['identity'][key] != old['identity'][key]
        changed['identity'][key] = old['identity'][key]
    assert changed == old
    assert old['identity']['model_sha256'] == 'ad320176bdd520aef392ad6c81a31fcd67f64f82199d50a62ec96d3a2046eaa4'
    wp = load_config(str(ROOT/'config/workpiece.yaml'))
    cam = load_config(str(ROOT/'config/camera.yaml'))
    image = cv2.cvtColor(np.full((240, 240, 3), (20, 150, 180), np.uint8), cv2.COLOR_HSV2BGR)
    mask = np.zeros((240, 240), np.uint8)
    cv2.circle(mask, (120, 120), 80, 255, -1)
    cv2.circle(mask, (120, 120), 8, 0, -1)
    mask[100, 100:102] = 0
    image[mask == 0] = 0
    outputs = []
    with TemporaryDirectory() as folder:
        folder = Path(folder)
        for threshold in (None, 10):
            policy = copy.deepcopy(load_config(str(ROOT/'config/part_dataset.yaml')).data)
            policy['experiment']['segmentation_overrides'] = {
                'segment.brass.min_area_px': 300, 'segment.brass.morph_open': 1,
                'segment.brass.morph_close': 1}
            policy['experiment']['tracking_overrides'] = {'backend': 'detection', 'detect_every_n': 1}
            if threshold is not None:
                policy['experiment']['hole_min_area_px'] = threshold
            def write(name, data):
                path = folder/name
                path.write_text(json.dumps(data), encoding='utf-8')
                return hashlib.sha256(path.read_bytes()).hexdigest()
            ph = write('policy.json', policy)
            mh = write('model.json', dict(policy=policy, policy_sha256=ph, variant='brass',
                templates={'P3_wheel': {'ranges': {'holes': [0, 10]}}}))
            settings = {'identity': dict(model='model.json', policy='policy.json', model_sha256=mh,
                policy_sha256=ph, workpiece_sha256=hashlib.sha256(Path(wp.path).read_bytes()).hexdigest(),
                camera_sha256=hashlib.sha256(Path(cam.path).read_bytes()).hexdigest())}
            adapter = FrozenPartIdentity(wp, cam, folder/'d.yaml', settings)
            _, audit = adapter.observe(image, EvidenceStamp(1, 100., image.shape[:2]), {'fx': 603.8248})
            assert len(audit['predictions']) == 1
            outputs.append(audit)
            settings['identity']['model_sha256'] = '0'*64
            try:
                FrozenPartIdentity(wp, cam, folder/'d.yaml', settings)
            except ValueError:
                pass
            else:
                raise AssertionError('来源不符必须拒绝')
    assert np.array_equal(outputs[0]['mask'], outputs[1]['mask'])
    before, after = [row['predictions'][0]['features'] for row in outputs]
    assert before['holes'] == 2 and after['holes'] == 1
    assert {k:v for k,v in before.items() if k != 'holes'} == {k:v for k,v in after.items() if k != 'holes'}
    print('D候选：仅四项来源变更、真实适配孔特征传递、默认兼容、掩膜不变和hash拒绝通过')


if __name__ == '__main__':
    main()
