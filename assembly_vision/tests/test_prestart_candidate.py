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
from assemvision.config import load_config
from assemvision.core.checks import EvidenceStamp
from assemvision.core.part_identity import FrozenPartIdentity

ROOT = Path(__file__).resolve().parents[1]


def main():
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
