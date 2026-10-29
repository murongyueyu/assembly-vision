# -*- coding: utf-8 -*-
"""E候选：真实入口来源校验、旧配置保留及同帧四步消费。"""
import copy
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from assemvision.config import load_config, Config
from assemvision.pipeline import Pipeline
from assemvision.types import Frame
from assemvision.core.tray_observer import TrayObserver
from assemvision.core.part_tracker import TrackState
from test_prestart_integration import fixture

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT/'config/process_sequence_size_candidate.yaml'


def main():
    old = load_config(str(ROOT/'config/process_sequence.yaml')).data
    new = load_config(str(CANDIDATE)).data
    restored = copy.deepcopy(new)
    restored['prestart'] = old['prestart']
    restored.pop('tracking_departure')
    assert restored == old and old['prestart'] == 'prestart_check.yaml'
    wp, cam = [load_config(str(ROOT/'config'/name)) for name in ('workpiece.yaml', 'camera.yaml')]
    # 真正构造适配层，验证配置路径、模型/policy及相机/workpiece来源，不开相机。
    real = Pipeline(cam, wp, load_config(str(ROOT/'config/calibration.yaml')), source_override='images',
                    path_override=str(ROOT/'data/synthetic_seq'), process_sequence_path=str(CANDIDATE))
    assert real.d_settings['identity']['model_sha256'] == '214991c9f31d99214d42466b94125242a56c998f599addea92ae66711f11e78a'
    assert real.d_settings['identity']['policy_sha256'] == '2af0b26d4969ced04bda10e1fcf670a6c31e276d83cf37465315c9a156e60266'
    assert real.sequence.order == ['slot_base', 'slot_column', 'slot_wheel', 'slot_cap']
    previous, observer, source, identity = fixture()
    with patch.object(TrayObserver, 'from_file', return_value=observer), \
         patch('assemvision.core.part_identity.FrozenPartIdentity', return_value=identity), \
         patch('assemvision.pipeline.make_source', return_value=source):
        pipe = Pipeline(cam, wp, Config(), process_sequence_path=str(CANDIDATE))
    # ★ 2026-10-25: 10-24 加了 D-prep 预分割线程后，_d_process_once 改从 _prepared 取帧；
    #   本用例是直接写 _c_input 的注入式测试，关掉 prep 线程以走原来的 _c_input 分支。
    pipe._prep_thread = None
    pipe.segmenter, pipe._d_intrinsics = previous.segmenter, previous._d_intrinsics
    image = np.zeros((480, 640, 3), 'uint8')
    objects = copy.deepcopy(identity.observe(image, None, None)[0])
    tracks = [TrackState(part_id=o.part, contour=o.contour.copy(), ever_seen=True,
                        detail={'visible_observed': 1.0}) for o in objects]
    identity.tracker = SimpleNamespace(tracks=tracks)
    clock, index = 100., 0
    def send(n):
        nonlocal clock, index
        for _ in range(n):
            clock += .16
            index += 1
            with patch('assemvision.pipeline.time.monotonic', return_value=clock):
                pipe._c_input = (Frame(index, clock, left=image), clock)
                assert pipe._c_process_once()
                state = pipe.build_ui_state()
        return state
    send(3)
    assert pipe.sequence.started and not pipe.sequence.done
    removed = set()
    for number, sid in enumerate(pipe.sequence.order, 1):
        removed.add(pipe.d_expected[sid])
        observer.observe(image).slots[sid]['state'] = 'empty'
        for track in tracks:
            if track.part_id == pipe.d_expected[sid]:
                track.contour = track.contour + np.array([0, 100], dtype=track.contour.dtype)
        identity.observe = lambda *a, **kw: ([o for o in objects if o.part not in removed],
                                             {'objects': [o for o in objects if o.part not in removed]})
        state = send(5)
        assert state.step == number and '缺件' not in state.banner
        assert not pipe._d_snapshot[1].allow_start and pipe.sequence.started
    assert len(pipe.sequence.events) == 4 and '推定完成' in state.title
    print('E尺寸候选：真实来源校验、原门槛保留、同帧四步/槽空接管通过（身份夹具非现场成绩）')
