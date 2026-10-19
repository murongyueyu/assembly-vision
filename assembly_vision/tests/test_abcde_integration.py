# -*- coding: utf-8 -*-
"""集成编排回归：真实入口绑定、四步门禁、B单帧选区和质量未知。"""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import app
from assemvision.config import Config, load_config
from assemvision.core.checks import EvidenceStamp
from assemvision.core.part_tracker import TrackState
from assemvision.core.tray_observer import TrayObserver
from assemvision.pipeline import Pipeline
from assemvision.integration import Integration
from assemvision.ui.base import UIState
from assemvision.types import Frame
from test_prestart_integration import fixture, ROOT


def main():
    previous, observer, source, identity = fixture()
    cam = load_config(str(ROOT/'config/camera.yaml'))
    wp = load_config(str(ROOT/'config/workpiece.yaml'))
    path = str(ROOT/'config/abcde_integration_candidate.yaml')
    real = Pipeline(cam, wp, load_config(str(ROOT/'config/calibration.yaml')),
                    source_override='images', path_override=str(ROOT/'data/synthetic_seq'), integration_path=path)
    assert real.source is not None and real.native_depth is real.integration.estimator
    assert not real.integration.camera.get('native_depth.baseline_accepted')
    assert real.integration.camera.get('native_depth.target_rect') is None
    assert not cam.get('depth.enabled')
    assert real.d_settings['identity']['model_sha256'].startswith('214991c9')
    for conflict in ('process_sequence_path', 'tray_observer_path', 'prestart_check_path'):
        try:
            Pipeline(cam, wp, Config(), integration_path=path, **{conflict:'unused'})
        except ValueError:
            pass
        else:
            raise AssertionError('集成入口未拒绝独立开关')
    with patch.object(TrayObserver, 'from_file', return_value=observer), \
         patch('assemvision.core.part_identity.FrozenPartIdentity', return_value=identity), \
         patch('assemvision.pipeline.make_source', return_value=source) as factory:
        pipe = Pipeline(cam, wp, Config(), integration_path=path)
    assert factory.call_args.args[0].get('depth.enabled')
    assert factory.call_args.args[0].get('realsense.controls') == cam.get('realsense.controls')
    pipe.segmenter, pipe._d_intrinsics = previous.segmenter, previous._d_intrinsics
    image = np.zeros((480,640,3), 'uint8')
    objects = copy.deepcopy(identity.observe(image,None,None)[0])
    tracks = [TrackState(part_id=o.part, contour=o.contour.copy(), ever_seen=True,
                        detail={'visible_observed':1.0}) for o in objects]
    identity.tracker = SimpleNamespace(tracks=tracks)
    clock, index = [100.], [0]
    def send(n):
        for _ in range(n):
            index[0] += 1
            clock[0] += .16
            pipe._c_input = (Frame(index[0],clock[0],left=image),clock[0])
            assert pipe._c_process_once()
        return pipe.build_ui_state()
    with patch('assemvision.pipeline.time.monotonic', side_effect=lambda:clock[0]):
        pipe.request_inspection()
        assert pipe.integration.result is None
        send(3)
        assert pipe.sequence.started
        removed = set()
        for number, sid in enumerate(pipe.sequence.order,1):
            removed.add(pipe.d_expected[sid])
            observer.observe(image).slots[sid]['state']='empty'
            for track in tracks:
                if track.part_id == pipe.d_expected[sid]:
                    track.contour += np.array([0,100],dtype=track.contour.dtype)
            identity.observe = lambda *a,**kw: ([o for o in objects if o.part not in removed],
                                               {'objects':[o for o in objects if o.part not in removed]})
            state = send(5)
            assert state.step == number and '缺件' not in state.banner
        assert len(pipe.sequence.events) == 4 and '质量待判' in state.title
        assert any('高度符合性：无法判断' in line for line in state.detail)
        pipe.request_inspection()
        assert pipe.integration.result is None  # 没有深度不补值。
        frame, result, *_ = pipe._d_snapshot
        yy, xx = np.mgrid[:480,:640]
        xyz = np.stack([xx,yy,np.full_like(xx,440)],axis=-1).astype('float32')
        xyz[300:320,300:320,2] = 372
        frame.depth_points_mm = xyz
        frame.depth_profile = copy.deepcopy(pipe.integration.camera.get('native_depth.profile'))
        with patch('assemvision.pipeline.cv2.selectROI', side_effect=[(20,20,100,100),(300,300,20,20)]), \
             patch('assemvision.pipeline.cv2.destroyWindow'):
            pipe.request_inspection()
        measured = pipe.integration.result
        assert measured['ok'] and abs(measured['height_mm']-68) < .001
        assert measured['frame_index'] == frame.index and measured['height_conforms'] is None
        assert pipe.integration.measurement[0] is frame
        # 设备来源不符、表面未选/选区重叠、取消均不能产生通过。
        bad = copy.copy(frame)
        bad.depth_profile = {'serial':'wrong'}
        assert not pipe.integration.measure(bad,result.stamp,(20,20,100,100),(300,300,20,20),0)['ok']
        try:
            pipe.integration.measure(frame,result.stamp,(20,20,100,100),(30,30,20,20),0)
        except ValueError:
            pass
        else:
            raise AssertionError('重叠测区未拒绝')
        try:
            wrong_stamp = EvidenceStamp(frame.index+1, result.stamp.captured_at, result.stamp.shape)
            pipe.integration.measure(frame,wrong_stamp,(20,20,100,100),(300,300,20,20),0)
        except ValueError:
            pass
        else:
            raise AssertionError('不同帧测量未拒绝')
        with patch('assemvision.pipeline.cv2.selectROI', return_value=(0,0,0,0)), \
             patch('assemvision.pipeline.cv2.destroyWindow'):
            pipe.request_inspection()
        assert pipe.integration.result is None and '取消' in pipe.integration.reason
        clock[0] += 2
        assert not pipe.get_prestart_result().allow_start
        assert not pipe.build_ui_state().polygons
        pipe.request_inspection()
        assert pipe.integration.result is None
        pipe.reset_process()
        assert not pipe.sequence.started and pipe.integration.measurement is None
    renderer = SimpleNamespace(run=lambda *a,**kw:0, close=lambda:None)
    with patch.object(app,'Pipeline',return_value=pipe), patch.object(app,'make_renderer',return_value=renderer) as render, \
         patch.object(pipe,'start'), patch.object(pipe,'stop'):
        assert app.main(['--integration',path,'--windowed','--max-frames','2']) == 0
        assert set(render.call_args.kwargs['actions']) == {'reset','inspect'}
    # 用户追加授权：仅人工确认顶盖的高度项采用±3mm，原B文件仍未接受。
    accepted_path = ROOT/'config/abcde_integration_height_3mm_candidate.yaml'
    accepted = Integration(str(accepted_path), cam)
    assert accepted.height_accepted and accepted.height_tolerance == 3.
    assert accepted.camera.get('native_depth.height_tolerance_mm') == 3.
    assert not accepted.camera.get('native_depth.baseline_accepted')
    assessed = accepted.measure(frame,result.stamp,(20,20,100,100),(300,300,20,20),0)
    assert assessed['height_conforms'] is True
    state = UIState(level='ok',step=4,total=4)
    accepted.decorate(state,True)
    assert state.level == 'ok' and '高度合格' in state.title
    assert any('紧固性：队友负责' in line for line in state.detail)
    baseline = accepted.camera.get('native_depth.baseline_height_mm')
    for difference in (-3.1, 3.1):
        changed_frame = copy.copy(frame)
        changed_frame.depth_points_mm = xyz.copy()
        changed_frame.depth_points_mm[300:320,300:320,2] = 440. - baseline - difference
        assessed = accepted.measure(changed_frame,result.stamp,(20,20,100,100),(300,300,20,20),0)
        assert assessed['ok'] and assessed['height_conforms'] is False
        state = UIState(level='ok',step=4,total=4)
        accepted.decorate(state,True)
        assert state.level == 'error' and '高度不合格' in state.title
    assert not accepted.measure(bad,result.stamp,(20,20,100,100),(300,300,20,20),0)['ok']
    assert accepted.result['height_conforms'] is None
    # 超过用户配置的最大容差时拒绝加载，不能悄悄恢复原15mm。
    invalid = copy.deepcopy(accepted.settings)
    invalid['height_assessment']['tolerance_mm'] = 15.
    for key in ('sequence','native_reference'):
        invalid[key] = str((accepted_path.parent/invalid[key]).resolve())
    with TemporaryDirectory() as temp:
        invalid_path = Path(temp)/'invalid.json'
        invalid_path.write_text(json.dumps(invalid,ensure_ascii=False),encoding='utf-8')
        try:
            Integration(str(invalid_path),cam)
        except ValueError:
            pass
        else:
            raise AssertionError('超过用户指定最大容差仍被接受')
    print('ABCDE：冻结绑定、同帧四步、原生深度接线、人工快照关联、质量未知及复位通过')
