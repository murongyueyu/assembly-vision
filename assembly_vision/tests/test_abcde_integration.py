# -*- coding: utf-8 -*-
"""集成编排回归：真实入口绑定、四步门禁、B单帧选区和质量未知。"""
import copy
import json
import threading
import time
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
from assemvision.pipeline import Pipeline, Shared
from assemvision.integration import Integration, assess_height
from assemvision.ui.base import UIState
from assemvision.types import Frame
from test_prestart_integration import fixture, ROOT


def demand_capture(cam):
    """真实采集线程按需交付一次新同帧深度，不复用旧彩色或跨复位帧。"""
    for mode in ('valid', 'missing', 'reset'):
        pipe = Pipeline.__new__(Pipeline)
        pipe.sh = Shared()
        pipe.sh.lock = threading.RLock()
        pipe.cam, pipe.c_max_age_s = cam, 1.
        pipe.c_error, pipe._e_generation = '', 0
        pipe._integration_depth_request = None
        pipe.integration = Integration(str(ROOT/'config/abcde_integration_height_3mm_candidate.yaml'), cam)
        pipe.sequence = SimpleNamespace(started=True, done=list(range(4)), order=list(range(4)))
        image = np.zeros((480,640,3), 'uint8')
        old = Frame(1, 1., left=image)
        now = time.monotonic()
        pipe._d_snapshot = (old, SimpleNamespace(stamp=EvidenceStamp(1,now,(480,640))), None, {}, None)
        yy, xx = np.mgrid[:480,:640]
        xyz = np.stack([xx,yy,np.full_like(xx,440)],axis=-1).astype('float32')
        xyz[300:320,300:320,2] = 372
        calls, delivered = [], []
        normal = threading.Event()
        def read(want_full=False):
            calls.append(want_full)
            time.sleep(.005)
            frame = Frame(len(calls)+1,float(len(calls)),left=image.copy())
            if want_full:
                if mode != 'missing':
                    frame.depth_points_mm = xyz.copy()
                    frame.depth_profile = copy.deepcopy(pipe.integration.camera.get('native_depth.profile'))
                delivered.append(frame)
                if mode == 'reset':
                    pipe._e_generation += 1
            else:
                normal.set()
            return frame
        pipe.source = SimpleNamespace(read=read)
        thread = threading.Thread(target=pipe._c_acq_loop)
        pipe._threads = [thread]
        thread.start()
        try:
            assert normal.wait(1.) and not any(calls)
            with patch('assemvision.pipeline.cv2.selectROI', side_effect=[(20,20,100,100),(300,300,20,20)]), \
                 patch('assemvision.pipeline.cv2.destroyWindow'):
                pipe.inspect_integration()
            assert calls.count(True) == 1
            if mode == 'valid':
                assert pipe.integration.result['height_conforms'] is True
                measured_frame = pipe.integration.measurement[0]
                assert np.array_equal(measured_frame.left, delivered[0].left)
                assert np.array_equal(measured_frame.depth_points_mm, delivered[0].depth_points_mm)
                assert pipe.integration.result['frame_index'] != old.index
                assert pipe.integration.result['frame_index'] == delivered[0].index
            else:
                assert pipe.integration.result is None
            assert pipe._integration_depth_request is None
        finally:
            pipe.sh.quit = True
            thread.join(1.)
            assert not thread.is_alive()


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
    # ★ 2026-10-25: 10-24 加了 D-prep 预分割线程后，_d_process_once 改从 _prepared 取帧；
    #   本用例是直接写 _c_input 的注入式测试，关掉 prep 线程以走原来的 _c_input 分支。
    pipe._prep_thread = None
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
    assert accepted.height_upper_limit == 68.5 and accepted.height_max == 68.5
    assert accepted.camera.get('native_depth.height_tolerance_mm') == 3.
    assert not accepted.camera.get('native_depth.baseline_accepted')
    assessed = accepted.measure(frame,result.stamp,(20,20,100,100),(300,300,20,20),0)
    assert assessed['height_conforms'] is True
    state = UIState(level='ok',step=4,total=4)
    accepted.decorate(state,True)
    assert state.level == 'ok' and '高度合格' in state.title
    assert any('紧固性：队友负责' in line for line in state.detail)
    assert any('上限：68.500 mm' in line for line in state.detail)
    assert not any('容差：±3' in line for line in state.detail)
    baseline = accepted.camera.get('native_depth.baseline_height_mm')
    # ★ 按原始数值比较，不能先格式化为68.50再判定；边界68.5包含。
    for height, expected in ((68.4999,True),(68.5,True),(68.5001,False),
                             (68.6,False),(70.,False),(60.,False),
                             (baseline-3.,True),(baseline-3.-.0001,False),
                             (None,None),(float('nan'),None),(float('inf'),None)):
        assert assess_height(height,baseline,3.,68.5) is expected, height
    assert assess_height(70.,baseline,3.) is True  # 旧无额外上限的接口仍保留。
    for height, expected in ((68.5,True),(68.5001,False)):
        with patch('assemvision.integration.NativeHeightEstimator.measure_frame',
                   return_value=dict(ok=True,height_mm=height,reason='ok')):
            boundary = accepted.measure(frame,result.stamp,(20,20,100,100),(300,300,20,20),0)
        assert boundary['height_conforms'] is expected
        assert boundary['height_max_mm'] == 68.5
        state = UIState(level='ok',step=4,total=4)
        accepted.decorate(state,True)
        assert state.level == ('ok' if expected else 'error')
    over = copy.copy(frame)
    over.depth_points_mm = xyz.copy()
    over.depth_points_mm[300:320,300:320,2] = 440. - 68.6
    over_result = accepted.measure(over,result.stamp,(20,20,100,100),(300,300,20,20),0)
    assert over_result['ok'] and over_result['height_conforms'] is False
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
        compatible = copy.deepcopy(invalid)
        compatible['height_assessment']['tolerance_mm'] = 3.
        compatible['height_assessment'].pop('upper_limit_mm')
        invalid_path.write_text(json.dumps(compatible,ensure_ascii=False),encoding='utf-8')
        legacy = Integration(str(invalid_path),cam)
        assert legacy.height_upper_limit is None and legacy.height_max == baseline+3.
        for limit in (float('nan'),float('inf'),60.):
            compatible['height_assessment']['upper_limit_mm'] = limit
            invalid_path.write_text(json.dumps(compatible,ensure_ascii=False),encoding='utf-8')
            try:
                Integration(str(invalid_path),cam)
            except ValueError:
                pass
            else:
                raise AssertionError('无效上限未拒绝')
    demand_capture(cam)
    pi_pipe = Pipeline(cam, wp, load_config(str(ROOT/'config/calibration.yaml')),
                       source_override='images', path_override=str(ROOT/'data/synthetic_seq'),
                       integration_path=str(ROOT/'config/abcde_integration_pi_functional_candidate.yaml'))
    assert pi_pipe.c_max_age_s == 3.
    assert pi_pipe.d_settings['freshness']['max_gap_s'] == 3.
    assert pi_pipe.sequence.detector.max_gap == pi_pipe.sequence.detector.max_age == 3.
    assert pi_pipe.identity.confirmation.max_gap == 3.
    assert pi_pipe.identity.confirmation.frames == real.identity.confirmation.frames == 7
    assert pi_pipe.identity.confirmation.min_score == real.identity.confirmation.min_score == .85
    assert real.identity.confirmation.max_gap == .25 and real.c_max_age_s == 1.
    assert pi_pipe.integration.height_upper_limit == 68.5
    assert any('Pi功能联调' in x for x in pi_pipe.build_ui_state().detail)
    pi_pipe.stop()
    performance = Pipeline(cam, wp, load_config(str(ROOT/'config/calibration.yaml')),
                          source_override='images', path_override=str(ROOT/'data/synthetic_seq'),
                          integration_path=str(ROOT/'config/abcde_integration_pi_performance_candidate.yaml'))
    assert performance.observer.workers == 4
    assert performance.deployment_runtime == dict(observer_workers=4,opencv_threads=1,ui_fps=8,
                                                correlation_backend='box_sum_cached')
    assert performance.observer.correlation_backend == 'box_sum_cached'
    assert performance.identity.confirmation.min_score == .85
    assert performance.identity.confirmation.frames == 7
    # ★ 2026-10-25: 上限由用户授权改过（70→65mm），不再写死 68.5；
    #   改成跟配置走，但仍验证「覆盖确实生效」（不是回落到冻结基线）。
    perf_upper = load_config(
        str(ROOT / 'config/abcde_integration_pi_performance_candidate.yaml')) \
        .get('height_assessment.upper_limit_mm')
    assert performance.integration.height_upper_limit == perf_upper
    frozen_baseline = performance.integration.camera.get('native_depth.baseline_height_mm')
    assert abs(perf_upper - frozen_baseline) > 0.5  # 覆盖生效，不是 68.025114 基线
    with patch.object(app, 'Pipeline', return_value=performance), \
         patch.object(app, 'make_renderer', return_value=renderer), \
         patch.object(performance, 'start'), patch.object(performance, 'stop'), \
         patch('cv2.setNumThreads') as thread_setting:
        assert app.main(['--integration',path,'--windowed','--max-frames','2']) == 0
        thread_setting.assert_called_once_with(1)
    performance.stop()
    assert performance.observer._executor is None
    print('ABCDE：冻结绑定、同帧四步、按需深度/复位拒绝、人工快照关联、质量未知通过')
