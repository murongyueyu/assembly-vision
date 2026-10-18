# -*- coding: utf-8 -*-
"""C主程序接线行为测试；不依赖相机或本机被忽略的硬件标定。"""
import copy
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assemvision.config import Config, load_config
from assemvision.core.tray_observer import TrayObservation, TrayObserver
from assemvision.core.tray_pose import TrayPose
from assemvision.pipeline import Pipeline
from assemvision.types import Frame, SegResult
import app

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    wp = load_config(str(ROOT / 'config/workpiece.yaml'))
    cam = load_config(str(ROOT / 'config/camera.yaml'))
    cam = Config(copy.deepcopy(cam.data))
    cam.set('depth.enabled', True)
    cam.set('depth.backend', 'realsense')
    cam.set('capture.warmup_frames', 0)
    tray = SimpleNamespace(slots=wp.get('slots'), mm_per_px=lambda: .7,
        slot_polygon_px=lambda *a, **kw: np.zeros((4, 2)))
    observer = SimpleNamespace(tray=tray, settings={'image_size': [640, 480]})
    source = SimpleNamespace(name='fixture', open=lambda: None, close=lambda: None,
                             read=lambda **kw: None)
    model = str(ROOT / 'config/tray_observation_v2.yaml')
    with patch.object(TrayObserver, 'from_file', return_value=observer), \
         patch('assemvision.pipeline.make_source', return_value=source) as factory, \
         patch('assemvision.pipeline.Qualifier', side_effect=AssertionError('旧D启动')), \
         patch('assemvision.pipeline.PreconditionNet', side_effect=AssertionError('旧E启动')), \
         patch('assemvision.pipeline.NativeHeightEstimator', side_effect=AssertionError('B启动')), \
         patch('assemvision.pipeline.PoseChecker', side_effect=AssertionError('旧位姿启动')), \
         patch('assemvision.pipeline.EventDetector', side_effect=AssertionError('事件启动')):
        pipe = Pipeline(cam, wp, Config(), tray_observer_path=model)
    assert factory.call_args.args[0].get('depth.enabled') is False
    assert cam.get('depth.enabled') is True
    assert not hasattr(pipe, 'qualifier') and not hasattr(pipe, 'net')
    return pipe, observer, source


def main():
    pipe, obs, source = fixture()
    assert app.parse_args([]).tray_observer is None
    assert not pipe._c_process_once()
    assert '等待首帧' in ' '.join(pipe.build_ui_state().detail)
    pipe.request_inspection(); pipe.reset_process()
    assert pipe.sh.full_request == ''
    masks = [np.full((480, 640), i, np.uint8) for i in (11, 22, 33)]
    calls = []
    def segment(image, **kwargs):
        calls.append(image)
        assert kwargs == {'ts': 123., 'scale': 1., 'mm_per_px': .7}
        # 模拟分割期间新帧到达，不能改变当前observe及UI对应原图。
        pipe._c_input = (Frame(2, 999., left=np.full_like(image, 88)), time.monotonic())
        return SegResult(metal_mask=masks[0], glove_mask=masks[1], brass_mask=masks[2])
    pipe.segmenter = SimpleNamespace(segment=segment)
    good = TrayObservation(found=True, orientation_ok=True, state='full',
        pose=TrayPose(ok=True), slots={s['id']: {'state': 'occupied'} for s in pipe.tray.slots})
    def observe(image, metal, glove, brass_mask):
        assert image is calls[-1]
        assert metal is masks[0] and glove is masks[1] and brass_mask is masks[2]
        return good
    obs.observe = observe
    first = Frame(1, 123., left=np.full((480, 640, 3), 77, np.uint8))
    pipe._c_input = (first, time.monotonic())
    assert pipe._c_process_once()
    state = pipe.build_ui_state()
    assert state.frame is first.left and state.stats['帧号'] == 1
    assert len(state.polygons) == 4 and '满盘' in state.title
    assert all('有件' in line for line in state.detail[:4])
    pipe._c_input = (first, time.monotonic())
    assert not pipe._c_process_once() and len(calls) == 1
    # 发布不能重置采集时间；即使计算刚完成，旧帧也须过期。
    pipe._c_input = (Frame(3, 123., left=first.left), time.monotonic() - 2 * pipe.c_max_age_s)
    pipe._c_process_once()
    state = pipe.build_ui_state()
    assert '观测过期' in state.detail and not state.polygons and '不确定' in state.title
    # 拒识清除此前四槽及角度，不依赖观察器内是否缓存。
    obs.observe = lambda *a, **kw: TrayObservation(reason='方向拒识')
    pipe._c_input = (Frame(4, 123., left=first.left), time.monotonic())
    pipe._c_process_once()
    state = pipe.build_ui_state()
    assert not state.polygons and all('不确定' in s for s in state.detail[:4])
    assert '方向估计' not in state.stats and '方向拒识' in state.detail
    # 错尺寸/缩放/无图不进入分割；可恢复的无效输入不是处理异常。
    for index, image, scale in [(5, first.left, .5), (6, None, 1.),
                               (7, first.left[:20], 1.)]:
        pipe._c_input = (Frame(index, 123., left=image, scale=scale), time.monotonic())
        pipe._c_process_once()
        assert '输入无效' in ' '.join(pipe.build_ui_state().detail)
    saved_segmenter = pipe.segmenter
    pipe.segmenter = SimpleNamespace(segment=lambda *a, **kw: SegResult(
        metal_mask=masks[0], brass_mask=masks[2]))
    pipe._c_input = (Frame(71, 123., left=first.left), time.monotonic())
    pipe._c_process_once()
    assert '输入无效' in ' '.join(pipe.build_ui_state().detail)
    pipe.segmenter = saved_segmenter
    obs.observe = observe
    pipe._c_input = (Frame(8, 123., left=first.left), time.monotonic())
    pipe._c_process_once()
    assert '满盘' in pipe.build_ui_state().title
    pipe.segmenter = SimpleNamespace(segment=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('boom')))
    pipe._c_input = (Frame(9, 123., left=first.left), time.monotonic())
    pipe._c_process_once()
    state = pipe.build_ui_state()
    assert pipe.sh.quit and '处理异常' in ' '.join(state.detail) and not state.polygons
    pipe._c_snapshot = (first, time.monotonic(), good, '')
    pipe._c_fail('采集异常：断开')
    assert not pipe.build_ui_state().polygons and pipe.build_ui_state().frame is None
    test_lifecycle()
    test_entry()
    # 显式模型失败不得回退旧入口；实际from_file执行文件/绑定验证。
    cam = load_config(str(ROOT / 'config/camera.yaml'))
    wp = load_config(str(ROOT / 'config/workpiece.yaml'))
    for model, cal in [('missing-c-model.yaml', Config()),
                       (str(ROOT / 'config/tray_observation_v2.yaml'),
                        Config({'tray_map': {'homography': np.eye(3).tolist(), 'scale': 1.}}))]:
        try:
            Pipeline(cam, wp, cal, tray_observer_path=model)
        except (ValueError, FileNotFoundError):
            pass
        else:
            raise AssertionError('显式C加载失败未拒绝')
    print('C主程序同帧/去重/拒识/过期/异常/生命周期/入口隔离测试通过')


def test_lifecycle():
    pipe, obs, src = fixture()
    closed, sent = [], []
    image = np.full((480, 640, 3), 20, np.uint8)
    def read(**kw):
        if not sent:
            sent.append(True)
            return Frame(1, 1., left=image)
        return None
    src.read = read
    src.close = lambda: closed.append(True)
    obs.observe = lambda *a, **kw: TrayObservation(reason='无方向')
    pipe.start()
    deadline = time.monotonic() + 2
    while pipe._c_snapshot is None and time.monotonic() < deadline:
        time.sleep(.005)
    assert pipe._c_snapshot is not None
    assert {t.name for t in pipe._threads} == {'C-acq', 'C-observe'}
    image[:] = 99
    assert np.all(pipe._c_snapshot[0].left == 20), '来源缓冲区不得改写已发布帧'
    pipe.stop()
    assert closed and not any(t.is_alive() for t in pipe._threads)
    pipe, obs, src = fixture()
    closed = []
    src.close = lambda: closed.append(True)
    src.open = lambda: (_ for _ in ()).throw(RuntimeError('open fail'))
    try:
        pipe.start()
    except RuntimeError:
        pass
    else:
        raise AssertionError('开流异常未报错')
    assert closed
    # 采集异常须终止线程且清除成功结果。
    pipe, obs, src = fixture()
    src.read = lambda **kw: (_ for _ in ()).throw(RuntimeError('read fail'))
    pipe.start()
    deadline = time.monotonic() + 1
    while not pipe.c_error and time.monotonic() < deadline:
        time.sleep(.005)
    pipe.stop()
    assert '采集异常' in pipe.c_error


def test_entry():
    # 实际main解析和路径传递，确保C没有动作入口；异常仍关闭renderer/来源。
    pipe, obs, src = fixture()
    captured = []
    def factory(*args, **kw):
        captured.append(kw)
        return pipe
    renderer = SimpleNamespace(run=lambda *a, **kw: 0, close=lambda: captured.append('renderer_closed'))
    with patch.object(app, 'Pipeline', side_effect=factory), \
         patch.object(app, 'make_renderer', return_value=renderer) as render, \
         patch.object(pipe, 'start'), patch.object(pipe, 'stop') as stop:
        assert app.main(['--tray-observer', 'config/tray_observation_v2.yaml',
                         '--windowed', '--max-frames', '2']) == 0
        assert render.call_args.kwargs['actions'] == {}
        assert render.call_args.kwargs['fullscreen'] is False
        assert Path(captured[0]['tray_observer_path']) == ROOT / 'config/tray_observation_v2.yaml'
        assert captured[-1] == 'renderer_closed' and stop.called
        renderer.run = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('render fail'))
        try:
            app.main(['--tray-observer', 'config/tray_observation_v2.yaml', '--windowed'])
        except RuntimeError:
            pass
        else:
            raise AssertionError('渲染异常丢失')
        assert stop.call_count == 2
        pipe.c_mode = False
        renderer.run = lambda *a, **kw: 0
        assert app.main(['--windowed', '--max-frames', '2']) == 0
        assert render.call_args.kwargs['actions']['inspect'] == pipe.request_inspection
        assert captured[-2]['tray_observer_path'] is None


if __name__ == '__main__':
    main()
