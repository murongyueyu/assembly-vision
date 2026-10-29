# -*- coding: utf-8 -*-
"""D主入口接线、同帧所有权、门禁撤销和生命周期；不依赖本机忽略数据。"""
import copy
import hashlib
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import app
from assemvision.config import Config, load_config
from assemvision.core.checks import EvidenceStamp
from assemvision.core.part_identity import FrozenPartIdentity
from assemvision.core.tray_observer import TrayObserver, TrayObservation
from assemvision.core.tray_pose import TrayPose
from assemvision.pipeline import Pipeline
from assemvision.types import Frame, SegResult
from test_prestart_check import fixture as d_fixture

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    settings, expected, polygons, objects, obs = d_fixture()
    obs.pose = TrayPose(ok=True)
    wp = load_config(str(ROOT/'config/workpiece.yaml'))
    cam = load_config(str(ROOT/'config/camera.yaml'))
    tray = SimpleNamespace(slots=wp.get('slots'), mm_per_px=lambda: .7,
        slot_polygon_px=lambda sid, **kw: polygons[sid])
    observer = SimpleNamespace(tray=tray, settings={'image_size':[640,480]}, observe=lambda *a, **kw: obs)
    source = SimpleNamespace(name='D-fixture', open=lambda: None, close=lambda: None,
        read=lambda **kw: None, color_intrinsics=lambda: {'fx':603.8248,'width':640,'height':480})
    identity = SimpleNamespace(observe=lambda image, stamp, intr:
        (copy.deepcopy(objects), {'objects':copy.deepcopy(objects)}), reset=lambda: None)
    with patch.object(TrayObserver, 'from_file', return_value=observer), \
         patch('assemvision.core.part_identity.FrozenPartIdentity', return_value=identity), \
         patch('assemvision.pipeline.make_source', return_value=source), \
         patch('assemvision.pipeline.Qualifier', side_effect=AssertionError('旧D启动')), \
         patch('assemvision.pipeline.PreconditionNet', side_effect=AssertionError('E启动')), \
         patch('assemvision.pipeline.EventDetector', side_effect=AssertionError('事件启动')), \
         patch('assemvision.pipeline.NativeHeightEstimator', side_effect=AssertionError('B启动')):
        pipe = Pipeline(cam, wp, Config(), prestart_check_path=str(ROOT/'config/prestart_check.yaml'))
    mask = np.zeros((480,640),'uint8')
    pipe.segmenter = SimpleNamespace(segment=lambda *a, **kw: SegResult(metal_mask=mask,
        glove_mask=mask.copy(), brass_mask=mask.copy()))
    # ★ 2026-10-25: 10-24 加了 D-prep 预分割线程后，_d_process_once 改从 _prepared 取帧；
    #   本用例是直接写 _c_input 的注入式测试，关掉 prep 线程以走原来的 _c_input 分支。
    pipe._prep_thread = None
    pipe._d_intrinsics = source.color_intrinsics()
    return pipe, observer, source, identity


def test_pairing():
    pipe, observer, source, identity = fixture()
    assert pipe.d_mode and pipe.c_mode and not hasattr(pipe,'net')
    assert not pipe.get_prestart_result().allow_start
    pipe.request_inspection(); pipe.reset_process()
    assert pipe.sh.full_request == ''
    image = np.full((480,640,3),77,'uint8')
    calls = []
    original = identity.observe
    clock = [100.]
    def observe(image_arg, stamp, intr):
        calls.append(stamp)
        assert image_arg is image
        pipe._c_input = (Frame(999,999.,left=image+1),clock[0]+.01)
        return original(image_arg,stamp,intr)
    identity.observe = observe
    with patch('assemvision.pipeline.time.monotonic', side_effect=lambda: clock[0]):
        for index in range(1,4):
            clock[0] = 100.+(index-1)*.16
            frame=Frame(index, 123., left=image)
            pipe._c_input=(frame,clock[0])
            assert pipe._c_process_once()
            state=pipe.build_ui_state()
            assert state.frame is image and state.stats['帧号']==index
            assert len(state.polygons)==8 and len(state.boxes)==4
            assert all(np.array_equal(state.polygons[i+4][0],o.contour) for i,o in enumerate(d_fixture()[3]))
            assert pipe.get_prestart_result().allow_start == (index==3)
            pipe._c_input=(frame,clock[0])
            assert not pipe._c_process_once()
        assert len(calls)==3
        clock[0]+=2
        assert not pipe.get_prestart_result().allow_start
        assert not pipe.build_ui_state().polygons
        clock[0]+=.1
        observer.observe=lambda *a,**kw: TrayObservation(reason='方向拒识')
        pipe._c_input=(Frame(4,123.,left=image),clock[0]); pipe._c_process_once()
        assert not pipe.get_prestart_result().valid
        # C拒识清槽框和关联；独立A本帧真实轮廓可保留为黄色，不沿用匹配颜色。
        state=pipe.build_ui_state()
        assert len(state.polygons)==4 and all(p[1]=='warn' for p in state.polygons)
        # 同帧手套只追加矩形框；不新增手的多边形。
        pipe._d_snapshot[-1]['glove_bbox']=(100,100,50,60)
        hand_state=pipe.build_ui_state()
        assert hand_state.boxes[-1]==((100,100,50,60),'手套遮挡','warn')
        assert len(hand_state.polygons)==len(state.polygons)
        pipe._c_input=(Frame(5,123.,left=image,scale=.5),clock[0]+.1)
        pipe._c_process_once()
        assert not pipe.get_prestart_result().allow_start
        identity.observe=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('identity failed'))
        pipe._c_input=(Frame(6,123.,left=image),clock[0]+.2);pipe._c_process_once()
        assert pipe.c_error and pipe.sh.quit
        assert not pipe.get_prestart_result().allow_start and not pipe.build_ui_state().polygons


def test_lifecycle_and_entry():
    pipe, observer, source, identity = fixture()
    sent, closed = [], []
    image=np.zeros((480,640,3),'uint8')
    def read(**kw):
        if not sent:
            sent.append(True)
            return Frame(1,1.,left=image)
        return None
    source.read=read; source.close=lambda:closed.append(True)
    pipe.start()
    deadline=time.monotonic()+2
    while pipe._d_snapshot is None and not pipe.c_error and time.monotonic()<deadline:
        time.sleep(.005)
    assert pipe._d_snapshot is not None
    image[:]=99
    assert np.all(pipe.build_ui_state().frame==0)
    pipe.stop()
    assert closed and not any(t.is_alive() for t in pipe._threads)
    pipe, observer, source, identity = fixture()
    source.read=lambda **kw: (_ for _ in ()).throw(RuntimeError('source failed'))
    pipe.start()
    deadline=time.monotonic()+2
    while not pipe.c_error and time.monotonic()<deadline:
        time.sleep(.005)
    pipe.stop()
    assert pipe.c_error and not pipe.get_prestart_result().allow_start
    pipe, observer, source, identity = fixture()
    closed=[];source.close=lambda:closed.append(True)
    source.open=lambda: (_ for _ in ()).throw(RuntimeError('open failed'))
    try:
        pipe.start()
    except RuntimeError:
        pass
    else:
        raise AssertionError('开流失败被忽略')
    assert closed
    pipe, observer, source, identity=fixture()
    renderer=SimpleNamespace(run=lambda *a,**kw:0,close=lambda:None)
    with patch.object(app,'Pipeline',return_value=pipe) as factory, \
         patch.object(app,'make_renderer',return_value=renderer) as render, \
         patch.object(pipe,'start'), patch.object(pipe,'stop') as stop:
        assert app.main(['--prestart-check','config/prestart_check.yaml','--windowed','--max-frames','2'])==0
        assert Path(factory.call_args.kwargs['prestart_check_path'])==ROOT/'config/prestart_check.yaml'
        assert render.call_args.kwargs['actions']=={} and not render.call_args.kwargs['fullscreen']
        assert stop.called
        renderer.run=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('render failed'))
        try:
            app.main(['--prestart-check','config/prestart_check.yaml','--windowed'])
        except RuntimeError:
            pass
        else:
            raise AssertionError('渲染失败被忽略')
        assert stop.call_count==2


def test_frozen_adapter_validation():
    # 自包含模型验证：正式绑定、policy一致性及无旧分类器回退。
    with TemporaryDirectory() as folder:
        folder=Path(folder)
        def write(name,data):
            path=folder/name;path.write_text(json.dumps(data),encoding='utf-8')
            return path,hashlib.sha256(path.read_bytes()).hexdigest()
        policy={'experiment':{'min_score':.55,'holes_tol':1.,'tracking_overrides':{'backend':'detection','detect_every_n':1}}}
        policy_path,policy_hash=write('policy.json',policy)
        model={'policy':policy,'policy_sha256':policy_hash,'templates':{'P1_base':{'ranges':{'area_mm2':[1,2]}}}}
        model_path,model_hash=write('model.json',model)
        wp_path,wp_hash=write('wp.json',{})
        cam_path,cam_hash=write('cam.json',{})
        cfg={'identity':{'model':model_path.name,'model_sha256':model_hash,
                        'policy':policy_path.name,'policy_sha256':policy_hash,
                        'workpiece_sha256':wp_hash,'camera_sha256':cam_hash}}
        wp,cam=Config({},str(wp_path)),Config({},str(cam_path))
        adapter=FrozenPartIdentity(wp,cam,folder/'d.yaml',cfg)
        assert adapter.tracker.backend=='detection'
        model_path.write_text('{}',encoding='utf-8')
        try:
            FrozenPartIdentity(wp,cam,folder/'d.yaml',cfg)
        except ValueError:
            pass
        else:
            raise AssertionError('错误模型hash被接受')


def main():
    test_pairing()
    test_lifecycle_and_entry()
    test_frozen_adapter_validation()
    assert app.parse_args([]).prestart_check is None
    print('D主程序同帧/去重/时效/异常撤销/入口隔离/窗口关闭/线程释放/冻结加载通过')


if __name__=='__main__':
    main()
