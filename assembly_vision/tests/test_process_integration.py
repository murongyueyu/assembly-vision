# -*- coding: utf-8 -*-
"""E主入口同帧接线、复位竞争、过期UI与旧C/D模式隔离。"""
import copy
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import app
from assemvision.config import load_config, Config
from assemvision.pipeline import Pipeline
from assemvision.core.tray_observer import TrayObserver
from assemvision.types import Frame
from test_prestart_integration import fixture, ROOT


def build():
    old, observer, source, identity = fixture()
    with patch.object(TrayObserver, 'from_file', return_value=observer), \
         patch('assemvision.core.part_identity.FrozenPartIdentity', return_value=identity), \
         patch('assemvision.pipeline.make_source', return_value=source), \
         patch('assemvision.pipeline.Qualifier', side_effect=AssertionError('旧D启动')), \
         patch('assemvision.pipeline.NativeHeightEstimator', side_effect=AssertionError('B启动')):
        pipe=Pipeline(old.cam, old.wp, Config(),
                      process_sequence_path=str(ROOT/'config/process_sequence.yaml'))
    pipe.segmenter=old.segmenter
    pipe._d_intrinsics=old._d_intrinsics
    return pipe, observer, identity


def main():
    old,_,_,_=fixture()
    for conflict in ('prestart_check_path','tray_observer_path'):
        try:
            Pipeline(old.cam,old.wp,Config(),
                     process_sequence_path=str(ROOT/'config/process_sequence.yaml'),
                     **{conflict:'config/unused.yaml'})
        except ValueError:
            pass
        else:
            raise AssertionError('E入口未拒绝互斥开关')
    pipe, observer, identity=build()
    clock=[100.]
    index=[0]
    image=np.zeros((480,640,3),'uint8')
    all_objects=copy.deepcopy(identity.observe(image,None,None)[0])
    def send(n):
        for _ in range(n):
            index[0]+=1;clock[0]+=.16
            pipe._c_input=(Frame(index[0],1.,left=image),clock[0])
            assert pipe._c_process_once()
        return pipe.build_ui_state()
    with patch('assemvision.pipeline.time.monotonic', side_effect=lambda:clock[0]):
        send(3)
        assert pipe.sequence.started and pipe.build_ui_state().total==4
        removed=set()
        for number,sid in enumerate(pipe.sequence.order,1):
            removed.add(pipe.d_expected[sid])
            observer.observe(image).slots[sid]['state']='empty'
            identity.observe=lambda *a,**kw: (
                [o for o in all_objects if o.part not in removed],
                {'objects':[o for o in all_objects if o.part not in removed]})
            st=send(5)
            assert st.step==number and st.level=='ok', (number,st)
            assert '缺件' not in st.banner
        assert '推定完成' in st.title
        clock[0]+=2
        st=pipe.build_ui_state()
        assert st.level=='warn' and st.step==4 and not st.polygons
        pipe.reset_process()
        assert not pipe.sequence.started and pipe.build_ui_state().step==0
        send(5);assert not pipe.sequence.started
        # 复位发生于A处理期间，旧一轮结果不得发布或重新开工。
        for row in observer.observe(image).slots.values():row['state']='occupied'
        identity.observe=lambda *a,**kw: (all_objects,{'objects':all_objects})
        send(3);assert pipe.sequence.started
        def racing(*args):
            pipe.reset_process()
            return all_objects,{'objects':all_objects}
        identity.observe=racing
        send(1);assert pipe._d_snapshot is None and not pipe.sequence.started
    pipe,_,_=build()
    renderer=SimpleNamespace(run=lambda *a,**kw:0,close=lambda:None)
    with patch.object(app,'Pipeline',return_value=pipe) as factory, \
         patch.object(app,'make_renderer',return_value=renderer) as render, \
         patch.object(pipe,'start'),patch.object(pipe,'stop') as stop:
        assert app.main(['--process-sequence','config/process_sequence.yaml','--windowed','--max-frames','2'])==0
        assert Path(factory.call_args.kwargs['process_sequence_path'])==ROOT/'config/process_sequence.yaml'
        assert set(render.call_args.kwargs['actions'])=={'reset'}
        assert stop.called and not render.call_args.kwargs['fullscreen']
    print('E接线：同帧四步、正常空槽不报缺件、过期、复位竞争及主入口通过')
    return 0
