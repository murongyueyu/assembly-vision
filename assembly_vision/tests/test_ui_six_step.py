"""六步真实状态、计时、复位隔离与完成卡到期回归。"""
from pathlib import Path
from types import SimpleNamespace
import threading
from unittest.mock import patch

import numpy as np

from assemvision.ui.workflow import SixStepWorkflow, duration_text
from assemvision.ui.base import UIState
from assemvision.ui.opencv_renderer import OpenCVRenderer


def evidence(step, generation=0, passed=True):
    return dict(step=step,generation=generation,ok=True,height_conforms=passed,
                length_mm=64.0 if step==2 else None,height_mm=68.5,frame_index=7)


def main():
    clock=[100.]
    w=SixStepWorkflow({'completion_seconds':5.5},clock=lambda:clock[0])
    results={}
    assert w.update(0,False,results,0)['elapsed']=='--:--'
    w.update(0,True,results,0)
    assert w.update(2,True,results,0)['completed']==2
    assert w.update(4,True,results,0)['completed']==2  # 取完四件不能跳过长度
    results['column']=evidence(2,passed=False)
    assert w.update(2,True,results,0)['statuses'][2]=='failed'
    results['column']=evidence(2)
    assert w.update(2,True,results,0)['completed']==3
    assert w.update(3,True,results,0)['completed']==4
    assert w.update(4,True,results,0)['completed']==5
    results['cap']=evidence(4,passed=False)
    assert not w.update(4,True,results,0)['complete']
    results['cap']=evidence(4)
    clock[0]=183.4
    v=w.update(4,True,results,0)
    assert v['complete'] and v['completed']==6 and v['elapsed']=='01:23.4'
    ident=v['completion_id']
    clock[0]=200.
    assert w.update(4,True,results,0)['elapsed']==v['elapsed']
    assert w.update(4,True,results,0)['completion_id']==ident
    assert not w.update(4,True,results,0,'error')['complete']
    w.update(3,True,results,0)
    assert 'cap' not in results and 'column' in results
    w.update(1,True,results,0)
    assert 'column' not in results
    results.update(column=evidence(2),cap=evidence(4))
    assert not w.update(4,True,results,1)['complete']  # 跨轮次旧结果不准通过
    assert duration_text(3601.9)=='01:00:01'
    for invalid in (4.9,6.1,float('nan')):
        try:
            SixStepWorkflow({'completion_seconds':invalid})
        except ValueError:
            pass
        else:
            raise AssertionError('展示时间无效却被接受')

    # 实际渲染：同一相机帧不变，完成卡仍准时消失，告警可立即撤销。
    state=UIState(frame=np.zeros((480,640,3),np.uint8),workflow=v)
    state.workflow['rules']=(('立柱','63.000～65.000 mm'),('顶盖','67.500～69.500 mm'))
    renderer=OpenCVRenderer(display=(1024,600),fullscreen=False)
    renderer.cache_unchanged=True
    renderer.render(state)
    t=[1000.]
    renderer._operator.clock=lambda:t[0]
    renderer._operator.seen=None
    active=renderer.render(state)
    t[0]=1005.49
    assert renderer._operator.popup_at==1000.
    renderer.render(state)
    t[0]=1005.5
    ended=renderer.render(state)
    assert not np.array_equal(active,ended)
    assert renderer._operator.popup_at==1000.  # 没有重新计时
    state.banner='测试故障'
    renderer.render(state)
    assert renderer._operator.popup_at is None
    renderer.close()

    # 集成回归：测顶盖保留长度结果；复位同时清两者和计时。
    from assemvision.integration import Integration
    from assemvision.config import load_config
    from assemvision.pipeline import Pipeline,Shared
    from assemvision.core.checks import EvidenceStamp
    from assemvision.types import Frame
    root=Path(__file__).resolve().parents[1]
    config=root/'config/abcde_integration_pi_ui6_candidate.yaml'
    if config.exists() and (root.parent/'ai_测试工具').exists():
        i=Integration(str(config),load_config(str(root/'config/camera.yaml')))
        i.results['column']=evidence(2)
        pipe=Pipeline.__new__(Pipeline)
        pipe.integration=i
        pipe.sh=Shared(); pipe.sh.lock=threading.RLock()
        pipe.sequence=SimpleNamespace(started=True,done=list(range(4)),verdict=SimpleNamespace(level='warn'))
        pipe._e_generation=0
        pipe.c_max_age_s=3.
        image=np.zeros((480,640,3),np.uint8)
        frame=Frame(7,0.,left=image)
        frame.depth_points_mm=np.zeros((480,640,3),np.float32)
        pipe._d_current_snapshot=lambda:(frame,SimpleNamespace(stamp=EvidenceStamp(7,100.,(480,640))),None,{},None)
        with patch('assemvision.pipeline.time.monotonic',return_value=100.), \
             patch('assemvision.pipeline.cv2.selectROI',side_effect=[(1,1,10,10),(20,20,10,10)]), \
             patch('assemvision.pipeline.cv2.destroyWindow'), \
             patch.object(i,'measure',side_effect=lambda *a,**kw:i.results.update(cap=evidence(4))):
            pipe.inspect_integration()
        assert set(i.results)=={'column','cap'}
        assert i.workflow.update(4,True,i.results,0)['complete']
        # 本段回归旧的“人工选区”兜底路径；正式配置的立柱阶段是 auto_zone。
        for stage in i.stages:
            stage['auto_zone']=False
        # 同轮选区期间提前取件只取消本次测量，不重启计时。
        original_start=i.workflow.started_at
        pipe.sequence.done=list(range(2))
        pipe._d_current_snapshot=lambda:(frame,SimpleNamespace(stamp=EvidenceStamp(7,100.,(480,640))),None,{},None)
        pipe.segmenter=SimpleNamespace(segment=lambda *a,**kw:None)
        pipe.tray=SimpleNamespace(mm_per_px=lambda *a:0.638)
        def advance(*a,**kw):
            pipe.sequence.done=list(range(3))
            return (1,1,10,10)
        with patch('assemvision.pipeline.time.monotonic',return_value=100.), \
             patch('assemvision.pipeline.cv2.selectROI',side_effect=advance), \
             patch('assemvision.pipeline.cv2.destroyWindow'):
            pipe.inspect_integration()
        assert i.workflow.started_at==original_start
        assert '进度已变化' in i.reason
        i.reset()
        assert not i.results and i.workflow.started_at is None
    print('test_ui_six_step 全部通过')


if __name__=='__main__':
    main()
