# -*- coding: utf-8 -*-
"""E追踪取件：部分遮挡、C未知延续、失去当前轮廓/换轨/断档拒绝及流程回退。"""
import copy
from pathlib import Path
import numpy as np
from assemvision.config import load_config
from assemvision.core.checks import EvidenceStamp, PrestartResult, PrestartSlot
from assemvision.core.fsm import PickSequence
from assemvision.core.part_tracker import TrackState

ROOT = Path(__file__).resolve().parents[1]


def rect(x, y):
    return np.array([[x,y],[x+20,y],[x+20,y+20],[x,y+20]], 'int32')


class Scene:
    def __init__(self):
        self.net = PickSequence(load_config(str(ROOT/'config/workpiece.yaml')),
            load_config(str(ROOT/'config/process_sequence_size_candidate.yaml')).data)
        self.polygons = {sid: rect(40+i*80,40) for i,sid in enumerate(self.net.expected)}
        self.tracks = {sid: TrackState(part_id=part, contour=rect(45+i*80,45),
            ever_seen=True, detail={'visible_observed':1.0})
            for i,(sid,part) in enumerate(self.net.expected.items())}
        self.origins = {sid: t.contour.copy() for sid,t in self.tracks.items()}
        self.index, self.clock, self.empty = 0, 100., set()
        self.feed(allow=True)
        assert self.net.started

    def feed(self, n=1, allow=False, c_valid=True, gap=.16, stale=False, stamp_mismatch=False):
        for _ in range(n):
            self.clock += gap
            self.index += 1
            stamp = EvidenceStamp(self.index,self.clock,(480,640))
            # D在开工后可以拒识/报警，不替代E当前追踪判据。
            self.result = PrestartResult(stamp,
                {sid:PrestartSlot(part,part,'match' if allow else 'missing' if sid in self.empty else 'unknown')
                 for sid,part in self.net.expected.items()}, allow_start=allow, valid=c_valid)
            tracking = (EvidenceStamp(0,self.clock,(480,640)) if stamp_mismatch else stamp,
                        list(self.tracks.values()), self.polygons if c_valid else {})
            self.args = (self.result,self.clock+(2 if stale else 0),tracking)
            verdict = self.net.update(*self.args)
        return verdict

    def move(self, sid, outside=True, partial=True):
        track = self.tracks[sid]
        track.contour = self.origins[sid] + np.array([0,100 if outside else 0])
        track.state = 'tracked' if partial else 'detected'
        track.detail.update(visible_observed=1.0,partial_observed=float(partial))
        track.lost_frames = 0
        if outside:self.empty.add(sid)
        else:self.empty.discard(sid)


def main():
    s=Scene()
    for i,sid in enumerate(s.net.order,1):
        # C失去方向时A仍在同帧正常追踪；期间不推进，恢复后无需放回重取。
        s.move(sid)
        s.feed(5,c_valid=False)
        assert len(s.net.done)==i-1
        s.feed(3)
        assert len(s.net.done)==i and len(s.net.events)==i
        for _ in range(4):
            s.net.update(*s.args)
        assert len(s.net.done)==i and len(s.net.events)==i
    assert '推定完成' in s.net.verdict.title
    s.move(s.net.order[1],False)
    s.feed(3)
    assert len(s.net.done)==1
    s.net.reset()
    assert not s.net.started and not s.net.events and not s.net.tracking.keys

    # 遮挡不算消失：同一轨迹遮挡时不推进，重新可见离槽后确认。
    s=Scene();sid=s.net.order[0]
    s.tracks[sid].state='occluded';s.tracks[sid].detail['visible_observed']=0.
    s.empty.add(sid);s.feed(5)
    assert not s.net.done
    s.move(sid);s.feed(3)
    assert len(s.net.done)==1
    # 真实追踪离槽一旦发生，随后单件装入底座不可见，只需当前空槽稳定确认。
    s=Scene();sid=s.net.order[0];s.move(sid);s.feed()
    s.tracks[sid].state='occluded';s.tracks[sid].detail['visible_observed']=0.
    s.feed(3)
    assert len(s.net.done)==1 and len(s.net.events)==1
    for failure in ('lost','replaced','gap','stale','mismatch'):
        s=Scene();sid=s.net.order[0]
        if failure=='lost':
            s.tracks[sid].state=failure
            s.tracks[sid].detail['visible_observed']=0.
            s.feed(c_valid=False)
        elif failure=='replaced':
            s.tracks[sid]=copy.deepcopy(s.tracks[sid])
            s.move(sid)
            s.feed()
        elif failure=='gap':s.feed(gap=1.)
        elif failure=='stale':s.feed(stale=True)
        else:s.feed(stamp_mismatch=True)
        s.move(sid)
        s.feed(5)
        assert not s.net.done, failure
        s.move(sid,False);s.feed(3)
        s.move(sid);s.feed(3)
        assert len(s.net.done)==1, failure
    s=Scene();s.move(s.net.order[2]);s.feed(3)
    assert not s.net.done and s.net.verdict.level=='error'
    s.move(s.net.order[2],False);s.feed(3)
    s.move(s.net.order[0]);s.move(s.net.order[1]);s.feed(3)
    assert not s.net.done and s.net.verdict.level=='error'
    # 料盘整体移动有当前C几何补偿，不应变成四个取件。
    s=Scene()
    for sid in s.polygons:
        s.polygons[sid]+=np.array([100,100])
        s.tracks[sid].contour+=np.array([100,100])
    s.feed(4);assert not s.net.done
    print('E追踪：跨C未知、遮挡暂停、观测离槽后安装遮挡确认、四步/越序/多取/回退/复位及五种失效拒绝通过')
