# -*- coding: utf-8 -*-
"""D独立判据：身份空间绑定、当前证据与新采集帧稳定门禁。"""
import copy
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from assemvision.config import load_config
from assemvision.core.checks import (EvidenceStamp, IdentityObject, PrestartGate,
                                    evaluate_prestart)
from assemvision.core.tray_observer import TrayObservation

ROOT = Path(__file__).resolve().parents[1]


def rect(x, y, w=10, h=10):
    return np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], 'int32')


def fixture():
    settings = load_config(str(ROOT / 'config/prestart_check.yaml')).data
    expected = {s['id']: s['part'] for s in load_config(str(ROOT/'config/workpiece.yaml')).get('slots')}
    polygons = {sid: rect(10 + i*40, 10, 30, 30) for i, sid in enumerate(expected)}
    objects = [IdentityObject(pid, rect(15+i*40, 15), True) for i, pid in enumerate(expected.values())]
    observation = TrayObservation(found=True, orientation_ok=True, pose=SimpleNamespace(ok=True),
        slots={sid: {'state': 'occupied', 'glove_fraction': 0.0} for sid in expected})
    return settings, expected, polygons, objects, observation


def main():
    cfg, expected, polys, objects, obs = fixture()
    stamp = EvidenceStamp(1, 10., (80, 200))
    def run(objects=objects, obs=obs, a=stamp, c=stamp, now=10.1, error=''):
        return evaluate_prestart(expected, polys, obs, objects, a, c, now, cfg, error)
    for order in permutations(expected.values()):
        candidate = copy.deepcopy(objects)
        for obj, pid in zip(candidate, order):
            obj.part = pid
        result = run(candidate)
        assert result.valid and not result.allow_start
        assert (result.state == 'ready') == (tuple(order) == tuple(expected.values()))
        for slot, pid in zip(result.slots.values(), order):
            assert slot.observed == pid
            assert slot.state == ('match' if slot.expected == pid else 'wrong')
    for i, sid in enumerate(expected):
        empty = copy.deepcopy(obs)
        empty.slots[sid]['state'] = 'empty'
        candidate = [o for j, o in enumerate(objects) if i != j]
        assert run(candidate, empty).slots[sid].state == 'missing'
        # 正确物体在盘外或另一槽都不能补本槽。
        external = copy.deepcopy(objects[i]); external.contour = rect(180, 60)
        assert run(candidate+[external], empty).slots[sid].state == 'missing'
        assert run(objects, empty).slots[sid].state == 'unknown'
        candidate = copy.deepcopy(objects); candidate[i].confirmed = False
        assert run(candidate).slots[sid].state == 'unknown'
        candidate[i].part = ''
        assert run(candidate).slots[sid].state == 'unknown'
        candidate[i].current = False
        assert run(candidate).slots[sid].state == 'unknown'
        hand = copy.deepcopy(empty); hand.slots[sid]['glove_fraction'] = .001
        assert run([], hand).slots[sid].state == 'unknown'
    duplicate = copy.deepcopy(objects[0]); duplicate.contour = rect(180, 60)
    assert run(objects+[duplicate]).slots[next(iter(expected))].state == 'unknown'
    extra = IdentityObject('', rect(28, 28, 3, 3))
    assert run(objects+[extra]).slots[next(iter(expected))].state == 'unknown'
    spanning = IdentityObject(objects[0].part, rect(15, 15, 70, 10), True)
    result = run([spanning]+objects[2:])
    assert all(s.state == 'unknown' for s in list(result.slots.values())[:2])
    outside = copy.deepcopy(objects); outside[0].contour = rect(180, 60)
    assert run(outside).state == 'wait'
    wrong = copy.deepcopy(objects); wrong[0].part = 'P_wrong'
    wrong[1].confirmed = False
    result = run(wrong)
    assert result.state == 'alarm' and list(result.slots.values())[1].state == 'unknown'
    band = copy.deepcopy(obs)
    sid = next(iter(expected))
    band.slots[sid].update(state='unknown', occupancy_fraction=.08, evidence='metal_or_brass')
    assert run(wrong, band).slots[sid].state == 'wrong'
    assert run(objects, band).slots[sid].state == 'unknown'
    band.slots[sid]['glove_fraction'] = .01
    assert run(wrong, band).slots[sid].state == 'unknown'
    # 大件居中充分覆盖小槽可报错放，中心在槽外或跨槽均不能归属。
    oversized = copy.deepcopy(objects)
    oversized[0] = IdentityObject('P_wrong', rect(2,2,46,46), True)
    assert run(oversized).slots[sid].state == 'wrong'
    offcenter = copy.deepcopy(oversized)
    offcenter[0].contour = rect(0,0,25,40)
    assert run(offcenter).slots[sid].state == 'unknown'
    for bad in (None, copy.deepcopy(obs)):
        if bad is not None:
            if bad is not None and bad.orientation_ok:
                bad.orientation_ok = False
        assert not run(obs=bad).valid
    old = copy.deepcopy(obs); old.observed_current = False
    assert not run(obs=old).valid
    for args in ({'c': EvidenceStamp(2,10.,(80,200))}, {'c': EvidenceStamp(1,10.01,(80,200))},
                 {'now': 12.}, {'now': 9.}, {'now': float('nan')}, {'error':'处理异常'}):
        result = run(**args)
        assert not result.valid and result.state == 'wait' and not result.allow_start
    gate = PrestartGate(expected, cfg)
    def step(index, ts, objects=objects, obs=obs):
        s = EvidenceStamp(index, ts, (80,200))
        raw = run(objects, obs, s, s, ts+.01)
        return gate.update(raw, ts+.01)
    first = step(1,10.)
    for _ in range(20):
        assert step(1,10.).stable_frames == 1
    assert not first.allow_start
    assert not step(2,10.16).allow_start
    assert step(3,10.32).allow_start
    assert not step(4,10.48,wrong).allow_start
    assert not step(5,10.64).allow_start
    assert not step(6,10.80).allow_start
    assert step(7,10.96).allow_start
    assert not gate.update(gate.last, 13.).allow_start
    assert not step(8,13.01).allow_start
    assert not step(9,13.17).allow_start
    assert step(10,13.34).allow_start
    assert not step(9,13.35).allow_start
    assert not step(11,13.34).allow_start  # 非递增采集时间清旧。
    assert not step(12,13.50).allow_start
    assert not step(13,13.66).allow_start
    assert step(14,13.83).allow_start
    assert not step(15,14.5).allow_start  # 断档重计。
    empty = copy.deepcopy(obs)
    sid = next(iter(expected)); empty.slots[sid]['state'] = 'empty'
    for idx in range(16,19):
        result = step(idx,14.5+(idx-15)*.1,objects[1:],empty)
        assert (result.state == 'alarm') == (idx == 18)
    rejected = copy.deepcopy(obs); rejected.orientation_ok = False
    assert not step(19,15.,obs=rejected).allow_start
    for idx in range(20,23):
        result = step(idx,15.1+(idx-20)*.16)
    assert result.allow_start
    print('D判据：24排列、分别缺件、全局借件防护、未知/遮挡/重复/跨槽、同帧/时效/恢复通过')


if __name__ == '__main__':
    main()
