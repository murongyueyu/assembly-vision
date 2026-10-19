# -*- coding: utf-8 -*-
"""操作事件检测：取件事件、装配事件。

「取件事件」= 某槽由有件变无件，且那一刻手套正压在该槽上。
   两个条件同时成立才算 —— 只看"由有变无"，光照突变/阈值漂移也会造成；
   只看"手在槽上"，路过也算。合起来才是"手伸过去把零件拿走了"。
"""
from __future__ import annotations

import time
import math
from typing import Dict, List, Optional, Sequence

from ..types import Event, SlotResult, SpokeResult


class ConfirmedPickDetector:
    """E取件边沿：正确在槽到可靠空槽，手退出后才确认；不推测隐藏动作。"""

    def __init__(self, expected, settings):
        self.expected = dict(expected)
        self.frames = settings['confirm_frames']
        self.hold = float(settings['confirm_hold_s'])
        self.max_gap = float(settings['max_gap_s'])
        self.max_age = float(settings['max_age_s'])
        if (type(self.frames) is not int or self.frames < 1 or not all(math.isfinite(v) and v > 0
                for v in (self.hold, self.max_gap, self.max_age))):
            raise ValueError('E确认参数必须为有限正数')
        self.reset()

    def reset(self):
        self.stable = {}
        self.picked = set()
        self.pending = {}
        self.last = None
        self.signature = None
        self.settled = False

    def interrupt(self):
        # ★ 中断期间可能发生取放，不可把恢复后的空槽当成观测到的取件边沿。
        self.stable.clear()
        self.picked.clear()
        self.pending.clear()
        self.settled = False

    def update(self, result, now):
        stamp = result.stamp
        signature = tuple((sid, row.state, row.observed) for sid, row in sorted(result.slots.items()))
        age = now - stamp.captured_at
        if (not result.valid or set(result.slots) != set(self.expected)
                or not math.isfinite(age) or not 0 <= age <= self.max_age):
            self.interrupt()
            return []
        if self.last is not None:
            index, captured = self.last
            gap = stamp.captured_at - captured
            if stamp.index == index and stamp.captured_at == captured and signature == self.signature:
                return []
            if stamp.index <= index or gap <= 0:
                self.interrupt()
                return []
            if gap > self.max_gap:
                self.interrupt()
        self.last = (stamp.index, stamp.captured_at)
        self.signature = signature
        events = []
        for sid, row in result.slots.items():
            state = row.state
            if state not in ('match', 'missing'):
                self.pending.pop(sid, None)
                continue
            candidate, count, since = self.pending.get(sid, (state, 0, stamp.captured_at))
            if candidate != state:
                count, since = 0, stamp.captured_at
            count += 1
            self.pending[sid] = (state, count, since)
            if count >= self.frames and stamp.captured_at - since >= self.hold:
                previous = self.stable.get(sid)
                self.stable[sid] = state
                if previous is not None and previous != state:
                    if state == 'missing':
                        self.picked.add(sid)
                    else:
                        self.picked.discard(sid)
                    events.append(Event(ts=stamp.captured_at,
                        kind='pick' if state == 'missing' else 'return',
                        slot=sid, part=self.expected[sid],
                        detail={'frame_index': stamp.index, 'inferred': True}))
        self.settled = all(row.state in ('match', 'missing') and
                           self.stable.get(sid) == row.state and
                           sid in self.pending and self.pending[sid][1] >= self.frames and
                           stamp.captured_at-self.pending[sid][2] >= self.hold
                           for sid, row in result.slots.items())
        return events


class EventDetector:
    def __init__(self, min_gap_s: float = 0.6) -> None:
        self._slot_present: Dict[str, bool] = {}
        self._preds: Dict[str, bool] = {}
        self._spoke_count: Optional[int] = None
        self.min_gap_s = float(min_gap_s)
        self._last_ts: Dict[str, float] = {}

    def update(self, ts: float,
               slots: Sequence[SlotResult],
               hand_on: Dict[str, bool],
               preds: Optional[Dict[str, bool]] = None,
               spokes: Optional[SpokeResult] = None) -> List[Event]:
        events: List[Event] = []

        # 取件：有 → 无，且手套压在槽上
        for sr in slots:
            prev = self._slot_present.get(sr.slot_id)
            cur = bool(sr.present)
            if prev is True and cur is False and hand_on.get(sr.slot_id, False):
                events.append(Event(ts=ts, kind="pick", part=sr.part, slot=sr.slot_id,
                                    detail={"coverage": sr.coverage}))
            self._slot_present[sr.slot_id] = cur

        # 装配事件：谓词由假变真
        if preds:
            for k, v in preds.items():
                if v and not self._preds.get(k, False):
                    if self._throttle("assemble:" + k, ts):
                        events.append(Event(ts=ts, kind="assemble", detail={"predicate": k}))
            self._preds.update(preds)

        # 辐条数变化单独记一条（这是重点指标）
        if spokes is not None:
            if self._spoke_count is not None and spokes.count != self._spoke_count:
                events.append(Event(ts=ts, kind="assemble",
                                    detail={"spoke_count": spokes.count,
                                            "was": self._spoke_count}))
            self._spoke_count = spokes.count

        return events

    def _throttle(self, key: str, ts: float) -> bool:
        last = self._last_ts.get(key, -1e9)
        if ts - last < self.min_gap_s:
            return False
        self._last_ts[key] = ts
        return True
