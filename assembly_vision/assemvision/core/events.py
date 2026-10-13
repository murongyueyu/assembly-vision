# -*- coding: utf-8 -*-
"""操作事件检测：取件事件、装配事件。

「取件事件」= 某槽由有件变无件，且那一刻手套正压在该槽上。
   两个条件同时成立才算 —— 只看"由有变无"，光照突变/阈值漂移也会造成；
   只看"手在槽上"，路过也算。合起来才是"手伸过去把零件拿走了"。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

from ..types import Event, SlotResult, SpokeResult


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
