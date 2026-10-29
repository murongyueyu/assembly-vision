# -*- coding: utf-8 -*-
"""跨帧跟踪 + 补帧。

用法：分割实际跑 10–15 fps，用这里的线性外推把框补到 25–30 fps 显示。
★ 补帧只动框的位置，不参与任何判定 —— 这点要写进文档，避免被质疑"补帧造假"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..types import Blob


@dataclass
class Track:
    id: int
    part: str = ""
    centroid: Tuple[float, float] = (0.0, 0.0)
    prev_centroid: Tuple[float, float] = (0.0, 0.0)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    area_px: float = 0.0
    age: int = 0
    missed: int = 0
    hits: int = 0

    def velocity(self) -> Tuple[float, float]:
        return (self.centroid[0] - self.prev_centroid[0],
                self.centroid[1] - self.prev_centroid[1])

    def projected(self, k: float) -> Tuple[float, float]:
        vx, vy = self.velocity()
        return (self.centroid[0] + vx * k, self.centroid[1] + vy * k)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class BlobTracker:
    """IoU 关联 + 上一帧位移外推。只跟踪最近活跃的几个目标，避免算力浪费。"""

    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 15,
                 center_gate_px: float = 120.0, max_tracks: int = 8) -> None:
        self.iou_thresh = float(iou_thresh)
        self.max_missed = int(max_missed)
        self.center_gate_px = float(center_gate_px)
        self.max_tracks = int(max_tracks)
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, blobs: Sequence[Blob], ts: float = 0.0) -> List[Track]:
        used = [False] * len(blobs)
        for tr in self.tracks:
            best, best_score = -1, 0.0
            for i, b in enumerate(blobs):
                if used[i]:
                    continue
                score = _iou(tr.bbox, b.bbox)
                d = float(np.hypot(b.centroid[0] - tr.centroid[0],
                                   b.centroid[1] - tr.centroid[1]))
                if score < self.iou_thresh and d > self.center_gate_px:
                    continue
                if score > best_score:
                    best, best_score = i, score
            if best >= 0:
                b = blobs[best]
                used[best] = True
                tr.prev_centroid = tr.centroid
                tr.centroid = b.centroid
                tr.bbox = b.bbox
                tr.area_px = b.area_px
                tr.age += 1
                tr.hits += 1
                tr.missed = 0
            else:
                tr.missed += 1

        for i, b in enumerate(blobs):
            if used[i]:
                continue
            if len(self.tracks) >= self.max_tracks:
                break
            self.tracks.append(Track(id=self._next_id, centroid=b.centroid,
                                     prev_centroid=b.centroid, bbox=b.bbox,
                                     area_px=b.area_px, age=1, hits=1))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return list(self.tracks)

    def projected(self, k: float) -> List[Tuple[int, Tuple[float, float]]]:
        """补帧用：返回 [(track_id, 外推中心), ...]。"""
        out = []
        for t in self.tracks:
            if t.hits >= 2:
                out.append((t.id, t.projected(k)))
            else:
                out.append((t.id, t.centroid))
        return out
