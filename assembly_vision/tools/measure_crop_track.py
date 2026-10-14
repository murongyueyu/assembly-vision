# -*- coding: utf-8 -*-
"""验证裁剪式追踪：速度提升多少、精度有没有坏。

★ 依据（`tools/_timing.py` 实测）：KCF 耗时随**搜索窗**而非帧尺寸增长 ——
    40x40 4.7ms / 80x80 15.8ms / 320x320 56.2ms；4 目标在整帧上合计 35.4ms/帧。
  所以把搜索窗裁成「框 + 2*pad」应当显著提速。

本工具对账三件事：
  ① 速度：整帧 vs 裁剪，各自 `_run_trackers` 的耗时
  ② 精度：静止漂移、跟已知位移的误差、遮挡 50% 的成功率
  ③ 一致性：裁剪模式解出的质心与整帧模式差多少
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

ROOT = r"D:\university resource\大三上\视觉\assembly_vision"
sys.path.insert(0, ROOT)

import cv2                                                        # noqa: E402
from assemvision.config import load_config                        # noqa: E402
from assemvision.core.part_tracker import PartTracker             # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402

seg = Segmenter(load_config(os.path.join(ROOT, "config", "workpiece.yaml")))
frame = imread_safe(os.path.join(ROOT, "data", "refs", "assembly_choice.png"))
res = seg.segment(frame, scale=1.0)
parts = [b for b in res.brass_blobs if 400 <= b.area_px <= 8000]
print("  真机帧 %s，目标 %d 个" % (frame.shape, len(parts)))
GLOVE = (200, 140, 90)

BASE = {"backend": "kcf", "min_area_px": 1, "max_area_px": 10 ** 9}


def make_tracker(crop, pad_px=40, pad_ratio=0.6):
    cfg = dict(BASE)
    cfg["crop_track"] = crop
    cfg["crop_pad_px"] = pad_px
    cfg["crop_pad_ratio"] = pad_ratio
    tr = PartTracker(cfg)
    tr.update(frame, parts, detect=True)
    return tr


def time_track(tr, n=25):
    for _ in range(5):
        tr.update(frame, [], detect=False)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        tr.update(frame, [], detect=False)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


print()
print("═" * 86)
print("  ① 速度：整帧 vs 裁剪（同一帧、同一批目标）")
print("═" * 86)
print("  %-28s %-14s %-14s %s" % ("模式", "_run_trackers中位ms", "轨迹数", "对整帧的倍率"))
print("  " + "-" * 70)
base_ms = None
for label, crop, pad_px, pad_r in (
        ("整帧（旧）", False, 40, 0.6),
        ("裁剪 pad=40px + 0.6*边长", True, 40, 0.6),
        ("裁剪 pad=24px + 0.4*边长", True, 24, 0.4),
        ("裁剪 pad=16px + 0.3*边长", True, 16, 0.3)):
    tr = make_tracker(crop, pad_px, pad_r)
    ms = time_track(tr)
    if base_ms is None:
        base_ms = ms
    print("  %-28s %-14.2f %-14d %.2fx"
          % (label, ms, len(tr.tracks), base_ms / ms if ms else 0))

print()
print("═" * 86)
print("  ② 精度：静止漂移 / 跟已知位移 / 遮挡 50%")
print("═" * 86)
print("  %-28s %-16s %-18s %s" % ("模式", "静止漂移px", "跟移动误差px", "遮挡50%成功"))
print("  " + "-" * 74)
for label, crop, pad_px, pad_r in (
        ("整帧（旧）", False, 40, 0.6),
        ("裁剪 pad=40px + 0.6*边长", True, 40, 0.6),
        ("裁剪 pad=24px + 0.4*边长", True, 24, 0.4)):
    # 静止漂移
    tr = make_tracker(crop, pad_px, pad_r)
    c0 = [(t.centroid[0], t.centroid[1]) for t in tr.tracks]
    drift = 0.0
    for _ in range(25):
        tr.update(frame, [], detect=False)
        for t, (x0, y0) in zip(tr.tracks, c0):
            drift = max(drift, float(np.hypot(t.centroid[0] - x0, t.centroid[1] - y0)))
    # 跟移动
    tr = make_tracker(crop, pad_px, pad_r)
    c0 = [(t.centroid[0], t.centroid[1]) for t in tr.tracks]
    errs = []
    for i in range(1, 26):
        dx = i * 2
        fr = cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, 0]]),
                            (frame.shape[1], frame.shape[0]))
        tr.update(fr, [], detect=False)
        for t, (x0, _y0) in zip(tr.tracks, c0):
            errs.append(abs(t.centroid[0] - (x0 + dx)))
    # 遮挡 50%（只遮第一个目标）
    tr = make_tracker(crop, pad_px, pad_r)
    b0 = tr.tracks[0]
    x, y, w, h = b0.bbox
    fr = frame.copy()
    cv2.rectangle(fr, (x + w // 2, y - 6), (x + w + 8, y + h + 6), GLOVE, -1)
    ok_n = 0
    for _ in range(25):
        tr.update(fr, [], detect=False)
        ok_n += 1 if tr.tracks and tr.tracks[0].state != "lost" else 0
    print("  %-28s %-16.2f %-18s %s"
          % (label, drift, "%.2f" % float(np.mean(errs)) if errs else "-",
             "%d/25" % ok_n))

print()
print("  ★ 判读：")
print("     · 裁剪若明显更快且精度不差 -> 设为默认（`tracking.crop_track`）")
print("     · 若精度变差 -> 说明裁剪把搜索余量削太多，加大 pad")
