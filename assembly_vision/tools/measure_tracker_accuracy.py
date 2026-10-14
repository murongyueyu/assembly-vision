# -*- coding: utf-8 -*-
"""追踪器精度对比：KCF vs MOSSE —— 换 MOSSE 省 250 倍时间，精度掉多少？

★ 背景（真机实测）：
    `_run_trackers` 轨迹数=1 -> 5.5ms，=4 -> 32.7ms（KCF 每目标约 5.5ms）
    而 MOSSE 单次只要 **0.022ms**（快 250 倍）。
    真机 demo 因此只有 6.8fps（均值被预热拉低，中位 31.7fps）。
  -> 该不该换，取决于**精度**。本工具就在真机帧上量：
       ① 静止时追踪框漂不漂（漂移量）
       ② 已知位移时跟不跟得上（跟丢率）
       ③ 遮挡后能不能恢复
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = r"D:\university resource\大三上\视觉\assembly_vision"
sys.path.insert(0, ROOT)

import cv2                                                        # noqa: E402
from assemvision.config import load_config                        # noqa: E402
from assemvision.core.part_tracker import PartTracker, make_tracker   # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402

seg = Segmenter(load_config(os.path.join(ROOT, "config", "workpiece.yaml")))
frame = imread_safe(os.path.join(ROOT, "data", "refs", "assembly_choice.png"))
res = seg.segment(frame, scale=1.0)
parts = [b for b in res.brass_blobs if 400 <= b.area_px <= 8000]
print("  真机参考帧: %s，目标 %d 个" % (frame.shape, len(parts)))

GLOVE = (200, 140, 90)


def test_drift(kind, box, n=40):
    """① 画面静止时追踪框漂多少（漂移越小越好）。"""
    tr, api = make_tracker(kind)
    if tr is None:
        return None, None
    tr.init(frame, box)
    cx0, cy0 = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
    drift = 0.0
    fail = 0
    for _ in range(n):
        try:
            ok, b = tr.update(frame)
        except Exception:                                          # noqa: BLE001
            fail += 1
            continue
        if not ok:
            fail += 1
            continue
        cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
        drift = max(drift, float(np.hypot(cx - cx0, cy - cy0)))
    return drift, fail


def test_move(kind, box, step=3, n=40):
    """② 已知位移（每帧右移 step px）时跟不跟得上。"""
    tr, api = make_tracker(kind)
    if tr is None:
        return None, None
    tr.init(frame, box)
    cx0 = box[0] + box[2] / 2.0
    errs = []
    fail = 0
    for i in range(1, n + 1):
        dx = i * step
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        fr = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        try:
            ok, b = tr.update(fr)
        except Exception:                                          # noqa: BLE001
            fail += 1
            continue
        if not ok:
            fail += 1
            continue
        errs.append(abs((b[0] + b[2] / 2.0) - (cx0 + dx)))
    return (float(np.mean(errs)) if errs else None), fail


def test_occlude(kind, box, frac=0.5, n=30):
    """③ 遮挡 50% 后还能不能跟（丢了算失败）。"""
    tr, api = make_tracker(kind)
    if tr is None:
        return None
    tr.init(frame, box)
    x, y, w, h = box
    ow = int(w * frac)
    fr = frame.copy()
    cv2.rectangle(fr, (x + w - ow, y - 6), (x + w + 8, y + h + 6), GLOVE, -1)
    ok_n = 0
    for _ in range(n):
        try:
            ok, _b = tr.update(fr)
        except Exception:                                          # noqa: BLE001
            continue
        ok_n += 1 if ok else 0
    return ok_n


print()
print("═" * 88)
print("  追踪器精度对比（真机帧；KCF vs MOSSE）")
print("═" * 88)
boxes = [cv2.boundingRect(b.contour) for b in parts[:4]]
for kind in ("kcf", "mosse"):
    tr, api = make_tracker(kind)
    print()
    print("  ── %s（%s）──" % (kind.upper(), api or "不可用"))
    if tr is None:
        print("     不可用")
        continue
    print("     %-8s %-14s %-16s %s" % ("目标", "静止漂移px", "跟移动(均误差px/丢)", "遮挡50%后成功"))
    for i, box in enumerate(boxes):
        d, f1 = test_drift(kind, box)
        m, f2 = test_move(kind, box)
        occ = test_occlude(kind, box)
        print("     #%-7d %-14s %-16s %s"
              % (i + 1,
                 "%.2f(丢%d)" % (d, f1) if d is not None else "-",
                 "%.1f(丢%d)" % (m, f2) if m is not None else "-",
                 "%d/30" % occ if occ is not None else "-"))

print()
print("  ★ 判读：")
print("     · MOSSE 静止漂移若明显大于 KCF -> 界面上的轮廓会抖（用户明确要求去抖）")
print("     · 跟移动误差大 / 丢得多 -> 不能用它做长时间跟手")
print("     · 若 MOSSE 精度可接受 -> 默认换 MOSSE，帧率从 ~7fps 提到 30fps 满帧")
print("     · 若不可接受 -> 保留 KCF，但把 detect_every 调大以摊薄，或只在少量目标时用 KCF")
