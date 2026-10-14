# -*- coding: utf-8 -*-
"""验证手（蓝手套）方框的渲染 —— 用合成手套，不依赖相机。

★ 用户要求：「还要能识别工人的手，工人的手用方框框就可以了（戴蓝手套）」
★ 为什么单独测渲染：真机跑的时候画面里未必有手（实测就拍到过"四个零件都在、
  没有手"的帧，那时显示 `手: 无` 是**正确**的）。所以必须用**可控的手套**
  来验证"有手时方框真的画出来"，否则分不清"没画"和"没手"。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = r"D:\university resource\大三上\视觉\assembly_vision"
sys.path.insert(0, ROOT)

import cv2                                                        # noqa: E402
from assemvision.config import load_config                        # noqa: E402
from assemvision.core.checks import Qualifier                     # noqa: E402
from assemvision.core.part_tracker import PartTracker             # noqa: E402
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter, glove_hits        # noqa: E402
from assemvision.imgio import imread_safe, imwrite_safe           # noqa: E402

wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
cal = load_config(os.path.join(ROOT, "config", "calibration.yaml"))
tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
               ref_scale=float(cal.get("tray_map.scale", 1.0)))
seg = Segmenter(wp)
mmpp = tray.mm_per_px(1.0)

# 槽位多边形（glove_hits 要用它判"手压在哪个槽"）
slot_polys = {}
for s in (wp.get("slots") or []):
    try:
        p = tray.slot_polygon_px(s["id"], 1.0)
        if p is not None:
            slot_polys[s["id"]] = p
    except Exception:                                              # noqa: BLE001
        pass
print("  槽位多边形 %d 个: %s" % (len(slot_polys), sorted(slot_polys)))
print("  手套分割阈值: %s" % dict((wp.get("segment", {}) or {}).get("glove", {}) or {}))

base = imread_safe(os.path.join(ROOT, "data", "refs", "assembly_choice.png"))
print("  底图 %s" % (base.shape,))
tracker = PartTracker({"backend": "kcf", "min_area_px": 400, "max_area_px": 8000})

outs = []
# 造几种手套形态：大小不同、位置不同（含"压在某个槽上"）
CASES = [
    ("方块 90x70",    (330, 200, 90, 70)),
    ("方块 140x110",  (300, 180, 140, 110)),
    ("方块 200x150",  (280, 170, 200, 150)),
    ("细长 260x60",   (250, 300, 260, 60)),      # 会被 max_aspect 过滤
    ("小 40x40",      (400, 400, 40, 40)),       # 会被 min_area 过滤
]
print()
print("  %-14s %-10s %-12s %-10s %s"
      % ("用例", "手套px", "glove_bbox", "手框画了", "手压在"))
print("  " + "-" * 74)
for label, (gx, gy, gw, gh) in CASES:
    frame = base.copy()
    # 蓝手套色（与 config 的 h 95~130 / v>=130 一致）
    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (200, 140, 90), -1)
    r = seg.segment(frame, scale=1.0)
    hits = glove_hits(r.glove_mask, slot_polys) if slot_polys else {}
    canvas = frame.copy()
    drew = tracker.draw_hand(canvas, r.glove_bbox, r.glove_area_px,
                             hand_on=hits, mm_per_px=mmpp)
    hit = ",".join(k.replace("slot_", "") for k, v in hits.items() if v) or "-"
    print("  %-14s %-10d %-12s %-10s %s"
          % (label, int(r.glove_area_px),
             str(r.glove_bbox) if r.glove_bbox else "None",
             "✓" if drew else "✗", hit))
    outs.append(canvas)

# 拼一张对照图
h = min(o.shape[0] for o in outs)
strip = np.concatenate([cv2.resize(o, (int(o.shape[1] * h / o.shape[0]), h))
                        for o in outs], axis=1)
out = os.path.join(ROOT, "data", "refs", "hand_box_check.png")
imwrite_safe(out, strip)
print()
print("  已存对照图 -> %s" % out)
print("  ★ 判读：")
print("     · 方块 90x70 应被识别并画出**青色方框 + 四角加粗 + 标签**")
print("     · 细长 260x60 应被 max_aspect=6.0 过滤 -> 无框（这是正确的，避免把")
print("       绿色台面的窄条误判成手）")
print("     · 小 40x40 应被 min_area_px=3000 过滤 -> 无框（避免噪声）")
