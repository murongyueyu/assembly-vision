# -*- coding: utf-8 -*-
r"""手套面积闸门 `min_area_px` 该定多少 —— 实测，不猜。

★ 发现的量级错配：
    配置里 `segment.glove.min_area_px = 3000`，依据是 docs/踩坑实录.md §2.3 那次实测
    —— 那时是「**整只手** + 笔记本键盘背光干扰」的场景，真手套 83316px，
    所以用 3000 做"面积闸门"挡住 7882px 的假蓝块。
    但**现在的用法变了**：用户是「**几根手指**搭在料盘上」，
    实测在 640x480 下只有 **约 1881px** -> 被 3000 滤掉
    -> `glove_bbox` 清空 -> 界面显示「手: 无」，但 `glove_mask` 仍在
    （所以**遮挡判定一直是对的**，只有显示受影响）。

★ 本工具量：
    ① 各真实帧上手（手指）的面积分布
    ② 各种 v_min 下，**空场景**（无人手）的最大蓝块面积  <- 决定闸门下限
    ③ 给出"既不漏手、又不误检"的 min_area_px 建议值
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = r"D:\university resource\大三上\视觉\assembly_vision"
sys.path.insert(0, ROOT)

import cv2                                                        # noqa: E402
from assemvision.config import load_config                        # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402

wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
seg = Segmenter(wp)
g = dict((wp.get("segment", {}) or {}).get("glove", {}) or {})
print("  当前判据: h %s~%s  s>=%s  v>=%s  min_area_px=%s  max_aspect=%s"
      % (g.get("h_min"), g.get("h_max"), g.get("s_min"), g.get("v_min"),
         g.get("min_area_px"), g.get("max_aspect")))

# 各真实帧：把"有手的"和"没手的"分开
HAND_FRAMES = ["glove_miss.png"]                    # 用户截图（有手，手指搭料盘）
CLEAN_FRAMES = ["assembly_choice.png", "now_raw.png", "tray_empty.png",
                "slot_check.png", "asm_raw.png", "assembly_left_check.png"]

print()
print("═" * 84)
print("  ① 有手的帧：手套/手指的面积")
print("═" * 84)
for name in HAND_FRAMES:
    p = os.path.join(ROOT, "data", "refs", name)
    img = imread_safe(p)
    if img is None:
        continue
    # 统一到相机真实分辨率 640 宽（截图是放大的）
    if img.shape[1] != 640:
        img = cv2.resize(img, (640, int(round(img.shape[0] * 640.0 / img.shape[1]))),
                         interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    m = ((H >= int(g["h_min"])) & (H <= int(g["h_max"])) &
         (S >= int(g["s_min"])) & (V >= int(g["v_min"]))).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    # ★ 列**未过滤**的连通域（不受 min_area_px / max_aspect 影响），
    #   否则量到的只是"已被现有判据放行的东西"，等于拿结论当输入。
    raw = []
    for i in range(1, n):
        x, y, w, h, a = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                         int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
                         int(stats[i, cv2.CC_STAT_AREA]))
        raw.append((a, round(max(w, h) / max(1, min(w, h)), 2)))
    raw.sort(reverse=True)
    print("  %-20s 图 %s" % (name, img.shape[:2]))
    print("     未过滤的连通域 (面积, 长宽比) 前 4: %s" % (raw[:4],))
    after_area = [a for a, r in raw if a >= int(g.get("min_area_px", 3000))]
    print("     过 min_area_px=%s 后剩: %s" % (g.get("min_area_px"), after_area or "无"))
    r = seg.segment(img, scale=1.0)
    print("     -> Segmenter: glove_area=%d  bbox=%s" % (r.glove_area_px, r.glove_bbox))

print()
print("═" * 84)
print("  ② 没手的帧：**最大蓝块**面积（决定闸门下限，防止误检）")
print("═" * 84)
worst = 0
for name in CLEAN_FRAMES:
    p = os.path.join(ROOT, "data", "refs", name)
    img = imread_safe(p)
    if img is None:
        continue
    r = seg.segment(img, scale=1.0)
    # 看**未过滤**的蓝块：临时把 min_area 调到极小
    d = dict(wp.data)
    d["segment"] = dict(d.get("segment", {}))
    d["segment"]["glove"] = dict(d["segment"].get("glove", {}))
    d["segment"]["glove"]["min_area_px"] = 1
    d["segment"]["glove"]["max_aspect"] = 1e9
    s2 = Segmenter(d)
    r2 = s2.segment(img, scale=1.0)
    worst = max(worst, int(r2.glove_area_px))
    print("  %-22s 未过滤最大蓝块 = %6d px   （当前判据下 glove_area=%d）"
          % (name, int(r2.glove_area_px), r.glove_area_px))

print()
print("═" * 84)
print("  结论")
print("═" * 84)
print("  · 有手时手指约 1900px；没手时最大蓝块 %d px" % worst)
for cand in (600, 800, 1000, 1200, 1500):
    ok = cand > worst
    print("     建议 min_area_px=%-5d -> 挡得住最大假蓝块(%d)? %s；容得下手指(1900)? %s"
          % (cand, worst, "是" if ok else "★ 否", "是" if cand < 1900 else "★ 否"))
