# -*- coding: utf-8 -*-
r"""修「手指碎成多块 -> 被 min_area_px 整只滤掉」的问题 —— 先量再改。

★ 实测（tools/measure_glove_area.py）：
    有手帧在 640x480 下的未过滤连通域: [(613,1.33), (502,2.04), (383,1.29), (278,1.27)]
    **手指碎成 4~5 块，没有一块超过 min_area_px=3000**
    -> 整只手被滤掉 -> glove_bbox=None -> 界面显示「手: 无」
    （但 glove_mask 仍在，所以**遮挡判定一直是对的**，只有显示受影响）
★ 而 6 个无手帧的未过滤最大蓝块 = **0 px**（一个蓝像素都没有）。

★ 所以正确修法是**两步**：
    ① 形态学**闭运算**把手指碎块连成一块（glove.morph_close 现在只有 5）
    ② 再把 min_area_px 降到与"连成一块后的手"匹配的量级
  本工具扫这两维，给出既有余量又不误检的组合。
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
g = dict((wp.get("segment", {}) or {}).get("glove", {}) or {})


def load(name, w=640):
    img = imread_safe(os.path.join(ROOT, "data", "refs", name))
    if img is None:
        return None
    if img.shape[1] != w:
        img = cv2.resize(img, (w, int(round(img.shape[0] * w / img.shape[1]))),
                         interpolation=cv2.INTER_AREA)
    return img


HAND = load("glove_miss.png")
CLEAN = [x for x in (load(n) for n in ("assembly_choice.png", "now_raw.png",
                                       "tray_empty.png", "slot_check.png",
                                       "asm_raw.png", "assembly_left_check.png"))
         if x is not None]
print("  有手帧 %s ；无手帧 %d 张" % (HAND.shape[:2], len(CLEAN)))


def largest_after(img, close_k, min_area):
    """按给定的形态学核与面积闸门，返回 (最大块面积, 块数, bbox)。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    m = ((H >= int(g["h_min"])) & (H <= int(g["h_max"])) &
         (S >= int(g["s_min"])) & (V >= int(g["v_min"]))).astype(np.uint8) * 255
    if int(g.get("morph_open", 3)) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(g["morph_open"]),) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    best = (0, (0, 0, 0, 0))
    cnt = 0
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        cnt += 1
        if a > best[0]:
            best = (a, (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                        int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])))
    return best[0], cnt, best[1]


print()
print("═" * 92)
print("  ① 闭运算核大小 对「手指连成一块」的效果（min_area 暂设 200 只看连通性）")
print("═" * 92)
print("  %-10s %-14s %-14s %s" % ("闭运算核", "最大块面积", "块数", "bbox"))
print("  " + "-" * 60)
for ck in (0, 5, 9, 15, 21, 31):
    a, c, bb = largest_after(HAND, ck, 200)
    print("  %-10s %-14d %-14d %s" % (ck if ck else "(不闭)", a, c, bb))

print()
print("═" * 92)
print("  ② 扫 (闭运算核, min_area_px)：有手要检出、无手要 0")
print("═" * 92)
print("  %-12s %-12s %-16s %s" % ("闭运算核", "min_area_px", "有手最大块", "无手最大块(应=0)"))
print("  " + "-" * 70)
for ck in (9, 15, 21):
    for ma in (400, 800, 1200, 1500, 2000):
        ah, _c, _b = largest_after(HAND, ck, ma)
        ac = 0
        for im in CLEAN:
            a2, _c2, _b2 = largest_after(im, ck, ma)
            ac = max(ac, a2)
        flag = "✓" if (ah > 0 and ac == 0) else ("★ 漏手" if ah == 0 else "★ 误检")
        print("  %-12d %-12d %-16d %-16d %s" % (ck, ma, ah, ac, flag))

print()
print("  ★ 判读：选「有手最大块 > 0」且「无手最大块 = 0」的组合，且留足余量。")
