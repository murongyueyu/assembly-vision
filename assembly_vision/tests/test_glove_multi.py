# -*- coding: utf-8 -*-
"""两只蓝手套同时入画 —— 钉住「不再只取最大连通域」这条改动（2026-10-27）。

★ 为什么必须有这一条：
  装配时**两手同时入画是常态**，而旧 `Segmenter.segment()` 的实现是
  `best = 最大块`，然后把 `glove_mask` 裁成只剩那一块。后果是双层的：
    ① 界面永远只有一个橙框 —— 第二只手套"消失"；
    ② `glove_mask` 被裁掉后，**遮挡判定也失去第二只手**
       （`glove_hits` / `tray_observer.max_glove_fraction` 吃的都是这个掩膜），
       另一只手压着料盘却判"没遮挡"。第 ② 条比第 ① 条隐蔽得多，所以本测试
       特意用**槽位遮挡**来验，而不只是数框。

本测试钉四件事：
  1. 两只手 -> `glove_bboxes` 长度 2，且按面积降序；
  2. `glove_mask` 是**并集**：偏小的那只手套所在 ROI 也必须判成被遮挡；
  3. 只有一只手时与旧版完全一致（长度 1、`glove_bbox` 非空）；
  4. `max_gloves: 1` 能整条退回旧行为（现场做 A/B 对比时用）。

另加一条负样本：低于 `min_area_px` 的蓝色噪点不能被算成第三只手 ——
否则"改成 2 只"就会变成"把反光也当手"。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check, ok, wp_config                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.core.segment import Segmenter, glove_hits      # noqa: E402

# 蓝手套的 HSV（与 tools/make_synthetic.py 的 GLOVE 同口径）：
#   H=110 落在配置的 h∈[91,130]，S=190 > s_min=80，V=200 > v_min=130
GLOVE_HSV = (110, 190, 200)
FRAME = (480, 640)          # h, w


def _glove_bgr() -> tuple:
    hsv = np.uint8([[[GLOVE_HSV[0], GLOVE_HSV[1], GLOVE_HSV[2]]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _frame(circles) -> np.ndarray:
    """黑底 + 若干蓝手套圆盘。circles = [(cx, cy, r), ...]"""
    img = np.zeros((FRAME[0], FRAME[1], 3), dtype=np.uint8)
    for cx, cy, r in circles:
        cv2.circle(img, (cx, cy), r, _glove_bgr(), thickness=cv2.FILLED)
    return img


def _square(cx: float, cy: float, half: float) -> np.ndarray:
    """以 (cx,cy) 为中心、半边长 half 的正方形 ROI 多边形。"""
    return np.array([[cx - half, cy - half], [cx + half, cy - half],
                     [cx + half, cy + half], [cx - half, cy + half]], dtype=np.float64)


def case_config_default() -> None:
    cfg = wp_config()
    seg = Segmenter(cfg)
    check(int(seg.glove.get("max_gloves", -1)) == 2,
          "配置 segment.glove.max_gloves 默认为 2")


def case_two_gloves() -> None:
    seg = Segmenter(wp_config())
    # 左大右小：面积降序必须稳定，`glove_bbox` 取最大那块
    big = (150, 240, 45)       # 面积 ~6362px
    small = (500, 240, 38)     # 面积 ~4536px
    res = seg.segment(_frame([big, small]))

    check(res.glove_count == 2, "两只手套 -> glove_count == 2")
    check(len(res.glove_bboxes) == 2, "两只手套 -> glove_bboxes 长度 2")
    check(res.glove_bbox is not None, "glove_bbox 仍非空（最大那块，老调用方兼容）")
    check(res.glove_bbox == res.glove_bboxes[0], "glove_bbox 等于列表中最大那块")
    check(res.glove_areas_px == sorted(res.glove_areas_px, reverse=True),
          "glove_areas_px 按面积降序")

    # 框的位置要各归各位：两个框的 x 中心分别落在左右两团附近
    cx0 = res.glove_bboxes[0][0] + res.glove_bboxes[0][2] / 2.0
    cx1 = res.glove_bboxes[1][0] + res.glove_bboxes[1][2] / 2.0
    check(abs(cx0 - big[0]) < 8 and abs(cx1 - small[0]) < 8,
          "两个框分别落在左右两只手套上（不重叠、不串位）")

    # ★ 最关键的一条：并集掩膜
    mask = res.glove_mask
    check(mask is not None and np.any(mask), "glove_mask 非空")
    for cx, cy, r in (big, small):
        area_px = int(np.count_nonzero(mask[cy - 3:cy + 4, cx - 3:cx + 4]))
        check(area_px == 49, f"glove_mask 覆盖 ({cx},{cy}) 处的手套（{area_px}/49 像素）")

    hits = glove_hits(mask, {"near_big": _square(big[0], big[1], 40),
                             "near_small": _square(small[0], small[1], 34)},
                      thresh=0.10)
    check(hits["near_big"], "较大的那只手套判为遮挡")
    check(hits["near_small"],
          "★ 较小的那只手套同样判为遮挡（旧实现此处为 False：掩膜被裁成只剩最大块）")


def case_single_glove() -> None:
    seg = Segmenter(wp_config())
    res = seg.segment(_frame([(320, 240, 45)]))
    check(res.glove_count == 1, "只有一只手 -> glove_count == 1（与旧版一致）")
    check(len(res.glove_bboxes) == 1, "只有一只手 -> glove_bboxes 长度 1")
    check(res.glove_bbox is not None and res.glove_area_px > 0,
          "只有一只手 -> glove_bbox / glove_area_px 照旧可用")


def case_switch_back_to_one() -> None:
    """max_gloves=1 必须能整条退回旧行为 —— 现场 A/B 对比的开关。"""
    seg = Segmenter(wp_config())
    seg.glove["max_gloves"] = 1
    res = seg.segment(_frame([(150, 240, 45), (500, 240, 38)]))
    check(res.glove_count == 1, "max_gloves=1 -> 只认一只")
    check(res.glove_bboxes and res.glove_bboxes[0][0] < 320,
          "max_gloves=1 -> 保留的是面积更大的左边那只")
    right = int(np.count_nonzero(res.glove_mask[:, 400:]))
    check(right == 0, f"max_gloves=1 -> 掩膜里不含右边那只（残留 {right} px）")


def case_noise_rejected() -> None:
    """低于 min_area_px 的蓝噪点不能凑成"第二只手"。"""
    seg = Segmenter(wp_config())
    small_px = int(seg.glove.get("min_area_px", 2000))
    # r=15 的圆盘面积 ~707px，远低于闸门；r=45 的 ~6362px 一定过
    res = seg.segment(_frame([(150, 240, 45), (500, 240, 15)]))
    check(res.glove_count == 1,
          f"面积低于闸门({small_px}px)的蓝点被滤掉 -> 仍只有 1 只")


def main() -> int:
    print("== 手套数量配置 ==")
    case_config_default()
    print("== 两只手套 ==")
    case_two_gloves()
    print("== 单只手套（回归） ==")
    case_single_glove()
    print("== 退回单只（开关） ==")
    case_switch_back_to_one()
    print("== 噪声不得凑数 ==")
    case_noise_rejected()
    ok("test_glove_multi 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
