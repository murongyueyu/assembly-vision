# -*- coding: utf-8 -*-
"""长度测量（2D 轮廓）：给"横放"的细长件量长度。

★ 为什么不用 B 的深度法（2026-10-28/29 实测）：
  立柱是 Ø15 的高反光细杆，D435 在杆身/杆端**一个有效深度点都没有**
  （40×20 px 区域有效点 = 0，周围满格），高度法只能拿到旁边的台面（3～6 mm）。
  横放之后"长度"完全在图像平面内 —— 用标定好的 mm/px 量金属掩膜的最长边即可，
  既不依赖深度，也不受杆身高反光影响。

★ 精度：GSD 0.86 mm/px 下 1 个像素 ≈ 0.86 mm，±3 mm 的窗口有 ~3.5 px 余量；
  因此掩膜边界宁可**少算**也不要多算（形态学不做外扩，只取最小外接矩形）。
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# 轮廓面积下限（像素²）：低于它认为测区里没有可量的零件
MIN_AREA_PX = 60.0
# 最小宽度（mm）：细长件的短边，低于它多半是断裂的碎片或是一条反光边
MIN_WIDTH_MM = 2.0


def measure_length(mask: np.ndarray, rect, mm_per_px: float) -> dict:
    """在 mask 的 rect 区域内取最大连通域的最小外接矩形，返回长/宽（mm）。

    rect 是 cv2.selectROI 的 (x, y, w, h)；mask 与坐标必须同源同尺寸。
    失败返回 ok=False + reason，绝不返回"看起来合理"的默认值。
    """
    result = dict(ok=False, length_mm=None, width_mm=None, angle_deg=None,
                  area_px=0.0, reason='', box_px=None)
    if mask is None or mask.ndim != 2:
        result['reason'] = '缺少同帧金属掩膜'
        return result
    if mm_per_px is None or not np.isfinite(mm_per_px) or mm_per_px <= 0:
        result['reason'] = '缺少有效 mm/px 尺度'
        return result
    try:
        x, y, w, h = [int(v) for v in rect]
    except (TypeError, ValueError):
        result['reason'] = '测区无效'
        return result
    if w <= 0 or h <= 0:
        result['reason'] = '测区为空'
        return result
    H, W = mask.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        result['reason'] = '测区越界'
        return result

    sub = (mask[y:y + h, x:x + w] > 0).astype(np.uint8)
    if int(sub.sum()) < MIN_AREA_PX:
        result['reason'] = '测区内金属像素太少'
        return result
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        result['reason'] = '测区内没有金属轮廓'
        return result
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < MIN_AREA_PX:
        result['reason'] = '金属轮廓过小（%.0f px²）' % area
        return result

    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    # ★ 像素中心 → 物理边缘：掩膜覆盖 N 个像素时真实跨度约 N 个像素，
    #   而 minAreaRect 给的是"最外侧像素中心之间"的距离（≈N-1），
    #   不补这一像素会系统性偏短（0.86 mm/px 下正好差 0.86 mm）。
    long_px, short_px = max(rw, rh) + 1.0, min(rw, rh) + 1.0
    length_mm = long_px * mm_per_px
    width_mm = short_px * mm_per_px
    result.update(length_mm=float(length_mm), width_mm=float(width_mm),
                  angle_deg=float(angle), area_px=area,
                  box_px=[int(x), int(y), int(w), int(h)])
    if width_mm < MIN_WIDTH_MM:
        result['reason'] = '轮廓过窄（宽 %.2f mm），可能是反光边或碎片' % width_mm
        return result
    result['ok'] = True
    result['reason'] = 'ok'
    return result
