# -*- coding: utf-8 -*-
"""形状特征：全部与位置无关；朝向相关的那几个只在『需要判方向』时才用。

注意取舍：既然料盘槽位与朝向几乎固定，完整性判定的主线可以是
「对齐后比差集/IoU」；但如果实际摆放有偏转，就用下面的旋转不变特征兜底。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def min_area_rect(contour: np.ndarray) -> Tuple[float, float, float]:
    """最小外接矩形，返回 (宽, 高, 角度)。宽高已按 长>=宽 归一。"""
    (_, _), (w, h), ang = cv2.minAreaRect(contour)
    if w < h:
        w, h = h, w
        ang += 90.0
    return float(w), float(h), float(ang % 180.0)


def fit_circle(pts: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float, float]:
    """Kasa 代数圆拟合。返回 ((cx,cy), r, rms残差)。点不足时返回 (None, 0, inf)。"""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(p) < 5:
        return None, 0.0, float("inf")
    x, y = p[:, 0], p[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, 0.0, float("inf")
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = float(np.sqrt(max(0.0, sol[2] + cx * cx + cy * cy)))
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rms = float(np.sqrt(np.mean((d - r) ** 2)))
    return (float(cx), float(cy)), r, rms


def count_holes(contour: np.ndarray, mask: Optional[np.ndarray] = None,
                min_area_px: float = 3.0) -> int:
    """数轮廓内部的孔洞个数。

    孔数变化是「完整性」最灵敏的判据之一（圆环少了内孔、圆盘多了裂纹都靠它）。
    mask 给了就直接用它（最准）；没给就只画轮廓线再洪泛填充找封闭区。
    ★ 注意不能画成填充轮廓 —— 那样孔就被填掉了，什么都数不出来。
    """
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return 0

    if mask is not None:
        sub = np.ascontiguousarray(mask[y:y + h, x:x + w])
    else:
        pad = 3
        sub = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
        shifted = contour.reshape(-1, 2) - np.array([x - pad, y - pad], dtype=contour.dtype)
        cv2.drawContours(sub, [shifted.reshape(-1, 1, 2)], -1, 255, 2)   # 只画轮廓线
        ff = sub.copy()
        m2 = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
        cv2.floodFill(ff, m2, (0, 0), 255)          # 从边界灌满外部
        sub = ((ff == 0) & (sub == 0)).astype(np.uint8) * 255   # 剩下没灌到的就是孔

    found = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = found[-2]
    if len(found) < 2 or found[-1] is None:
        return 0
    hier = found[-1][0]
    n = 0
    for i in range(len(contours)):
        if hier[i][3] != -1 and cv2.contourArea(contours[i]) >= min_area_px:
            n += 1
    return n


def straightness(contour: np.ndarray, n_bins: int = 20) -> float:
    """弯曲度：沿主轴把轮廓切成若干段，取每段的中心线，量中心线偏离直线的最大量，
    再除以长度。0 = 完全直；值越大越弯。

    ★ 为什么不直接用"轮廓点到拟合线的最大垂距"：对一根直杆，那个量约等于
      半宽/长度（例如 15mm 宽、85mm 长的立柱会得到 0.088），看起来像弯的。
      量中心线才能把"粗细"和"弯曲"分开。
    """
    p = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(p) < 12:
        return 0.0
    # 主轴用 PCA（比 minAreaRect 对弯曲形状更稳）
    d = p - p.mean(axis=0)
    cov = d.T @ d / max(1, len(p))
    try:
        _evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return 0.0
    u = evecs[:, -1]                 # 最大特征值方向 = 主轴
    v = evecs[:, 0]
    tu = d @ u
    tv = d @ v
    lo, hi = float(tu.min()), float(tu.max())
    span = hi - lo
    if span < 1e-6:
        return 0.0

    edges = np.linspace(lo, hi, int(max(4, n_bins)) + 1)
    idx = np.clip(np.digitize(tu, edges) - 1, 0, len(edges) - 2)
    ub, vb = [], []
    for k in range(len(edges) - 1):
        sel = idx == k
        if int(sel.sum()) < 2:
            continue
        ub.append(float(tu[sel].mean()))
        vb.append(float(tv[sel].mean()))
    if len(ub) < 4:
        return 0.0
    ub_a = np.asarray(ub)
    vb_a = np.asarray(vb)
    A = np.stack([ub_a, np.ones_like(ub_a)], axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, vb_a, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    resid = vb_a - A @ sol
    return float(np.abs(resid).max() / span)


def end_width_ratio(contour: np.ndarray, frac: float = 0.2) -> float:
    """两端宽度比：沿主轴两端各取 1/frac 段，比较两端的垂直跨度（大/小）。

    辐条一端球头、一端螺纹 —— 球头端明显更宽，所以这个比值能判「装反」。
    完全对称的零件该值接近 1。

    ★ 主轴用 PCA 定，不要用 cv2.minAreaRect 的角度：OpenCV 不同版本里那个角
      可能是短轴方向，会让整段退化（实测会稳定返回 1.00，看起来"很正常"）。
    """
    p = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(p) < 12:
        return 1.0
    d = p - p.mean(axis=0)
    cov = d.T @ d / max(1, len(p))
    try:
        _evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return 1.0
    u = evecs[:, -1]
    v = evecs[:, 0]
    tu = d @ u
    tv = d @ v
    lo, hi = float(tu.min()), float(tu.max())
    span = hi - lo
    if span < 1e-6:
        return 1.0
    k = max(0.05, min(0.49, float(frac)))
    lo_sel = tu <= lo + k * span
    hi_sel = tu >= hi - k * span
    if not lo_sel.any() or not hi_sel.any():
        return 1.0
    w_lo = float(np.ptp(tv[lo_sel]))
    w_hi = float(np.ptp(tv[hi_sel]))
    lo_w, hi_w = min(w_lo, w_hi), max(w_lo, w_hi)
    return float(hi_w / lo_w) if lo_w > 1e-6 else 1.0


def contour_features(contour: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """一次算出全部特征。数值单位是像素，比例类无量纲。"""
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    rw, rh, ang = min_area_rect(contour)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    holes = count_holes(contour, mask)

    return {
        "area_px": area,
        "perimeter_px": peri,
        # 圆度：正圆=1，正方形≈0.785
        "circularity": float(4.0 * np.pi * area / (peri * peri)) if peri > 1e-6 else 0.0,
        # 凸包实度：完整凸形状≈1，有缺口会下降
        "solidity": float(area / hull_area) if hull_area > 1e-6 else 0.0,
        "rect_w": rw,
        "rect_h": rh,
        "aspect": float(rw / rh) if rh > 1e-6 else 0.0,
        "rect_angle": ang,
        "holes": float(holes),
        "straightness": straightness(contour),
        "end_width_ratio": end_width_ratio(contour),
        "elongation": float(max(rw, rh) / max(1e-6, min(rw, rh))),
    }


def diff_area(a: np.ndarray, b: np.ndarray) -> float:
    """两个掩膜差集的面积（像素）。用于「与标准件比差集」这条主线。"""
    if a is None or b is None:
        return 0.0
    d = cv2.absdiff(a, b)
    return float(np.count_nonzero(d))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    inter = float(np.count_nonzero(cv2.bitwise_and(a, b)))
    union = float(np.count_nonzero(cv2.bitwise_or(a, b)))
    return inter / union if union > 0 else 0.0
