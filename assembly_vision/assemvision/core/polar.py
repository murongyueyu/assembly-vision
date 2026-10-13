# -*- coding: utf-8 -*-
"""极坐标展开与辐条计数 —— 本项目唯一的重点指标。

做法：以轮盘中心为极点，在"球头端所在的那个半径环带"里，沿 0°→360° 统计
金属覆盖率，得到一条曲线；完整轮盘应该有 8 个连通峰。
   * 数的是"连通高于阈值的段"个数，不是局部极大值 —— 辐条根部容易连成一片
   * 阈值自适应：中值 + k×MAD，比固定阈值抗光照与反光差异
   * 多半径投票：避免某个半径恰好遇到遮挡或高光
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..types import SpokeResult


def polar_profile(mask: np.ndarray, center_px: Sequence[float],
                  r_min_px: float, r_max_px: float,
                  n_angles: int = 720, n_radii: int = 25) -> np.ndarray:
    """返回形状 (n_angles,) 的覆盖率曲线：每个角度上、半径环带内的前景占比。"""
    h, w = mask.shape[:2]
    th = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    rs = np.linspace(float(r_min_px), float(r_max_px), max(2, int(n_radii)))
    TH, RS = np.meshgrid(th, rs)
    cx, cy = float(center_px[0]), float(center_px[1])
    xi = np.clip(np.round(cx + RS * np.cos(TH)).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(cy + RS * np.sin(TH)).astype(np.int32), 0, h - 1)
    vals = mask[yi, xi] > 0
    return vals.mean(axis=0).astype(np.float64)


def _adaptive_threshold(curve: np.ndarray, k_mad: float) -> float:
    """自适应阈值：中值 + k×MAD，并且**加一个上限**。

    ★ 两个必须防的退化情形：
      1) MAD 几乎为 0（曲线大部分角度是同一个值），直接用标准差当尺度会把阈值
         推到峰顶之上，什么都数不出来 → 退回用高分位数定尺度。
      2) 无论哪种尺度，阈值都不能超过「中值到峰顶距离」的 60% —— 否则一旦峰值
         一侧不够高，整圈就全被判成背景。
    """
    med = float(np.median(curve))
    mad = float(np.median(np.abs(curve - med)))
    peak = float(np.percentile(curve, 95.0))
    if (peak - med) < 0.10:
        return float("inf")                  # 这一圈根本没有辐条

    if mad < 1e-3:
        scale = (float(np.percentile(curve, 90.0)) - med) / 1.4826
    else:
        scale = mad
    thr = med + float(k_mad) * 1.4826 * max(scale, 1e-6)
    return min(thr, med + 0.6 * (peak - med))


def count_peaks(curve: np.ndarray, k_mad: float = 3.0,
                min_width_deg: float = 6.0) -> List[Tuple[float, float, float]]:
    """数覆盖率曲线上的连通峰。

    返回 [(中心角度°, 峰宽°, 峰顶均值), ...]，角度已在 0..360 内。
    """
    n = len(curve)
    if n < 8:
        return []
    step_deg = 360.0 / n
    thr = _adaptive_threshold(curve, k_mad)
    if not np.isfinite(thr):
        return []

    b = curve > thr
    if not b.any():
        return []
    if b.all():                      # 整圈都高于阈値：退化成"一片"，不当作峰
        return [(0.0, 360.0, float(curve.mean()))]

    # 旋转到以 False 开头，把环形问题变成线性问题
    shift = 0
    if b[0]:
        shift = int(np.argmin(b))
        b = np.roll(b, -shift)
        curve = np.roll(curve, -shift)

    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if b[i]:
            j = i
            while j < n and b[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1

    out: List[Tuple[float, float, float]] = []
    for s, e in runs:
        width_deg = (e - s) * step_deg
        if width_deg < float(min_width_deg):
            continue
        center_deg = ((s + e) / 2.0 * step_deg - shift * step_deg) % 360.0
        out.append((center_deg, width_deg, float(curve[s:e].mean())))
    out.sort(key=lambda t: t[0])
    return out


def count_spokes(mask: np.ndarray, center_px: Sequence[float],
                 r_min_px: float, r_max_px: float,
                 required: int = 8, n_radii_bands: int = 3,
                 k_mad: float = 3.0, min_width_deg: float = 6.0,
                 n_angles: int = 720,
                 baseline_angles_deg: Optional[Sequence[float]] = None) -> SpokeResult:
    """多半径投票数辐条，并指出缺失角度。"""
    res = SpokeResult(count=0, required=int(required))
    r0, r1 = float(r_min_px), float(r_max_px)
    if r1 <= r0 or mask is None or mask.size == 0:
        return res

    bands = int(max(1, n_radii_bands))
    edges = np.linspace(r0, r1, bands + 1)
    counts: List[int] = []
    peaks_per_band: List[List[Tuple[float, float, float]]] = []
    for k in range(bands):
        prof = polar_profile(mask, center_px, edges[k], edges[k + 1],
                             n_angles=n_angles, n_radii=9)
        pk = count_peaks(prof, k_mad=k_mad, min_width_deg=min_width_deg)
        counts.append(len(pk))
        peaks_per_band.append(pk)
        if k == bands // 2:
            res.curve = prof

    # 投票：取出现次数最多的计数。
    # ★ 先剔除"一个峰都没数到"的环带 —— 那说明这个半径上根本没有对比度
    #   （例如环带选到了整圈都是金属的位置），不能让它投票说"有 0 根辐条"。
    valid = [c for c in counts if c > 0]
    pool = valid if valid else counts
    vals, freq = np.unique(np.array(pool), return_counts=True)
    # 并列时取较小值（宁少不多，避免漏报）
    best = int(vals[int(np.argmax(freq))])
    if freq.max() > 1:
        best = int(min(vals[freq == freq.max()]))
    res.count = best

    # 报告用的峰角度：选与多数一致的、峰最多的那个半径带
    cand = [pb for c, pb in zip(counts, peaks_per_band) if len(pb) == best]
    pk = max(cand, key=len) if cand else peaks_per_band[bands // 2]
    res.peaks_deg = [round(p[0], 1) for p in pk]

    # 指出缺失角度：用第一个峰做相位参考，生成理想均布位置，找没有峰靠近的
    if best > 0 and best < required:
        if baseline_angles_deg:
            ideal = list(baseline_angles_deg)
        else:
            base = res.peaks_deg[0]
            ideal = [(base + i * 360.0 / required) % 360.0 for i in range(required)]
        found = list(res.peaks_deg)
        for ang in ideal:
            if not any(_ang_diff(ang, f) < (360.0 / required) * 0.5 for f in found):
                res.missing_deg.append(round(ang, 1))

    res.ok = (best == int(required))
    return res


def _ang_diff(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def spoke_band_px(mm_per_px: float, r_min_mm: float, r_max_mm: float) -> Tuple[float, float]:
    """把毫米的半径环带换算成像素（用实测 GSD）。"""
    if mm_per_px <= 0:
        raise ValueError("mm_per_px 必须为正")
    return (float(r_min_mm) / float(mm_per_px), float(r_max_mm) / float(mm_per_px))
