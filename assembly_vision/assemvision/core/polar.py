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


def spoke_jitter_deg(peaks, expected: int) -> float:
    """峰角度的**角度均匀度**：间隔相对 360/expected 的抖动（度）。

    越小越像"真的均布辐条"。返回 `inf` 表示峰数不对或不足 2 个峰。
    ★ 与 `c5_assembly_baseline.sweep_band` 用的是同一个判据
      （那边实测：真辐条抖动 5.9°，凑出来的 8 峰抖动 33°）。
    """
    if expected <= 0 or peaks is None or len(peaks) != int(expected):
        return float("inf")
    angs = sorted(float(p[0] if isinstance(p, (tuple, list)) else p) for p in peaks)
    n = len(angs)
    ideal = 360.0 / n
    gaps = [((angs[(i + 1) % n] - angs[i]) % 360.0) for i in range(n)]
    return float(np.sqrt(np.mean([(g - ideal) ** 2 for g in gaps])))


def count_spokes(mask: np.ndarray, center_px: Sequence[float],
                 r_min_px: float, r_max_px: float,
                 required: int = 8, n_radii_bands: int = 3,
                 k_mad: float = 3.0, min_width_deg: float = 6.0,
                 n_angles: int = 720,
                 baseline_angles_deg: Optional[Sequence[float]] = None,
                 select: str = "uniform", max_jitter_deg: float = 15.0) -> SpokeResult:
    """多半径投票数辐条，并指出缺失角度。

    ★★ `select` 为什么必须有（真机实测踩到的**假 NG**，务必看清）：

    原来是「`n_radii_bands` 个子带各自数峰 -> 取众数 -> **并列时取较小值**」
    （注释写的是"宁少不多，避免漏报"）。这个偏置在**没有真值可依**时是对的，
    但在这里会**稳定地少数**：

    ```
    真机装配体实测（30~35mm 切成 3 个带，每带仅 1.67mm ≈ 1.9px）：
      带(mm)        med    p95-med  阈值    峰数
      30.0~31.7    0.000   1.000   0.600    8   <- ★ 球头只落在最内侧这个带
      31.7~33.3    0.000   1.000   0.600    6
      33.3~35.0    0.000   1.000   0.600    5
    -> 计数 [8,6,5] 三个全不同、**没有众数** -> 并列取较小值 -> **5**
    -> `5 < required(8)` -> **明明装好的轮盘被判"缺辐条"**
    ```

    而同一帧**整条 30~35mm 一次数**（c5 的做法）是 **8 峰、抖动 5.9°**。
    **两边做法不一致，所以一个数出 8、一个数出 5。**

    ★ 怎么改才对：**按"角度均匀度"挑最好的带**，而不是让 3 个带投票取最小值。
      球头只有 Ø5mm，在 0.86mm/px 下**只占约 6px**，落进哪个子带是碰运气；
      但**真辐条的角度是均布的**，所以"抖动最小"是比"票数最多"可靠得多的判据。
      这也让 `check_assembly` 与 c5 的 `sweep_band` **用上同一个判据**。

    select:
      "uniform"（默认）—— 在「各子带」与「整条带」里挑抖动最小的；抖动相同再要峰多的
      "vote"          —— 老行为（众数 + 并列取小），保留以便对比与回退
    max_jitter_deg:
      抖动超过它就认为这条带**根本没数出规整辐条**，宁可退回"峰最多但抖动大"的结果
      （这样"真的缺辐条"仍会被如实报出来，不会因为我们挑剔而变成 0）。
    """
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

    if str(select).lower() == "vote":
        # ── 老行为（众数 + 并列取小）—— 保留以便对比与回退 ──────────
        # ★ 先剔除"一个峰都没数到"的环带 —— 那说明这个半径上根本没有对比度
        #   （例如环带选到了整圈都是金属的位置），不能让它投票说"有 0 根辐条"。
        valid = [c for c in counts if c > 0]
        pool = valid if valid else counts
        vals, freq = np.unique(np.array(pool), return_counts=True)
        # 并列时取较小值（宁少不多，避免漏报）
        best = int(vals[int(np.argmax(freq))])
        if freq.max() > 1:
            best = int(min(vals[freq == freq.max()]))
        cand = [pb for c, pb in zip(counts, peaks_per_band) if len(pb) == best]
        pk = max(cand, key=len) if cand else peaks_per_band[bands // 2]
        res.count = int(best)
        res.peaks_deg = [round(float(p[0]), 1) for p in pk]
    else:
        # ── 默认：按角度均匀度挑最好的带（见 docstring 里的实测依据）──
        # 候选 = 各子带 + 整条带（整条带正是 c5 扫出来的那个用法）
        full_prof = polar_profile(mask, center_px, r0, r1,
                                  n_angles=n_angles, n_radii=25)
        full_pk = count_peaks(full_prof, k_mad=k_mad, min_width_deg=min_width_deg)
        cands: List[Tuple[np.ndarray, List[Tuple[float, float, float]]]] = \
            [(polar_profile(mask, center_px, edges[k], edges[k + 1],
                            n_angles=n_angles, n_radii=9), peaks_per_band[k])
             for k in range(bands)]
        cands.append((full_prof, full_pk))

        # 排序键：抖动小优先 -> 峰多优先（抖动同为 inf 时才轮到峰数）
        pick = min(cands, key=lambda cp: (spoke_jitter_deg(cp[1], required),
                                          -len(cp[1])))
        # ★ 即使最好的候选抖动也超阈值 = 这一圈本来就不规整（真缺辐条/被遮挡）。
        #   此时**退回"峰最多"**那个，让缺陷被如实报出来，
        #   而不是因为我们挑剔就报成 0（那会变成另一种假 NG）。
        if spoke_jitter_deg(pick[1], required) > float(max_jitter_deg):
            pick = max(cands, key=lambda cp: len(cp[1]))

        res.curve = pick[0]
        res.count = len(pick[1])
        res.peaks_deg = [round(float(p[0]), 1) for p in pick[1]]
        res.jitter_deg = spoke_jitter_deg(pick[1], int(required))

    # 指出缺失角度：用第一个峰做相位参考，生成理想均布位置，找没有峰靠近的
    best = int(res.count)
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
