# -*- coding: utf-8 -*-
"""合格性检查（要求①）与装配体缺件检查（要求④）。

关键前提：料盘槽与零件几乎贴合，所以
    有件 → 槽内黄铜覆盖率 ≈ 0.95~1.00
    缺件 → ≈ 0.00
两个状态差一个数量级，阈值可以取得很宽（默认 0.45）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..config import Config
from ..types import (AssemblyResult, Blob, SegResult, SLOT_NG, SLOT_OK,
                     SLOT_SUSPECT, SlotResult, SpokeResult)
from .polar import count_spokes, spoke_band_px
from .roi import TrayMap, coverage_ratio, circle_polygon, polygon_mask


class Qualifier:
    def __init__(self, wp_cfg: Config, tray: TrayMap,
                 calib: Optional[Config] = None) -> None:
        self.cfg = wp_cfg
        self.tray = tray
        self.calib = calib
        chk = wp_cfg.get("check", {}) or {}
        self.cov_present = float(chk.get("coverage_present", 0.45))
        self.cov_absent = float(chk.get("coverage_absent", 0.15))
        self.spk = dict(chk.get("spokes", {}) or {})
        self.col = dict(chk.get("column", {}) or {})
        r = (wp_cfg.get("assembly", {}) or {}).get("radii_mm", {}) or {}
        self.r_spoke = tuple(r.get("spoke_band", [12.0, 24.0]))
        self.r_spoke_out = tuple(r.get("spoke_band_outer", [31.0, 37.0]))
        self.r_ring = tuple(r.get("ring_band", [26.0, 30.0]))
        self.r_hub = float(r.get("hub", 13.0))
        self.r_cap = float(r.get("cap", 9.55))
        self.r_outer = float(r.get("outer", 37.2))
        self.r_plug = float(r.get("center_plug", 6.0))    # 判立柱用的中心小圆半径

    # ══════════════════════════════════════════════════════════
    # 要求① 单画面料盘逐位核验
    # ══════════════════════════════════════════════════════════
    def check_slots(self, seg: SegResult, scale: float = 1.0) -> List[SlotResult]:
        out: List[SlotResult] = []
        for slot in self.tray.slots:
            sid = slot["id"]
            part = str(slot.get("part", ""))
            poly = self.tray.slot_polygon_px(sid, scale)
            cov = coverage_ratio(seg.brass_mask, poly)

            sr = SlotResult(slot_id=sid, part=part, coverage=cov, roi_px=poly)
            sr.present = cov >= self.cov_present
            blob = self._blob_in(seg, poly)
            if blob is not None:
                sr.features = dict(blob.features)

            if cov <= self.cov_absent:
                sr.verdict = SLOT_NG
                sr.reasons.append("缺件：%s 槽内黄铜覆盖率仅 %.0f%%"
                                  % (sid, cov * 100))
            elif cov < self.cov_present:
                sr.verdict = SLOT_SUSPECT
                sr.reasons.append("覆盖率 %.0f%% 落在阈值带内，请人工复核" % (cov * 100))
            else:
                sr.verdict = SLOT_OK

            # 立柱额外判"细长比"与"弯曲"（装配过程中它是可见的）
            if part == "P2_column" and blob is not None and sr.verdict == SLOT_OK:
                self._check_column_shape(sr, blob, scale)
            out.append(sr)
        return out

    def _check_column_shape(self, sr: SlotResult, blob: Blob, scale: float) -> None:
        f = blob.features
        mmpp = self.tray.mm_per_px(scale)
        asp = float(f.get("elongation", 0.0))
        length_mm = float(f.get("rect_w", 0.0)) * mmpp
        asp_min = float(self.col.get("aspect_min", 3.0))
        len_min = float(self.col.get("min_length_mm", 40.0))
        if asp < asp_min:
            sr.verdict = SLOT_NG
            sr.reasons.append("立柱细长比 %.2f < %.2f，形状异常" % (asp, asp_min))
        if length_mm < len_min:
            sr.verdict = SLOT_NG
            sr.reasons.append("立柱长度约 %.1f mm < %.1f mm" % (length_mm, len_min))

    @staticmethod
    def _blob_in(seg: SegResult, poly_px: np.ndarray) -> Optional[Blob]:
        """取质心落在该 ROI 内、面积最大的前景块。"""
        if seg.brass_mask is None or not seg.brass_blobs:
            return None
        roi = polygon_mask(seg.brass_mask.shape[:2], poly_px)
        best, best_a = None, 0.0
        for b in seg.brass_blobs:
            cx, cy = int(round(b.centroid[0])), int(round(b.centroid[1]))
            h, w = roi.shape[:2]
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            if roi[cy, cx] == 0:
                continue
            overlap = float(np.count_nonzero(
                cv2.bitwise_and(polygon_mask(roi.shape[:2], poly_px),
                                _blob_mask(roi.shape[:2], b))))
            if overlap > best_a:
                best, best_a = b, overlap
        return best

    # ══════════════════════════════════════════════════════════
    # 辐条计数（重点指标）
    # ══════════════════════════════════════════════════════════
    def check_spokes(self, seg: SegResult, slot_id: str = "slot_wheel",
                     scale: float = 1.0,
                     baseline_angles: Optional[Sequence[float]] = None) -> SpokeResult:
        required = int(self.spk.get("required", 8))
        res = SpokeResult(count=0, required=required)
        blob = self._blob_in(seg, self.tray.slot_polygon_px(slot_id, scale))
        if blob is None or seg.brass_mask is None:
            return res

        mmpp = self.tray.mm_per_px(scale)
        r0, r1 = spoke_band_px(mmpp,
                               float(self.spk.get("r_min_mm", self.r_spoke[0])),
                               float(self.spk.get("r_max_mm", self.r_spoke[1])))
        return count_spokes(seg.brass_mask, blob.centroid, r0, r1,
                            required=required,
                            n_radii_bands=int(self.spk.get("n_radii", 3)),
                            k_mad=float(self.spk.get("peak_k_mad", 3.0)),
                            min_width_deg=float(self.spk.get("min_peak_width_deg", 8.0)),
                            baseline_angles_deg=baseline_angles)

    # ══════════════════════════════════════════════════════════
    # 装配位的布尔量（给流程判断用，比 check_assembly 轻）
    # ══════════════════════════════════════════════════════════
    def assembly_flags(self, seg: SegResult, scale: float = 1.0,
                       column_by_depth: Optional[bool] = None) -> Dict[str, object]:
        out: Dict[str, object] = {"base": False, "column": False, "wheel": False,
                                  "cap": False, "spokes": 0, "coverage": 0.0}
        poly = self.tray.assembly_polygon_px(scale)
        if poly is None or seg.brass_mask is None:
            return out
        blob = self._blob_in(seg, poly)
        if blob is None:
            return out

        mmpp = self.tray.mm_per_px(scale)
        cx, cy = blob.centroid
        cov = coverage_ratio(seg.brass_mask, poly)
        out["coverage"] = cov
        out["base"] = cov > 0.20

        # ── 立柱：判"底座中心孔被堵住" ────────────────────────────
        # ★ 不能用"细长前景"来判！立柱是竖立的杆，俯视（含 20° 斜俯视）下
        #   它投影在底座圆盘内部 —— 85mm 长的杆在 20° 倾角下只投影约 29mm，
        #   小于底座半径 29.65mm，整根杆都被底座盖住，看不出细长。
        #   物理上立柱是旋进底座中心螺纹孔的，所以"中心孔被堵住"就是它在位。
        plug_r = float(self.r_plug) / mmpp
        plug_cov = _disc_coverage(seg.brass_mask, (cx, cy), plug_r)
        elong = float(blob.features.get("elongation", 0.0))
        out["center_plugged"] = plug_cov > 0.70
        out["column_plug_coverage"] = plug_cov
        out["column_elongation"] = elong
        out["column"] = bool(out["center_plugged"]
                             or elong >= 2.2
                             or (column_by_depth is True))

        spk = count_spokes(seg.brass_mask, (cx, cy),
                           self.r_spoke[0] / mmpp, self.r_spoke[1] / mmpp,
                           required=int(self.spk.get("required", 8)),
                           n_radii_bands=int(self.spk.get("n_radii", 3)),
                           k_mad=float(self.spk.get("peak_k_mad", 3.0)),
                           min_width_deg=float(self.spk.get("min_peak_width_deg", 6.0)))
        out["spokes"] = spk.count
        out["wheel"] = spk.count >= 4

        # 顶盖：中心被实心覆盖，且轴承孔被盖住。
        # ★ 必须叠加"轮盘已在位"——物理上顶盖压在轮盘上，轮盘不在就不可能装顶盖；
        #   否则只装了一根立柱时中心也是亮的，会误判成"顶盖已装"并触发顺序误报。
        cap_r = self.r_cap / mmpp
        cap_cov = _disc_coverage(seg.brass_mask, (cx, cy), cap_r)
        cap_holes = _disc_holes(seg.brass_mask, (cx, cy), cap_r)
        out["cap_coverage"] = cap_cov
        out["cap_holes"] = cap_holes
        out["cap"] = bool(out["wheel"] and cap_cov > 0.60 and cap_holes == 0)
        return out

    # ══════════════════════════════════════════════════════════
    # 要求④ 装配体缺件检查
    # ══════════════════════════════════════════════════════════
    def check_assembly(self, seg: SegResult, scale: float = 1.0,
                       wheel_height_mm: Optional[float] = None) -> AssemblyResult:
        asm = self.cfg.get("assembly", {}) or {}
        # ★ 深度基线的字段路径历史上不一致：
        #   c5_assembly_baseline.py 写的是 assembly.wheel_height_mm，
        #   这里原本只读 assembly.baseline.wheel_height_mm → 基线永远读不到，
        #   终检会静默输出「本次无深度数据，立柱未参与判定」，立柱判据等于失效。
        #   现在两个路径都认，优先新的 baseline.*；c5 也已改成同时写两处。
        base_h = asm.get("baseline", {}) or {}
        h_ref = base_h.get("wheel_height_mm", None)
        if h_ref is None:
            h_ref = asm.get("wheel_height_mm", None)
        res = AssemblyResult(wheel_height_mm=wheel_height_mm, height_baseline_mm=h_ref)
        poly = self.tray.assembly_polygon_px(scale)
        if poly is None:
            res.verdict = SLOT_SUSPECT
            res.reasons.append("未配置装配位")
            return res
        if seg.brass_mask is None:
            res.verdict = SLOT_NG
            res.reasons.append("装配位没有检测到任何金属")
            return res

        blob = self._blob_in(seg, poly)
        if blob is None:
            res.verdict = SLOT_NG
            res.missing.append("全部零件")
            res.reasons.append("装配位为空")
            return res

        mmpp = self.tray.mm_per_px(scale)
        cx, cy = blob.centroid
        res.center_px = (cx, cy)
        res.outer_radius_px = self.r_outer / mmpp

        m = seg.brass_mask
        # 外圈 / 轮毂 / 顶盖 三处覆盖率
        res.ring_coverage = _ring_coverage(m, (cx, cy), self.r_ring[0] / mmpp,
                                          self.r_ring[1] / mmpp)
        res.hub_coverage = _disc_coverage(m, (cx, cy), self.r_hub / mmpp)
        cap_r = self.r_cap / mmpp
        res.cap_coverage = _disc_coverage(m, (cx, cy), cap_r)

        # 顶盖：中心是否被实心覆盖（内孔数 = 0 表示轴承孔被盖住）
        res.cap_holes = _disc_holes(m, (cx, cy), cap_r)

        # 辐条：★ 装配体上必须数"球头端伸出外圈的凸起"，不能数"臂"。
        #   臂与臂之间的空隙里看到的是下方的底座圆盘（同样是黄铜），
        #   在装配体上用臂的环带数只会得到"整圈都是金属"→ 一个峰都数不出来。
        spk = count_spokes(m, (cx, cy),
                           self.r_spoke_out[0] / mmpp, self.r_spoke_out[1] / mmpp,
                           required=int(self.spk.get("required", 8)),
                           n_radii_bands=int(self.spk.get("n_radii", 3)),
                           k_mad=float(self.spk.get("peak_k_mad", 3.0)),
                           min_width_deg=float(self.spk.get("min_peak_width_deg", 6.0)))
        res.spoke = spk
        if spk.count < spk.required:
            res.missing.append("辐条 %d/%d" % (spk.count, spk.required))
            res.reasons.append("辐条峰数 %d < %d（角度位置见 peaks/missing）"
                               % (spk.count, spk.required))

        # 顶盖
        if res.cap_coverage < 0.5:
            res.missing.append("顶盖铭牌")
            res.reasons.append("中心区覆盖率仅 %.0f%%，顶盖可能未装" % (res.cap_coverage * 100))
        elif res.cap_holes > 0:
            res.reasons.append("中心仍可见 %d 个孔（轴承孔未被完全盖住）" % res.cap_holes)

        # 立柱：终检时用轮盘高度
        tol = float(self.col.get("wheel_height_tol_mm", 15.0))
        if wheel_height_mm is not None and res.height_baseline_mm is not None:
            dh = abs(float(wheel_height_mm) - float(res.height_baseline_mm))
            if dh > tol:
                res.missing.append("立柱")
                res.reasons.append("轮盘高度与基线差 %.1f mm > %.1f mm，立柱可能未装"
                                   % (dh, tol))
        elif wheel_height_mm is None:
            res.reasons.append("本次无深度数据，立柱未参与判定")

        res.complete = len(res.missing) == 0
        res.verdict = SLOT_OK if res.complete else SLOT_NG
        return res


# ──────────────────────────────────────────────────────────────
def _blob_mask(shape, blob: Blob) -> np.ndarray:
    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.drawContours(m, [blob.contour], -1, 255, cv2.FILLED)
    return m


def _disc_coverage(mask: np.ndarray, center, radius_px: float) -> float:
    if radius_px <= 1:
        return 0.0
    poly = circle_polygon(center, 2.0 * radius_px)
    return coverage_ratio(mask, poly)


def _ring_coverage(mask: np.ndarray, center, r_in_px: float, r_out_px: float) -> float:
    if r_out_px <= r_in_px:
        return 0.0
    outer = polygon_mask(mask.shape[:2], circle_polygon(center, 2.0 * r_out_px))
    inner = polygon_mask(mask.shape[:2], circle_polygon(center, 2.0 * r_in_px))
    ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
    total = int(np.count_nonzero(ring))
    if total <= 0:
        return 0.0
    return float(np.count_nonzero(cv2.bitwise_and(mask, ring))) / total


def _disc_holes(mask: np.ndarray, center, radius_px: float) -> int:
    """中心圆盘区域内，前景掩膜上有几个内孔（>0 说明中心是空的）。"""
    if radius_px <= 1:
        return 0
    sub = polygon_mask(mask.shape[:2], circle_polygon(center, 2.0 * radius_px))
    inter = cv2.bitwise_and(mask, sub)
    total = int(np.count_nonzero(sub))
    if total <= 0:
        return 0
    fill = float(np.count_nonzero(inter)) / total
    if fill < 0.5:
        return 1                      # 中心基本是空的 → 当作有一个"大孔"
    found = cv2.findContours(inter, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = found[-2]
    if len(found) < 2 or found[-1] is None:
        return 0
    hier = found[-1][0]
    n = sum(1 for i in range(len(contours))
            if hier[i][3] != -1 and cv2.contourArea(contours[i]) > 8.0)
    return int(n)
