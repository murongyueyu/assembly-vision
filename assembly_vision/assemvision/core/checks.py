# -*- coding: utf-8 -*-
"""合格性检查（要求①）与装配体缺件检查（要求④）。

关键前提：料盘槽与零件几乎贴合，所以
    有件 → 槽内黄铜覆盖率 ≈ 0.95~1.00
    缺件 → ≈ 0.00
两个状态差一个数量级，阈值可以取得很宽（默认 0.45）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

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
    def _classifier(self):
        """按需构造零件特征分类器（模板优先取标定文件，没有就用内置初值）。"""
        if getattr(self, "_clf", None) is not None:
            return self._clf
        try:
            from .part_id import PartClassifier, PartTemplate, default_templates
        except Exception:                                    # noqa: BLE001
            self._clf = False
            return False
        tpls = None
        try:
            stored = self.calib.get("part_templates") if self.calib else None
            if isinstance(stored, dict):
                tpls = []
                for part, v in stored.items():
                    if not isinstance(v, dict) or "ranges" not in v:
                        continue
                    # ★ 逐零件权重：立柱这种细长弯曲件要对 area/solidity 降权，
                    #   而轮盘/底座仍靠 solidity 区分 —— 必须能按零件覆盖。
                    #   实测踩过：这里漏传 weights，配置里写死了也不生效。
                    w = v.get("weights")
                    # ★★ 逐零件的**离散容忍带**：`holes` 是整数计数，真机上同一个
                    #   零件会数出 0~4（镜面过曝把掩膜打穿）。容忍带放在
                    #   weights 里（`holes_tol: 2`），因为它和权重一样是
                    #   "这个零件的这个特征该信几分"的同类信息。
                    #   依据与实测数字见 part_id.DISCRETE_FEATURES 与
                    #   docs/踩坑实录.md §2.30。
                    tol = {}
                    if isinstance(w, dict) and "holes_tol" in w:
                        try:
                            tol["holes"] = float(w["holes_tol"])
                        except (TypeError, ValueError):
                            pass
                        w = {k: x for k, x in w.items() if k != "holes_tol"}
                    tpls.append(PartTemplate(
                        part=part, name=str(v.get("name", "")),
                        ranges={k: tuple(x) for k, x in v["ranges"].items()},
                        discrete_tol=tol,
                        weights=({k: float(x) for k, x in w.items()}
                                 if isinstance(w, dict) else {})))
        except Exception:                                    # noqa: BLE001
            tpls = None
        if not tpls:
            tpls = default_templates()
        # ★ `holes_tol` 是**策略**（允许数错几个），默认 1；知识在模板区间里。
        self._clf = PartClassifier(tpls, min_score=float(
            self.cfg.get("part_id.min_score", 0.55)),
            holes_tol=float(self.cfg.get("part_id.holes_tol", 1.0)))
        return self._clf

    def _tray_area_px(self, scale: float) -> float:
        try:
            from .part_id import tray_area_px
            return tray_area_px(self.tray, scale)
        except Exception:                                    # noqa: BLE001
            return 0.0

    def _nominal_centers(self) -> Dict[str, Tuple[float, float]]:
        """每个槽的**标称**中心（tray mm），按零件名索引。

        标称 = "零件正确放在槽里时"的位置。它是料盘位姿估计的基准。
        运行时实测到的质心与它的差，就是料盘被挪动/旋转的量。
        """
        out: Dict[str, Tuple[float, float]] = {}
        for s in self.tray.slots:
            c = s.get("center")
            if c is None:
                continue
            out[str(s.get("part", s.get("id", "")))] = (float(c[0]), float(c[1]))
        return out

    def _slot_with_center(self, slot_id: str, center_mm, scale: float,
                          angle_offset_deg: float = 0.0):
        """返回某个槽在**指定中心/朝向**处的多边形（用于料盘位姿补偿）。"""
        return self.tray.slot_polygon_px(
            slot_id, scale, center_override=center_mm,
            angle_offset_deg=angle_offset_deg)

    def _nominal_tray_px(self, scale: float = 1.0):
        """标定时料盘的中心与边长（像素），给"料盘轮廓定位"当参考点。

        ★ 直接用 c3 单应把料盘四角投出来算 —— 不需要额外标定。
        """
        quad = self.tray.mm_to_px(
            np.array([[0.0, 0.0], [160.0, 0.0], [160.0, 160.0], [0.0, 160.0]]),
            scale)
        cx = float(quad[:, 0].mean())
        cy = float(quad[:, 1].mean())
        side = float((np.linalg.norm(quad[1] - quad[0])
                      + np.linalg.norm(quad[2] - quad[1])) / 2.0)
        return (cx, cy), side

    def _pose_from_outline(self, bgr, scale: float):
        """用料盘**自身轮廓**估位姿（零件不够时的兜底）。

        ★ 为什么需要：零件星座法要求画面里认得出零件。装配过程中零件被逐个拿走，
          最后料盘会空 —— 那时位姿估计 ok=False，ROI 全错位（实测踩过）。
          料盘本身总在画面里，所以再加这条依据。
        """
        try:
            from .tray_detect import detect_tray, pose_from_tray_detect
        except Exception:                                    # noqa: BLE001
            return None
        cfg = self.cfg.get("tray_detect", {}) or {}
        det = detect_tray(bgr,
                          v_max=int(cfg.get("v_max", 140)),
                          s_max=int(cfg.get("s_max", 170)),
                          min_area_ratio=float(cfg.get("min_area_ratio", 0.01)))
        if not det.ok:
            return det, None
        (ncx, ncy), nside = self._nominal_tray_px(scale)
        pose = pose_from_tray_detect(
            det, (ncx, ncy), nside, self.tray.mm_per_px(scale),
            nominal_angle_deg=float(cfg.get("nominal_angle_deg", det.angle_deg)),
            use_scale=bool(cfg.get("use_scale", False)))
        return det, pose

    def segment_with_roi(self, segmenter, bgr, scale: float = 1.0,
                         prev_pose=None, margin_mm: Optional[float] = None):
        """两遍分割：先用几何 ROI 粗分 -> 定位姿 -> 用更准的 ROI 细分。

        ★ 为什么是"两遍"：ROI 需要位姿（料盘可能被挪动），
          而位姿又需要分割结果 —— 先有鸡还是先有蛋。
          解法：第一遍用**几何 ROI**（料盘/装配位外扩，不依赖位姿）粗分，
          够用来估位姿；第二遍用估出的位姿修正 ROI 再细分，把木桌彻底挡住。

        返回 (seg, pose, pose_src, identified)。
        """
        if margin_mm is None:
            margin_mm = float(self.cfg.get("segment.work_roi_margin_mm", 25.0))
        # ① 几何 ROI（不依赖位姿）粗分
        roi0 = None
        if margin_mm > 0:
            roi0 = self.work_roi_mask(bgr.shape, scale, pose=None, margin_mm=margin_mm)
        seg = segmenter.segment(bgr, scale=scale, work_roi=roi0)
        pose, src, ident = self.pose_estimate(seg, scale, bgr=bgr, prev_pose=prev_pose)
        # ② 用位姿再算一次 ROI，重新分割（ROI 更贴合实际料盘位置）
        if margin_mm > 0 and pose is not None and getattr(pose, "ok", False):
            roi1 = self.work_roi_mask(bgr.shape, scale, pose=pose, margin_mm=margin_mm)
            if roi1 is not None:
                seg2 = segmenter.segment(bgr, scale=scale, work_roi=roi1)
                # 只有"分割结果没变差"才采用第二遍（防止 ROI 反而把零件切掉）
                s1 = int((seg.brass_mask > 0).sum())
                s2 = int((seg2.brass_mask > 0).sum())
                if s2 >= 0.8 * max(1, s1):
                    seg = seg2
                    pose, src, ident = self.pose_estimate(seg, scale, bgr=bgr,
                                                          prev_pose=pose)
        return seg, pose, src, ident

    def work_roi_mask(self, shape, scale: float = 1.0, pose=None,
                      margin_mm: float = 25.0):
        """工位全局 ROI：**只在"料盘 + 装配位"范围内检测黄铜**。

        ★ 为什么必须有（AGENTS.md 2.2 的老问题，一直没做，实测终于踩到）：
          木桌的 H/S 与黄铜重叠。实测某帧里**画面顶部的木桌整块被判成黄铜**，
          产生一个 **13269mm²** 的假连通域（真零件最大才 3281mm²），
          把识别和面积测量全带偏了。

        ★ 为什么不用调阈值：
          单独收紧 h_max/s_max 会削掉零件自己的高光（实测零件 S 上限 109、
          H 上限 45，与桌面区间本就重叠），改阈值按下葫芦起瓢。
          而 ROI 是**位置即身份** —— 与颜色、光照都无关。

        传入 pose 时按位姿修正后的槽位/装配位算，否则用标称坐标。
        """
        h, w = shape[:2]
        m = np.zeros((h, w), np.uint8)
        try:
            quad = self.tray.mm_to_px(
                np.array([[-margin_mm, -margin_mm], [160.0 + margin_mm, -margin_mm],
                          [160.0 + margin_mm, 160.0 + margin_mm],
                          [-margin_mm, 160.0 + margin_mm]]), scale)
            if pose is not None and getattr(pose, "ok", False):
                quad = self.tray.mm_to_px(
                    pose.apply(np.array([[-margin_mm, -margin_mm],
                                         [160.0 + margin_mm, -margin_mm],
                                         [160.0 + margin_mm, 160.0 + margin_mm],
                                         [-margin_mm, 160.0 + margin_mm]])), scale)
            cv2.fillPoly(m, [np.round(quad).astype(np.int32)], 255)
            # 装配位：一个圆，外扩 margin
            asm = self.cfg.get("assembly", {}) or {}
            if asm and asm.get("enabled", False):
                ac = np.asarray(asm.get("center", [0.0, 0.0]), dtype=np.float64).reshape(1, 2)
                if pose is not None and getattr(pose, "ok", False):
                    ac = pose.apply(ac)
                r = float(asm.get("radius", 45.0)) + margin_mm
                pts = np.array([[ac[0, 0] + r * np.cos(t), ac[0, 1] + r * np.sin(t)]
                                for t in np.linspace(0.0, 2.0 * np.pi, 64)], dtype=np.float64)
                cv2.fillPoly(m, [np.round(self.tray.mm_to_px(pts, scale)).astype(np.int32)], 255)
        except Exception:                                        # noqa: BLE001
            return None
        return m

    def pose_estimate(self, seg: SegResult, scale: float = 1.0,
                      bgr: Optional[np.ndarray] = None, prev_pose=None):
        """估料盘位姿。优先级：零件星座(>=2) > 料盘轮廓 > prev_pose > 标称。

        返回 (pose, pose_src, identified)。`identified` 是"零件名 -> 记录"，
        调用方可以复用它，避免把识别算两遍。

        ★ 把这段从 check_slots 里拆出来，是为了能"**先定位姿、再带工位 ROI 分割**"：
          ROI 需要位姿（料盘可能被挪动），而分割又必须在判定之前 —— 所以分两步。
        """
        clf = self._classifier()
        identified = {}
        if clf:
            from .part_id import features_from_blob
            mmpp = self.tray.mm_per_px(scale)
            a_tray = self._tray_area_px(scale)
            for b in (seg.brass_blobs or []):
                if getattr(b, "area_px", 0.0) < 60:
                    continue
                feats = features_from_blob(b, roi_area_px=a_tray, mm_per_px=mmpp,
                                           bgr=bgr, mask=seg.brass_mask)
                got, score, _ = clf.classify(feats)
                if not got:
                    continue
                mm = self.tray.px_to_mm(
                    np.array([[b.centroid[0], b.centroid[1]]], dtype=np.float64), scale)[0]
                prev = identified.get(got)
                if prev is None or b.area_px > prev["blob"].area_px:
                    identified[got] = {"centroid_mm": (float(mm[0]), float(mm[1])),
                                       "blob": b, "score": float(score), "feats": feats}

        from .tray_pose import estimate_pose
        nominal = self._nominal_centers()
        observed = {p: v["centroid_mm"] for p, v in identified.items()}
        pose = estimate_pose(observed, nominal)
        if getattr(pose, "ok", False) and pose.n_used >= 2:
            return pose, "parts", identified
        # 零件不够（被拿走/没放）-> 退回料盘轮廓
        det = None
        if bgr is not None:
            got = self._pose_from_outline(bgr, scale)
            if got is not None:
                det, pose2 = got
                if pose2 is not None and getattr(pose2, "ok", False):
                    self._last_outline = det
                    return pose2, "outline", identified
        self._last_outline = det
        if prev_pose is not None and getattr(prev_pose, "ok", False):
            return prev_pose, "cached", identified
        if getattr(pose, "ok", False):
            return pose, "parts(1)", identified
        return pose, "", identified

    def check_slots(self, seg: SegResult, scale: float = 1.0,
                    bgr: Optional[np.ndarray] = None,
                    pose=None) -> List[SlotResult]:
        """逐槽核验。

        ★★ 两遍流程（2026-10 加入料盘位姿自动补偿）
          第一遍：**不依赖槽位坐标**，按形状/面积/色域特征识别每个黄铜块是什么零件；
                  把「实测质心」与「标称质心」配对，解出料盘的刚体变换（旋转+平移）。
          第二遍：用变换修正所有槽位 ROI，再算覆盖率与最终判定。

        为什么必须这样：只靠固定槽位坐标有个硬前提 —— 料盘位置和朝向不变。
        实测踩过：料盘在 c3 标定后被挪了约 18mm（相机没动），
        四个槽的 ROI 全偏 18~20mm，覆盖率和判定全错。
        """
        out: List[SlotResult] = []
        clf = self._classifier()

        # ── 位姿与零件识别（都交给 pose_estimate，避免两处各算一遍识别）──
        if pose is None:
            pose, pose_src, identified = self.pose_estimate(seg, scale, bgr=bgr)
        else:
            # 调用方给了位姿（pipeline 的缓存），仍要做识别来取特征
            _p, _s, identified = self.pose_estimate(seg, scale, bgr=bgr, prev_pose=pose)
            pose_src = "given"
        det = getattr(self, "_last_outline", None)

        # ── 第二遍：逐槽（用位姿修正后的 ROI）──
        for slot in self.tray.slots:
            sid = slot["id"]
            part = str(slot.get("part", ""))
            if getattr(pose, "ok", False):
                c_now = pose.apply(np.asarray(slot.get("center", [0.0, 0.0]),
                                              dtype=np.float64).reshape(1, 2))[0]
                poly = self._slot_with_center(sid, c_now, scale,
                                              angle_offset_deg=pose.theta_deg)
            else:
                poly = self.tray.slot_polygon_px(sid, scale)
            cov = coverage_ratio(seg.brass_mask, poly)

            # ★★ 覆盖率阈值按"该槽的理论最大覆盖率"缩放，不能全局一刀切。
            #   原因：覆盖率 = 零件面积 / 槽面积，这个比值**每种零件天生不同**：
            #     轮盘/底座/顶盖 ~0.95+（圆件填满圆槽）
            #     立柱只有 ~0.51（85x15 的细杆 vs 91x28 的宽胶囊槽）
            #   用统一的 0.45 卡，立柱完整在位也只有 0.40 -> **必然误判存疑**（实测踩过）。
            cov_scale = 1.0
            if self.cfg.get("check.coverage_scale_by_max", True) and clf:
                try:
                    from .part_id import max_coverage_for
                    m = max_coverage_for(sid, clf.templates, self.tray, scale)
                    cov_scale = float(min(1.0, max(0.05, m)))
                except Exception:                            # noqa: BLE001
                    cov_scale = 1.0
            thr_present = self.cov_present * cov_scale
            thr_absent = self.cov_absent * cov_scale

            sr = SlotResult(slot_id=sid, part=part, coverage=cov, roi_px=poly)
            sr.present = cov >= thr_present
            blob = self._blob_in(seg, poly)
            # ★ 优先用"第一遍已识别到该零件"的块：
            #   即使 ROI 因位姿估计有残余误差没框住它，也能拿到正确特征。
            idrec = identified.get(part)
            if idrec is not None:
                blob = idrec["blob"]
            if blob is not None:
                sr.features = dict(blob.features)
            sr.features["cov_thr_present"] = float(thr_present)
            sr.features["cov_scale"] = float(cov_scale)
            if getattr(pose, "ok", False):
                sr.features["pose_theta_deg"] = float(pose.theta_deg)
                sr.features["pose_dx_mm"] = float(pose.tx)
                sr.features["pose_dy_mm"] = float(pose.ty)
                sr.features["pose_n_parts"] = int(pose.n_used)
                sr.features["pose_resid_mm"] = float(pose.residual_mm)
                sr.features["pose_src"] = pose_src or ("parts" if pose.n_used else "?")
            if det is not None and det.ok:
                sr.features["tray_det_fill"] = float(det.fill)
                sr.features["tray_det_side_px"] = float((det.size_px[0] + det.size_px[1]) / 2)

            # ── 零件特征识别：这里直接用第一遍的结果 ──
            if blob is not None and clf:
                from .part_id import features_from_blob, slot_area_px
                a_slot = slot_area_px(self.tray, sid, scale)
                feats = features_from_blob(
                    blob, roi_area_px=a_slot,
                    mm_per_px=self.tray.mm_per_px(scale),
                    bgr=bgr, mask=seg.brass_mask)
                got, score = None, 0.0
                if idrec is not None and idrec["blob"] is blob:
                    # 这个块在第一遍已经认出身份了，直接沿用（避免重复算两遍）
                    got, score = part, float(idrec["score"])
                else:
                    got, score, _ = clf.classify(feats)
                sr.features["id_part"] = got or ""
                sr.features["id_score"] = float(score or 0.0)
                # ★ `length_mm` / `short_side_mm` 必须一起带出来：
                #   它是**权重最高的形状特征**（2.5 / 1.5，合计 4.0 of 18.1），
                #   而且是"细长/弯曲件上唯一可信的特征"（见 part_id 的说明）。
                #   实测踩过：`features_from_blob` 明明算了，但这里白名单没列，
                #   于是 `SlotResult.features` 里看不到它 —— 诊断工具
                #   （measure_features_real / measure_holes_*）都少了最关键一项。
                #   ★ 只加"透出"、不改任何判据。
                for k in ("area_mm2", "area_frac", "length_mm", "short_side_mm",
                          "c_hue_mean", "c_sat_mean", "c_val_mean", "c_hue_std",
                          "c_sat_hi_frac"):
                    if k in feats:
                        sr.features[k] = float(feats[k])
                if got is not None and part and got != part:
                    if cov > thr_absent:
                        sr.verdict = SLOT_NG
                        sr.reasons.append(
                            "零件不符：该槽应为 %s，但特征识别为 %s（得分 %.2f）"
                            % (part, got, score))
                        out.append(sr)
                        continue

            if cov <= thr_absent:
                sr.verdict = SLOT_NG
                sr.reasons.append("缺件：%s 槽内黄铜覆盖率仅 %.0f%%（阈值 %.0f%%）"
                                  % (sid, cov * 100, thr_absent * 100))
            elif cov < thr_present:
                sr.verdict = SLOT_SUSPECT
                sr.reasons.append("覆盖率 %.0f%% 落在阈值带内（%.0f%%~%.0f%%），请人工复核"
                                  % (cov * 100, thr_absent * 100, thr_present * 100))
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
    def assembly_polygon_px(self, scale: float = 1.0, pose=None):
        """装配位 ROI 多边形（像素）。★ 所有需要装配位的地方都必须走这里。

        ★★ 为什么必须用 pose 参数（实测踩过，是个真 bug，与"开放位置"无关）：
          `work_roi_mask()` 里**装配位是跟着位姿修正的**，而
          `check_assembly()` / `assembly_flags()` 原来直接用
          `self.tray.assembly_polygon_px()`（标称坐标）——
          **同一个装配位在两处算出来的圈不一样**。
          料盘一挪，ROI 跟着走了、判定用的圈没走，
          "圈住的"和"判的"就不是同一个地方。
        """
        if pose is None or not getattr(pose, "ok", False):
            return self.tray.assembly_polygon_px(scale)
        asm = self.cfg.get("assembly", {}) or {}
        if not asm or not asm.get("enabled", False):
            return None
        c = np.asarray(asm.get("center", [0.0, 0.0]), dtype=np.float64).reshape(1, 2)
        ac = pose.apply(c)[0]
        r = float(asm.get("radius", 45.0))
        t = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        pts = np.stack([ac[0] + r * np.cos(t), ac[1] + r * np.sin(t)], axis=1)
        return self.tray.mm_to_px(pts, scale)

    def assembly_flags(self, seg: SegResult, scale: float = 1.0,
                       column_by_depth: Optional[bool] = None,
                       pose=None) -> Dict[str, object]:
        out: Dict[str, object] = {"base": False, "column": False, "wheel": False,
                                  "cap": False, "spokes": 0, "coverage": 0.0}
        poly = self.assembly_polygon_px(scale, pose=pose)
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
                       wheel_height_mm: Optional[float] = None,
                       pose=None) -> AssemblyResult:
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
        poly = self.assembly_polygon_px(scale, pose=pose)
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
        #
        # ★★ 而且**必须用 metal 掩膜，不能用 brass 掩膜**（c5 实测踩过）：
        #   球头只有 Ø5mm，在 0.86mm/px 下只有 6 像素宽，边缘被绿垫抗锯齿
        #   混色后色相从 H=30 跳到 **H=59**，被 brass 的 `h_max=45` 整条挡掉，
        #   于是辐条数恒为 0（实测 20 帧全是 0）。
        #   换 metal 掩膜（"亮 + 不是绿垫"，不看色相上限）后：
        #     brass 在 28mm 处 8 峰但角度抖 33°（凑的）
        #     metal 在 30~35mm 带 8 峰、角度抖 **5.9°**（真的）—— 见 AGENTS.md 2.13
        #
        # ★★ 回落条件是"有没有内容"，**不是"是不是 None"**（实测踩过）：
        #   metal 判据要求"不是绿垫"，所以在**深色底**（合成图、黑垫）上
        #   正确地什么都匹配不到 -> mask 是**全 0 的数组而不是 None**。
        #   写成 `is not None` 就会拿这个空掩膜去数辐条 -> 恒为 0，
        #   端到端离线测试因此从 OK 掉成「辐条 0/8」。
        spoke_mask = m
        mm_ = seg.metal_mask
        if mm_ is not None and int(np.count_nonzero(mm_)) > 200:
            spoke_mask = mm_
        spk = count_spokes(spoke_mask, (cx, cy),
                           self.r_spoke_out[0] / mmpp, self.r_spoke_out[1] / mmpp,
                           required=int(self.spk.get("required", 8)),
                           n_radii_bands=int(self.spk.get("n_radii", 3)),
                           k_mad=float(self.spk.get("peak_k_mad", 3.0)),
                           min_width_deg=float(self.spk.get("min_peak_width_deg", 6.0)),
                           # ★ 必须走配置：默认的 "uniform" 是按角度均匀度挑半径带，
                           #   修的是"装好的轮盘被判缺辐条"的真机假 NG
                           #   （见 polar.count_spokes 的 docstring 与 workpiece.yaml）
                           select=str(self.spk.get("select", "uniform")),
                           max_jitter_deg=float(self.spk.get("max_jitter_deg", 15.0)))
        res.spoke = spk
        if spk.count < spk.required:
            res.missing.append("辐条 %d/%d" % (spk.count, spk.required))
            res.reasons.append("辐条峰数 %d < %d（角度位置见 peaks/missing）"
                               % (spk.count, spk.required))

        # ── ★ 用 c5 实测基线比"轮盘完整度"（P0-1 接线）────────────────
        # 判据本体抽到 `spoke_vs_baseline()` 里，便于离线测试（见 tests/test_assembly.py）
        bad, why = spoke_vs_baseline(
            spk.count, asm.get("spoke_peak_count", None),
            slack=int(self.spk.get("baseline_slack", 2)))
        if bad:
            res.missing.append("轮盘辐条 %d/基线%s" % (spk.count, asm.get("spoke_peak_count")))
            res.reasons.append(why)

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
def spoke_vs_baseline(count: int, baseline, slack: int = 2) -> Tuple[bool, str]:
    """把实测辐条数与 c5 标定的基线比较。返回 (是否异常, 理由文字)。

    ★ 为什么只比"辐条数"这一条基线，而不是 ring/hub/cap 覆盖率：
      实测（合成图各种缺件组合）ring 0.994~0.994、hub 0.880~0.998、
      cap 0.874~1.000 —— **连缺顶盖都还有 0.874，几乎没有区分度**；
      真机上 c5 实测三个数全是 ~1.0（1.0 / nan→1.0 / 1.0）。
      拿它们当阈值等于没判。**不要为了"用上基线"而加没有区分度的判据。**

    ★ 容差 2 的依据：c5 真机实测 12 帧
      `[8,8,8,8,6,6,7,8,8,6,8,8]` —— **完整在位就是 6~8，散布 ±1**。
      取 2 表示"少 2 根以上才报"。

    ★ 明确写下的代价：真少 2 根（数出 6）**不会**被发现，因为"完整在位"
      本身就出现过 6。这是"不误报"优先于"不漏报"的取舍；要收紧就传 slack=1，
      但那与本项目的实测散布直接冲突。

    边界：
      baseline 无效（None/0/非数字）-> 不判（返回 False，附说明）
      count == 0 -> 不在这里判。0 根说明整个轮盘没装，由
                    `spk.count < spk.required` 那条更基础的判据负责，
                    这里再报一次只会重复。
    """
    try:
        base = int(baseline) if baseline is not None else 0
    except (TypeError, ValueError):
        base = 0
    if base <= 0:
        return False, "未标定辐条基线（先跑 c5），本条不判"
    c = int(count)
    if c <= 0:
        return False, "辐条 0 根由基础判据负责，这里不重复报"
    if c < base - int(slack):
        return True, ("辐条峰数 %d 比标定基线 %d 少 %d 根（容差 %d），"
                      "轮盘可能被遮挡或不完整"
                      % (c, base, base - c, int(slack)))
    return False, "辐条 %d 相对基线 %d 在容差内" % (c, base)


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
