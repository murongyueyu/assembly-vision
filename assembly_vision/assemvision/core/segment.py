# -*- coding: utf-8 -*-
"""黄铜零件分割 + 蓝色手套分割。

核心思路（本项目能不上深度学习的原因）：
  黄铜 vs 黑底/白底 → 用 HSV 的【饱和度 S】分离，比灰度阈值抗反光、抗阴影
  蓝手套 vs 其他     → 用 HSV 的【色相 H】分离，因为画面里保证没有相近颜色

黑底的关键：S 是相对量，暗部只要有一点色偏 S 就会很高。
  所以必须用「S ≥ 60 且 V ≥ 80」双条件，V 下限专门用来挡暗部色偏。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import Config
from ..types import Blob, SegResult
from .features import contour_features
from .roi import polygon_mask


class Segmenter:
    def __init__(self, wp_cfg: Config) -> None:
        seg = wp_cfg.get("segment", {}) or {}
        self.brass = dict(seg.get("brass", {}) or {})
        self.glove = dict(seg.get("glove", {}) or {})
        self.background = str(seg.get("background", "black"))

    # ── 对外主入口 ────────────────────────────────────────────
    def segment(self, bgr: np.ndarray, ts: float = 0.0, scale: float = 1.0,
                want_features: bool = True,
                work_roi: Optional[np.ndarray] = None) -> SegResult:
        """分割。

        ★ work_roi（工位全局 ROI）：
          传入一个与 bgr 同尺寸的 0/255 掩膜时，**黄铜只在掩膜内检测**。
          这是对付"桌面/墙面被当成黄铜"的正解（见 AGENTS.md 2.2）：
          实测踩过 —— 木桌的 H/S 与黄铜重叠，画面顶部的木桌整块被涂成黄铜，
          产生 13269mm² 的假连通域（真零件最大才 3281mm²）。

          为什么不用"调阈值"：单独收紧 h_max/s_max 会削掉零件自己的高光
          （实测零件 S 上限 109、H 上限 45，与桌面区间本就重叠），
          而 ROI 是**位置即身份**，与颜色无关，稳得多。

          ROI 由调用方算（`Qualifier.work_roi_mask()`），因为它要知道料盘单应
          和装配位；`core/` 不去猜这些。
        """
        t0 = time.time()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        brass_mask = self._mask(hsv, self.brass)
        glove_mask = self._mask(hsv, self.glove)
        if work_roi is not None:
            if work_roi.shape[:2] != brass_mask.shape[:2]:
                work_roi = cv2.resize(work_roi, (brass_mask.shape[1], brass_mask.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            brass_mask = cv2.bitwise_and(brass_mask, work_roi)

        res = SegResult(ts=ts, scale=scale, brass_mask=brass_mask, glove_mask=glove_mask)

        # 手套：取最大连通域，套面积与长宽比闸门（滤掉蓝色噪声与屏幕反光）
        blobs_g = self._blobs(glove_mask, min_area_px=float(self.glove.get("min_area_px", 3000)))
        max_area = float(self.glove.get("max_area_px", 200000))
        max_aspect = float(self.glove.get("max_aspect", 6.0))
        best = None
        for b in blobs_g:
            if b.area_px > max_area:
                continue
            _, _, w, h = b.bbox
            aspect = max(w, h) / max(1.0, float(min(w, h)))
            if aspect > max_aspect:
                continue
            if best is None or b.area_px > best.area_px:
                best = b
        if best is not None:
            res.glove_bbox = best.bbox
            res.glove_area_px = best.area_px
            keep = np.zeros_like(glove_mask)
            cv2.drawContours(keep, [best.contour], -1, 255, thickness=cv2.FILLED)
            res.glove_mask = keep

        # 黄铜：全部保留（可能同时有多个零件在画面里）
        blobs_b = self._blobs(brass_mask, min_area_px=float(self.brass.get("min_area_px", 150)))
        if want_features:
            for b in blobs_b:
                b.features = contour_features(b.contour, brass_mask)
        res.brass_blobs = blobs_b

        res.detail = {"seg_ms": (time.time() - t0) * 1000.0}
        return res

    # ── 内部工具 ──────────────────────────────────────────────
    def _mask(self, hsv: np.ndarray, spec: Dict[str, float]) -> np.ndarray:
        """按 HSV 区间取掩膜。

        ★ 上限必须齐全：h_max / s_max / v_max 都要能配。
          黑底时代只写了 h_max、把 S/V 上限写死 255 是够的；换成绿幕之后不够了 ——
          绿幕 S≈255、色相 H≈83，而黄铜 S 中值只有 87，
          靠「饱和度上限 s_max」挡绿幕比调 h_max 稳得多（见 README 第六节）。
          这里默认值保持 255/179，所以老配置的行为完全不变。
        """
        lo = np.array([int(spec.get("h_min", 0)), int(spec.get("s_min", 0)),
                       int(spec.get("v_min", 0))], dtype=np.uint8)
        hi = np.array([int(spec.get("h_max", 179)), int(spec.get("s_max", 255)),
                       int(spec.get("v_max", 255))], dtype=np.uint8)
        m = cv2.inRange(hsv, lo, hi)

        ko = int(spec.get("morph_open", 3))
        kc = int(spec.get("morph_close", 5))
        if ko > 1:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
        if kc > 1:
            # 闭运算专门用来补金属镜面高光造成的空洞
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
        return m

    @staticmethod
    def _blobs(mask: np.ndarray, min_area_px: float = 0.0,
               mode: int = cv2.RETR_EXTERNAL) -> List[Blob]:
        out: List[Blob] = []
        # 新版 OpenCV 返回 (contours, hierarchy)
        found = cv2.findContours(mask, mode, cv2.CHAIN_APPROX_NONE)
        contours = found[-2]
        for c in contours:
            a = float(cv2.contourArea(c))
            if a < min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(c)
            mm = cv2.moments(c)
            if mm["m00"] > 0:
                cx, cy = mm["m10"] / mm["m00"], mm["m01"] / mm["m00"]
            else:
                cx, cy = x + w / 2.0, y + h / 2.0
            out.append(Blob(contour=c, bbox=(x, y, w, h),
                            centroid=(float(cx), float(cy)), area_px=a))
        out.sort(key=lambda b: b.area_px, reverse=True)
        return out


def glove_hits(mask: Optional[np.ndarray], polygons: Dict[str, np.ndarray],
               thresh: float = 0.10) -> Dict[str, bool]:
    """手套掩膜是否压在各个 ROI 上。thresh 是重叠面积占比阈值。

    用"重叠比例"而不是"手套中心是否落在区内"：拿零件时是手指去捏，
    手套中心不一定在槽内。
    """
    hits: Dict[str, bool] = {}
    if mask is None:
        return {k: False for k in polygons}
    for k, poly in polygons.items():
        m = polygon_mask(mask.shape[:2], poly)
        area = float(np.count_nonzero(m))
        if area <= 0:
            hits[k] = False
            continue
        inter = np.count_nonzero(cv2.bitwise_and(mask, m))
        hits[k] = (inter / area) > thresh
    return hits
