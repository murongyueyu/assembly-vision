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
        # ★ 金属掩膜（不靠色相）：见 `_metal_mask()` 的说明与 AGENTS.md 2.13
        self.metal = dict(seg.get("metal", {}) or {})
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

        # ── 金属掩膜：不靠色相，只判"亮 + 不是绿垫" ──────────────
        # ★ 为什么单独做一条通路（c5 实测踩过，见 AGENTS.md 2.13）：
        #   装配体上的球头只有 Ø5mm，在 0.86mm/px 下只有 6 像素宽。
        #   边缘与绿垫抗锯齿混色后色相从黄铜的 H=30 跳到 **H=59**
        #   （被绿垫染成淡绿），于是被 brass 的 `h_max=45` 整条挡掉。
        #   结果：装配位上数辐条一个峰都数不出来（实测评分数 0/8）。
        #
        #   而"不是绿垫"这个判据**不含色相上限**，所以能救回这些像素：
        #     实测同一帧 brass 掩膜在 28mm 处 8 峰但角度抖 33°（凑的），
        #     metal 掩膜在 30~35mm 带 8 峰、角度抖仅 **5.9°**（真的）。
        #
        #   ★ 判据必须同时有 V 下限：木桌也是"非绿"，光看非绿会把桌子算进来。
        #     `v_min` 挡住暗部；同时外层还有 work_roi 兜着（位置即身份）。
        res.metal_mask = self._metal_mask(hsv, work_roi)

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

        # ── 过曝自检 ──────────────────────────────────────────────
        # ★★ 为什么必须查这个（c5 实测踩过）：过曝的后果**不是"少切一点"，
        #   而是整条判据链静默失效** ——
        #     过曝 -> 黄铜的色相被推向绿/青 -> 被判成"绿垫" -> 从 metal 掩膜剔除
        #          -> 装配体上数辐条恒为 0 峰
        #   实测某次光照下：整幅均值 BGR 217/244/97、近全白像素 **8.0%**、
        #   装配位 V 中值 **255**（完全饱和），而界面上只是"看着亮一点"，
        #   完全看不出判据已经废了。
        #   所以这里把过曝量记进 detail，让上层能报警而不是默默出错。
        #   依据：曝光 -6.0 时近全白像素 0.00%，-5.0 时 7.96%。
        blown = float((bgr.min(axis=2) >= 250).mean())
        res.detail["blown_ratio"] = blown
        res.detail["v_mean"] = float(hsv[..., 2].mean())
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

    def _metal_mask(self, hsv: np.ndarray,
                    work_roi: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """金属掩膜：**不靠色相**，判据 = 亮 + 不是绿垫。

        配置（`segment.metal`，全部有默认值，老配置行为不变）：
            enabled     是否启用（默认 True）
            v_min       亮度下限，挡暗部（默认 150）
            green_h_min / green_h_max / green_s_min
                        "绿垫"的 HSV 区间（默认 60 / 100 / 120，与实测吻合）

        ★ 为什么不复用 brass 掩膜：brass 的被测对象是**黄铜的色相**，
          而装配体上球头（Ø5mm = 6px）的边缘被绿垫混色后色相跳到 H=59，
          整颗球都被 h_max=45 挡掉。金属掩膜不看色相上限，所以能救回来。

        ★ 为什么不能用"非绿"直接当金属：木桌/皮肤也是非绿。
          所以必须配 V 下限，并且调用方应传 work_roi（位置即身份）。
        """
        m_cfg = dict(self.metal)
        if not bool(m_cfg.get("enabled", True)):
            return None
        g_lo, g_hi = int(m_cfg.get("green_h_min", 60)), int(m_cfg.get("green_h_max", 100))
        g_smin = int(m_cfg.get("green_s_min", 120))
        v_min = int(m_cfg.get("v_min", 150))
        s_max = int(m_cfg.get("s_max", 100))

        h = hsv[..., 0]
        s = hsv[..., 1]
        v = hsv[..., 2]
        # ★★ 判据的准确形式："**既不像黑底、也不像绿垫**"（实测踩过两次）
        #   ① 只写 `~green` —— green 用的是「H 在绿区间**且** S>120」，
        #      于是**木桌**（H=19, S=122, V=210）因为色相不在绿区间而成了"非绿"
        #      -> 被判成金属，顶部木桌区 metal 占比 **99.9%**，辐条扫优全废。
        #   ② 改成 `~green & V>=v_min` 后又踩一次：**黑底**（H≈0, S≈0, V≈0）
        #      也是"非绿"，且合成图的深色底 V 可能 >=150，
        #      于是合成圈整幅变金属 -> 端到端离线从 OK 掉成「辐条 0/8」。
        #   现在的形式对两种背景都安全：
        #      黑底：S 低 -> `s > g_smin` 不成立 -> green=False -> metal=False ✓
        #      绿垫：H 在绿区间且 S≈255 -> green=True  -> metal=False ✓
        #      木桌/皮肤：S≈122 但 H 不在绿区间 -> green=False，
        #                 但 S 超出 s_max=130 -> metal=False ✓
        #      黄铜/球头：V 高、S ≤109 -> metal=True ✓
        green = ((h >= g_lo) & (h <= g_hi) & (s > g_smin))
        metal = (v >= v_min) & (s <= s_max) & (~green)

        m = (metal.astype(np.uint8)) * 255
        if work_roi is not None:
            if work_roi.shape[:2] != m.shape[:2]:
                work_roi = cv2.resize(work_roi, (m.shape[1], m.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            m = cv2.bitwise_and(m, work_roi)
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
