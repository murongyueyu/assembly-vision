# -*- coding: utf-8 -*-
"""料盘坐标 ↔ 图像像素的映射，以及槽位 ROI 的生成与覆盖率计算。

坐标系约定：tray mm，X 向右、Y 向下，料盘左上角为原点。
单应矩阵 H（3x3）把 tray mm → 左眼图像的【全分辨率】像素。
实时通道用缩放后的图，只要把像素坐标乘 scale 即可（等比缩放）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import Config


# ──────────────────────────────────────────────────────────────
# 几何图元
# ──────────────────────────────────────────────────────────────
def circle_polygon(center: Sequence[float], diameter: float, margin: float = 0.0,
                   n: int = 64) -> np.ndarray:
    """圆 ROI 多边形（mm）。margin 是各方向外扩量。"""
    r = (float(diameter) + 2.0 * float(margin)) / 2.0
    t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    c = np.asarray(center, dtype=np.float64)
    return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], axis=1)


def capsule_polygon(center: Sequence[float], length: float, width: float,
                    angle_deg: float, margin: float = 0.0,
                    arc_pts: int = 24) -> np.ndarray:
    """长圆槽（胶囊形）ROI 多边形（mm）。

    length 是两端圆头最外缘之间的总长度，width 是槽宽。
    angle_deg 是长轴相对 X 轴的角度（度）。
    """
    a = np.deg2rad(float(angle_deg))
    u = np.array([np.cos(a), np.sin(a)], dtype=np.float64)
    v = np.array([-np.sin(a), np.cos(a)], dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)

    w2 = (float(width) + 2.0 * float(margin)) / 2.0
    half_axis = max(0.0, (float(length) - float(width)) / 2.0)
    p1 = c + u * half_axis
    p2 = c - u * half_axis

    t1 = np.linspace(-np.pi / 2.0, np.pi / 2.0, arc_pts)
    arc1 = p1[None, :] + w2 * (np.cos(t1)[:, None] * u[None, :]
                               + np.sin(t1)[:, None] * v[None, :])
    t2 = np.linspace(np.pi / 2.0, 3.0 * np.pi / 2.0, arc_pts)
    arc2 = p2[None, :] + w2 * (np.cos(t2)[:, None] * u[None, :]
                               + np.sin(t2)[:, None] * v[None, :])
    return np.vstack([arc1, arc2])


def polygon_mask(shape: Tuple[int, int], polygon_px: np.ndarray) -> np.ndarray:
    """在给定画幅上把多边形栅格化成 0/255 掩膜。"""
    m = np.zeros(shape[:2], dtype=np.uint8)
    if polygon_px is None or len(polygon_px) < 3:
        return m
    pts = np.round(np.asarray(polygon_px, dtype=np.float64)).astype(np.int32)
    cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], 255)
    return m


def coverage_ratio(mask: np.ndarray, polygon_px: np.ndarray) -> float:
    """多边形 ROI 内的前景覆盖率 = 前景像素 / ROI 像素。

    槽与零件几乎贴合时：有件≈1.0、无件≈0.0，所以阈值可以取很宽。
    """
    if mask is None:
        return 0.0
    roi = polygon_mask(mask.shape[:2], polygon_px)
    total = int(np.count_nonzero(roi))
    if total <= 0:
        return 0.0
    inter = int(np.count_nonzero(cv2.bitwise_and(mask, roi)))
    return inter / float(total)


# ──────────────────────────────────────────────────────────────
# 料盘映射
# ──────────────────────────────────────────────────────────────
class TrayMap:
    """管理 tray mm → 图像像素；并生成各槽位 / 装配位的 ROI 多边形。"""

    def __init__(self, wp_cfg: Config,
                 homography: Optional[np.ndarray] = None,
                 ref_scale: float = 1.0,
                 px_per_mm: Optional[float] = None,
                 origin_px: Optional[Sequence[float]] = None) -> None:
        self.cfg = wp_cfg
        self.slots: List[dict] = list(wp_cfg.get("slots", []) or [])
        self._slot_by_id: Dict[str, dict] = {s["id"]: s for s in self.slots}
        self.ref_scale = float(ref_scale)

        if homography is not None:
            self.H = np.asarray(homography, dtype=np.float64).reshape(3, 3)
            self.synthetic = False
        else:
            # 没有标定时，用「每毫米多少像素 + 原点」造一个仿射映射。
            # 主要给合成图测试用；真实运行必须来自标定。
            s = float(px_per_mm or 1.0)
            ox, oy = (origin_px if origin_px is not None else (0.0, 0.0))
            self.H = np.array([[s, 0.0, float(ox)],
                               [0.0, s, float(oy)],
                               [0.0, 0.0, 1.0]], dtype=np.float64)
            self.synthetic = True

    # ---------- 坐标变换 ----------
    def mm_to_px(self, pts_mm: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """mm → 像素。scale 是目标图相对 ref_scale 的缩放比。"""
        pts = np.asarray(pts_mm, dtype=np.float64).reshape(-1, 1, 2)
        px = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        k = float(scale) / self.ref_scale
        if abs(k - 1.0) > 1e-9:
            px = px * k
        return px

    def px_to_mm(self, pts_px: np.ndarray, scale: float = 1.0) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        k = float(scale) / self.ref_scale
        if abs(k - 1.0) > 1e-9:
            pts = pts / k
        inv = np.linalg.inv(self.H)
        return cv2.perspectiveTransform(pts, inv).reshape(-1, 2)

    def mm_per_px(self, scale: float = 1.0) -> float:
        """局部尺度：**每像素代表多少毫米**（GSD）。

        ★ 注意方向：这个值通常是 0.3~0.5。若要"每毫米多少像素"，取它的倒数。
        曾经在这里写反过，结果辐条环带被算到轮毂里面去了。
        """
        c = np.asarray(self.cfg.get("tray.size", [160.0, 160.0]), dtype=np.float64) / 2.0
        p0 = self.mm_to_px(c[None, :], scale)[0]
        p1 = self.mm_to_px((c + np.array([1.0, 0.0]))[None, :], scale)[0]
        px_per_mm = float(np.linalg.norm(p1 - p0))
        return 1.0 / px_per_mm if px_per_mm > 1e-9 else 0.0

    # ---------- ROI 生成 ----------
    def slot_polygon_mm(self, slot_id: str) -> np.ndarray:
        s = self._slot_by_id[slot_id]
        m = float(s.get("roi_margin_mm", 0.0))
        if s.get("type") == "circle":
            return circle_polygon(s["center"], s["diameter"], m)
        if s.get("type") == "capsule":
            return capsule_polygon(s["center"], s["length"], s["width"],
                                   s.get("angle_deg", 0.0), m)
        raise ValueError("未知槽型 %r（应为 circle / capsule）" % s.get("type"))

    def slot_polygon_px(self, slot_id: str, scale: float = 1.0) -> np.ndarray:
        return self.mm_to_px(self.slot_polygon_mm(slot_id), scale)

    def slot_polygons_px(self, scale: float = 1.0) -> Dict[str, np.ndarray]:
        return {s["id"]: self.slot_polygon_px(s["id"], scale) for s in self.slots}

    def assembly_polygon_mm(self) -> Optional[np.ndarray]:
        asm = self.cfg.get("assembly", None)
        if not asm or not asm.get("enabled", False):
            return None
        return circle_polygon(asm["center"], 2.0 * float(asm.get("radius", 45.0)), 0.0)

    def assembly_polygon_px(self, scale: float = 1.0) -> Optional[np.ndarray]:
        poly = self.assembly_polygon_mm()
        return None if poly is None else self.mm_to_px(poly, scale)

    def slot_of(self, slot_id: str) -> dict:
        return self._slot_by_id[slot_id]

    def part_of(self, slot_id: str) -> str:
        return str(self._slot_by_id[slot_id].get("part", ""))
