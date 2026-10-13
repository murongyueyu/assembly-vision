# -*- coding: utf-8 -*-
"""位姿自检：用料盘上的 4 个标记，每 0.5 s 校一次相机有没有被碰动。

为什么必须做这件事：
    相机纯旋转 θ 角，图像会整体平移约 f_px×θ 像素（与点在画面里的位置基本无关）。
    f_px≈987 时，1° ≈ 17 px。按"ROI 位移不超过 10 px"倒推，角度容差只有 ±0.6°。
    这个量级在机械上做不到长期稳定，所以不能只靠把相机锁死 ——
    必须"动了能发现、并且能自动补偿"。

自检与自动补偿是同一套代码：标记检测顺手就把单应 H 重新算出来了。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import Config
from ..types import PoseState


def _aruco_api(name: str):
    """兼容 OpenCV 新旧两套 aruco 接口。"""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("需要 opencv-contrib-python（cv2.aruco 不可用）")
    a = cv2.aruco
    dict_id = getattr(a, name, None)
    if dict_id is None:
        raise ValueError("未知的 aruco 字典 %r" % name)
    if hasattr(a, "getPredefinedDictionary"):            # OpenCV >= 4.7
        d = a.getPredefinedDictionary(dict_id)
        params = a.DetectorParameters()
        return a.ArucoDetector(d, params), "new"
    d = a.Dictionary_get(dict_id)                        # 旧接口
    params = a.DetectorParameters_create()
    return (d, params), "old"


class PoseChecker:
    def __init__(self, cam_cfg: Config, wp_cfg: Config,
                 calib: Optional[Config] = None) -> None:
        pc = cam_cfg.get("pose_check", {}) or {}
        self.enabled = bool(pc.get("enabled", True))
        self.warn_px = float(pc.get("warn_px", 3.0))
        self.fail_px = float(pc.get("fail_px", 20.0))
        self.min_markers = int(pc.get("min_markers", 3))

        fid = wp_cfg.get("fiducials", {}) or {}
        self.positions: List[Sequence[float]] = list(fid.get("positions", []) or [])
        self.marker_mm = float(fid.get("marker_size_mm", 20.0))
        self.ids_cfg = list(pc.get("marker_ids", [0, 1, 2, 3]))

        self.calib = calib
        self.ref_px: Optional[np.ndarray] = None
        self.ref_diag_px: Optional[float] = None
        self.ref_edge_ratio: Optional[float] = None
        self._detector = None
        self._api = ""
        if self.enabled:
            try:
                self._detector, self._api = _aruco_api(str(pc.get("aruco_dict", "DICT_4X4_50")))
            except Exception as e:  # noqa: BLE001
                print("[pose] aruco 不可用，位姿自检关闭:", e)
                self.enabled = False

        if calib is not None and "fiducials" in calib.data:
            f = calib.get("fiducials")
            if f.get("ref_px"):
                self.ref_px = np.asarray(f["ref_px"], dtype=np.float64)
            self.ref_diag_px = f.get("ref_diag_px")
            self.ref_edge_ratio = f.get("ref_edge_ratio")

    # ── 检测 ──────────────────────────────────────────────────
    def detect(self, bgr: np.ndarray) -> Tuple[List[int], List[np.ndarray]]:
        if not self.enabled or bgr is None:
            return [], []
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self._api == "new":
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            d, params = self._detector  # type: ignore[misc]
            corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=params)
        if ids is None:
            return [], []
        out_ids = [int(i) for i in np.asarray(ids).ravel()]
        return out_ids, list(corners)

    def marker_centers(self, bgr: np.ndarray) -> Dict[int, Tuple[float, float]]:
        ids, corners = self.detect(bgr)
        out: Dict[int, Tuple[float, float]] = {}
        for i, c in zip(ids, corners):
            p = np.asarray(c, dtype=np.float64).reshape(-1, 2)
            out[i] = (float(p[:, 0].mean()), float(p[:, 1].mean()))
        return out

    # ── 自检 ──────────────────────────────────────────────────
    def check(self, bgr: np.ndarray) -> PoseState:
        st = PoseState()
        if not self.enabled:
            st.ok, st.message = True, "位姿自检已关闭"
            return st
        centers = self.marker_centers(bgr)
        st.markers_found = len(centers)
        if len(centers) < self.min_markers:
            st.ok, st.fail = False, True
            st.message = "只检测到 %d 个标记（需要 ≥%d），无法确认位姿" \
                         % (len(centers), self.min_markers)
            return st

        ids = sorted(centers.keys())
        cur = np.array([centers[i] for i in ids], dtype=np.float64)

        if self.ref_px is not None and len(self.ref_px) == len(cur):
            shift = np.linalg.norm(cur - self.ref_px, axis=1)
            st.max_shift_px = float(shift.max())
        else:
            st.max_shift_px = 0.0

        # 高度变化 → 表现为缩放；倾斜变化 → 表现为透视（对边长度比）
        if len(ids) >= 4:
            quad = _order_quad(cur)
            d1 = np.linalg.norm(quad[2] - quad[0])
            d2 = np.linalg.norm(quad[3] - quad[1])
            diag = (d1 + d2) / 2.0
            e01 = np.linalg.norm(quad[1] - quad[0])
            e23 = np.linalg.norm(quad[2] - quad[3])
            e12 = np.linalg.norm(quad[2] - quad[1])
            e30 = np.linalg.norm(quad[0] - quad[3])
            st.edge_ratio = float(max(e01, e23) / max(1e-6, min(e01, e23)))
            if self.ref_diag_px:
                st.scale_change = float(diag / float(self.ref_diag_px) - 1.0)

        if st.max_shift_px > self.fail_px:
            st.ok, st.fail = False, True
            st.message = "相机被移动了 %.0f px（超过 %.0f），请重新标定" \
                         % (st.max_shift_px, self.fail_px)
        elif st.max_shift_px > self.warn_px:
            st.ok, st.warn = True, True
            st.message = "相机被轻微移动 %.1f px，已自动补偿 ROI" % st.max_shift_px
        else:
            st.ok = True
            st.message = "位姿正常（最大偏移 %.1f px）" % st.max_shift_px

        if abs(st.scale_change) > 0.025 and st.ok:
            st.warn = True
            st.message += "；疑似高度变化 %.1f%%" % (st.scale_change * 100)
        return st

    # ── 用标记重算单应（自动补偿 ROI）─────────────────────────
    def current_homography(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """用当前标记位置重算 tray mm → 像素 的单应。

        只用标记中心（4 点刚好定一个单应）。比用角点简单，且标记贴在料盘上
        位置固定，中心受透视影响很小。
        """
        if not self.positions:
            return None
        centers = self.marker_centers(bgr)
        if len(centers) < 4:
            return None
        src, dst = [], []
        for idx, i in enumerate(self.ids_cfg):
            if i in centers and idx < len(self.positions):
                src.append(self.positions[idx])
                dst.append(centers[i])
        if len(src) < 4:
            return None
        H, _ = cv2.findHomography(np.asarray(src, np.float64),
                                  np.asarray(dst, np.float64), 0)
        return H

    def set_reference(self, bgr: np.ndarray) -> bool:
        """标定时调用：记录标记参考位置。"""
        centers = self.marker_centers(bgr)
        if len(centers) < self.min_markers:
            return False
        ids = sorted(centers.keys())
        self.ref_px = np.array([centers[i] for i in ids], dtype=np.float64)
        if len(ids) >= 4:
            quad = _order_quad(self.ref_px)
            d1 = np.linalg.norm(quad[2] - quad[0])
            d2 = np.linalg.norm(quad[3] - quad[1])
            self.ref_diag_px = float((d1 + d2) / 2.0)
        return True

    def reference_dict(self) -> Dict[str, object]:
        return {
            "ref_px": self.ref_px.tolist() if self.ref_px is not None else [],
            "ref_diag_px": self.ref_diag_px,
            "ref_edge_ratio": self.ref_edge_ratio,
        }


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """把 4 个点按 左上→右上→右下→左下 排序。"""
    p = np.asarray(pts, dtype=np.float64)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    order = np.argsort(ang)              # 按角度排序（图像 y 向下，顺序为 左上→右上→右下→左下）
    q = p[order]
    # 旋转到以最靠左上的点开头
    start = int(np.argmin(q[:, 0] + q[:, 1]))
    return np.roll(q, -start, axis=0)
