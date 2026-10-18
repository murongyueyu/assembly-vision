# -*- coding: utf-8 -*-
"""B 模块：D435 原生深度/三维法向高度，以及保留的旧双目兼容算法。

当前 D435 使用 NativeHeightEstimator，显式选择裸桌面参考区和目标顶面区。
以下 B=120mm、视差和 work_scale 的说明仅适用于旧 HBV 双目路径。

B=120mm 长基线 → 精度好，但近距离视差很大：
    1280x960 下 30cm 处视差约 395 px，超出 StereoBM 的 256 上限
    640x480 下只有 197 px，40-50cm 时是 118-148 px → 用 numDisparities=160 刚好
所以深度一律在 work_scale（默认 0.5）上算。

★ 明确不做的事：判「顶盖是否拧紧」（行程 1-2 mm，与深度噪声同量级）。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ..config import Config


def native_depth_mm(raw: np.ndarray, meters_per_unit: float) -> np.ndarray:
    """SDK Z16 -> 毫米；0、负数和非有限值始终表示无观测。"""
    if raw.ndim != 2 or not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("原生深度必须是二维数组，depth_scale 必须为正数")
    out = raw.astype(np.float32) * (1000.0 * meters_per_unit)
    out[~np.isfinite(out) | (raw <= 0)] = np.nan
    return out


class NativeHeightEstimator:
    """仅处理数组的 B 模块：显式桌面参考区 -> 三维平面 -> 法向高度。

    points_mm 必须来自 SDK 对同一对齐深度帧的反投影，不能套旧相机内参。
    ★ 参考区只能包含裸桌面；目标区应选轮盘可见顶面小片，避开孔洞/背景/手。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.min_points = int(cfg.get("native_depth.min_points"))
        self.min_ratio = float(cfg.get("native_depth.min_valid_ratio"))
        self.min_z = float(cfg.get("native_depth.min_distance_mm"))
        self.max_z = float(cfg.get("native_depth.max_distance_mm"))
        self.residual_limit = float(cfg.get("native_depth.max_plane_residual_mm"))
        self.spread_limit = float(cfg.get("native_depth.max_height_spread_mm"))
        self.max_points = int(cfg.get("native_depth.max_fit_points"))
        self.iterations = int(cfg.get("native_depth.fit_iterations"))
        if (self.min_points < 3 or self.max_points < self.min_points
                or not 0 < self.min_ratio <= 1 or self.min_z <= 0
                or self.max_z <= self.min_z or self.residual_limit <= 0
                or self.spread_limit <= 0 or self.iterations < 1):
            raise ValueError("原生深度质量参数无效")

    def measure_frame(self, frame):
        """运行时绑定设备和对齐内参；缺少本机标定时拒用旧基线。"""
        if frame is None or frame.depth_points_mm is None:
            return dict(ok=False, height_mm=None, reason="本帧没有原生深度")
        if frame.depth_profile != self.cfg.get("native_depth.profile", None):
            return dict(ok=False, height_mm=None, reason="深度设备/内参与标定不匹配")
        ref = self.cfg.get("native_depth.reference_rect", None)
        target = self.cfg.get("native_depth.target_rect", None)
        if ref is None or target is None:
            return dict(ok=False, height_mm=None, reason="尚未选择桌面参考区和目标区")
        shape = frame.depth_points_mm.shape[:2]
        return self.measure(frame.depth_points_mm, rectangle_mask(shape, ref), rectangle_mask(shape, target))

    def measure(self, points_mm, reference_mask, target_mask):
        """失败返回 ok=False 和原因，绝不沿用上次高度。"""
        result = dict(ok=False, height_mm=None, reason="")
        xyz = np.asarray(points_mm)
        ref, target = np.asarray(reference_mask, bool), np.asarray(target_mask, bool)
        if xyz.ndim != 3 or xyz.shape[2] != 3 or ref.shape != xyz.shape[:2] or target.shape != ref.shape:
            raise ValueError("三维点与参考/目标掩膜尺寸不一致")
        if np.any(ref & target):
            result["reason"] = "参考区与目标区重叠"
            return result
        valid = np.isfinite(xyz).all(axis=2) & (xyz[..., 2] >= self.min_z) & (xyz[..., 2] <= self.max_z)
        for name, mask in (("reference", ref), ("target", target)):
            count = int(np.count_nonzero(mask & valid))
            ratio = count / max(1, int(mask.sum()))
            result[name + "_points"] = count
            result[name + "_valid_ratio"] = ratio
            if count < self.min_points or ratio < self.min_ratio:
                result["reason"] = name + " 有效深度不足"
                return result
        pts = xyz[ref & valid].astype(np.float64)
        if len(pts) > self.max_points:
            pts = pts[np.linspace(0, len(pts) - 1, self.max_points, dtype=int)]
        original = pts.copy()
        # Z = ax + by + c 是三维平面；不同于旧版 Z = au + bv + c。
        # 显式参考区使工件不参与拟合，残差按法向毫米计。
        for _ in range(self.iterations):
            A = np.column_stack((pts[:, :2], np.ones(len(pts))))
            abc, _, rank, _ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
            if rank < 3:
                result["reason"] = "参考点共线，无法拟合桌面"
                return result
            normal = np.array([-abc[0], -abc[1], 1.0])
            length = np.linalg.norm(normal)
            normal /= length
            offset = float(abc[2] / length)
            residuals = original @ normal - offset
            mad = float(np.median(np.abs(residuals - np.median(residuals))))
            keep = np.abs(residuals - np.median(residuals)) <= max(self.residual_limit, 3.0 * mad)
            if int(keep.sum()) < self.min_points or float(keep.mean()) < self.min_ratio:
                result["reason"] = "桌面参考内点不足"
                return result
            pts = original[keep]
        # 最后一次筛选后重新拟合，避免使用上一轮系数。
        A = np.column_stack((pts[:, :2], np.ones(len(pts))))
        abc, _, rank, _ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
        if rank < 3:
            result["reason"] = "桌面参考退化"
            return result
        normal = np.array([-abc[0], -abc[1], 1.0])
        length = np.linalg.norm(normal)
        normal /= length
        offset = float(abc[2] / length)
        rms = float(np.sqrt(np.mean((pts @ normal - offset) ** 2)))
        result.update(plane_normal=normal.tolist(), plane_offset_mm=offset,
                      plane_rms_mm=rms, reference_inliers=int(len(pts)))
        if rms > self.residual_limit:
            result["reason"] = "桌面平面残差超限"
            return result
        heights = offset - xyz[target & valid].astype(np.float64) @ normal
        spread = float(np.percentile(heights, 90) - np.percentile(heights, 10))
        result["height_spread_mm"] = spread
        if spread > self.spread_limit:
            result["reason"] = "目标高度散布超限，请避开孔洞、背景和手"
            return result
        result.update(ok=True, height_mm=float(np.median(heights)), reason="ok")
        return result


def rectangle_mask(shape, rect):
    """彩色全分辨率像素矩形 [x,y,w,h]；越界拒绝，避免静默裁剪。"""
    if len(rect) != 4 or any(not np.isfinite(v) or int(v) != v for v in rect):
        raise ValueError("ROI 必须为四个整数")
    x, y, w, h = map(int, rect)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > shape[1] or y + h > shape[0]:
        raise ValueError("ROI 越界或为空")
    mask = np.zeros(shape[:2], bool)
    mask[y:y+h, x:x+w] = True
    return mask


def compare_native_height(height_mm, baseline_mm, tolerance_mm):
    """纯高度判据：None 表示未知，不能把空洞当作缺立柱。"""
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("高度容差必须为正数")
    if height_mm is None or baseline_mm is None:
        return None
    if not np.isfinite(height_mm) or not np.isfinite(baseline_mm):
        return None
    return bool(abs(height_mm - baseline_mm) <= tolerance_mm)


def _as_dist(d) -> np.ndarray:
    """把畸变系数整理成 OpenCV 5.0 要求的 (1,N) 二维数组。

    ★ 为什么需要：stereoRectify 在 OpenCV 5.0 里对 1-D 的畸变系数会报
        cv2.error: (-215:Assertion failed) a_size.width == len in cv::gemm
      而 YAML/JSON 只能存成一维列表，读回来必然是 1-D，所以必须在这里 reshape。
    """
    a = np.asarray(d, dtype=np.float64).ravel()
    if a.size == 0:
        a = np.zeros(5, dtype=np.float64)
    return a.reshape(1, -1)


def _as_vec31(t) -> np.ndarray:
    """把平移向量整理成 OpenCV 5.0 要求的 (3,1) 列向量。

    ★★ 这是最容易踩的坑（实测三种形状的表现）：
        T shape=(3,)    -> stereoRectify ✗  断言失败 a_size.width == len in cv::gemm
        T shape=(3,1)   -> stereoRectify ✓
      1-D 的 T 在 OpenCV 4.x 能过，5.0 不行。而 YAML 读回来必然是 1-D，
      所以**必须在代码里 reshape**。
    """
    a = np.asarray(t, dtype=np.float64).ravel()
    if a.size == 0:
        a = np.zeros(3, dtype=np.float64)
    return a.reshape(3, 1)


class DepthEstimator:
    def __init__(self, cam_cfg: Config, calib: Optional[Config] = None) -> None:
        self.cam = cam_cfg
        self.calib = calib
        self.scale = float(cam_cfg.get("depth.work_scale", 0.5))
        self.num_disp = int(cam_cfg.get("depth.num_disparities", 160))
        self.block = int(cam_cfg.get("depth.block_size", 15))
        self.min_valid = float(cam_cfg.get("depth.min_valid_ratio", 0.2))
        self.enabled = bool(cam_cfg.get("depth.enabled", True))

        self.matcher = None
        self.map1_l = self.map2_l = self.map1_r = self.map2_r = None
        self.fx = 0.0
        self.baseline_mm = 0.0
        self.out_size: Tuple[int, int] = (0, 0)
        self.ready = False

    # ── 初始化 ────────────────────────────────────────────────
    def setup(self) -> bool:
        if not self.enabled:
            return False
        if self.calib is None or "stereo" not in self.calib.data:
            print("[depth] 没有双目标定结果，深度功能关闭（先跑 calib/c2_stereo.py）")
            return False
        st = self.calib.get("stereo")
        try:
            K1 = np.asarray(st["K_left"], dtype=np.float64).reshape(3, 3)
            K2 = np.asarray(st["K_right"], dtype=np.float64).reshape(3, 3)
            R = np.asarray(st["R"], dtype=np.float64).reshape(3, 3)
            T = _as_vec31(st["T"])          # ★ 必须 (3,1)，1-D 在 5.0 下会崩
            W, H = [int(v) for v in st["image_size"]]
            # ★★ 畸变系数必须是 (1,N) 二维，不能是 1-D！
            #   OpenCV 5.0 的 stereoRectify 对 1-D 的 D 会触发
            #     cv2.error: (-215:Assertion failed) a_size.width == len in cv::gemm
            #   实测对比（同一组参数）：
            #     D shape=(1,5) -> ✓ 成功
            #     D shape=(5,)  -> ✗ 断言失败
            #   而 YAML/JSON 存不了 (1,5)，读回来必然是 1-D 列表，
            #   所以**必须在代码里 reshape**，不能指望文件里存对。
            #   （早期把 D 存成 .ravel() 的结果，标定文件看着没问题但一跑就崩）
            D1 = _as_dist(st["D_left"])
            D2 = _as_dist(st["D_right"])
        except Exception as e:  # noqa: BLE001
            print("[depth] 标定文件字段不全:", e)
            return False

        self.baseline_mm = float(np.linalg.norm(T)) or float(st.get("baseline_mm", 0.0))
        w = int(round(W * self.scale))
        h = int(round(H * self.scale))

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, (W, H), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

        # 直接在缩放后的分辨率上生成映射表：把新内参的焦距/主点按比例缩小
        P1s = P1.copy()
        P2s = P2.copy()
        P1s[0, 0] *= self.scale; P1s[0, 2] *= self.scale
        P1s[1, 1] *= self.scale; P1s[1, 2] *= self.scale
        P2s[0, 0] *= self.scale; P2s[0, 2] *= self.scale
        P2s[1, 1] *= self.scale; P2s[1, 2] *= self.scale

        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1s, (w, h), cv2.CV_32FC1)
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2s, (w, h), cv2.CV_32FC1)
        self.fx = float(P1s[0, 0])
        self.out_size = (w, h)
        self.Q = Q

        # ★ 匹配器选择：默认 SGBM。
        #   实测（绿幕工位、黑色 3D 打印料盘）StereoBM 在料盘区只有 9% 有效像素、
        #   视差散布高达 75 px（纯噪声），过不了 min_valid_ratio 闸门 → 深度整帧被丢。
        #   根因是料盘表面几乎无纹理（Lap 方差 156，只有整幅的 6%），
        #   局部块匹配器无料可比；SGBM 靠半全局平滑把料盘区有效像素提到 46%、散布降到 6.6 px。
        kind = str(self.cam.get("depth.matcher", "sgbm")).lower()
        if kind in ("bm", "stereobm"):
            self.matcher = cv2.StereoBM_create(numDisparities=self.num_disp,
                                               blockSize=self.block)
            self.matcher.setMinDisparity(0)
            self.matcher.setTextureThreshold(10)
            self.matcher.setSpeckleWindowSize(100)
            self.matcher.setSpeckleRange(4)
        else:
            self.matcher = cv2.StereoSGBM_create(
                minDisparity=0, numDisparities=self.num_disp, blockSize=self.block,
                P1=8 * self.block * self.block, P2=32 * self.block * self.block,
                disp12MaxDiff=1, uniquenessRatio=10,
                speckleWindowSize=300, speckleRange=4)
        self.matcher_kind = "bm" if kind in ("bm", "stereobm") else "sgbm"
        self.ready = True
        print("[depth] 就绪：匹配器 %s，基线 %.1f mm，输出 %dx%d，fx=%.1f，视差档数 %d"
              % (self.matcher_kind.upper(), self.baseline_mm, w, h, self.fx, self.num_disp))
        return True

    # ── 计算 ──────────────────────────────────────────────────
    def compute(self, left_bgr: np.ndarray, right_bgr: np.ndarray
                ) -> Optional[np.ndarray]:
        """返回深度图（mm，float32，无效处为 nan）。"""
        if not self.ready or left_bgr is None or right_bgr is None:
            return None
        gl = cv2.remap(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY), self.map1_l, self.map2_l,
                       cv2.INTER_LINEAR)
        gr = cv2.remap(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY), self.map1_r, self.map2_r,
                       cv2.INTER_LINEAR)
        disp = self.matcher.compute(gl, gr).astype(np.float32) / 16.0
        valid = disp > 0.5
        if float(valid.mean()) < self.min_valid:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(valid, self.fx * self.baseline_mm / np.maximum(disp, 1e-6),
                             np.nan)
        return depth.astype(np.float32)

    # ── 高度图与读取 ──────────────────────────────────────────
    @staticmethod
    def fit_plane(depth: np.ndarray, mask: Optional[np.ndarray] = None,
                  iters: int = 3) -> Optional[Tuple[float, float, float]]:
        """拟合 Z = a*u + b*v + c（深度平面）。用迭代最小二乘剔除离群点。"""
        h, w = depth.shape[:2]
        vv, uu = np.mgrid[0:h, 0:w]
        sel = np.isfinite(depth)
        if mask is not None:
            sel &= (mask > 0)
        if int(sel.sum()) < 50:
            return None
        u = uu[sel].astype(np.float64); v = vv[sel].astype(np.float64)
        z = depth[sel].astype(np.float64)
        abc = np.array([0.0, 0.0, float(np.median(z))])
        for _ in range(max(1, iters)):
            A = np.stack([u, v, np.ones_like(u)], axis=1)
            abc, *_ = np.linalg.lstsq(A, z, rcond=None)
            resid = z - A @ abc
            s = float(np.median(np.abs(resid))) * 3.0 + 1e-6
            keep = np.abs(resid) < max(s, 3.0)
            if keep.sum() < 50:
                break
            u, v, z = u[keep], v[keep], z[keep]
        return float(abc[0]), float(abc[1]), float(abc[2])

    @staticmethod
    def height_map(depth: np.ndarray, plane: Tuple[float, float, float]) -> np.ndarray:
        """h = 平面深度 − 实测深度（正值 = 比平面更靠近相机，即更高）。"""
        h, w = depth.shape[:2]
        vv, uu = np.mgrid[0:h, 0:w]
        z_plane = plane[0] * uu + plane[1] * vv + plane[2]
        return (z_plane - depth).astype(np.float32)

    @staticmethod
    def height_at(hmap: np.ndarray, center_px: Tuple[float, float],
                  radius_px: float) -> Optional[float]:
        """取圆盘区域内的中值高度（mm）。中值比均值抗空洞与离群点。"""
        h, w = hmap.shape[:2]
        cx, cy = float(center_px[0]), float(center_px[1])
        r = max(1.0, float(radius_px))
        x0, x1 = int(max(0, cx - r)), int(min(w, cx + r + 1))
        y0, y1 = int(max(0, cy - r)), int(min(h, cy + r + 1))
        if x1 <= x0 or y1 <= y0:
            return None
        sub = hmap[y0:y1, x0:x1]
        vv, uu = np.mgrid[y0:y1, x0:x1]
        sel = ((uu - cx) ** 2 + (vv - cy) ** 2) <= r * r
        vals = sub[sel & np.isfinite(sub)]
        if vals.size < 20:
            return None
        return float(np.median(vals))
