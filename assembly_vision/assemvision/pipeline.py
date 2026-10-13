# -*- coding: utf-8 -*-
"""线程编排：把 sources / core / ui 串起来跑。

线程划分（4 核）
    T1 采集      30 Hz   唯一读相机的人（V4L2 只能一个线程读）
    T2 分割      12-15 Hz 黄铜 + 蓝手套掩膜
    T3 逻辑      12 Hz    覆盖率 / 辐条 / 谓词 / 流程 / 事件
    T4 抽检      事件触发  全分辨率合格性 + 齐套性
    T5 深度      1 Hz     轮盘高度（判立柱）
    T6 位姿自检  2 Hz     标记检测 + 单应补偿
    UI 主线程    25-30 Hz

共享状态只放小对象与掩膜，不放大图队列；需要全分辨率帧时由采集线程代抓。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config, load_config, merge
from .core.checks import Qualifier
from .core.depth import DepthEstimator
from .core.events import EventDetector
from .core.fsm import Debouncer, PreconditionNet, evaluate_predicates
from .core.pose_check import PoseChecker
from .core.segment import Segmenter, glove_hits
from .core.roi import TrayMap
from .core.tracker import BlobTracker
from .sources.factory import make_source
from .types import (ERROR, IDLE, OK, WARN, AssemblyResult, Event, Frame,
                    SegResult, SlotResult, SpokeResult, Verdict)
from .ui.base import UIState


class Shared:
    """带锁的共享状态。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame_small: Optional[Frame] = None
        self.frame_full_left: Optional[np.ndarray] = None
        self.frame_full_right: Optional[np.ndarray] = None
        self.frame_full_index: int = -1
        self.seg: Optional[SegResult] = None
        self.tracks: List = []
        self.slots: List[SlotResult] = []
        self.spokes: Optional[SpokeResult] = None
        self.assembly: Optional[AssemblyResult] = None
        self.verdict: Verdict = Verdict()
        self.events: List[Event] = []
        self.pose_msg: str = ""
        self.wheel_height_mm: Optional[float] = None
        self.stats: Dict[str, object] = {}
        self.asm_flags: Dict[str, object] = {}
        self.full_request: str = ""            # "" | "tray" | "assembly"
        self.full_result: Optional[AssemblyResult] = None
        self.quit = False


class Pipeline:
    def __init__(self, cam_cfg: Config, wp_cfg: Config,
                 calib: Optional[Config] = None,
                 source_override: Optional[str] = None,
                 path_override: Optional[str] = None) -> None:
        self.cam = cam_cfg
        self.wp = wp_cfg
        self.calib = calib
        self.sh = Shared()

        # ── 组装各模块 ──
        H = None
        ref_scale = 1.0
        if calib is not None and "tray_map" in calib.data:
            H = np.asarray(calib.get("tray_map.homography"), dtype=np.float64)
            ref_scale = float(calib.get("tray_map.scale", 1.0))
        else:
            print("[pipeline] 没有料盘标定（calibration.yaml 缺失或字段不全），"
                  "ROI 用『每毫米1像素』的仿射兜底 —— 只能看连通性，不能判覆盖率。")
        self.tray = TrayMap(wp_cfg, homography=H, ref_scale=ref_scale,
                            px_per_mm=1.0, origin_px=(0.0, 0.0))
        self.segmenter = Segmenter(wp_cfg)
        self.qualifier = Qualifier(wp_cfg, self.tray, calib)
        self.net = PreconditionNet(wp_cfg,
                                   timeout_s=float(wp_cfg.get("ui.step_timeout_s", 15.0)))
        # 去抖：默认按"连续 N 帧"。★ 低帧率下必须改成按时间 ——
        # 本机实测 2560x960（真双目）只有约 4.5 fps，按 2 帧去抖就要等 0.44 s，
        # 按 6 帧就要等 1.3 s，取件事件与流程判断会明显迟滞。
        # camera.yaml 里 frame_rate_hint 写实测帧率；低于 10 fps 就切到 hold_s。
        self.deb = self._make_debouncer(cam_cfg, wp_cfg)
        self.detector = EventDetector()
        self.tracker = BlobTracker()
        # ★ 料盘位姿缓存：零件被全部拿走时（装配完成后）没有基准点了，
        #   沿用最后一次成功的位姿，避免槽位忽然跳回标称位置。
        self._last_pose = None
        # ★ 工位全局 ROI 的缓存（见 core/checks.work_roi_mask）
        #   T3 每轮用**上轮位姿**算一次，T2 下一帧用它来分割。
        #   为什么要缓存、不每帧重算：ROI 需要位姿，位姿又需要分割结果 ——
        #   是个先有鸡还是先有蛋的问题。料盘位置逐帧变化很慢，
        #   用"上一轮的位姿"算 ROI 完全够（顶多差一帧的挪动量）。
        #   初值 None 表示"还没算过"——第一帧不限制，之后立刻收敛。
        self._work_roi = None
        self.source = make_source(cam_cfg, source_override, path_override)

        self.depth: Optional[DepthEstimator] = None
        if bool(cam_cfg.get("depth.enabled", True)):
            self.depth = DepthEstimator(cam_cfg, calib)
        self.pose = PoseChecker(cam_cfg, wp_cfg, calib)

        self._threads: List[threading.Thread] = []
        self._seg_busy = False
        self._last_seg_ts = 0.0
        self._inspect_ts = 0.0
        self._inspect_waiting = False
        self._wheel_ts = 0.0
        self._pose_ts = 0.0
        self._fps_t = time.time()
        self._fps_n = 0
        self._fps = 0.0

    # ══════════════════════════════════════════════════════════
    @staticmethod
    def _make_debouncer(cam_cfg: Config, wp_cfg: Config) -> Debouncer:
        """按实测帧率决定去抖方式：高帧率按帧，低帧率按时间。

        为什么需要：去抖的物理意义是"稳定持续一段时间"，不是"连续若干帧"。
        帧率一变，帧数阈值就悄悄失效 —— 30 fps 下 6 帧是 0.2 s，
        4.5 fps 下同样是 6 帧却要 1.3 s，操作员会觉得系统反应迟钝。
        """
        n = int(wp_cfg.get("check.drone_consistency", 2))
        hint = float(cam_cfg.get("capture.frame_rate_hint", 0) or 0)
        if 0 < hint < 10.0:
            hold = max(0.15, n * 0.05)      # 低帧率：按时间，别按帧
            print("[pipeline] 实测帧率约 %.1f fps，去抖改为按时间（hold_s=%.2fs，"
                  "原按 %d 帧）" % (hint, hold, n))
            return Debouncer(n_frames=1, hold_s=hold)
        return Debouncer(n_frames=n)

    # ══════════════════════════════════════════════════════════
    def start(self) -> None:
        self.source.open()
        if hasattr(self.source, "apply_locked_controls"):
            self.source.apply_locked_controls()          # type: ignore[attr-defined]
            # ★ 预热：改完曝光后相机要若干帧才生效，前几帧可能是黑的/过曝的。
            #   不丢这几帧，开头的分割与深度都会基于垃圾帧。
            for _ in range(int(self.cam.get("capture.warmup_frames", 12))):
                self.source.read(want_full=False)
        if self.depth is not None:
            self.depth.setup()

        self._threads = [
            threading.Thread(target=self._acq_loop, name="T1-acq", daemon=True),
            threading.Thread(target=self._seg_loop, name="T2-seg", daemon=True),
            threading.Thread(target=self._logic_loop, name="T3-logic", daemon=True),
            threading.Thread(target=self._side_loop, name="T5-side", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self.sh.quit = True
        for t in self._threads:
            t.join(timeout=1.0)
        self.source.close()

    # ══════════════════════════════════════════════════════════
    # T1 采集 —— 唯一读相机的人
    # ══════════════════════════════════════════════════════════
    def _acq_loop(self) -> None:
        while not self.sh.quit:
            want = ""
            with self.sh.lock:
                want = self.sh.full_request
            f = self.source.read(want_full=bool(want))
            if f is None:
                time.sleep(0.005)
                continue
            self._fps_n += 1
            now = time.time()
            if now - self._fps_t >= 1.0:
                self._fps = self._fps_n / (now - self._fps_t)
                self._fps_n = 0
                self._fps_t = now
            with self.sh.lock:
                self.sh.frame_small = f
                if want and f.left_full is not None:
                    self.sh.frame_full_left = f.left_full
                    self.sh.frame_full_right = f.right_full
                    self.sh.frame_full_index = f.index
                    self.sh.full_request = ""
                self.sh.stats["FPS"] = "%.0f" % self._fps

    # ══════════════════════════════════════════════════════════
    # T2 分割
    # ══════════════════════════════════════════════════════════
    def _seg_loop(self) -> None:
        period = 1.0 / float(self.cam.get("segmentation.hz", 12.0))
        while not self.sh.quit:
            t0 = time.time()
            with self.sh.lock:
                f = self.sh.frame_small
            if f is None or f.left is None:
                time.sleep(0.005)
                continue
            # ★ 传 work_roi 把黄铜检测**限制在工位范围内**。
            #   实测踩过：木桌的 HSV 与黄铜重叠，画面顶部的木桌整块被判成黄铜，
            #   产生 13269mm² 的假连通域（真零件最大才 3281mm²），
            #   把零件识别和面积测量全带偏。ROI 是"位置即身份"，比调阈值稳。
            #   首次运行 _work_roi 还是 None（不限制），T3 一轮后就填上。
            roi = None
            with self.sh.lock:
                if self._work_roi is not None and self._work_roi.shape[:2] == f.left.shape[:2]:
                    roi = self._work_roi
            seg = self.segmenter.segment(f.left, ts=f.ts, scale=f.scale, work_roi=roi)
            seg.hand_on = glove_hits(seg.glove_mask,
                                     self.tray.slot_polygons_px(f.scale))
            with self.sh.lock:
                self.sh.seg = seg
                self.sh.stats["seg"] = "%.0fms" % seg.detail.get("seg_ms", 0.0)
            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))

    # ══════════════════════════════════════════════════════════
    # T3 逻辑：覆盖率 / 辐条 / 谓词 / 流程 / 事件
    # ══════════════════════════════════════════════════════════
    def _logic_loop(self) -> None:
        period = 1.0 / float(self.cam.get("logic.hz", 12.0))
        while not self.sh.quit:
            t0 = time.time()
            with self.sh.lock:
                seg = self.sh.seg
                f = self.sh.frame_small
            if seg is None or f is None:
                time.sleep(0.01)
                continue
            scale = f.scale

            # ★ 传 bgr=left 让 check_slots 能算**色域特征**（颜色天然旋转不变，
            #   用来补形状特征区分不了的场景）。不传的话只做形状识别。
            # ★ 传入缓存的上次位姿：零件被全部拿走时（装配完成）就没有基准点了，
            #   这时沿用上一次成功的位姿，避免槽位忽然跳回标称位置。
            slots = self.qualifier.check_slots(seg, scale, bgr=f.left,
                                               pose=self._last_pose)
            # 位姿可用且零件够多时更新缓存（只认到 1 个零件时置信度低，不覆盖）
            try:
                fe0 = slots[0].features if slots else {}
                n_used = int(fe0.get("pose_n_parts", 0))
                if n_used >= 2:
                    from .core.tray_pose import TrayPose
                    self._last_pose = TrayPose(
                        theta_deg=float(fe0.get("pose_theta_deg", 0.0)),
                        tx=float(fe0.get("pose_dx_mm", 0.0)),
                        ty=float(fe0.get("pose_dy_mm", 0.0)),
                        n_used=n_used,
                        residual_mm=float(fe0.get("pose_resid_mm", 0.0)),
                        ok=True)
            except Exception:                                # noqa: BLE001
                pass
            # ★ 刷新工位 ROI（给下一帧的分割用）。
            #   带 margin 外扩，料盘小范围挪动/抖动不会切到零件。
            #   pose 用刚更新的缓存（优先），没有就用本轮结果 —— 都不行时置 None
            #   （宁可"不限制"也不要"限制错"，限制错会把真零件挡在外面）。
            try:
                roi_margin = float(self.wp.get("segment.work_roi_margin_mm", 25.0))
                if roi_margin > 0:
                    m = self.qualifier.work_roi_mask(
                        (f.left.shape[0], f.left.shape[1]), scale,
                        pose=self._last_pose, margin_mm=roi_margin)
                    with self.sh.lock:
                        self._work_roi = m
            except Exception:                                # noqa: BLE001
                with self.sh.lock:
                    self._work_roi = None
            spokes = self.qualifier.check_spokes(seg, scale=scale)   # 主检位置：料盘槽
            self.tracker.update(seg.brass_blobs, ts=f.ts)

            # 装配位布尔量：流程判断一律判"装配位上有没有"，不判料盘槽里有没有
            with self.sh.lock:
                wh = self.sh.wheel_height_mm
            col_depth = None
            if wh is not None:
                asm_cfg = self.wp.get("assembly", {}) or {}
                base_h = float((asm_cfg.get("baseline", {}) or {}).get("base_height_mm", 5.0))
                col_depth = bool(wh > (base_h + 30.0))
            flags = self.qualifier.assembly_flags(seg, scale, column_by_depth=col_depth)

            ctx = {
                "asm": lambda key: bool(flags.get(key, False)),
                "slot_present": lambda sid: _slot_present(slots, sid),
                "spoke_count": flags.get("spokes", 0),
                "spoke_count_tray": spokes.count,
            }
            raw = evaluate_predicates(self.wp.get("predicates", {}) or {}, ctx)
            preds = self.deb.update(raw, f.ts)

            frozen = bool(seg.glove_bbox) and any(seg.hand_on.values())
            verdict = self.net.step(preds, f.ts, frozen=frozen)
            events = self.detector.update(f.ts, slots, seg.hand_on, preds, spokes)

            # 每步完成 / 全部完成时触发一次全分辨率抽检
            need_inspect = self.net.all_done() and (time.time() - self._inspect_ts) > 3.0

            with self.sh.lock:
                self.sh.slots = slots
                self.sh.spokes = spokes
                self.sh.verdict = verdict
                if events:
                    self.sh.events = (self.sh.events + events)[-50:]
                self.sh.stats["spokes"] = "%d/%d" % (spokes.count, spokes.required)
                self.sh.asm_flags = flags
                if need_inspect and not self.sh.full_request:
                    self.sh.full_request = "assembly"
                    self._inspect_waiting = True
                full = self.sh.frame_full_left
                full_idx = self.sh.frame_full_index
                consumed = (self.sh.full_request == "")
                cur_idx = f.index
            if need_inspect:
                self._inspect_ts = time.time()

            # 全分辨率帧已拿到、且足够新鲜 → 跑一次齐套性检查（注意：在锁外调用）
            fresh = (cur_idx - full_idx) <= 5 if full_idx >= 0 else False
            if self._inspect_waiting and consumed and full is not None:
                if fresh:
                    self._inspect_waiting = False
                    self._run_inspection(full, scale)
                else:
                    print("[inspect] 全分辨率帧已过期（差 %d 帧），重新抓一帧" % (cur_idx - full_idx))
                    self._inspect_waiting = False
                    with self.sh.lock:
                        self.sh.full_request = "assembly"
                    self._inspect_waiting = True

            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))

    def _run_inspection(self, full: np.ndarray, scale: float) -> None:
        """用一帧全分辨率图跑齐套性检查。★ 必须在锁外调用。"""
        seg_full = self.segmenter.segment(full, ts=time.time(), scale=1.0,
                                         want_features=False)
        with self.sh.lock:
            wh = self.sh.wheel_height_mm
        asm = self.qualifier.check_assembly(seg_full, scale=1.0, wheel_height_mm=wh)
        with self.sh.lock:
            self.sh.assembly = asm
        print("[inspect] 齐套性: %s%s" % (asm.verdict,
                                         ("  缺失: " + "、".join(asm.missing))
                                         if asm.missing else ""))

    # ══════════════════════════════════════════════════════════
    # T5 低频：深度 + 位姿自检
    # ══════════════════════════════════════════════════════════
    def _side_loop(self) -> None:
        d_every = float(self.cam.get("depth.every_n_frames", 30)) / 30.0
        p_every = float(self.cam.get("pose_check.every_n_frames", 15)) / 30.0
        while not self.sh.quit:
            now = time.time()
            if now - self._pose_ts >= p_every:
                self._pose_ts = now
                self._do_pose()
            if self.depth is not None and self.depth.ready and now - self._wheel_ts >= d_every:
                self._wheel_ts = now
                self._do_depth()
            time.sleep(0.05)

    def _do_pose(self) -> None:
        with self.sh.lock:
            full = self.sh.frame_full_left
        if full is None:
            with self.sh.lock:
                self.sh.full_request = "pose"
            return
        st = self.pose.check(full)
        with self.sh.lock:
            self.sh.pose_msg = st.message
            self.sh.stats["位姿"] = "OK" if st.ok else "异常"

    def _do_depth(self) -> None:
        with self.sh.lock:
            lf = self.sh.frame_full_left
            rf = self.sh.frame_full_right
        if lf is None or rf is None:
            with self.sh.lock:
                self.sh.full_request = "depth"
            return
        assert self.depth is not None
        depth = self.depth.compute(lf, rf)
        if depth is None:
            return
        plane = self.depth.fit_plane(depth)
        if plane is None:
            return
        hmap = self.depth.height_map(depth, plane)
        # 装配位中心（按半分辨率换算）
        poly = self.tray.assembly_polygon_px(1.0)
        if poly is None:
            return
        c = poly.mean(axis=0) * self.depth.scale
        h = self.depth.height_at(hmap, (c[0], c[1]), float(self.qualifier.r_outer
                                                           / self.tray.mm_per_px(1.0)
                                                           * self.depth.scale))
        with self.sh.lock:
            self.sh.wheel_height_mm = h
            if h is not None:
                self.sh.stats["轮盘高度"] = "%.1fmm" % h

    # ══════════════════════════════════════════════════════════
    # 给界面调用的动作（快捷键：空格 / R）
    # ══════════════════════════════════════════════════════════
    def request_inspection(self, kind: str = "assembly") -> None:
        """手动触发一次全分辨率抽检（等操作者按空格，或演示时救场用）。"""
        with self.sh.lock:
            self.sh.full_request = kind
        self._inspect_waiting = True
        self._inspect_ts = time.time()
        print("[pipeline] 手动触发抽检:", kind)

    def reset_process(self) -> None:
        """复位：清空装配状态记忆、事件与终检结果，等待下一次装配。"""
        self.net.reset()
        self.deb.reset()
        self.detector = EventDetector()
        with self.sh.lock:
            self.sh.assembly = None
            self.sh.events = []
            self.sh.verdict = Verdict()
            self.sh.wheel_height_mm = None
        self._inspect_waiting = False
        print("[pipeline] 流程已复位")

    # ══════════════════════════════════════════════════════════
    # UI 数据组装
    # ══════════════════════════════════════════════════════════
    def build_ui_state(self) -> UIState:
        with self.sh.lock:
            f = self.sh.frame_small
            seg = self.sh.seg
            slots = list(self.sh.slots)
            verdict = self.sh.verdict
            spokes = self.sh.spokes
            asm = self.sh.assembly
            stats = dict(self.sh.stats)
            pose = self.sh.pose_msg

        st = UIState()
        st.frame = f.left if f is not None else None
        st.level = verdict.level
        st.title = verdict.title
        st.detail = list(verdict.detail)
        st.step, st.total = verdict.step, verdict.total
        st.source_name = self.source.name

        if spokes is not None:
            st.detail.append("辐条峰数 %d/%d" % (spokes.count, spokes.required))
        if pose:
            st.detail.append(pose)
        if asm is not None:
            st.detail.append("齐套性：%s%s" % (asm.verdict,
                                            ("，缺失 " + "、".join(asm.missing))
                                            if asm.missing else ""))
        st.stats = {
            "FPS": stats.get("FPS", "-"),
            "分割": stats.get("seg", "-"),
            "辐条": stats.get("spokes", "-"),
            "轮盘高度": stats.get("轮盘高度", "-"),
        }

        if st.frame is None:
            return st
        scale = f.scale if f is not None else 1.0

        # 槽位 ROI：有件绿、缺件红、存疑黄
        for sr in slots:
            key = OK if sr.verdict == "OK" else (ERROR if sr.verdict == "NG" else WARN)
            st.polygons.append((sr.roi_px, key, 2))
        apoly = self.tray.assembly_polygon_px(scale)
        if apoly is not None:
            st.polygons.append((apoly, IDLE, 2))

        # 前景框
        if seg is not None and seg.brass_blobs:
            for b in seg.brass_blobs[:6]:
                part = self._label_for(b.centroid, scale)
                st.boxes.append((b.bbox, part, OK))
            if seg.glove_bbox is not None:
                st.boxes.append((seg.glove_bbox, "手套", WARN))

        if verdict.level == ERROR:
            st.banner = verdict.title
        return st

    def _label_for(self, centroid: Tuple[float, float], scale: float) -> str:
        best, best_d = "", 1e18
        for s in self.tray.slots:
            p = self.tray.mm_to_px([s["center"]], scale)[0]
            d = float(np.hypot(p[0] - centroid[0], p[1] - centroid[1]))
            if d < best_d:
                best, best_d = str(s.get("part", "")), d
        return best


# ── 谓词辅助 ──────────────────────────────────────────────────
def _slot_present(slots: List[SlotResult], sid: str) -> bool:
    for s in slots:
        if s.slot_id == sid:
            return bool(s.present)
    return False


def _column_visible(slots: List[SlotResult]) -> bool:
    for s in slots:
        if s.part == "P2_column":
            f = s.features or {}
            return bool(s.present) and float(f.get("elongation", 0.0)) >= 3.0
    return False
