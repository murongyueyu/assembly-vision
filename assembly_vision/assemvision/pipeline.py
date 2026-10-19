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

from pathlib import Path
import threading
import copy
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config, load_config, merge
from .core.checks import Qualifier
from .core.depth import DepthEstimator, NativeHeightEstimator
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
        self.depth_frame: Optional[Frame] = None
        self.depth_updated_at = 0.0
        self.depth_captured_at = 0.0
        self.stats: Dict[str, object] = {}
        self.asm_flags: Dict[str, object] = {}
        self.full_request: str = ""            # "" | "tray" | "assembly"
        self.full_result: Optional[AssemblyResult] = None
        self.quit = False


class Pipeline:
    def __init__(self, cam_cfg: Config, wp_cfg: Config,
                 calib: Optional[Config] = None,
                 source_override: Optional[str] = None,
                 path_override: Optional[str] = None,
                 tray_observer_path: Optional[str] = None,
                 tray_observer_runtime_path: Optional[str] = None,
                 prestart_check_path: Optional[str] = None,
                 process_sequence_path: Optional[str] = None,
                 integration_path: Optional[str] = None) -> None:
        self.cam = cam_cfg
        self.wp = wp_cfg
        self.calib = calib
        self.sh = Shared()
        self.native_depth = None
        self._depth_index = -1
        self.integration = None
        if integration_path is not None:
            if any(p is not None for p in (process_sequence_path, prestart_check_path, tray_observer_path)):
                raise ValueError('ABCDE集成入口与独立C/D/E入口互斥')
            from .integration import Integration
            self.integration = Integration(integration_path, cam_cfg)
            process_sequence_path = self.integration.sequence_path
        self.e_mode = process_sequence_path is not None
        if self.e_mode:
            self.sh.lock = threading.RLock()
            self._e_generation = 0
            if prestart_check_path is not None or tray_observer_path is not None:
                raise ValueError('E入口与独立C/D入口互斥')
            from .core.fsm import PickSequence
            self.e_settings = load_config(process_sequence_path).data
            self.sequence = PickSequence(wp_cfg, self.e_settings)
            prestart_check_path = str((Path(process_sequence_path).resolve().parent /
                                      self.e_settings['prestart']).resolve())
        self.d_mode = prestart_check_path is not None
        if self.d_mode and tray_observer_path is not None:
            raise ValueError('D开工模式与独立C观察模式互斥')
        self.c_mode = tray_observer_path is not None or self.d_mode
        self.c_error = ""
        self._threads = []
        if self.c_mode:
            # ★ C/D独立分支：不构造旧D/E、深度或自检；仅D增加冻结v5身份适配层。
            from .core.tray_observer import TrayObserver
            if self.d_mode:
                from .core.checks import PrestartGate
                from .core.part_identity import FrozenPartIdentity
                self.d_settings = load_config(prestart_check_path).data
                self.d_expected = {s['id']: s['part'] for s in wp_cfg.get('slots')}
                self.d_gate = PrestartGate(self.d_expected, self.d_settings)
                self.identity = FrozenPartIdentity(wp_cfg, cam_cfg, prestart_check_path, self.d_settings)
                tray_observer_path = str((Path(prestart_check_path).resolve().parent /
                                          self.d_settings['identity']['observer']).resolve())
                self._d_snapshot = None
                self._d_intrinsics = None
                self._d_source_kind = source_override or cam_cfg.get('source')
            runtime = tray_observer_runtime_path or str(
                Path(__file__).resolve().parents[1] / "config/tray_observer_runtime.yaml")
            self.c_max_age_s = float(load_config(runtime).get("display.max_age_s"))
            if self.d_mode:
                self.c_max_age_s = self.d_settings['freshness']['max_age_s']
            if not np.isfinite(self.c_max_age_s) or self.c_max_age_s <= 0:
                raise ValueError("C显示有效期必须为有限正数")
            self.observer = TrayObserver.from_file(wp_cfg, calib, tray_observer_path)
            self.segmenter = Segmenter(wp_cfg)
            self.tray = self.observer.tray
            source_cfg = Config(copy.deepcopy(cam_cfg.data))
            # 只覆盖C来源的内存参数，不改正式相机控件/配置，也不启动深度流。
            if self.integration is not None:
                source_cfg = self.integration.camera
                self.native_depth = self.integration.estimator
            else:
                source_cfg.set("depth.enabled", False)
            self.source = make_source(source_cfg, source_override, path_override)
            self.depth = None
            self._c_input = None
            self._c_snapshot = None
            self._c_last_index = None
            return
        if (bool(cam_cfg.get("depth.enabled", False))
                and cam_cfg.get("depth.backend", "") == "realsense"):
            self.native_depth = NativeHeightEstimator(cam_cfg)

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
        qualifier_cfg = wp_cfg
        if self.native_depth is not None:
            # B 接口：判据实际读取 workpiece 的 assembly 字段，不能只写 calib。
            # 只覆盖内存副本的高度基线/容差，不改变正式 YAML 和其他判据。
            qualifier_cfg = Config(copy.deepcopy(wp_cfg.data))
            height = (cam_cfg.get("native_depth.baseline_height_mm", None)
                      if cam_cfg.get("native_depth.baseline_accepted", False) else None)
            qualifier_cfg.set("assembly.baseline.wheel_height_mm", height)
            qualifier_cfg.set("assembly.wheel_height_mm", height)
            qualifier_cfg.set("check.column.wheel_height_tol_mm", cam_cfg.get("native_depth.height_tolerance_mm"))
        self.qualifier = Qualifier(qualifier_cfg, self.tray, calib)
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
        if bool(cam_cfg.get("depth.enabled", True)) and self.native_depth is None:
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
        if self.c_mode:
            try:
                self.source.open()
                if hasattr(self.source, "apply_locked_controls"):
                    self.source.apply_locked_controls()
                    for _ in range(int(self.cam.get("capture.warmup_frames", 12))):
                        self.source.read(want_full=False)
                if self.d_mode:
                    if self._d_source_kind in ('images', 'image', 'video'):
                        self._d_intrinsics = self.d_settings['identity']['replay_intrinsics']
                    else:
                        self._d_intrinsics = self.source.color_intrinsics()
                    if not self._d_intrinsics or self._d_intrinsics.get('fx', 0) <= 0:
                        raise ValueError('D缺少当前SDK内参或显式同工位回放内参')
                    expected = tuple(self.observer.settings['image_size'])
                    if (self._d_intrinsics.get('width'), self._d_intrinsics.get('height')) != expected:
                        raise ValueError('D内参分辨率与C模板不一致，禁止外推')
                self._threads = [
                    threading.Thread(target=self._c_acq_loop, name="D-acq" if self.d_mode else "C-acq", daemon=True),
                    threading.Thread(target=self._c_loop, name="D-check" if self.d_mode else "C-observe", daemon=True)]
                for thread in self._threads:
                    thread.start()
            except Exception:
                self.stop()
                raise
            return
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
            if t.ident is not None:
                t.join(timeout=1.0)
        self.source.close()
        if self.c_mode:
            # 读帧可能阻塞到SDK超时；关闭来源后再确认全部退出。
            for t in self._threads:
                if t.ident is not None:
                    t.join(timeout=1.0)
            if any(t.is_alive() for t in self._threads):
                raise RuntimeError("C线程未正常退出")

    def _c_fail(self, reason, frame=None, captured_at=0.0):
        # ★ 异常覆盖成功快照；采集失败不与正在计算的旧成功竞争发布。
        with self.sh.lock:
            self.c_error = reason
            self._c_snapshot = (frame, captured_at, None, reason)
            self.sh.quit = True
        print(("[D] " if self.d_mode else "[C] ") + reason, flush=True)

    def _c_acq_loop(self):
        try:
            while not self.sh.quit:
                frame = self.source.read(want_full=self.integration is not None)
                captured_at = time.monotonic()
                if frame is None:
                    time.sleep(0.005)
                    continue
                # 所有权归本次采集，避免SDK或来源复用缓冲区改写待处理画面。
                frame = copy.copy(frame)
                frame.left = None if frame.left is None else frame.left.copy()
                if self.integration is not None:
                    frame.depth_mm = None if frame.depth_mm is None else frame.depth_mm.copy()
                    frame.depth_points_mm = (None if frame.depth_points_mm is None
                                             else frame.depth_points_mm.copy())
                    frame.depth_profile = copy.deepcopy(frame.depth_profile)
                with self.sh.lock:
                    self._c_input = (frame, captured_at)
        except Exception as exc:
            self._c_fail("采集异常：" + str(exc))

    def _c_process_once(self):
        if self.d_mode:
            return self._d_process_once()
        with self.sh.lock:
            item = self._c_input
            if self.c_error or item is None or item[0].index == self._c_last_index:
                return False
            frame, captured_at = item
            self._c_last_index = frame.index
        observation, reason = None, ""
        try:
            expected = tuple(self.observer.settings['image_size'])
            if (frame.left is None or frame.left.dtype != np.uint8
                    or frame.left.ndim != 3 or frame.left.shape[2] != 3
                    or frame.size != expected or frame.scale != 1.0):
                reason = "输入无效：C模式需要绑定分辨率的原始BGR图像及scale=1"
            else:
                seg = self.segmenter.segment(
                    frame.left, ts=frame.ts, scale=1.0,
                    mm_per_px=self.observer.tray.mm_per_px())
                masks = (seg.metal_mask, seg.glove_mask, seg.brass_mask)
                if any(mask is None or mask.shape != frame.left.shape[:2]
                       or mask.dtype != np.uint8 for mask in masks):
                    reason = "输入无效：缺少同帧、同尺寸的金属/手套/黄铜掩膜"
                else:
                    observation = self.observer.observe(
                        frame.left, seg.metal_mask, seg.glove_mask, brass_mask=seg.brass_mask)
                    reason = observation.reason
        except Exception as exc:
            self._c_fail("处理异常：" + str(exc), frame, captured_at)
            return True
        with self.sh.lock:
            if not self.c_error:
                # ★ 一次发布对应帧和全部观测；UI绝不读取最新采集帧作底图。
                self._c_snapshot = (frame, captured_at, observation, reason)
        return True

    def _d_process_once(self):
        from .core.checks import EvidenceStamp, evaluate_prestart, prestart_unknown
        with self.sh.lock:
            item = self._c_input
            if self.c_error or item is None or item[0].index == self._c_last_index:
                return False
            frame, captured_at = item
            self._c_last_index = frame.index
            generation = self._e_generation if self.e_mode else None
        shape = frame.left.shape[:2] if frame.left is not None else (0, 0)
        stamp = EvidenceStamp(frame.index, captured_at, shape)
        observation, polygons, audit = None, {}, None
        try:
            expected = tuple(self.observer.settings['image_size'])
            if (frame.left is None or frame.left.dtype != np.uint8 or frame.left.ndim != 3
                    or frame.left.shape[2] != 3 or frame.size != expected or frame.scale != 1.0):
                result = prestart_unknown(stamp, self.d_expected, 'D需要绑定分辨率的原始BGR及scale=1')
                self.identity.reset()
            else:
                # ★ 同一原图两套冻结分割分别使用；C不能改用v5实验掩膜。
                seg = self.segmenter.segment(frame.left, ts=frame.ts, scale=1.,
                                               mm_per_px=self.tray.mm_per_px())
                if any(m is None or m.shape != shape or m.dtype != np.uint8
                       for m in (seg.metal_mask, seg.glove_mask, seg.brass_mask)):
                    raise ValueError('缺少同帧同尺寸C三掩膜')
                objects, audit = self.identity.observe(frame.left, stamp, self._d_intrinsics)
                # ★ 手只用同帧冻结分割的矩形框显示，不拟合或绘制手的轮廓。
                audit['glove_bbox'] = seg.glove_bbox
                observation = self.observer.observe(frame.left, seg.metal_mask, seg.glove_mask,
                                                     brass_mask=seg.brass_mask)
                if observation.orientation_ok and observation.pose is not None:
                    for slot in self.tray.slots:
                        center = observation.pose.apply([slot['center']])[0]
                        polygons[slot['id']] = self.tray.slot_polygon_px(slot['id'], center_override=center,
                                                    angle_offset_deg=observation.pose.theta_deg)
                result = evaluate_prestart(self.d_expected, polygons, observation, objects,
                                            stamp, stamp, time.monotonic(), self.d_settings)
            with self.sh.lock:
                if not self.c_error:
                    if self.e_mode and generation != self._e_generation:
                        return True
                    result = self.d_gate.update(result, time.monotonic())
                    if self.e_mode:
                        # ★ E候选消费同帧实际A轨迹；部分遮挡可见跟随不等于零件消失。
                        tracking = None
                        if self.sequence.tracking is not None and audit is not None:
                            tracking = (stamp, self.identity.tracker.tracks,
                                        polygons if observation.observed_current else {})
                        self.sequence.update(result, time.monotonic(), tracking=tracking)
                        if self.integration is not None and len(self.sequence.done) != len(self.sequence.order):
                            self.integration.reset()
                    self._d_snapshot = (frame, result, observation, polygons, audit)
        except Exception as exc:
            self._c_fail('D处理异常：' + str(exc), frame, captured_at)
        return True

    def _d_current_snapshot(self):
        from .core.checks import EvidenceStamp
        with self.sh.lock:
            if self._d_snapshot is None:
                stamp = EvidenceStamp(-1, 0., (0, 0))
                return None, self.d_gate.invalidate(stamp, self.c_error or '等待首帧'), None, {}, None
            frame, result, observation, polygons, audit = self._d_snapshot
            age = time.monotonic() - result.stamp.captured_at
            if self.c_error or not np.isfinite(age) or not 0 <= age <= self.c_max_age_s:
                result = self.d_gate.invalidate(result.stamp, self.c_error or '观测过期')
                self._d_snapshot = (frame, result, None, {}, None)
                if self.e_mode:
                    self.sequence.interrupt(result.reason)
                return self._d_snapshot
            return self._d_snapshot

    def get_prestart_result(self):
        """D动态开工信号；E历史开工另行锁存。读取时禁止沿用旧放行。"""
        if not self.d_mode:
            raise RuntimeError('未启用D开工模式')
        return copy.deepcopy(self._d_current_snapshot()[1])

    def _build_d_ui_state(self):
        frame, result, observation, polygons, audit = self._d_current_snapshot()
        st = UIState(frame=frame.left if frame else None)
        st.level = OK if result.allow_start else ERROR if result.state == 'alarm' else WARN
        st.title = ('D：可以开始组装' if result.allow_start else
                    'D：错放/缺件报警' if result.state == 'alarm' else 'D：等待确认')
        names = self.wp.get('parts')
        label = lambda pid: names.get(pid, {}).get('name', pid) if pid else '未知'
        states = {'match': '匹配', 'missing': '缺件', 'wrong': '错件', 'unknown': '不确定'}
        alarms = []
        for sid, row in result.slots.items():
            st.detail.append('%s槽：%s' % (label(row.expected), states[row.state]))
            st.detail.append('当前：空槽' if row.state == 'missing' else '当前身份：' + label(row.observed))
            if row.state in ('missing', 'wrong'):
                alarms.append(('缺少' + label(row.expected)) if row.state == 'missing' else
                              ('应为%s，实为%s' % (label(row.expected), label(row.observed))))
            if result.valid and sid in polygons:
                # C槽位细灰线，A当前真实轮廓另画粗线，避免把槽形当零件轮廓。
                st.polygons.append((polygons[sid], IDLE, 1))
        if audit:
            for index, obj in enumerate(audit.get('objects', [])):
                if not obj.current:
                    continue
                rows = [r for r in result.slots.values() if r.object_index == index]
                state = rows[0].state if len(rows) == 1 else 'unknown'
                color = ERROR if state == 'wrong' else OK if state == 'match' else WARN
                st.polygons.append((obj.contour.copy(), color, 3))
                identity_label = label(obj.part) if obj.confirmed else '身份待确认'
                if obj.source == 'partial':
                    identity_label += '（当前残片）'
                st.boxes.append((cv2.boundingRect(obj.contour), identity_label, color))
            if audit.get('glove_bbox') is not None:
                st.boxes.append((audit['glove_bbox'], '手套遮挡', WARN))
        if result.reason:
            st.detail.insert(0, result.reason)
        st.detail.append('仅开工联检，未启动工序')
        unknown_reasons = [row.reason for row in result.slots.values() if row.state == 'unknown']
        st.banner = '；'.join(alarms) or result.reason or (unknown_reasons[0] if unknown_reasons else '')
        st.source_name = self.source.name
        st.stats = {'帧号': result.stamp.index, '稳定帧': result.stable_frames,
                    '允许开始': '是' if result.allow_start else '否'}
        if self.e_mode:
            with self.sh.lock:
                verdict = copy.deepcopy(self.sequence.verdict)
                started = self.sequence.started
            st.level, st.title = verdict.level, verdict.title
            st.step, st.total = verdict.step, verdict.total
            st.detail = verdict.detail + (st.detail[:-1] if not started else [])
            st.banner = verdict.title if verdict.level == ERROR else ''
            st.stats['流程已开工'] = '是' if started else '否'
            if self.integration is not None:
                self.integration.decorate(st, started)
        return st

    def inspect_integration(self):
        """UI主线程选冻结同帧测区；采集和流程窗口保持运行。"""
        with self.sh.lock:
            self.integration.reset()
            frame, result, _, _, _ = self._d_current_snapshot()
            if (frame is None or not self.sequence.started or len(self.sequence.done) != len(self.sequence.order)
                    or time.monotonic() - result.stamp.captured_at > self.c_max_age_s):
                self.integration.reason = '待测：需要当前有效观测及取件4/4'
                return
            generation = self._e_generation
            stamp = result.stamp
            # 帧对象已归本次采集所有，不用后续画面替换选区底图。
        if frame.depth_points_mm is None:
            self.integration.reason = '无法测量：本帧无原生深度'
            return
        rects = []
        for name in ('Reference: bare green mat only', 'Target: current assembly CAP top only'):
            window = '%s | frame %s | ENTER confirm / C cancel' % (name, frame.index)
            rect = cv2.selectROI(window, frame.left, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(window)
            if rect[2] <= 0 or rect[3] <= 0:
                self.integration.reason = '待测：人工选区已取消'
                return
            rects.append(rect)
        with self.sh.lock:
            if generation != self._e_generation or len(self.sequence.done) != len(self.sequence.order):
                self.integration.reset()
                return
            try:
                self.integration.measure(frame, stamp, *rects, generation)
            except ValueError as exc:
                self.integration.reset()
                self.integration.reason = '无法测量：' + str(exc)

    def _c_loop(self):
        while not self.sh.quit:
            if not self._c_process_once():
                time.sleep(0.005)

    def _build_c_ui_state(self):
        with self.sh.lock:
            snapshot = self._c_snapshot
        st = UIState(source_name=self.source.name, level=WARN)
        st.title = "C观察：不确定"
        labels = {'empty': '空', 'occupied': '有件', 'unknown': '不确定'}
        totals = {'empty': '空盘', 'full': '满盘', 'partial': '部分占用', 'unknown': '不确定'}
        names = {'slot_base': '底座槽', 'slot_column': '立柱槽',
                 'slot_wheel': '轮盘槽', 'slot_cap': '顶盖槽'}
        observation, valid = None, False
        reason = "等待首帧观测"
        if snapshot is not None:
            frame, captured_at, observation, reason = snapshot
            st.frame = frame.left if frame is not None else None
            age = max(0.0, time.monotonic() - captured_at)
            st.stats = {"帧号": frame.index if frame else "-", "观测年龄": "%.2fs" % age}
            if self.c_error:
                reason = self.c_error
            elif age > self.c_max_age_s:
                reason = "观测过期"
            else:
                valid = bool(observation is not None and observation.observed_current
                             and observation.orientation_ok and observation.pose is not None)
            if valid:
                st.title = "C观察：" + totals[observation.state]
                st.stats["方向估计"] = "%.1f°" % (observation.pose.theta_deg % 360)
        for slot in self.tray.slots:
            sid = slot['id']
            state = observation.slots.get(sid, {}).get('state', 'unknown') if valid else 'unknown'
            st.detail.append(names.get(sid, sid) + "：" + labels[state])
            if valid:
                center = observation.pose.apply([slot['center']])[0]
                polygon = self.tray.slot_polygon_px(
                    sid, center_override=center, angle_offset_deg=observation.pose.theta_deg)
                # 用空/有件文字说明；颜色不承担合格或身份判据。
                st.polygons.append((polygon, IDLE if state == 'empty' else WARN, 2))
        if reason:
            st.detail.append(reason)
        st.detail.extend(["有件仅表示占用", "不代表身份正确或合格"])
        return st

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
                if want and self.native_depth is not None:
                    self.sh.depth_frame = f
                    self.sh.depth_captured_at = time.monotonic()
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
            # ★★ mm_per_px 必须传（2026-10 补）：它是**尺寸闸门**的尺度来源，
            #   也是"尺度唯一来源 = TrayMap.mm_per_px()"这条铁律的落点 ——
            #   `core/` 不去猜尺度，只接受调用方给的数（含 scale）。
            seg = self.segmenter.segment(f.left, ts=f.ts, scale=f.scale, work_roi=roi,
                                         mm_per_px=self.tray.mm_per_px(f.scale))
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
            wh = self._current_height()
            col_depth = None
            if wh is not None:
                asm_cfg = self.wp.get("assembly", {}) or {}
                base_h = float((asm_cfg.get("baseline", {}) or {}).get("base_height_mm", 5.0))
                if self.native_depth is not None:
                    threshold = self.cam.get("native_depth.column_threshold_mm", None)
                    col_depth = None if threshold is None else bool(wh > float(threshold))
                else:
                    col_depth = bool(wh > (base_h + 30.0))
            # ★★ 必须传 pose：`work_roi_mask()` 里装配位是**跟着位姿修正的**，
            #   如果这里不传，就会出现"ROI 跟着料盘走了、判定用的圈没走"——
            #   圈住的和判的不是同一个地方（实测踩过，见 checks.assembly_polygon_px）。
            flags = self.qualifier.assembly_flags(seg, scale, column_by_depth=col_depth,
                                                  pose=self._last_pose)

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
                                         want_features=False,
                                         # ★ mm_per_px 同理（尺寸闸门 + metal 掩膜裁剪）
                                         mm_per_px=self.tray.mm_per_px(1.0))
        wh = self._current_height()
        # ★ 同样要传 pose（理由见 assembly_flags 那处注释）
        asm = self.qualifier.check_assembly(seg_full, scale=1.0, wheel_height_mm=wh,
                                            pose=self._last_pose)
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
            if (self.native_depth is not None or (self.depth is not None and self.depth.ready)) and now - self._wheel_ts >= d_every:
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
        if self.native_depth is not None:
            with self.sh.lock:
                f = self.sh.depth_frame
                captured_at = self.sh.depth_captured_at
                if not self.sh.full_request:
                    self.sh.full_request = "depth"
            if f is None or f.index == self._depth_index:
                return
            self._depth_index = f.index
            try:
                result = self.native_depth.measure_frame(f)
            except (ValueError, np.linalg.LinAlgError) as exc:
                result = dict(ok=False, height_mm=None, reason=str(exc))
            with self.sh.lock:
                self.sh.wheel_height_mm = result["height_mm"] if result["ok"] else None
                self.sh.depth_updated_at = captured_at
                self.sh.stats["深度状态"] = result["reason"]
                self.sh.stats["轮盘高度"] = ("%.1fmm" % result["height_mm"] if result["ok"] else "未知")
            return
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

    def _current_height(self):
        """B 接口：深度失效或过期必须为未知，不能沿用旧高度。"""
        with self.sh.lock:
            if self.native_depth is not None:
                age = time.monotonic() - self.sh.depth_updated_at
                if age > float(self.cam.get("native_depth.max_age_s")):
                    self.sh.stats["轮盘高度"] = "未知（深度过期）"
                    return None
            return self.sh.wheel_height_mm

    # ══════════════════════════════════════════════════════════
    # 给界面调用的动作（快捷键：空格 / R）
    # ══════════════════════════════════════════════════════════
    def request_inspection(self, kind: str = "assembly") -> None:
        """手动触发一次全分辨率抽检（等操作者按空格，或演示时救场用）。"""
        if self.integration is not None:
            return self.inspect_integration()
        if self.c_mode:
            return
        with self.sh.lock:
            self.sh.full_request = kind
        self._inspect_waiting = True
        self._inspect_ts = time.time()
        print("[pipeline] 手动触发抽检:", kind)

    def reset_process(self) -> None:
        """复位：清空装配状态记忆、事件与终检结果，等待下一次装配。"""
        if self.e_mode:
            with self.sh.lock:
                self.sequence.reset()
                self._e_generation += 1
                from .core.checks import PrestartGate
                self.d_gate = PrestartGate(self.d_expected, self.d_settings)
                self._d_snapshot = None
                if self.integration is not None:
                    self.integration.reset()
            return
        if self.c_mode:
            return
        self.net.reset()
        self.deb.reset()
        self.detector = EventDetector()
        with self.sh.lock:
            self.sh.assembly = None
            self.sh.events = []
            self.sh.verdict = Verdict()
            self.sh.wheel_height_mm = None
            self.sh.depth_frame = None
            self.sh.depth_updated_at = 0.0
        self._inspect_waiting = False
        print("[pipeline] 流程已复位")

    # ══════════════════════════════════════════════════════════
    # UI 数据组装
    # ══════════════════════════════════════════════════════════
    def build_ui_state(self) -> UIState:
        if self.e_mode:
            with self.sh.lock:
                return self._build_d_ui_state()
        if self.d_mode:
            return self._build_d_ui_state()
        if self.c_mode:
            return self._build_c_ui_state()
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
