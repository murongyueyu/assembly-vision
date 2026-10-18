# -*- coding: utf-8 -*-
"""Intel RealSense D435 —— 走 Intel 的 `pyrealsense2`，**不能用 `cv2.VideoCapture`**。

★★ 两条硬约束（都是实测出来的，见 docs/踩坑实录.md §2.30 / §2.31）：
  1. **D435 不是普通 UVC 相机**：OpenCV 按索引枚举不到它
     （MSMF / DSHOW / 默认后端 × 索引 0~5 全部 `isOpened()=False`），
     必须让 Intel SDK 直接对话硬件。→ 所以本文件是"换一套取图"，不是"改个参数"。
  2. **必须插 USB 3 口**：USB 2.1 链路上连 `424x240@6` 都拿不到一帧
     —— 不是带宽不够，是**这条链路根本不工作**。开流后会读
     `usb_type_descriptor` 自检，不是 3.x 就大声报警。
  ★ 但 `cv2.imshow` 照用：取图与显示无关，SDK 给的就是 numpy BGR 数组
    （`rs.format.bgr8`，不需要自己 `cvtColor`）。

与工程其它层的分工（分层铁律）：
  * 本文件只回答"图从哪来"，与 `uvc_stereo.py` 同层，**不碰 core/、不碰界面**。
  * B 模块显式设置 depth.enabled=true 且 depth.backend=realsense 时，
    同流采集原生深度并对齐彩色；仅 want_full 时生成毫米深度和 SDK 三维点。
    默认仍只开彩色。right/right_full 始终 None，绝不把深度塞进右眼。
  * `pose_check` / 全分辨率抽检照常：见下面 `want_full` 的说明。

分辨率与"全分辨率"：
  * 只有一个彩色传感器 ⇒ **没有"更高分辨率的 left_full"可给**，
    `left_full` 就是同一帧原图。所以全分辨率抽检**不会更清楚**，
    也不会更模糊 —— 想更清楚必须整体换分辨率（640x480 ⇄ 1280x720）并重跑标定。
  * 本工程的算法**全程只用左眼**（分割/覆盖率/辐条都在 `f.left` 上做），
    D435 的彩色流就是那只"左眼"。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..types import Frame
from .base import CameraSource, scale_frame

# ★★ USB 链路自检的判据：descriptor 必须是 3.x（实测 USB 2.1 下**一帧都拿不到**）
#   见 docs/踩坑实录.md §2.31。别把 "2.1 但分辨率低一点也许能跑" 当退路 —— 实测不能。
_USB3_PREFIX = "3"

# 配置里可以写短名，这里映射到 rs.option 的属性名。
# ★ 只列真正会用到的那些；写错的名字不会被静默吞掉，会明确报"SDK 没有这个选项"。
_CTRL_ALIASES: Dict[str, str] = {
    "auto_exposure": "enable_auto_exposure",
    "exposure": "exposure",
    "gain": "gain",
    "auto_white_balance": "enable_auto_white_balance",
    "white_balance": "white_balance",
    "backlight_compensation": "backlight_compensation",
    "brightness": "brightness",
    "contrast": "contrast",
    "gamma": "gamma",
    "hue": "hue",
    "saturation": "saturation",
    "sharpness": "sharpness",
    "power_line_frequency": "power_line_frequency",
    "laser_power": "laser_power",
    "emitter_enabled": "emitter_enabled",
}

# ★ 这些是 **DSHOW 口径**的旧名字（HBV 那台相机的 camera.yaml 里就是这么写的）。
#   量纲完全不同：DSHOW 的 -6.0 与 RealSense 的 exposure（约 1~10000）不是一回事，
#   直接抄过来会把画面弄成全黑/全白。所以这里**只做提示，不做别名**。
_HBV_CTRL_HINTS: Dict[str, str] = {
    "exposure_absolute": "exposure（RealSense 量纲约 1~10000，不是 DSHOW 的负数）",
    "white_balance_temperature": "white_balance（注意色温量纲也不同）",
    "white_balance_automatic": "enable_auto_white_balance（1=自动，0=手动）",
}


def import_rs():
    """延迟导入 `pyrealsense2`。

    ★ 为什么不在模块顶部 import：没装 SDK 的机器（离线回归、树莓派上还没装时）
      也要能 `import assemvision.sources.d435` —— 否则 `factory` 一 import 就炸，
      连"你现在用的是离线源"这种正常用法都跑不了。
    """
    try:
        import pyrealsense2 as rs                                       # noqa: PLC0415
    except Exception as e:                                             # noqa: BLE001
        raise RuntimeError(
            "用 D435 必须装 Intel 的 pyrealsense2 —— "
            "cv2.VideoCapture 永远打不开它（见 docs/踩坑实录.md §2.30）。\n"
            "  装法：pip install pyrealsense2\n"
            "  原始错误：%s: %s" % (type(e).__name__, e))
    return rs


def sdk_available() -> bool:
    """SDK 装了没。离线测试与工具用它决定走哪条分支（不抛异常）。"""
    try:
        import_rs()
        return True
    except RuntimeError:
        return False


def usb_is3(descriptor: str) -> bool:
    """`usb_type_descriptor` 是不是 USB 3.x（'3.2' / '3.1' / '3.0'）。

    ★ 拿不到描述符时返回 True（= 不报警）—— 宁可漏报，也不要在读不到信息时
      把一条能用的链路说成坏的（实测刚插拔时 `get_info` 会抛异常）。
    """
    d = str(descriptor or "").strip()
    if not d or d == "?":
        return True
    return d.startswith(_USB3_PREFIX)


class RealSenseSource(CameraSource):
    """D435 的彩色流来源。接口与 `UvcStereoSource` 完全一致（可插拔）。"""

    def __init__(self, cam_cfg: Config) -> None:
        super().__init__(scale_realtime=float(cam_cfg.get("scale.realtime", 1.0)))
        self.cfg = cam_cfg

        # ── 分辨率/帧率：以 realsense.* 为准，缺了才退回 capture.* ──
        #   ★ 默认取 640x480@30：实测纯读帧 30.0fps（60fps 也能开），
        #     且 360mm 处 GSD 0.795mm/px，与 HBV 的 0.8623 同一量级
        #     —— 换机后先别同时改分辨率，免得"看不清"和"没重标定"两件事混在一起。
        #     要上 1280x720（GSD 0.397）就改这里，并重跑 c1/c3/c4/c5/c6。
        self.w = int(cam_cfg.get("realsense.width", cam_cfg.get("capture.width", 640)))
        self.h = int(cam_cfg.get("realsense.height", cam_cfg.get("capture.height", 480)))
        self.fps = int(cam_cfg.get("realsense.fps", cam_cfg.get("capture.fps", 30)))
        # 多台相机时按序列号选；空 = 用第一台
        self.serial = str(cam_cfg.get("realsense.serial", "") or "").strip()
        # 等一帧最多多久（毫秒）。给短一点：相机掉了要能快点让上层发现。
        self.timeout_ms = int(cam_cfg.get("realsense.timeout_ms", 1000))
        self.native_depth = (bool(cam_cfg.get("depth.enabled", False))
                             and cam_cfg.get("depth.backend", "") == "realsense")
        self.depth_scale = None
        self.align = self.pointcloud = None
        # ★ 开流失败要重试：实测刚插拔/刚复位时 `query_devices()` 会抛
        #   `Mutex_TotalFailure` 或 HResult 0x800703e3，等一两秒重来就好（实录 §2.31）。
        #   ★ 但重试**救不了"相机没插"** —— 那种情况会明确报"看不到相机"。
        self.open_retries = max(1, int(cam_cfg.get("realsense.open_retries", 3)))
        self.retry_wait_s = float(cam_cfg.get("realsense.retry_wait_s", 1.5))

        self.rs = None
        self.pipe = None
        self.profile = None
        self.usb = "?"
        self.dev_name = "?"
        self.dev_serial = "?"
        self.fw = "?"
        self._read_fail = 0
        self._warned_usb = False

    # ── 生命周期 ──────────────────────────────────────────────
    def open(self) -> None:
        rs = import_rs()
        self.rs = rs
        last: Optional[Exception] = None
        for attempt in range(1, self.open_retries + 1):
            try:
                self._open_once(rs)
                self._print_ready()
                return
            except Exception as e:                                     # noqa: BLE001
                last = e
                print("[d435] 第 %d/%d 次开流失败：%s"
                      % (attempt, self.open_retries, str(e).strip()))
                self.close()
                if attempt < self.open_retries:
                    time.sleep(self.retry_wait_s)
        raise RuntimeError(
            "打不开 D435（试了 %d 次）。最后错误：%s\n"
            "  排查顺序（都是实测踩过的，见 docs/踩坑实录.md §2.30~§2.33）：\n"
            "    ① 相机插好了没、线是不是数据线（不是只供电的）\n"
            "    ② **必须插 USB 3 口**（USB 2.1 链路下一帧都拿不到，不是带宽问题）\n"
            "    ③ 关掉 RealSense Viewer / 相机 App —— 开着会**独占**设备\n"
            "    ④ 还是 busy 就硬件复位：python tools/probe_d435.py --reset\n"
            "    ⑤ 换了机器/环境：确认 pip 里装了 pyrealsense2（本模块自带提示）"
            % (self.open_retries, last))

    def _open_once(self, rs) -> None:
        """枚举设备 → 开彩色流。异常往外抛，由 open() 决定重试。"""
        dev = self._pick_device(rs)

        # 设备信息（★ 每一项都可能抛 Mutex_TotalFailure，绝不能让它把开流拦下）
        self.dev_name = self._info(dev, "name")
        self.dev_serial = self._info(dev, "serial_number")
        self.fw = self._info(dev, "firmware_version")
        self.usb = self._info(dev, "usb_type_descriptor")

        cfg = rs.config()
        if self.dev_serial not in ("?", ""):
            cfg.enable_device(self.dev_serial)
        # ★ 用 bgr8：SDK 直接给 BGR，省一次 cvtColor，也与 cap.read() 的输出一致
        cfg.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps)
        if self.native_depth:
            cfg.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, self.fps)
        self.pipe = rs.pipeline()
        try:
            self.profile = self.pipe.start(cfg)
            if self.native_depth:
                self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())
                if not np.isfinite(self.depth_scale) or self.depth_scale <= 0:
                    raise ValueError("设备 depth_scale 无效")
                self.align = rs.align(rs.stream.color)
                self.pointcloud = rs.pointcloud()
        except Exception as e:                                         # noqa: BLE001
            raise RuntimeError(
                "开流失败（请求 %dx%d@%d）：%s\n"
                "  ★ 先确认这个模式真的存在：跑 python tools/probe_d435.py 列一遍。"
                "D435 的彩色只有 1920x1080 / 1280x720 / 960x540 / 848x480 / "
                "640x480 / 640x360 / 424x240 —— 比如 1280x480 就**不是**它的模式，"
                "SDK 只会报 Couldn't resolve requests。"
                % (self.w, self.h, self.fps, str(e).strip())) from e

    def _pick_device(self, rs):
        """枚举设备并挑一台。★ 枚举本身会抛，交给外层重试。"""
        devs: List = []
        try:
            devs = list(rs.context().query_devices())
        except Exception as e:                                         # noqa: BLE001
            raise RuntimeError(
                "枚举 RealSense 设备失败：%s\n"
                "  ★ 这个报错（Mutex_TotalFailure / HResult 0x800703e3）"
                "最常见的原因就是**没有可用设备**：相机没插、插在 USB 2.1 口上、"
                "或设备刚被拔插。" % str(e).strip()) from e
        if not devs:
            raise RuntimeError(
                "SDK 看不到任何 RealSense 相机。\n"
                "  检查：① 相机插好没 ② 换 USB 3 口（USB 2.1 下拿不到帧）\n"
                "        ③ 关掉 RealSense Viewer（会独占设备）④ 拔插一次")
        if not self.serial:
            return devs[0]
        for d in devs:
            if self._info(d, "serial_number") == self.serial:
                return d
        found = "、".join(self._info(d, "serial_number") for d in devs)
        raise RuntimeError("按 serial=%s 找不到相机；当前能看到的是：%s"
                           % (self.serial, found))

    def _info(self, dev, field: str) -> str:
        """读一条设备信息。★ 读不到就回 '?'，**不抛**（刚插拔时真的会抛）。

        实测：刚插拔/刚复位时 `get_info()` 会抛 `Mutex_TotalFailure` ——
        如果这里不兜住，一次读信息失败就会把整个开流流程带崩（我第一版探针就这么崩的）。
        """
        try:
            rs = self.rs if self.rs is not None else import_rs()
            return str(dev.get_info(getattr(rs.camera_info, field)))
        except Exception:                                              # noqa: BLE001
            return "?"

    def _print_ready(self) -> None:
        print("[d435] 彩色流 %dx%d@%d  设备=%s SN=%s 固件=%s USB=%s"
              % (self.w, self.h, self.fps, self.dev_name, self.dev_serial,
                 self.fw, self.usb))
        if not usb_is3(self.usb):
            # ★★ 这一条必须喊出来：USB 2.1 链路下实测**一帧都拿不到**，
            #    而症状看起来像"相机坏了/分辨率太高"，很容易查错方向（实录 §2.31）。
            self._warned_usb = True
            print("  " + "!" * 70)
            print("  [d435] ★★ USB 链路 = %s，不是 3.x！实测这种链路上**一帧都拿不到**"
                  "（连 424x240@6 都是 0 帧）。" % self.usb)
            print("  [d435]    → 换到 USB 3 口（蓝口/SS 口）再跑，别去调分辨率。")
            print("  " + "!" * 70)
        elif self.fps >= 60:
            print("  [d435] ★ 60fps 模式：每帧只给算法 16.7ms，而 KCF 约 8ms/目标 ——"
                  " 4 个目标就吃满 32ms。帧率上去了**算法不会变快**（实录 §2.37）。")

    def read(self, want_full: bool = False) -> Optional[Frame]:
        if self.pipe is None:
            return None                       # 没开流：老实返回 None，不抛（上层照常处理）
        try:
            frames = self.pipe.wait_for_frames(self.timeout_ms)
        except Exception as e:                                         # noqa: BLE001
            self._read_fail += 1
            # ★ 别刷屏：只在第 1/10/100… 次报，但计数一直留着
            if self._read_fail in (1, 10, 100, 1000):
                print("[d435] 取帧失败（第 %d 次）：%s%s"
                      % (self._read_fail, str(e).strip(),
                         "  ★ 链路 = %s，不是 3.x 时先换 USB 3 口" % self.usb
                         if not usb_is3(self.usb) else ""))
            return None
        # 只在低频全分辨率请求时计算对齐与点云，不占用 A 模块默认彩色路径。
        aligned = self.align.process(frames) if frames and self.native_depth and want_full else frames
        color = aligned.get_color_frame() if aligned else None
        if not color:
            return None
        img = np.asanyarray(color.get_data())
        if img is None or img.size == 0:
            return None
        self._read_fail = 0
        frame = self._make_frame(img, want_full)
        if self.native_depth and want_full:
            depth = aligned.get_depth_frame()
            if depth:
                from ..core.depth import native_depth_mm
                raw = np.asanyarray(depth.get_data())
                if raw.shape != img.shape[:2]:
                    raise ValueError("对齐后深度与彩色尺寸不一致")
                frame.depth_mm = native_depth_mm(raw, self.depth_scale)
                points = self.pointcloud.calculate(depth)
                xyz = np.asanyarray(points.get_vertices()).view(np.float32).reshape(*raw.shape, 3).copy()
                xyz *= 1000.0
                xyz[~np.isfinite(frame.depth_mm)] = np.nan
                frame.depth_points_mm = xyz
                intr = depth.profile.as_video_stream_profile().get_intrinsics()
                frame.depth_profile = dict(serial=self.dev_serial, width=int(intr.width),
                                           height=int(intr.height), fx=float(intr.fx), fy=float(intr.fy),
                                           ppx=float(intr.ppx), ppy=float(intr.ppy),
                                           model=str(intr.model), coeffs=list(intr.coeffs),
                                           depth_scale=self.depth_scale)
        return frame

    def _make_frame(self, bgr: np.ndarray, want_full: bool) -> Frame:
        """组装 Frame。

        ★ 只有一个彩色传感器：`left` 与 `left_full` **同源**（前者按
          `scale.realtime` 缩放）。所以 `want_full=True` 不额外花钱，
          也给不出更清楚的图 —— 别指望靠它提升抽检精度。
        ★ `right` 一律 None：D435 的深度是固件算的深度图，**不是第二只眼睛的图像**，
          塞进 right 会让 SGBM 拿深度图当右目去匹配，出来的全是噪声。
        """
        return Frame(index=self._next_index(), ts=self._ts(),
                     left=scale_frame(bgr, self.scale_realtime),
                     right=None,
                     left_full=bgr if want_full else None,
                     right_full=None,
                     scale=self.scale_realtime)

    def close(self) -> None:
        if self.pipe is not None:
            try:
                self.pipe.stop()
            except Exception:                                          # noqa: BLE001
                pass                            # 已经停了/设备已拔，都属于正常
            self.pipe = None
        self.profile = None
        self.align = self.pointcloud = None
        self.depth_scale = None

    @property
    def name(self) -> str:
        return ("RealSenseSource(SN=%s, %dx%d@%d, USB=%s)"
                % (self.dev_serial, self.w, self.h, self.fps, self.usb))

    # ── 开机自检：把配置里的相机选项写进相机 ──────────────────
    def apply_locked_controls(self) -> None:
        """应用 `realsense.controls`（曝光/增益/白平衡…）。

        ★★ 与 HBV 的差别（**别把旧值抄过来**）：
          * 曝光是 `rs.option.exposure`，量纲约 **1~10000**，
            **不是** DSHOW 的 `-6.0`；
          * 而且**必须同时把 `enable_auto_exposure` 设成 0**，
            否则给多少都白给（自动曝光会把画面拉回去）。
        ★ 必须在开流之后调用 —— RealSense 的传感器选项要流在跑才生效。
        ★ 在工位调曝光之前，`realsense.controls` 建议留空 `{}`：
          用相机自己的自动曝光先把画面看起来，等
          `用户存档/d435_site_report.py` 量出"过曝≈0%"再把值填进来锁死。
        """
        ctrls = self.cfg.get("realsense.controls", {}) or {}
        if not ctrls:
            print("[d435] realsense.controls 为空 —— 跳过控件锁定"
                  "（自动曝光/自动白平衡仍开着，HSV 阈值会随光照漂）")
            return
        if self.profile is None or self.rs is None:
            print("[d435] 还没开流，跳过控件锁定（RealSense 的选项要流在跑才生效）")
            return

        sensors = self._target_sensors()
        if not sensors:
            print("[d435] 没有可写的传感器，跳过控件锁定")
            return

        # ★★ 顺序：先把所有 enable_*（自动开关）写掉，再写具体数值 ——
        #    实测"先给值、后关自动"等于白设（自动曝光会把值覆盖掉）。
        items = [(str(k), v) for k, v in ctrls.items()]
        items.sort(key=lambda kv: 0 if _opt_name(kv[0]).startswith("enable_") else 1)
        done, skipped = 0, []
        for key, val in items:
            opt_name = _opt_name(key)
            opt = getattr(self.rs.option, opt_name, None)
            if opt is None:
                hint = _HBV_CTRL_HINTS.get(key.lower())
                skipped.append("%s（SDK 没有这个选项%s）"
                               % (key, "；RealSense 里应写 " + hint if hint else ""))
                continue
            written = False
            for s in sensors:
                if not self._supports(s, opt):
                    continue
                try:
                    s.set_option(opt, float(val))
                    back = self._read_back(s, opt)
                    print("[d435] %s <- %s（回读 %s）[%s]"
                          % (opt_name, val, back, self._sensor_name(s)))
                    written = True
                    done += 1
                except Exception as e:                                 # noqa: BLE001
                    skipped.append("%s=%s 写入失败：%s" % (key, val, str(e).strip()))
            if not written:
                skipped.append("%s=%s：没有传感器支持这个选项" % (key, val))
        print("[d435] 控件已写 %d 项；跳过/异常 %d 项" % (done, len(skipped)))
        for s in skipped:
            print("  [d435] 跳过：%s" % s)

    def _target_sensors(self) -> List:
        """挑要写选项的传感器：优先 RGB Camera（彩色画面归它管）。

        ★ 不无脑写全部传感器：`exposure` / `gain` 在深度传感器上也存在，
          写过去会同时改红外曝光，跟彩色画面没关系，徒增副作用。
        """
        try:
            sensors = list(self.profile.get_device().sensors)
        except Exception:                                              # noqa: BLE001
            return []
        rgb = [s for s in sensors if "rgb" in self._sensor_name(s).lower()]
        return rgb or sensors

    @staticmethod
    def _sensor_name(sensor) -> str:
        try:
            return str(sensor.get_info(import_rs().camera_info.name))
        except Exception:                                              # noqa: BLE001
            return "?"

    @staticmethod
    def _supports(sensor, opt) -> bool:
        try:
            return bool(sensor.supports(opt))
        except Exception:                                              # noqa: BLE001
            return False

    @staticmethod
    def _read_back(sensor, opt) -> str:
        try:
            return "%.1f" % float(sensor.get_option(opt))
        except Exception:                                              # noqa: BLE001
            return "?"

    # ── 给标定/自检工具用：彩色内参 ────────────────────────────
    def color_intrinsics(self) -> Optional[Dict[str, float]]:
        """彩色流的内参（fx/fy/ppx/ppy + 尺寸）。

        ★ 换机后 **c2 双目标定不用重跑**：这些是相机出厂标定，深度也由固件算好。
          标定链里真正要重跑的是 c1（GSD，可从 fx 直接推）/ c3~c6。
        """
        if self.profile is None or self.rs is None:
            return None
        try:
            stream = self.profile.get_stream(self.rs.stream.color)
            intr = stream.as_video_stream_profile().get_intrinsics()
            return {"width": float(intr.width), "height": float(intr.height),
                    "fx": float(intr.fx), "fy": float(intr.fy),
                    "ppx": float(intr.ppx), "ppy": float(intr.ppy)}
        except Exception:                                              # noqa: BLE001
            return None

    def gsd_mm_per_px(self, distance_mm: float) -> Optional[float]:
        """按内参推"距离 distance_mm 处"的地面分辨率 mm/px（给 c1 抄用）。"""
        intr = self.color_intrinsics()
        if not intr or intr.get("fx", 0) <= 0:
            return None
        return float(distance_mm) / float(intr["fx"])


# ── 小工具 ──────────────────────────────────────────────────────
def _opt_name(key: str) -> str:
    """配置里的短名 -> rs.option 的属性名。"""
    k = str(key or "").strip().lower()
    return _CTRL_ALIASES.get(k, k)
