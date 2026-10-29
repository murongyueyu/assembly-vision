# -*- coding: utf-8 -*-
"""D435 相机来源的**接口契约**测试（不需要相机、不需要真 SDK）。

★ 为什么要有这个模块：
  `sources/d435.py` 是**只能在真机上跑**的那条路径 —— 相机不在手边时一行都执行不到。
  本项目已经因为"只能手工跑的路径没人跑"吃过亏（`app.py --windowed` 一启动连崩两次，
  见 tests/test_ui_renderer.py 的文件头）。所以这里用一个**假 SDK** 把契约钉住：

    ① `factory` 真的会造出 `RealSenseSource`（换机后 `source: d435` 要能生效）
    ② **Frame 契约**：left / right / left_full / scale / index / ts 与 pipeline 的假设一致
    ③ `want_full=True` 时给 `left_full`（全分辨率抽检与位姿自检靠它）
    ④ ★ `right` 必须是 None —— D435 没有"第二只眼睛的图像"，
       它的深度是固件算出来的，塞进 right 会让 SGBM 拿深度图当右目匹配
    ⑤ 没 `open()` 时 `read()` 返回 None 而不是抛（pipeline 的采集线程靠它兜底）
    ⑥ 没装 `pyrealsense2` 时的报错要说清"装什么"（而不是一个裸 ImportError）
    ⑦ ★ USB 不是 3.x 要**明确报警**（实录 §2.31：USB 2.1 链路上**一帧都拿不到**）
    ⑧ ★ 控件写入顺序：先 `enable_auto_exposure=0` 再给 `exposure`（反了等于白设）
    ⑨ 选项只写 RGB 传感器；未知选项名要**明确报跳过**，不能静默吞掉
    ⑩ `tools/probe_camera.py`（UVC 专用）遇到 D435 要拦住并指路
"""
from __future__ import annotations

import io
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check                                     # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assemvision.config import Config, load_config            # noqa: E402
from assemvision.sources import d435                          # noqa: E402
from assemvision.sources.d435 import RealSenseSource          # noqa: E402
from assemvision.sources.factory import make_source           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ══════════════════════════════════════════════════════════════
# 假 SDK：只实现 d435.py 真正用到的那几个接口
# ══════════════════════════════════════════════════════════════
class _FakeSensor:
    def __init__(self, name: str, options: dict) -> None:
        self._name = name
        self.options = dict(options)
        self.log = []                     # [(opt, value)] —— 用来断言写入顺序

    def get_info(self, key) -> str:
        return self._name

    def supports(self, opt) -> bool:
        return opt in self.options

    def set_option(self, opt, val) -> None:
        if opt not in self.options:
            raise RuntimeError("option not supported")
        self.options[opt] = float(val)
        self.log.append((opt, float(val)))

    def get_option(self, opt) -> float:
        return float(self.options.get(opt, 0.0))

    def get_stream_profiles(self):
        return []


class _FakeDevice:
    def __init__(self, sn: str, sensors, usb: str = "3.2",
                 name: str = "Intel RealSense D435") -> None:
        self.sensors = list(sensors)
        self._info = {"INFO_SN": sn, "INFO_NAME": name, "INFO_USB": usb,
                      "INFO_FW": "5.17.3.10"}
        self.reset_called = False

    def get_info(self, key) -> str:
        if key not in self._info:
            raise RuntimeError("no such info field")     # 真 SDK 也会抛
        return self._info[key]

    def hardware_reset(self) -> None:
        self.reset_called = True


class _FakeIntrinsics:
    def __init__(self, w=640, h=480, fx=608.0, fy=608.0) -> None:
        self.width, self.height = w, h
        self.fx, self.fy = fx, fy
        self.ppx, self.ppy = w / 2.0, h / 2.0
        self.model = "brown_conrady"


class _FakeVideoProfile:
    def __init__(self, intr) -> None:
        self._intr = intr

    def get_intrinsics(self):
        return self._intr


class _FakeProfile:
    def __init__(self, dev, intr) -> None:
        self._dev, self._intr = dev, intr

    def get_device(self):
        return self._dev

    def get_stream(self, stream):
        return self

    def as_video_stream_profile(self):
        return _FakeVideoProfile(self._intr)


class _FakeColorFrame:
    def __init__(self, arr) -> None:
        self._arr = arr

    def get_data(self):
        return self._arr


class _FakeFrameset:
    def __init__(self, arr) -> None:
        self._arr = arr

    def get_color_frame(self):
        return _FakeColorFrame(self._arr) if self._arr is not None else None


class _FakeConfig:
    def __init__(self) -> None:
        self.device = None
        self.streams = []

    def enable_device(self, serial) -> None:
        self.device = serial

    def enable_stream(self, stream, w, h, fmt, fps) -> None:
        self.streams.append((stream, w, h, fmt, fps))


class _FakePipeline:
    def __init__(self, sdk) -> None:
        self.sdk = sdk
        self.stopped = False

    def start(self, cfg):
        self.sdk.start_calls += 1
        self.sdk.last_config = cfg
        if self.sdk.fail_starts > 0:
            self.sdk.fail_starts -= 1
            raise RuntimeError("Device or resource busy")
        return _FakeProfile(self.sdk.devices[0], _FakeIntrinsics())

    def wait_for_frames(self, timeout_ms):
        if self.sdk.raise_on_read:
            raise RuntimeError("stop() 之后不该再读")
        if not self.sdk.frames:
            return None                        # 真 SDK 会抛；两种都要能兜住
        return _FakeFrameset(self.sdk.frames.pop(0))

    def stop(self) -> None:
        self.stopped = True


class _FakeSDK:
    """够用的假 pyrealsense2。"""

    def __init__(self, frames=None, sensors=None, devices=None, usb="3.2",
                 fail_starts=0, enum_raises=False, serials=("262322070302",)) -> None:
        self.frames = list(frames or [])
        self.sensors = list(sensors or [_FakeSensor(
            "RGB Camera", {"OPT_AE": 1.0, "OPT_EXPOSURE": 100.0, "OPT_WB": 4000.0}),
            _FakeSensor("Stereo Module", {"OPT_AE": 1.0, "OPT_EXPOSURE": 100.0})])
        # ★ 注意区分 None 与 []：devices=[] 表示"一台都没有"，要真的走那条分支
        self.devices = (list(devices) if devices is not None
                        else [_FakeDevice(sn, self.sensors, usb=usb) for sn in serials])
        self.fail_starts = fail_starts
        self.enum_raises = enum_raises
        self.raise_on_read = False
        self.start_calls = 0
        self.last_config = None

        self.stream = SimpleNamespace(color="COLOR")
        self.format = SimpleNamespace(bgr8="BGR8")
        self.option = SimpleNamespace(enable_auto_exposure="OPT_AE",
                                      exposure="OPT_EXPOSURE",
                                      gain="OPT_GAIN",
                                      enable_auto_white_balance="OPT_AWB",
                                      white_balance="OPT_WB")
        self.camera_info = SimpleNamespace(name="INFO_NAME", serial_number="INFO_SN",
                                           firmware_version="INFO_FW",
                                           usb_type_descriptor="INFO_USB")

    def config(self):
        return _FakeConfig()

    def pipeline(self):
        return _FakePipeline(self)

    def context(self):
        if self.enum_raises:
            raise RuntimeError("CreateNamedMutex returned Mutex_TotalFailure")
        return SimpleNamespace(query_devices=lambda: list(self.devices))


class _patched_sdk:
    """把 `d435.import_rs` 临时换成假 SDK（用完还原）。"""

    def __init__(self, sdk: _FakeSDK) -> None:
        self.sdk = sdk
        self.old = None

    def __enter__(self) -> _FakeSDK:
        self.old = d435.import_rs
        d435.import_rs = lambda: self.sdk
        return self.sdk

    def __exit__(self, *exc) -> bool:
        d435.import_rs = self.old
        return False


class _capture:
    """抓 print 输出（断言"有没有报警"用）。"""

    def __enter__(self):
        self.buf = io.StringIO()
        self.old = sys.stdout
        sys.stdout = self.buf
        return self

    def __exit__(self, *exc) -> bool:
        sys.stdout = self.old
        return False

    @property
    def text(self) -> str:
        return self.buf.getvalue()


def _cam_cfg(**over) -> Config:
    """最小可用配置：不依赖 config/camera.yaml。"""
    data = {"source": "d435",
            "scale": {"realtime": 1.0},
            "capture": {"warmup_frames": 1},
            "realsense": {"width": 640, "height": 480, "fps": 30,
                          "open_retries": 1, "retry_wait_s": 0.0,
                          "timeout_ms": 50, "controls": {}}}
    for k, v in over.items():
        cur = data
        keys = k.split(".")
        for key in keys[:-1]:
            cur = cur.setdefault(key, {})
        cur[keys[-1]] = v
    return Config(data, "<test>")


def _one_frame(w=640, h=480):
    """一张可辨认的假彩色帧（不是全 0，避免把"没给图"和"给了黑图"混起来）。"""
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = 20
    img[10:20, 10:20, 2] = 200
    return img


# ══════════════════════════════════════════════════════════════
def main() -> int:
    # ① factory 真的会造出 D435 源（换机后 source: d435 要能生效）
    print("== factory 换源 ==")
    for name in ("d435", "D435", "d435i", "realsense", "rs"):
        src = make_source(_cam_cfg(), name)
        check(isinstance(src, RealSenseSource), "source=%-9s -> RealSenseSource" % name)
    from assemvision.sources.uvc_stereo import UvcStereoSource
    check(isinstance(make_source(_cam_cfg(), "uvc"), UvcStereoSource),
          "source=uvc 仍然是旧相机（换机后可回退）")
    try:
        make_source(_cam_cfg(), "nonsense")
        check(False, "未知来源名应报 ValueError")
    except ValueError as e:
        check("d435" in str(e), "未知来源名的报错里要列出 d435 这个合法值")

    # ② 真配置：camera.yaml 真的换到 D435 了，且关键项齐
    print("\n== config/camera.yaml ==")
    cam = load_config(os.path.join(ROOT, "config", "camera.yaml"))
    check(str(cam.get("source")) in ("d435", "realsense"),
          "camera.yaml 的 source 已换成 D435（当前 %r）" % cam.get("source"))
    real = RealSenseSource(cam)
    check((real.w, real.h, real.fps) == (640, 480, 30),
          "D435 默认走 640x480@30（实测 30.0fps）—— 当前 %dx%d@%d"
          % (real.w, real.h, real.fps))
    check(float(cam.get("capture.frame_rate_hint", 0)) >= 10.0,
          "frame_rate_hint 要 ≥10（D435 实测 30fps，去抖不必切成按时间）")
    check(cam.get("depth.enabled") is False,
          "换机后 depth.enabled 应为 false —— D435 的深度由固件算，"
          "接进 core/depth.py 属 B 模块（本层只给彩色）")

    # ③ Frame 契约（假 SDK 跑通 open -> read）
    print("\n== Frame 契约 ==")
    sdk = _FakeSDK(frames=[_one_frame() for _ in range(3)])
    with _patched_sdk(sdk):
        src = RealSenseSource(_cam_cfg())
        with _capture() as cap:
            src.open()
            f1 = src.read()
            f2 = src.read()
            f3 = src.read()
            f_empty = src.read()           # 帧用完了 -> None，不许抛
        print("   " + cap.text.strip().splitlines()[0])

    check(f1 is not None and f2 is not None and f3 is not None,
          "open 之后连续三帧都要拿到（假 SDK 给了三帧）")
    check(f1.left is not None and f1.left.shape == (480, 640, 3),
          "left 是 640x480x3 的 BGR 图（scale.realtime=1.0）")
    check(f1.right is None and f1.right_full is None,
          "★ right/right_full 必须是 None —— D435 没有第二只眼睛的图像")
    check(abs(f1.scale - 1.0) < 1e-9, "scale 要如实回报 scale.realtime")
    check((f2.index, f3.index) == (f1.index + 1, f1.index + 2),
          "index 逐帧递增（%d -> %d -> %d）" % (f1.index, f2.index, f3.index))
    check(f2.ts >= f1.ts, "ts 单调不减")
    check(f1.left_full is None, "want_full=False 时不给 left_full（别白拷一份）")
    check(f_empty is None, "取不到帧时 read() 返回 None 而不是抛（采集线程靠它兜底）")
    check(sdk.start_calls == 1, "只开了一次流（open_retries=1）")
    check(sdk.last_config.streams
          and sdk.last_config.streams[0][1:3] == (640, 480),
          "开流请求的分辨率来自配置（%s）" % (sdk.last_config.streams,))

    # ③b want_full 单独一段（免得读空帧把上面的帧计数搅乱）
    sdk = _FakeSDK(frames=[_one_frame()])
    with _patched_sdk(sdk):
        src_full = RealSenseSource(_cam_cfg())
        with _capture():
            src_full.open()
            f_full = src_full.read(want_full=True)
    check(f_full is not None and f_full.left_full is not None,
          "want_full=True 时要给 left_full（全分辨率抽检/位姿自检靠它）")
    check(f_full.left_full.shape == (480, 640, 3),
          "left_full 是原图尺寸（只有一个彩色传感器，left 与 left_full 同源）")

    # ④ 缩放的实时通道 vs 全分辨率
    print("\n== 实时通道缩放 ==")
    sdk = _FakeSDK(frames=[_one_frame()])
    with _patched_sdk(sdk):
        src = RealSenseSource(_cam_cfg(**{"scale.realtime": 0.5}))
        with _capture():
            src.open()
            f = src.read(want_full=True)
    check(f.left.shape[:2] == (240, 320), "scale.realtime=0.5 -> left 缩成 320x240")
    check(f.left_full.shape[:2] == (480, 640), "left_full 仍是原始 640x480")
    check(abs(f.scale - 0.5) < 1e-9, "scale 报 0.5")

    # ⑤ 没 open 就 read / 锁控件：不能抛
    print("\n== 未开流时的行为 ==")
    src = RealSenseSource(_cam_cfg())
    check(src.read() is None, "没 open() 时 read() 返回 None（不抛）")
    with _capture() as cap:
        src.apply_locked_controls()
    check("跳过" in cap.text, "没开流时锁控件要明确说跳过（不静默）")
    check(src.color_intrinsics() is None, "没开流时内参返回 None")
    check(src.gsd_mm_per_px(360.0) is None, "拿不到内参时 GSD 也返回 None")

    # ⑥ 没装 pyrealsense2：报错要说清装什么
    print("\n== 没装 SDK 时的报错 ==")
    saved = sys.modules.get("pyrealsense2", "___absent___")
    sys.modules["pyrealsense2"] = None            # 让 import 直接失败
    try:
        try:
            d435.import_rs()
            check(False, "没装 SDK 时 import_rs() 应当报 RuntimeError")
        except RuntimeError as e:
            msg = str(e)
            check("pyrealsense2" in msg and "2.30" in msg,
                  "报错里要说清装 pyrealsense2、并指向实录 §2.30")
        check(d435.sdk_available() is False, "sdk_available() 在没装 SDK 时返回 False（不抛）")
    finally:
        if saved == "___absent___":
            sys.modules.pop("pyrealsense2", None)
        else:
            sys.modules["pyrealsense2"] = saved
    check(d435.sdk_available() is True, "本机装了 pyrealsense2（sdk_available()=True）")

    # ⑦ USB 链路判据 + 报警
    print("\n== USB 链路自检 ==")
    check(d435.usb_is3("3.2") and d435.usb_is3("3.1") and d435.usb_is3("3.0"),
          "3.x 都算达标")
    check(not d435.usb_is3("2.1") and not d435.usb_is3("2.0"), "2.x 判为不达标")
    check(d435.usb_is3("?") and d435.usb_is3(""),
          "★ 读不到描述符时**不报警**（刚插拔时 get_info 会抛，宁可漏报）")
    sdk = _FakeSDK(frames=[_one_frame()], usb="2.1")
    with _patched_sdk(sdk):
        src = RealSenseSource(_cam_cfg())
        with _capture() as cap:
            src.open()
    check("USB 3" in cap.text and "一帧都拿不到" in cap.text,
          "★ USB 2.1 时必须大声报警（实录 §2.31：这种链路一帧都拿不到）")
    check("2.1" in src.name, "name 里带上 USB 链路（%s）" % src.name)

    # ⑧ 控件写入顺序 + 只写 RGB 传感器
    print("\n== 控件写入（曝光/白平衡）==")
    rgb = _FakeSensor("RGB Camera", {"OPT_AE": 1.0, "OPT_EXPOSURE": 100.0,
                                     "OPT_WB": 4000.0, "OPT_AWB": 1.0})
    depth_sensor = _FakeSensor("Stereo Module", {"OPT_AE": 1.0, "OPT_EXPOSURE": 100.0})
    sdk = _FakeSDK(frames=[_one_frame()], sensors=[depth_sensor, rgb])
    # ★ 故意把 exposure 写在 enable_auto_exposure 前面 —— 代码必须自己纠正顺序
    cfg = _cam_cfg(**{"realsense.controls": {"exposure": 700,
                                             "enable_auto_exposure": 0,
                                             "white_balance": 4600}})
    with _patched_sdk(sdk):
        src = RealSenseSource(cfg)
        with _capture() as cap:
            src.open()
            src.apply_locked_controls()
    check([o for o, _v in rgb.log][:2] == ["OPT_AE", "OPT_EXPOSURE"],
          "★ 必须先关自动曝光、再给曝光值（实测反了等于白设）——实际顺序 %s"
          % [o for o, _v in rgb.log])
    check(rgb.options["OPT_EXPOSURE"] == 700.0, "曝光值真的写进去了")
    check(rgb.options["OPT_AE"] == 0.0, "自动曝光真的关掉了")
    check(rgb.options["OPT_WB"] == 4600.0, "白平衡也写进去了")
    check(depth_sensor.log == [],
          "★ 不写深度传感器（写过去只会改红外曝光，跟彩色画面无关）")
    check("回读" in cap.text, "要打印回读值（确认真的生效，而不是只说设过了）")

    # ⑨ 旧名字/未知选项要明确报跳过
    print("\n== 选项名写错时 ==")
    sdk = _FakeSDK(frames=[_one_frame()])
    cfg = _cam_cfg(**{"realsense.controls": {"exposure_absolute": 120,
                                             "nonsense_option": 1}})
    with _patched_sdk(sdk):
        src = RealSenseSource(cfg)
        with _capture() as cap:
            src.open()
            src.apply_locked_controls()
    check("跳过" in cap.text and "exposure" in cap.text,
          "★ DSHOW 口径的旧名字（exposure_absolute）要报出来并指路，不能静默吞掉")
    check("nonsense_option" in cap.text, "未知选项名也要报出来")

    # ⑩ 空 controls = 用相机自动曝光，要说明白
    sdk = _FakeSDK(frames=[_one_frame()])
    with _patched_sdk(sdk):
        src = RealSenseSource(_cam_cfg())
        with _capture() as cap:
            src.open()
            src.apply_locked_controls()
    check("跳过控件锁定" in cap.text and "自动曝光" in cap.text,
          "controls 为空时要提示「仍在用自动曝光、阈值会漂」（不装作已锁定）")

    # ⑪ 开流失败要重试，且最终报错要把排查顺序写全
    print("\n== 开流失败与重试 ==")
    sdk = _FakeSDK(frames=[_one_frame()], fail_starts=1)
    with _patched_sdk(sdk):
        src = RealSenseSource(_cam_cfg(**{"realsense.open_retries": 2}))
        with _capture() as cap:
            src.open()
            f = src.read()
    check(sdk.start_calls == 2 and f is not None,
          "第 1 次 busy 后重试成功（实录 §2.31：刚插拔的 busy 重试即可）")

    sdk = _FakeSDK(frames=[_one_frame()], fail_starts=99)
    with _patched_sdk(sdk):
        src = RealSenseSource(_cam_cfg(**{"realsense.open_retries": 2}))
        try:
            with _capture():
                src.open()
            check(False, "一直 busy 时应当抛 RuntimeError")
        except RuntimeError as e:
            msg = str(e)
            check("USB 3" in msg and "Viewer" in msg and "--reset" in msg,
                  "最终报错要写全排查顺序（USB 3 口 / 关 Viewer / --reset）")

    # ⑫ 枚举不到设备 / 枚举本身抛异常
    print("\n== 看不到相机时 ==")
    sdk = _FakeSDK(devices=[])
    with _patched_sdk(sdk):
        src = RealSenseSource(_cam_cfg())
        try:
            with _capture():
                src.open()
            check(False, "没有设备时应当抛 RuntimeError")
        except RuntimeError as e:
            check("看不到任何 RealSense" in str(e),
                  "要说清「SDK 看不到相机」，别只说失败")
    sdk = _FakeSDK(enum_raises=True)
    with _patched_sdk(sdk):
        src = RealSenseSource(_cam_cfg())
        try:
            with _capture():
                src.open()
            check(False, "枚举抛异常时应当抛 RuntimeError")
        except RuntimeError as e:
            check("Mutex_TotalFailure" in str(e) or "枚举" in str(e),
                  "枚举异常要原样带出（%s）" % str(e).splitlines()[0])

    # ⑬ 设备信息读不出来也不能把开流拦下（实测 get_info 会抛）
    print("\n== 设备信息读不出来时 ==")
    sdk = _FakeSDK(frames=[_one_frame()])
    sdk.devices[0]._info = {"INFO_SN": "262322070302"}     # 其它字段一律抛
    with _patched_sdk(sdk):
        src = RealSenseSource(_cam_cfg())
        with _capture() as cap:
            src.open()
            f = src.read()
    check(f is not None and src.usb == "?" and src.fw == "?",
          "★ get_info 抛异常时用 '?' 兜住，**不能**因此开不了流（实录：刚插拔真的会抛）")
    check("USB=?" in src.name, "name 里如实写 '?'（凭据不足就别假装知道）")

    # ⑭ 多台相机按序列号选
    print("\n== 按序列号选相机 ==")
    dev_a = _FakeDevice("111111111111", [])
    dev_b = _FakeDevice("222222222222", [])
    sdk = _FakeSDK(frames=[_one_frame()], devices=[dev_a, dev_b])
    with _patched_sdk(sdk):
        src = RealSenseSource(_cam_cfg(**{"realsense.serial": "222222222222"}))
        with _capture():
            src.open()
    check(src.name.startswith("RealSenseSource(SN=222222222222"),
          "按 serial 选到了第二台（%s）" % src.name)
    sdk = _FakeSDK(frames=[_one_frame()], devices=[dev_a, dev_b])
    with _patched_sdk(sdk):
        src = RealSenseSource(_cam_cfg(**{"realsense.serial": "999999999999"}))
        try:
            with _capture():
                src.open()
            check(False, "serial 对不上时应当报错")
        except RuntimeError as e:
            check("111111111111" in str(e),
                  "serial 对不上时要列出**当前能看到哪些**（%s）" % str(e).splitlines()[0])

    # ⑮ 内参 / GSD（c1 要用它推 GSD）
    print("\n== 内参 ==")
    sdk = _FakeSDK(frames=[_one_frame()])
    with _patched_sdk(sdk):
        src = RealSenseSource(_cam_cfg())
        with _capture():
            src.open()
        intr = src.color_intrinsics()
        gsd = src.gsd_mm_per_px(360.0)
        src.close()
    check(intr is not None and intr["fx"] == 608.0, "从 profile 读到内参（fx=%s）"
          % (None if intr is None else intr["fx"]))
    check(gsd is not None and abs(gsd - 360.0 / 608.0) < 1e-9,
          "GSD = 距离/fx（%.4f mm/px）" % (gsd or -1))
    check(sdk.start_calls == 1, "close() 不重复开流")

    # ⑯ UVC 探针遇到 D435 要拦住（否则白跑一遍再报"打不开"）
    print("\n== 工具之间的指路 ==")
    import probe_camera                                          # noqa: E402
    with _capture() as cap:
        blocked = probe_camera.guard_not_realsense("d435")
    check(blocked is True and "probe_d435" in cap.text,
          "probe_camera.py（UVC 专用）遇到 D435 要拦下并指向 probe_d435.py")
    check(probe_camera.guard_not_realsense("uvc") is False, "source=uvc 时照常放行")

    # ⑰ 新工具本身能 import、解析函数正确（这类脚本最容易"写完没人跑"）
    print("\n== probe_d435 工具 ==")
    import probe_d435                                            # noqa: E402
    check(probe_d435.parse_modes("640x480, 1280*720") == [(640, 480), (1280, 720)],
          "模式解析（兼容 x/*、逗号与空格）")
    check(probe_d435.parse_ints("30,60") == [30, 60], "帧率列表解析")
    white = np.full((10, 10, 3), 255, np.uint8)
    black = np.zeros((10, 10, 3), np.uint8)
    check(abs(probe_d435.exposure_stats(white)[0] - 1.0) < 1e-6,
          "全白图 -> 过曝 100%")
    check(probe_d435.exposure_stats(black)[0] == 0.0, "全黑图 -> 过曝 0%")

    print("\nD435 相机来源契约测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
