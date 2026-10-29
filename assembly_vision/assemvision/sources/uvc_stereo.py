# -*- coding: utf-8 -*-
"""真实双目相机（HBV-1714-2 S2.0）—— 输出单路 UVC 拼接图，用 cv2.VideoCapture 读。

三种采集方式预留说明（config/camera.yaml 的 capture.decoder）：
  full      先用这个：全尺寸 MJPG 解码，Pi 4B 约 15-25 ms/帧，分线程后不影响 30fps
  reduced2  TODO 需要从 V4L2 取裸 MJPEG 再用 cv2.imdecode(..., IMREAD_REDUCED_COLOR_2)
            解码开销只有全解的一半左右，且同一份裸流可解出两种分辨率。
            实现要点：用 fcntl.ioctl + VIDIOC_DQBUF 取 mmap 缓冲区，属底层操作，
            等整机跑通后再补。接口已经留好（_open_raw / _read_raw）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

import cv2
import numpy as np

from ..config import Config
from ..types import Frame
from .base import CameraSource, split_stereo

# 后端名 -> OpenCV 常量。★ 这是 Windows 上能不能出图的关键：
#   V4L2 是 Linux 专用接口，在 Windows 上 VideoCapture(0, CAP_V4L2) 必然 isOpened()=False。
#   实测（本机 OpenCV 4.12）：CAP_V4L2 -> False，CAP_DSHOW -> True。
_BACKENDS = {
    "v4l2": "CAP_V4L2",         # Linux
    "dshow": "CAP_DSHOW",       # Windows（DirectShow，本项目实测必须用这个）
    "msmf": "CAP_MSMF",         # Windows Media Foundation
    "any": "CAP_ANY",
}


def resolve_backend(name: str) -> int:
    """把配置里的后端名解析成 OpenCV 常量。"""
    key = str(name or "any").strip().lower()
    attr = _BACKENDS.get(key)
    if attr is None or not hasattr(cv2, attr):
        return cv2.CAP_ANY
    return int(getattr(cv2, attr))


def default_backend_name() -> str:
    """按当前平台给出合适的默认后端名。"""
    return "v4l2" if sys.platform.startswith("linux") else "dshow"


class UvcStereoSource(CameraSource):
    def __init__(self, cam_cfg: Config) -> None:
        super().__init__(scale_realtime=float(cam_cfg.get("scale.realtime", 0.5)))
        self.cfg = cam_cfg
        self.cap: Optional[cv2.VideoCapture] = None
        self.decoder = str(cam_cfg.get("capture.decoder", "full"))
        self.mode = str(cam_cfg.get("capture.mode", "side_by_side"))
        self.left_index = int(cam_cfg.get("capture.left_index", 0))
        self.w = int(cam_cfg.get("capture.width", 2560))
        self.h = int(cam_cfg.get("capture.height", 960))

    # ── 生命周期 ──────────────────────────────────────────────
    def _backend_name(self) -> str:
        """决定用哪个后端。

        capture.backend 就写目标平台的名字（Linux 写 v4l2，Windows 写 dshow），
        也可以显式用 capture.backend_windows 单独覆盖 Windows。
        """
        name = str(self.cfg.get("capture.backend", "") or "").strip()
        if sys.platform.startswith("win"):
            name = str(self.cfg.get("capture.backend_windows", name) or "").strip()
        if not name:
            name = default_backend_name()
        return name

    def open(self) -> None:
        dev = self.cfg.get("device", 0)
        name = self._backend_name()
        backend = resolve_backend(name)
        self.cap = cv2.VideoCapture(dev, backend)

        # 首选后端打不开就退回 CAP_ANY 再试一次 —— 有些平台/驱动只认自动选择
        if not self.cap.isOpened() and backend != cv2.CAP_ANY:
            self.cap.release()
            print("[camera] 后端 %s 打不开，改用 CAP_ANY 重试" % name)
            self.cap = cv2.VideoCapture(dev, cv2.CAP_ANY)

        if not self.cap.isOpened():
            hint = ("检查 /dev/video* 是否存在、是否被占用"
                    if sys.platform.startswith("linux")
                    else "检查设备管理器的相机、是否被其它程序（含浏览器/会议软件）占用，"
                         "以及 capture.backend 是否写成了 dshow")
            raise RuntimeError("打不开相机 device=%r（后端 %s）。%s" % (dev, name, hint))

        fourcc = str(self.cfg.get("capture.fourcc", "MJPG"))
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        self.cap.set(cv2.CAP_PROP_FPS, float(self.cfg.get("capture.fps", 30)))
        # 尽量少缓冲，降低延迟
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        aw, ah = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("[camera] 实际打开: %dx%d fourcc=%s 后端=%s" % (aw, ah, fourcc, name))
        if (aw, ah) != (self.w, self.h):
            print("[camera] ★ 注意：请求 %dx%d 但拿到 %dx%d。"
                  "相机可能不支持该模式，请跑 tools/probe_camera.py 探一遍支持的分辨率，"
                  "再把 camera.yaml 的 capture.width/height 改成实际值。"
                  % (self.w, self.h, aw, ah))

    def read(self, want_full: bool = False) -> Optional[Frame]:
        if self.cap is None:
            return None
        ok, img = self.cap.read()
        if not ok or img is None:
            return None
        left, right = split_stereo(img, self.mode, self.left_index)
        return self.make_frame(left, right, self._next_index(), self._ts(), want_full)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def name(self) -> str:
        return "UvcStereoSource(device=%s)" % self.cfg.get("device", 0)

    # ── 开机自检：锁定相机控件 ────────────────────────────────
    def apply_locked_controls(self) -> None:
        """把 config 里 locked_controls 逐条应用，锁手动曝光与白平衡。

        ★ 为什么必须锁：HSV 阈值全靠光照稳定。自动曝光/自动白平衡一动，
          黄铜的饱和度和手套的色相就跟着漂，阈值就白标了。

        两条路径：
          Linux  用 v4l2-ctl（控件名以 `v4l2-ctl -d /dev/video0 --list-ctrls` 为准）
          Windows 用 cv2.VideoCapture.set(CAP_PROP_*)（DSHOW 下部分控件驱动会忽略，
                  返回 False 不代表失败，所以这里只提示不报错）
        """
        ctrls = self.cfg.get("locked_controls", {}) or {}
        if sys.platform.startswith("linux"):
            self._apply_controls_v4l2(ctrls)
        else:
            self._apply_controls_opencv(ctrls)

    def _apply_controls_v4l2(self, ctrls: dict) -> None:
        dev = self.cfg.get("device", 0)
        node = "/dev/video%d" % dev if isinstance(dev, int) else str(dev)
        for key, val in ctrls.items():
            cmd = ["v4l2-ctl", "-d", node, "--set-ctrl=%s=%s" % (key, val)]
            try:
                r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   timeout=3)
                if r.returncode != 0:
                    print("[camera] 控件设置失败 %s=%s -> %s"
                          % (key, val, r.stdout.decode("utf-8", "ignore").strip()))
            except FileNotFoundError:
                print("[camera] 找不到 v4l2-ctl（apt install v4l-utils），跳过控件锁定")
                return
            except Exception as e:  # noqa: BLE001
                print("[camera] 控件设置异常 %s: %s" % (key, e))

    # yaml 里的控件名 -> OpenCV 属性。None 表示 OpenCV 没有对应属性，只能提示手工设。
    _CTRL_TO_PROP = {
        "auto_exposure": ("CAP_PROP_AUTO_EXPOSURE", {1: 0.25, 0: 0.75}),
        "exposure_absolute": ("CAP_PROP_EXPOSURE", None),
        "gain": ("CAP_PROP_GAIN", None),
        "white_balance_automatic": ("CAP_PROP_AUTO_WB", {0: 0.0, 1: 1.0}),
        "white_balance_temperature": ("CAP_PROP_WB_TEMPERATURE", None),
    }

    def _apply_controls_opencv(self, ctrls: dict) -> None:
        if self.cap is None or not self.cap.isOpened():
            print("[camera] 相机未打开，跳过控件锁定")
            return

        # ★★ 曝光单独处理，而且必须"先关自动、再给值"：
        #   实测这台相机**只设 auto_exposure 画面毫无变化**（仍被自动曝光拉满），
        #   必须同时把 EXPOSURE 设成一个手动值才生效。
        #   实测有效范围 -11 ~ -1：-5 最佳（mean≈153）、-8 偏暗、-11 全黑、
        #   -2 及以上严重过曝。
        #   ★ 曝光正确是后续一切的前提：过曝会让各通道削顶、画面颜色失真发青，
        #     还会因为高熵画面把 MJPG 编码器压垮，帧率从 ~50fps 掉到 ~4.5fps。
        wexp = self.cfg.get("exposure_windows", {}) or {}
        try:
            exp_val = float(wexp.get("exposure", 0.0) or 0.0)
        except (TypeError, ValueError):
            exp_val = 0.0
        if exp_val != 0.0 and hasattr(cv2, "CAP_PROP_EXPOSURE"):
            auto_off = wexp.get("auto_exposure_off", 0.25)
            if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(auto_off))
            ok = self.cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)
            print("[camera] 曝光设为 %.1f（手动，先关自动）set=%s 回读=%s"
                  % (exp_val, ok, self.cap.get(cv2.CAP_PROP_EXPOSURE)))

        for key, val in ctrls.items():
            if str(key) in ("auto_exposure", "exposure_absolute"):
                continue                      # 上面已按正确顺序处理过
            spec = self._CTRL_TO_PROP.get(str(key))
            if spec is None:
                print("[camera] 未知控件 %s（Windows 下请手工在相机工具里设置）" % key)
                continue
            attr, mapping = spec
            if not hasattr(cv2, attr):
                print("[camera] OpenCV 没有 %s，跳过 %s" % (attr, key))
                continue
            if mapping is not None:
                try:
                    val = mapping[int(val)]
                except (KeyError, ValueError, TypeError):
                    pass
            try:
                ok = self.cap.set(getattr(cv2, attr), float(val))
            except Exception as e:  # noqa: BLE001
                print("[camera] 控件设置异常 %s=%s: %s" % (key, val, e))
                continue
            # ★ 返回 False 不等于失败：DSHOW 下很多控件由驱动自己管，
            #   这里只报出来，让使用者知道该不该去相机工具里手工锁。
            if not ok:
                print("[camera] 控件 %s=%s 未被驱动接受（返回 False）。"
                      "若光照/白平衡漂移，请在 Windows 相机设置里手工锁死。" % (key, val))
        print("[camera] 已尝试锁定 %d 个控件（Windows/OpenCV 路径）" % len(ctrls))

    # ── TODO: V4L2 裸流 + 降采样解码 ──────────────────────────
    def _open_raw(self, node: str) -> None:
        raise NotImplementedError(
            "V4L2 裸流采集尚未实现。需要 fcntl.ioctl 走 VIDIOC_S_FMT/REQBUFS/QBUF/DQBUF "
            "取 MJPEG 缓冲区，再用 cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_2) 解码。"
            "先用 capture.decoder=full。")

    def _read_raw(self) -> bytes:
        raise NotImplementedError
