# -*- coding: utf-8 -*-
"""按配置造出图像来源。命令行参数优先于配置文件。

支持的来源（`camera.yaml` 的 `source:`，或命令行 `--source`）：
    uvc                 普通 UVC 双目相机（HBV-1714-2，走 cv2.VideoCapture/dshow）
    d435 / realsense    Intel RealSense D435（走 pyrealsense2，**必须 USB 3**）
    video / images      离线：视频文件 / 图片目录

★ 为什么 D435 要单列一条：它**不是普通 UVC 相机**，OpenCV 按索引根本枚举不到
  （见 docs/踩坑实录.md §2.30）。换机后 `source:` 要一起改，否则会以为"相机坏了"。
"""
from __future__ import annotations

from typing import Optional

from ..config import Config
from .base import CameraSource

# D435 的几种写法都认（配置里写 d435，命令行里顺手写 realsense 也不会错）
_REALSENSE_NAMES = ("d435", "d435i", "realsense", "rs")


def make_source(cam_cfg: Config,
                source: Optional[str] = None,
                path: Optional[str] = None) -> CameraSource:
    src = (source or cam_cfg.get("source", "uvc")).lower()

    if src == "uvc":
        from .uvc_stereo import UvcStereoSource
        return UvcStereoSource(cam_cfg)

    if src in _REALSENSE_NAMES:
        # ★ 延迟导入：没装 pyrealsense2 的机器上，用 uvc/video/images 的人不该被拖累
        from .d435 import RealSenseSource
        return RealSenseSource(cam_cfg)

    if src in ("video", "images", "image"):
        from .video_file import VideoFileSource
        p = path or cam_cfg.get("path", "")
        if not p:
            raise ValueError("source=%s 时必须给 --path" % src)
        return VideoFileSource(cam_cfg, p)

    raise ValueError("未知的图像来源 source=%r（uvc | d435/realsense | video | images）" % src)
