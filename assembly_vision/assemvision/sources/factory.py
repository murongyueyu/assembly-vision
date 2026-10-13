# -*- coding: utf-8 -*-
"""按配置造出图像来源。命令行参数优先于配置文件。"""
from __future__ import annotations

from typing import Optional

from ..config import Config
from .base import CameraSource


def make_source(cam_cfg: Config,
                source: Optional[str] = None,
                path: Optional[str] = None) -> CameraSource:
    src = (source or cam_cfg.get("source", "uvc")).lower()

    if src == "uvc":
        from .uvc_stereo import UvcStereoSource
        return UvcStereoSource(cam_cfg)

    if src in ("video", "images", "image"):
        from .video_file import VideoFileSource
        p = path or cam_cfg.get("path", "")
        if not p:
            raise ValueError("source=%s 时必须给 --path" % src)
        return VideoFileSource(cam_cfg, p)

    raise ValueError("未知的图像来源 source=%r（uvc | video | images）" % src)
