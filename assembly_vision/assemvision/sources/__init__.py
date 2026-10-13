# -*- coding: utf-8 -*-
"""图像来源层。核心算法完全不关心这里是真相机还是视频文件。"""
from __future__ import annotations

from .base import CameraSource, split_stereo, scale_frame
from .factory import make_source

__all__ = ["CameraSource", "split_stereo", "scale_frame", "make_source"]
