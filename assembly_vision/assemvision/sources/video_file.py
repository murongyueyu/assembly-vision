# -*- coding: utf-8 -*-
"""离线图像来源：视频文件 / 图片目录 / 单张图片。

这是"相机和支架没到也能开发"的关键 —— 所有算法都能喂这里跑。
"""
from __future__ import annotations

import glob
import os
import time
from typing import List, Optional

import cv2
import numpy as np

from ..config import Config
from ..imgio import imread_safe
from ..types import Frame
from .base import CameraSource, split_stereo

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


class VideoFileSource(CameraSource):
    """视频或图片序列。

    camera.yaml 里的 capture.mode / left_index 同样生效：
    如果视频本身就是左右拼接的双目录像，会自动拆成左右眼。

    ★ 默认按 video.fps 实时播放（video.pace: true）。
      否则文件会以几百帧/秒被读完，而逻辑线程只有 12 Hz ——
      它看到的将是序列里的随机帧，去抖、事件、流程判断全部失效。
    """

    def __init__(self, cam_cfg: Config, path: str) -> None:
        super().__init__(scale_realtime=float(cam_cfg.get("scale.realtime", 0.5)))
        self.cfg = cam_cfg
        self.path = path
        self.mode = str(cam_cfg.get("capture.mode", "side_by_side"))
        self.left_index = int(cam_cfg.get("capture.left_index", 0))
        self.loop = bool(cam_cfg.get("video.loop", True))
        self.pace = bool(cam_cfg.get("video.pace", True))
        # ★ 离线素材默认是"单眼"画面（一段普通视频/一张图），不做左右眼拆分。
        #   只有当你录下的本身就是双目拼接画面时才设 video.stereo: true。
        self.stereo = bool(cam_cfg.get("video.stereo", False))
        self.play_fps = float(cam_cfg.get("video.fps", 0)) or \
            float(cam_cfg.get("capture.fps", 30))
        self.cap: Optional[cv2.VideoCapture] = None
        self.images: List[str] = []
        self._cursor = 0
        self._next_t = 0.0

    def open(self) -> None:
        if os.path.isdir(self.path):
            self.images = sorted(
                p for p in glob.glob(os.path.join(self.path, "*"))
                if p.lower().endswith(IMG_EXT))
            if not self.images:
                raise FileNotFoundError("目录里没有图片: %s" % self.path)
            print("[video] 图片序列 %d 张: %s" % (len(self.images), self.path))
            return

        if not os.path.exists(self.path):
            raise FileNotFoundError("视频不存在: %s" % self.path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError("打不开视频: %s" % self.path)
        print("[video] 打开 %s  %dx%d @%.1ffps"
              % (self.path,
                 int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 self.cap.get(cv2.CAP_PROP_FPS)))

    def read(self, want_full: bool = False) -> Optional[Frame]:
        img = None
        if self.images:
            if self._cursor >= len(self.images):
                if not self.loop:
                    return None
                self._cursor = 0
            img = imread_safe(self.images[self._cursor])
            self._cursor += 1
        else:
            if self.cap is None:
                return None
            ok, frame = self.cap.read()
            if not ok:
                if not self.loop:
                    return None
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if not ok:
                    return None
            img = frame

        if img is None:
            return None
        self._sleep_pace()
        if self.stereo:
            left, right = split_stereo(img, self.mode, self.left_index)
        else:
            left, right = img, img      # 单眼素材：两眼都给整幅（深度自然不可用）
        return self.make_frame(left, right, self._next_index(), self._ts(),
                               want_full or not self.images)

    def _sleep_pace(self) -> None:
        """按 video.fps 节流，让离线播放的速度和真相机一致。"""
        if not self.pace or self.play_fps <= 0:
            return
        now = time.time()
        if self._next_t <= 0.0:
            self._next_t = now
        self._next_t += 1.0 / self.play_fps
        dt = self._next_t - now
        if dt > 0:
            time.sleep(dt)
        elif dt < -1.0:                 # 落后太多就重新对齐，避免追赶时暴冲
            self._next_t = now

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def name(self) -> str:
        return "VideoFileSource(%s)" % os.path.basename(self.path)


class ImageFolderSource(VideoFileSource):
    """只是语义更清楚的别名。"""
