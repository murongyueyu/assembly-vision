# -*- coding: utf-8 -*-
"""兼容中文路径的图片读写。

★ 为什么需要这个：OpenCV 在 Windows 上用窄字符 API 打开文件，路径里只要有中文，
  cv2.imread / cv2.imwrite 就会**静默失败**（不抛异常，只是返回 None / 返回 False）。
  本项目的目录名是中文，所以全项目统一走这里的函数。

Linux（树莓派）本来没问题，用这个也不会有额外代价。
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import cv2
import numpy as np


def imread_safe(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """读图，支持中文路径。失败返回 None。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, flags)
    return img


def imwrite_safe(path: str, img: np.ndarray,
                 params: Optional[Sequence[int]] = None) -> bool:
    """写图，支持中文路径。"""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, img, list(params or []))
    except cv2.error:
        return False
    if not ok:
        return False
    try:
        buf.tofile(path)
    except OSError:
        return False
    return True


def videowriter_safe(path: str, fourcc: str, fps: float,
                     size: tuple, is_color: bool = True):
    """VideoWriter 同样不认中文路径，这里先用临时 ASCII 名再改名。

    简单做法：写到同目录的临时文件中转（文件名用 ascii），关闭时改名。
    返回的 writer 多一个 .finalize() 方法。
    """
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, "_tmp_%d%s" % (abs(hash(path)) % 100000,
                                         os.path.splitext(path)[1] or ".mp4"))
    w = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*fourcc), float(fps), size, is_color)

    class _W:
        def write(self, frame):
            w.write(frame)

        def release(self):
            w.release()
            try:
                if os.path.exists(path):
                    os.remove(path)
                os.replace(tmp, path)
            except OSError:
                pass

        def finalize(self):
            self.release()

    return _W()
