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
    返回的 writer 多一个 `.finalize()` 方法。

    ★★ 两条实测踩过的坑（2026-10 加）：
      ① **`cv2.VideoWriter` 打不开时不抛异常** —— `isOpened()` 为 False，
         之后每次 `write()` **静默什么都不做**，调用方于是以为「录上了」，
         其实一个文件都没生成。⇒ 这里**当场查 `isOpened()` 并打清楚**。
      ② **改名只发生在 `release()` 里** —— 也就是**只有正常退出（按 q）才生成
         最终文件**；中途 Ctrl+C / 直接关窗口，目录里只剩一个 `_tmp_*.mp4`。
         ⇒ 这里把临时名也报出来，至少能找回来。
         ★ 用法忠告：**录完按 `q` 退出，别关窗口。**
    """
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".mp4"
    tmp = os.path.join(d, "_tmp_%d%s" % (abs(hash(path)) % 100000, ext))
    w = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*fourcc), float(fps), size, is_color)
    if not w.isOpened():
        print("  ★★ VideoWriter 打不开：fourcc=%s -> %s" % (fourcc, tmp))
        print("     常见原因：① 目标目录不可写（受限进程写不进 assembly_vision\\，实录 2.40）")
        print("               ② 这个 fourcc 在本机 OpenCV 里没编进（可试 mp4v / XVID / MJPG）")
        print("     ★ VideoWriter **不抛异常** —— 不查 isOpened() 就会以为录上了。")
    else:
        print("  ● 正在录制 -> %s" % tmp)
        print("    ★ 录完请按 q 正常退出，才会改名成 %s" % os.path.basename(path))

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
