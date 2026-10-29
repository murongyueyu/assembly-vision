# -*- coding: utf-8 -*-
"""CameraSource 抽象与双目拼接图拆分工具。"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np

from ..types import Frame


class CameraSource(ABC):
    """所有图像来源的统一接口。

    实现者只需要保证：
      * open() 成功后 read() 能返回 Frame（或 None 表示暂时取不到）
      * read() 返回的 left/right 是『单眼』图像，已经按 mode 拆好
      * want_full=True 时尽量给出全分辨率的 left_full（给不了就留 None）
    """

    def __init__(self, scale_realtime: float = 0.5) -> None:
        self.scale_realtime = float(scale_realtime)
        self._index = 0
        self._t0 = time.time()

    # ---- 生命周期 ----
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self, want_full: bool = False) -> Optional[Frame]: ...

    def close(self) -> None:  # 子类按需覆盖
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # ---- 供子类使用的小工具 ----
    def _next_index(self) -> int:
        self._index += 1
        return self._index

    def _ts(self) -> float:
        return time.time() - self._t0

    def make_frame(self, left_full: np.ndarray, right_full: Optional[np.ndarray],
                   index: int, ts: float, full: bool) -> Frame:
        left_small = scale_frame(left_full, self.scale_realtime)
        right_small = scale_frame(right_full, self.scale_realtime) if right_full is not None else None
        return Frame(index=index, ts=ts,
                     left=left_small, right=right_small,
                     left_full=left_full if full else None,
                     right_full=right_full if full else None,
                     scale=self.scale_realtime)


# ──────────────────────────────────────────────────────────────
def split_stereo(img: np.ndarray, mode: str = "side_by_side",
                 left_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """把 UVC 输出的拼接图拆成左右眼。

    mode        side_by_side（左右并排） | top_bottom（上下堆叠）
    left_index  0 = 左眼在左/上；1 = 左眼在右/下
                ★ 这一项必须用纸巾挡住一个镜头实测确认，规格书不会写。
    """
    h, w = img.shape[:2]
    if mode == "side_by_side":
        half = w // 2
        a, b = img[:, :half], img[:, half:half * 2]
    elif mode == "top_bottom":
        half = h // 2
        a, b = img[:half, :], img[half:half * 2, :]
    else:
        raise ValueError("未知的拼接方式 mode=%r（应为 side_by_side / top_bottom）" % mode)
    return (a, b) if left_index == 0 else (b, a)


def scale_frame(img: np.ndarray, scale: float) -> np.ndarray:
    """按比例缩放，scale=1.0 时原样返回（不做无谓的拷贝）。"""
    if img is None:
        return img
    if abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)
