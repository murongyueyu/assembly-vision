# -*- coding: utf-8 -*-
"""中文文字渲染（★ OpenCV 的 Hershey 字体**没有中文字形** ⇒ 全是乱码）。

★★ 为什么单独写这个（实测踩过）：
  `cv2.putText` 用的是 Hershey 矢量字体，**只覆盖拉丁字母/数字**。
  中文传进去会渲染成一堆乱码或方块 —— 用户实测反馈
  「滑块没有中文我根本看不懂，中文显示是乱码」。
  ⇒ 用 PIL（Pillow）加载系统中文（微软雅黑 msyh.ttc）把中文画出来，
    再转成 numpy 贴回 OpenCV 图上。每帧只渲一次、**带缓存**，不影响帧率。

★ 用法：
    from 中文文字 import put_cn, put_cn_bar
    put_cn(img_bgr, "轮盘 wheel", (20, 40), 22, (0,0,255))
    img = put_cn_bar(img, "料盘 OK | 检出 4/4", 26)
"""
from __future__ import annotations

import os

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:                                                  # noqa: BLE001
    _HAS_PIL = False

# ★ 按优先级找中文字体（Windows 自带）
_FONT_CANDS = [
    r"C:\Windows\Fonts\msyh.ttc",       # 微软雅黑（最好看）
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",     # 黑体
    r"C:\Windows\Fonts\Deng.ttf",       # 等线
    r"C:\Windows\Fonts\simsun.ttc",     # 宋体
]
_FONT_PATH = next((p for p in _FONT_CANDS if os.path.exists(p)), None)
_CACHE: dict = {}
# ★ 渲染缓存：同一段文字+字号只渲一次（滑条拖来拖去也不会掉帧）
_TEXCACHE: dict = {}


def font(size: int):
    key = ("f", int(size))
    if key not in _CACHE:
        if _HAS_PIL and _FONT_PATH:
            try:
                _CACHE[key] = ImageFont.truetype(_FONT_PATH, int(size))
            except Exception:                                      # noqa: BLE001
                _CACHE[key] = None
        else:
            _CACHE[key] = None
    return _CACHE[key]


def has_cn() -> bool:
    return bool(_HAS_PIL and _FONT_PATH)


def put_cn(img: np.ndarray, text: str, org, size: int = 22,
           color=(255, 255, 255), bg=None, pad: int = 2) -> np.ndarray:
    """在 BGR 图上写**中文**（org 是**左上角**，与 cv2.putText 的基线不同）。

    ★ 带缓存：相同 (文字, 字号, 颜色) 只渲染一次 —— 实时循环里这样才不掉帧。
    ★ PIL 不可用或没有中文字体时**自动退回 cv2.putText**（至少不崩）。
    """
    if not has_cn():
        cv2.putText(img, text, (int(org[0]), int(org[1]) + int(size)),
                    cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, color, 1, cv2.LINE_AA)
        return img
    key = (text, int(size), tuple(color), None if bg is None else tuple(bg))
    if key not in _TEXCACHE:
        f = font(size)
        tmp = Image.new("RGBA", (10, 10))
        d = ImageDraw.Draw(tmp)
        try:
            box = d.textbbox((0, 0), text, font=f)
        except Exception:                                          # noqa: BLE001
            box = (0, 0, len(text) * size // 2, size)
        w = max(1, box[2] - box[0]) + pad * 2
        h = max(1, box[3] - box[1]) + pad * 2
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dr = ImageDraw.Draw(im)
        if bg is not None:
            dr.rectangle([0, 0, w - 1, h - 1], fill=tuple(bg) + (255,))
        dr.text((pad - box[0], pad - box[1]), text, font=f,
                fill=(int(color[2]), int(color[1]), int(color[0]), 255))
        _TEXCACHE[key] = np.array(im)          # RGBA
    rgba = _TEXCACHE[key]
    h, w = rgba.shape[:2]
    x0, y0 = int(org[0]), int(org[1])
    H, W = img.shape[:2]
    if x0 >= W or y0 >= H:
        return img
    x1, y1 = min(W, x0 + w), min(H, y0 + h)
    sub = rgba[:y1 - y0, :x1 - x0]
    a = sub[:, :, 3:4].astype(np.float32) / 255.0
    roi = img[y0:y1, x0:x1].astype(np.float32)
    rgb = sub[:, :, :3].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - a) + rgb * a).astype(np.uint8)
    return img


def text_size(text: str, size: int):
    if not has_cn():
        return (int(len(text) * size * 0.55), int(size * 1.2))
    f = font(size)
    try:
        box = ImageDraw.Draw(Image.new("RGB", (8, 8))).textbbox((0, 0), text, font=f)
        return (box[2] - box[0], box[3] - box[1])
    except Exception:                                              # noqa: BLE001
        return (int(len(text) * size * 0.55), int(size))


def put_cn_bar(img: np.ndarray, text: str, size: int = 24,
               fg=(255, 255, 255), bgcol=(0, 0, 0)) -> np.ndarray:
    """在图**底部**加一条状态栏并写中文，返回拼好的新图。"""
    tw, th = text_size(text, size)
    h = max(30, th + 12)
    bar = np.zeros((h, img.shape[1], 3), np.uint8)
    bar[:] = bgcol
    put_cn(bar, text, (10, max(2, (h - th) // 2)), size, fg)
    return np.vstack([img, bar])
