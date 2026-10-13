# -*- coding: utf-8 -*-
"""默认渲染器：OpenCV 全屏窗口 + numpy 拼版 + PIL 画中文。

零额外依赖（除 Pillow）。布局：
    ┌──────────────────────────────┬────────────┐
    │                              │ 步骤进度    │
    │      主视频区（约 80%）        │ ●──●──○──○ │
    │                              │            │
    │  黄铜件：绿框                  │   大圆形    │
    │  蓝手套：蓝框                  │   状态灯    │
    │  槽位 ROI：细线                 │            │
    │  缺陷处：红框                  │  结论文字   │
    │                              │  原因明细   │
    ├──────────────────────────────┤            │
    │ FPS / 检测 / 延迟 / 进度       │            │
    └──────────────────────────────┴────────────┘
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:                                   # pragma: no cover
    _HAS_PIL = False

from ..types import ERROR, IDLE, OK, WARN
from .base import (DEFAULT_COLORS, LEVEL_GLYPH, Renderer, UIState,
                   find_cjk_font)

WHITE = (235, 235, 235)
GREY = (150, 150, 150)
PANEL_BG = (38, 38, 38)
CANVAS_BG = (22, 22, 22)
GLOVE_COLOR = (230, 120, 40)          # 蓝手套用橙色框，避免和手套本身同色


class _Text:
    """PIL 中文绘制。收集所有文本，一次成像，避免逐条来回转换的开销。"""

    def __init__(self, font_path: Optional[str], scale: float = 1.0) -> None:
        self.ok = _HAS_PIL and font_path is not None
        self.font_path = font_path
        self._cache: Dict[int, "ImageFont.FreeTypeFont"] = {}
        self._scale = float(scale)

    def font(self, size: int):
        size = max(10, int(round(size * self._scale)))
        if size not in self._cache:
            self._cache[size] = ImageFont.truetype(self.font_path, size)
        return self._cache[size]

    def draw(self, canvas: np.ndarray,
             items: Sequence[Tuple[str, Tuple[int, int], Tuple[int, int, int], int, str]]) -> None:
        """items: (文本, 锚点像素, 颜色BGR, 字号, 锚点方位 l/m/r + a/t/m/b)"""
        if not items:
            return
        if not self.ok:
            for text, org, color, size, _anchor in items:
                cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.4, size / 30.0), color, 1, cv2.LINE_AA)
            return
        img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for text, org, color, size, anchor in items:
            rgb = (int(color[2]), int(color[1]), int(color[0]))
            d.text((int(org[0]), int(org[1])), str(text), font=self.font(size),
                   fill=rgb, anchor=anchor or "la")
        canvas[:, :, :] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class OpenCVRenderer(Renderer):
    def __init__(self, window: str = "装配流程检查", fullscreen: bool = True,
                 display: Tuple[int, int] = (1920, 1080), panel_ratio: float = 0.20,
                 font_path: str = "", colors: Optional[Dict[str, Sequence[int]]] = None,
                 video_fps: int = 30) -> None:
        self.window = window
        self.fullscreen = bool(fullscreen)
        self.W, self.H = int(display[0]), int(display[1])
        self.panel_ratio = float(panel_ratio)
        self.fps = int(video_fps)
        self.font_path = find_cjk_font(font_path)
        self.text = _Text(self.font_path, scale=self.H / 1080.0)
        self._inited = False
        if colors:
            self.set_colors(colors)
        if not self.font_path:
            print("[ui] 没找到中文字体，界面将退回英文/拼音标签。"
                  "可在 workpiece.yaml 的 ui.font_path 指定字体文件。")

    # ── 布局参数 ──────────────────────────────────────────────
    def _layout(self):
        pw = int(self.W * self.panel_ratio)
        vw = self.W - pw
        bar_h = int(self.H * 0.055)
        vh = self.H - bar_h
        return vw, vh, pw, bar_h

    # ── 主入口 ────────────────────────────────────────────────
    def render(self, state: UIState) -> np.ndarray:
        vw, vh, pw, bar_h = self._layout()
        canvas = np.full((self.H, self.W, 3), CANVAS_BG, dtype=np.uint8)
        texts: List[Tuple[str, Tuple[int, int], Tuple[int, int, int], int, str]] = []

        # 1) 主视频区
        sx = sy = 1.0
        ox = oy = 0.0
        if state.frame is not None:
            fh, fw = state.frame.shape[:2]
            sx = min(vw / float(fw), vh / float(fh))
            nw, nh = int(fw * sx), int(fh * sx)
            ox, oy = (vw - nw) / 2.0, (vh - nh) / 2.0
            small = cv2.resize(state.frame, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas[int(oy):int(oy) + nh, int(ox):int(ox) + nw] = small

            for poly, ckey, thick in state.polygons:
                p = _map_poly(poly, sx, ox, oy)
                if p is not None:
                    cv2.polylines(canvas, [p], True, state.color(ckey), int(thick), cv2.LINE_AA)

            for bbox, label, ckey in state.boxes:
                x, y, w, h = bbox
                p1 = (int(x * sx + ox), int(y * sx + oy))
                p2 = (int((x + w) * sx + ox), int((y + h) * sx + oy))
                col = state.color(ckey)
                cv2.rectangle(canvas, p1, p2, col, 2, cv2.LINE_AA)
                if label:
                    texts.append((label, (p1[0], max(0, p1[1] - 6)), col, 20 * self.H / 1080.0, "ld"))

        # 2) 严重告警横幅
        if state.banner:
            cv2.rectangle(canvas, (0, 0), (vw, int(vh * 0.075)), DEFAULT_COLORS[ERROR], -1)
            texts.append((state.banner, (20, int(vh * 0.037)), WHITE,
                          30 * self.H / 1080.0, "lm"))

        # 3) 底部状态条
        y0 = vh
        cv2.rectangle(canvas, (0, y0), (vw, self.H), (30, 30, 30), -1)
        stats = "  |  ".join("%s %s" % (k, v) for k, v in state.stats.items())
        texts.append((stats, (14, y0 + bar_h // 2), GREY, 19, "lm"))
        if state.source_name:
            texts.append((state.source_name, (vw - 14, y0 + bar_h // 2), GREY, 18, "rm"))

        # 4) 右侧面板
        self._draw_panel(canvas, state, vw, pw, texts)

        self.text.draw(canvas, texts)
        return canvas

    # ── 右侧面板 ──────────────────────────────────────────────
    def _draw_panel(self, canvas, state: UIState, x0: int, pw: int,
                    texts: List) -> None:
        cv2.rectangle(canvas, (x0, 0), (self.W, self.H), PANEL_BG, -1)
        cv2.line(canvas, (x0, 0), (x0, self.H), (70, 70, 70), 1)
        cx = x0 + pw // 2
        pad = int(pw * 0.10)

        # 4.1 步骤进度
        top = int(self.H * 0.035)
        n = max(1, int(state.total or 1))
        r = max(6, int(pw * 0.035))
        span = pw - 2 * pad
        texts.append(("装配进度  %d/%d" % (state.step, state.total),
                      (cx, top - int(r * 2.0)), GREY, 20, "mm"))
        if n > 1:
            for i in range(n):
                px = int(x0 + pad + span * i / (n - 1))
                py = top + r
                if i < state.step:
                    col = DEFAULT_COLORS[OK]
                elif i == state.step:
                    col = DEFAULT_COLORS[WARN]
                else:
                    col = (80, 80, 80)
                if i > 0:
                    pprev = int(x0 + pad + span * (i - 1) / (n - 1))
                    cv2.line(canvas, (pprev, py), (px, py), (80, 80, 80), 2)
                cv2.circle(canvas, (px, py), r, col, -1, cv2.LINE_AA)

        # 4.2 大圆形状态灯（颜色最醒目的部分）
        lamp_r = int(pw * 0.28)
        lamp_y = int(self.H * 0.32)
        col = state.color(state.level)
        cv2.circle(canvas, (cx, lamp_y), lamp_r, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, lamp_y), lamp_r, (20, 20, 20), 3, cv2.LINE_AA)
        glyph = LEVEL_GLYPH.get(state.level, "?")
        texts.append((glyph, (cx, lamp_y), (255, 255, 255),
                      int(lamp_r * 1.25), "mm"))

        # 4.3 结论
        y = lamp_y + lamp_r + int(self.H * 0.055)
        texts.append((state.title or "就绪", (cx, y), WHITE,
                      int(self.H * 0.030), "mm"))

        # 4.4 原因明细
        y += int(self.H * 0.045)
        cv2.line(canvas, (x0 + pad, y - int(self.H * 0.012)),
                 (self.W - pad, y - int(self.H * 0.012)), (70, 70, 70), 1)
        size = int(self.H * 0.021)
        line_h = int(size * 1.75)
        for i, line in enumerate(state.detail[:12]):
            if y + i * line_h > self.H - 12:
                break
            texts.append((line, (x0 + pad, y + i * line_h), (215, 215, 215), size, "la"))

    # ── 显示 ──────────────────────────────────────────────────
    def show(self, canvas: Optional[np.ndarray]) -> bool:
        if canvas is None:
            return True
        if not self._inited:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            if self.fullscreen:
                cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            self._inited = True
        cv2.imshow(self.window, canvas)
        key = cv2.waitKey(max(1, int(1000 / max(1, self.fps)))) & 0xFF
        if key in (27, ord("q")):            # ESC / q 退出
            return False
        return True

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.window)
        except Exception:                     # noqa: BLE001
            pass


def _map_poly(poly, sx: float, ox: float, oy: float):
    if poly is None or len(poly) < 3:
        return None
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    p = p * sx + np.array([ox, oy])
    return np.round(p).astype(np.int32).reshape(-1, 1, 2)
