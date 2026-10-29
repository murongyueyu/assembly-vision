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

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
        self._bmp: "OrderedDict" = OrderedDict()   # 文字位图缓存（见 _raster）
        self._bmp_max = 256

    def font(self, size: int):
        size = max(10, int(round(size * self._scale)))
        if size not in self._cache:
            self._cache[size] = ImageFont.truetype(self.font_path, size)
        return self._cache[size]

    def _raster(self, text: str, size: int, anchor: str,
                color: Tuple[int, int, int]):
        """把一条文字光栅化成 RGBA 位图并缓存，返回 (图像, 相对锚点的 dx, dy)。

        ★ 2026-10 Pi4B 实测：每帧对每条文字走一次 PIL 往返约 3.5ms，
          6~17 条就是 20~60ms，是 render 的最大单项。同一文字（零件名、
          状态行）内容不变，位图可以复用 —— 命中后每条约 0.05ms。
          用 LRU 上限，避免底部状态栏里「帧号」这种每秒都变的文本撑爆内存。
        """
        key = (text, size, anchor, color)
        ent = self._bmp.get(key)
        if ent is not None:
            self._bmp.move_to_end(key)
            return ent
        font = self.font(size)
        try:
            bb = font.getbbox(text, anchor=anchor)
        except Exception:                       # noqa: BLE001  锚点不支持时退回估值
            bb = (0, 0, len(text) * int(size), int(size))
        w = max(1, bb[2] - bb[0])
        h = max(1, bb[3] - bb[1])
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(img).text(
            (-bb[0], -bb[1]), text, font=font,
            fill=(int(color[2]), int(color[1]), int(color[0]), 255), anchor=anchor)
        ent = (np.asarray(img), int(bb[0]), int(bb[1]))
        self._bmp[key] = ent
        if len(self._bmp) > self._bmp_max:
            self._bmp.popitem(last=False)
        return ent

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

        ch, cw = canvas.shape[:2]
        for text, org, color, size, anchor in items:
            arr, dx, dy = self._raster(str(text), int(size), anchor or "la",
                                       (int(color[0]), int(color[1]), int(color[2])))
            h, w = arr.shape[:2]
            x0, y0 = int(org[0]) + dx, int(org[1]) + dy
            cx0, cy0 = max(0, x0), max(0, y0)
            cx1, cy1 = min(cw, x0 + w), min(ch, y0 + h)
            if cx1 <= cx0 or cy1 <= cy0:
                continue                        # 完全在画布外
            src = arr[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            sub = canvas[cy0:cy1, cx0:cx1]
            a = src[:, :, 3].astype(np.uint16)[:, :, None]
            fg = src[:, :, :3][:, :, ::-1].astype(np.uint16)     # RGBA 的 RGB -> BGR
            out = (sub.astype(np.uint16) * (255 - a) + fg * a + 127) // 255
            sub[:] = out.astype(np.uint8)


class OpenCVRenderer(Renderer):
    def __init__(self, window: str = "装配流程检查", fullscreen: bool = True,
                 display: Tuple[int, int] = (1920, 1080), panel_ratio: float = 0.20,
                 font_path: str = "", colors: Optional[Dict[str, Sequence[int]]] = None,
                 video_fps: int = 30,
                 actions: Optional[Dict[str, Callable]] = None) -> None:
        """★ `actions` 必须与 `QtRenderer` **同名同义**（实测踩过）：

        `app.py` 是无条件把 `actions={...}` 传给 `make_renderer()` 的，
        而 OpenCV 渲染器原本没有这个参数 -> `app.py --windowed` **一启动就崩**：
            TypeError: OpenCVRenderer.__init__() got an unexpected keyword argument 'actions'
        因为 `--windowed` 的默认渲染器就是 opencv，所以这条路径当时**根本跑不通**。

        现在两个渲染器接口一致，并且这里**真的把动作接到按键上**：
            i  手动抽检一次（Qt 版也有这个按钮）
            r  复位流程
            q / ESC  退出
        """
        # ★ 必须调基类 __init__：`Renderer.__init__` 设了 `self.frames_drawn = 0`，
        #   而 `Renderer.run()` 靠它做 `--max-frames` 计数。
        #   漏掉的症状（实测踩过，是 app.py --windowed 的**第二个**崩）：
        #     AttributeError: 'OpenCVRenderer' object has no attribute 'frames_drawn'
        #   QtRenderer 是 QMainWindow 子类、链路里自然调到了基类，
        #   所以只有 OpenCV 渲染器有这个坑 —— 即 --windowed 的默认路径。
        super().__init__()
        self.window = window
        self.fullscreen = bool(fullscreen)
        self.W, self.H = int(display[0]), int(display[1])
        self.panel_ratio = float(panel_ratio)
        self.fps = int(video_fps)
        self.font_path = find_cjk_font(font_path)
        self.text = _Text(self.font_path, scale=self.H / 1080.0)
        self.actions: Dict[str, Callable] = dict(actions or {})
        self._inited = False
        self._operator = None
        self.cache_unchanged = False
        self.cache_hits = self.canvases_built = 0
        self._cache_key = self._cache_frame = self._cache_canvas = None
        self._panel_key = None                 # 右侧面板缓存（见 _render_frame）
        self._panel_cache: Optional[np.ndarray] = None
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
        if state.workflow is not None:
            if self._operator is None:
                from .operator_view import OperatorView
                # ★ 文字直接按最终画布像素绘制，消除小屏下重复缩放。
                self._operator = OperatorView(_Text(self.font_path, scale=1.0))
            return self._operator.render(state, self.W, self.H)
        if not self.cache_unchanged:
            return self._render_frame(state)
        # ★ 仅显式Pi候选开启；帧对象归采集所有且不被原地修改。
        # 同帧仍逐项比较文字、框和轮廓；过期/复位/测量变更立即重绘。
        key = (state.level,state.title,tuple(state.detail),state.step,state.total,
               tuple((k,repr(v)) for k,v in state.stats.items()),state.banner,state.source_name,
               tuple((tuple(box),label,color) for box,label,color in state.boxes),
               tuple((p.shape,p.dtype.str,p.tobytes(),color,int(thick)) for p,color,thick in state.polygons),
               tuple((k,tuple(v)) for k,v in DEFAULT_COLORS.items()))
        if self._cache_frame is state.frame and self._cache_key == key:
            self.cache_hits += 1
            return self._cache_canvas.copy()
        canvas = self._render_frame(state)
        self._cache_frame, self._cache_key, self._cache_canvas = state.frame,key,canvas
        return canvas.copy()

    def _render_frame(self, state: UIState) -> np.ndarray:
        self.canvases_built += 1
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
            # ★ 直接缩放到目标 ROI（2026-10 Pi4B 实测）：旧版先 resize 出中间
            #   缓冲再整块赋值，多一次 4.2MB 拷贝（实测 5.7ms/帧）。写进 dst
            #   后连这一趟拷贝一起省掉，像素结果与原写法逐字节相同。
            roi = canvas[int(oy):int(oy) + nh, int(ox):int(ox) + nw]
            cv2.resize(state.frame, (nw, nh), dst=roi, interpolation=cv2.INTER_AREA)

            for poly, ckey, thick in state.polygons:
                p = _map_poly(poly, sx, ox, oy)
                if p is not None:
                    # ★ 轮廓用 LINE_8 不用 LINE_AA（2026-10 Pi4B 实测）：
                    #   轮盘轮廓有 500+ 个点，抗锯齿绘制在 Pi 上要 5ms 以上，
                    #   而轮廓只是提示位置，硬边完全够用。
                    cv2.polylines(canvas, [p], True, state.color(ckey), int(thick), cv2.LINE_8)

            for bbox, label, ckey in state.boxes:
                x, y, w, h = bbox
                p1 = (int(x * sx + ox), int(y * sx + oy))
                p2 = (int((x + w) * sx + ox), int((y + h) * sx + oy))
                col = state.color(ckey)
                cv2.rectangle(canvas, p1, p2, col, 2, cv2.LINE_8)
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

        # 4) 右侧面板（整块缓存）
        # ★ 2026-10 Pi4B 实测：面板里最多 12 条明细，每帧重画要 30~40ms，
        #   而静态观察时内容几乎不变。把面板整块（底色 + 图形 + 文字）缓存，
        #   命中时直接贴回（约 0.3ms），像素与原路径逐字节相同。
        panel_key = (state.level, state.title, tuple(state.detail),
                     state.step, state.total,
                     tuple((k, tuple(v)) for k, v in DEFAULT_COLORS.items()))
        if self._panel_cache is not None and self._panel_key == panel_key:
            canvas[:, vw:] = self._panel_cache
        else:
            panel_texts: List[Tuple[str, Tuple[int, int], Tuple[int, int, int], int, str]] = []
            self._draw_panel(canvas, state, vw, pw, panel_texts)
            self.text.draw(canvas, panel_texts)
            self._panel_cache = canvas[:, vw:].copy()
            self._panel_key = panel_key

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
                cv2.circle(canvas, (px, py), r, col, -1, cv2.LINE_8)

        # 4.2 大圆形状态灯（颜色最醒目的部分）
        lamp_r = int(pw * 0.28)
        lamp_y = int(self.H * 0.32)
        col = state.color(state.level)
        cv2.circle(canvas, (cx, lamp_y), lamp_r, col, -1, cv2.LINE_8)
        cv2.circle(canvas, (cx, lamp_y), lamp_r, (20, 20, 20), 3, cv2.LINE_8)
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
            else:
                # ★ 2026-10 Pi 实测：`--windowed` 下 WINDOW_NORMAL 不会跟着画布
                #   尺寸走，1920x1080 的画面被开成约 400x350，现场根本看不清。
                #   显式 resizeWindow 到画布尺寸；减掉标题栏(48)与窗口边框(4)，
                #   再拉回左上角 —— 否则窗口继承上次位置，右侧面板会被裁掉。
                cv2.resizeWindow(self.window, max(240, self.W - 4),
                                 max(240, self.H - 48))
                cv2.moveWindow(self.window, 0, 0)
            self._inited = True
        cv2.imshow(self.window, canvas)
        # ★ 用 pollKey 而不是 waitKey（2026-10 Pi4B 实测）：旧版按 1000/fps 阻塞，
        #   30fps 时每帧白等 **33~35ms**；改成 waitKey(1) 后还剩约 4ms 的等待开销。
        #   pollKey 完全不等待，实测端到端再省 ~4ms（38.4ms → 34.3ms）。
        #   两个都负责 pump 窗口事件，按键不会丢。
        try:
            key = cv2.pollKey() & 0xFF
        except AttributeError:                  # 老版本 OpenCV 没有 pollKey
            key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):            # ESC / q 退出
            return False
        # ★ 动作键（与 QtRenderer 的按钮同义）—— 接口必须一致，
        #   否则 app.py 传 actions 时两个渲染器行为不一样，很难排查。
        if key == ord("f"):
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord("i") and "inspect" in self.actions:
            try:
                self.actions["inspect"]()
                print("[ui] 已请求手动抽检（i）")
            except Exception as e:                        # noqa: BLE001
                print("[ui] 抽检失败: %s" % e)
        elif key == ord("r") and "reset" in self.actions:
            try:
                self.actions["reset"]()
                print("[ui] 已复位流程（r）")
            except Exception as e:                        # noqa: BLE001
                print("[ui] 复位失败: %s" % e)
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
