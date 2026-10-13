# -*- coding: utf-8 -*-
"""PyQt5 渲染器：左 80% 主画面 + 右 20% 面板。

布局（按需求）
    ┌──────────────────────────────┬────────────────┐
    │                              │  ①● ②● ③○ ④○  │  步骤进度
    │       主视频区（约 80%）        │                │
    │   黄铜件绿框 / 蓝手套橙框        │   ┌────────┐   │
    │   槽位 ROI 细线 / 缺陷红框       │   │  大圆形 │   │  状态灯
    │                              │   └────────┘   │
    │                              │   结论（大字）   │
    │                              │  ────────────  │
    │                              │   原因明细列表   │
    ├──────────────────────────────┤                │
    │ FPS │ 分割 │ 辐条 │ 进度        │   来源         │  底部状态条
    └──────────────────────────────┴────────────────┘

快捷键：空格=手动重判、R=复位流程、F=全屏、Q/ESC=退出
（重判与复位通过 actions 字典注入，渲染器本身不碰算法）

为什么 render() 会返回窗口截图：这样 --save-video 在 Qt 下也能用，
报告/答辩演示时可以录一段界面视频。
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from ..types import IDLE
from .base import Renderer, UIState, find_cjk_font

# 让 QApplication 在离屏/无显示环境下也能跑（做界面预览图时会用到）
if "QT_QPA_PLATFORM" not in os.environ and not os.environ.get("DISPLAY") \
        and sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class QtRenderer(Renderer):
    def __init__(self, window: str = "装配流程检查", fullscreen: bool = True,
                 display: Tuple[int, int] = (1920, 1080), panel_ratio: float = 0.20,
                 font_path: str = "", colors: Optional[Dict[str, Sequence[int]]] = None,
                 video_fps: int = 30, actions: Optional[Dict[str, Callable]] = None,
                 min_panel_px: int = 340) -> None:
        super().__init__()
        try:
            from PyQt5.QtWidgets import QApplication
        except ImportError as e:                      # pragma: no cover
            raise ImportError(
                "QtRenderer 需要 PyQt5：pip install PyQt5\n"
                "或者改用默认的 OpenCV 渲染器（--renderer opencv）。") from e

        self.window_title = window
        self.fullscreen = bool(fullscreen)
        self.display = (int(display[0]), int(display[1]))
        self.panel_ratio = float(panel_ratio)
        self.min_panel_px = int(min_panel_px)
        self.fps = int(max(1, video_fps))
        self.actions = dict(actions or {})
        if colors:
            self.set_colors(colors)

        # 字体：优先用配置/自动找到的中文字体，保证中文不会变方块
        self.font_path = find_cjk_font(font_path)
        self.font_family = ""
        self._apply_app_font = None

        self.app = QApplication.instance() or QApplication(sys.argv[:1])
        self._setup_font()

        from .qt_widgets import (BottomBar, ReasonPanel, VideoCanvas)
        from PyQt5.QtWidgets import (QFrame, QHBoxLayout, QMainWindow,
                                     QVBoxLayout, QWidget)

        class _Win(QMainWindow):
            def keyPressEvent(self, ev):
                self._on_key(ev)

        self.win = _Win()
        self.win.setWindowTitle(window)
        self.win._on_key = self._on_key

        central = QWidget(self.win)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        left = QWidget(central)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        self.canvas = VideoCanvas(left)
        self.bottom = BottomBar(left)
        lv.addWidget(self.canvas, 1)
        sep = QFrame(left)
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3a3a;")
        lv.addWidget(sep)
        lv.addWidget(self.bottom, 0)

        self.panel = ReasonPanel(central)

        root.addWidget(left,
                       int(round((1.0 - self.panel_ratio) * 1000)))
        root.addWidget(self.panel, int(round(self.panel_ratio * 1000)))
        self.win.setCentralWidget(central)

        scale = self.display[1] / 1080.0
        self.canvas.set_font_family(self.font_family, scale)
        self.panel.set_fonts(self.font_family, scale)
        self.bottom.set_fonts(self.font_family, scale)

        w = min(self.display[0], int((1.0 - self.panel_ratio) * 100 + self.min_panel_px))
        self.win.resize(self.display[0], self.display[1])

    # ── 字体 ──────────────────────────────────────────────────
    def _setup_font(self) -> None:
        from PyQt5.QtGui import QFont, QFontDatabase
        fam = ""
        if self.font_path:
            fid = QFontDatabase.addApplicationFont(self.font_path)
            fams = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
            if fams:
                fam = fams[0]
        if not fam:
            f = QFont()
            fam = f.defaultFamily()
        self.font_family = fam
        app_font = QFont(fam)
        app_font.setPointSize(11)
        self.app.setFont(app_font)
        if not self.font_path:
            print("[ui] 没有找到中文字体文件，界面中文可能显示为方块。"
                  "可在 workpiece.yaml 的 ui.font_path 指定字体路径。")

    # ── Renderer 接口 ─────────────────────────────────────────
    def render(self, state: UIState) -> Optional[np.ndarray]:
        """把状态推进控件，并（可选地）返回窗口截图供录像用。"""
        from PyQt5.QtWidgets import QApplication
        self.canvas.apply(state)
        self.panel.apply(state)
        self.bottom.apply(state)
        QApplication.processEvents()
        return None                      # 截图走 snapshot()，避免每帧都抓

    def snapshot(self) -> Optional[np.ndarray]:
        """抓一张当前窗口画面（用于录像 / 生成界面预览图）。

        ★ Format_RGB32 在内存里的字节序其实是 B,G,R,A（小端），
          所以取前三个字节就已经是 BGR，不要再翻一次 —— 翻了会红蓝颠倒。
        """
        pm = self.win.grab()
        img = pm.toImage().convertToFormat(4)      # 4 = Format_RGB32
        w, h = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine() // 4, 4)
        return np.ascontiguousarray(arr[:, :w, :3])    # 已是 BGR

    def show(self, _canvas: Optional[np.ndarray] = None) -> bool:
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()
        return not self._quit

    # ── 主循环（Qt 自己拥有事件循环）────────────────────────────
    def run(self, state_provider, fps: int = 30, save_video: str = "",
            max_frames: int = 0, quit_flag=None) -> int:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication

        self._quit = False
        if self.fullscreen:
            self.win.showFullScreen()
        else:
            self.win.show()

        writer = None
        counter = {"n": 0}

        def tick():
            if quit_flag is not None and quit_flag.is_set():
                self._quit = True
            if self._quit:
                timer.stop()
                self.app.quit()
                return
            st = state_provider()
            self.render(st)
            counter["n"] += 1
            if save_video and counter["n"] % 2 == 0:      # 每 2 帧录一次，降体积
                nonlocal_writer[0] = self._write(nonlocal_writer[0], save_video)
            if max_frames and counter["n"] >= max_frames:
                self._quit = True

        nonlocal_writer = [writer]
        timer = QTimer()
        timer.timeout.connect(tick)
        timer.start(max(10, int(1000 / self.fps)))
        try:
            self.app.exec_()
        except KeyboardInterrupt:
            pass
        finally:
            timer.stop()
            if nonlocal_writer[0] is not None:
                nonlocal_writer[0].release()
                print("[ui] 界面录像已保存:", save_video)
        return 0

    def _write(self, writer, path: str):
        snap = self.snapshot()
        if snap is None:
            return writer
        if writer is None:
            from ..imgio import videowriter_safe
            h, w = snap.shape[:2]
            writer = videowriter_safe(path, "mp4v", float(self.fps) / 2.0, (w, h))
            print("[ui] 界面录像 ->", path)
        writer.write(snap)
        return writer

    # ── 按键 ──────────────────────────────────────────────────
    def _on_key(self, ev) -> None:
        from PyQt5.QtCore import Qt
        k = ev.key()
        if k in (Qt.Key_Q, Qt.Key_Escape):
            self._quit = True
            self.app.quit()
        elif k == Qt.Key_Space:
            cb = self.actions.get("inspect")
            if cb:
                cb()
                print("[ui] 手动触发一次判定")
        elif k == Qt.Key_R:
            cb = self.actions.get("reset")
            if cb:
                cb()
        elif k == Qt.Key_F:
            if self.win.isFullScreen():
                self.win.showNormal()
            else:
                self.win.showFullScreen()

    def close(self) -> None:
        try:
            self.win.close()
        except Exception:                     # noqa: BLE001
            pass
