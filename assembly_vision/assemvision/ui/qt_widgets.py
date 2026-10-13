# -*- coding: utf-8 -*-
"""PyQt5 控件：主视频画布、步骤进度条、大圆形状态灯、原因面板、底部状态条。

设计要点
  * 每个控件只管"把 UIState 里的某几项画出来"，不做任何算法判断
  * 颜色统一从 UIState.color(key) 取（key = ok/warn/error/idle）
  * 视频区自己做 letterbox，并把 ROI 多边形与检测框按同一缩放映射过去
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import (QBrush, QColor, QFont, QImage, QPainter, QPen,
                         QPolygonF)
from PyQt5.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy,
                             QVBoxLayout, QWidget)

from ..types import ERROR, IDLE, OK, WARN
from .base import LEVEL_GLYPH, UIState

PANEL_BG = QColor(38, 38, 38)
CANVAS_BG = QColor(22, 22, 22)
TEXT = QColor(232, 232, 232)
TEXT_DIM = QColor(150, 150, 150)
LINE = QColor(72, 72, 72)
LAMP_BORDER = QColor(18, 18, 18)


def to_qcolor(bgr: Sequence[int]) -> QColor:
    """UIState 里的颜色是 BGR（OpenCV 习惯），Qt 要 RGB。"""
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return QColor(r, g, b)


def state_color(state: UIState, key: str) -> QColor:
    return to_qcolor(state.color(key))


def bgr_to_qimage(img: np.ndarray) -> Optional[QImage]:
    """numpy BGR -> QImage（RGB888）。必须保住底层 buffer 的引用，否则会花屏。"""
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    qimg._dsh_buf = rgb          # 关键：防止 numpy 缓冲被回收
    return qimg


# ══════════════════════════════════════════════════════════════
class VideoCanvas(QWidget):
    """主视频区：letterbox 显示 + 叠加 ROI 多边形与检测框。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img: Optional[QImage] = None
        self._polys: List[Tuple[Sequence, QColor, int]] = []
        self._boxes: List[Tuple[Sequence, str, QColor]] = []
        self._lamp: Optional[Tuple[QColor, str, str]] = None   # 左上角小状态灯
        self._banner: str = ""
        self._banner_col = QColor(198, 40, 40)
        self._label_font = QFont()
        self._label_font.setPointSize(11)

    def set_font_family(self, family: str, scale: float = 1.0) -> None:
        self._label_font = QFont(family)
        self._label_font.setPointSize(max(9, int(11 * scale)))
        self.update()

    def apply(self, state: UIState) -> None:
        self._img = bgr_to_qimage(state.frame) if state.frame is not None else None
        self._polys = [(p, state_color(state, k), t) for p, k, t in state.polygons]
        self._boxes = [(b, lab, state_color(state, k)) for b, lab, k in state.boxes]
        lvl = state.level
        self._lamp = (state_color(state, lvl), LEVEL_GLYPH.get(lvl, "-"), state.title)
        self._banner = state.banner or ""
        self._banner_col = state_color(state, ERROR)
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), CANVAS_BG)
        if self._img is None:
            p.setPen(TEXT_DIM)
            p.drawText(self.rect(), Qt.AlignCenter, "等待画面…")
            return

        W, H = self.width(), self.height()
        iw, ih = self._img.width(), self._img.height()
        s = min(W / iw, H / ih)
        nw, nh = iw * s, ih * s
        ox, oy = (W - nw) / 2.0, (H - nh) / 2.0
        p.drawImage(QRectF(ox, oy, nw, nh), self._img)

        p.setRenderHint(QPainter.Antialiasing, True)
        for poly, col, th in self._polys:
            if poly is None or len(poly) < 3:
                continue
            pts = [QPointF(ox + float(x) * s, oy + float(y) * s) for x, y in poly]
            p.setPen(QPen(col, max(1, th)))
            p.setBrush(Qt.NoBrush)
            p.drawPolygon(QPolygonF(pts))

        for bbox, label, col in self._boxes:
            x, y, w, h = bbox
            r = QRectF(ox + x * s, oy + y * s, w * s, h * s)
            p.setPen(QPen(col, 2))
            p.drawRect(r)
            if label:
                p.setFont(self._label_font)
                p.drawText(QPointF(r.left(), max(12.0, r.top() - 5)), label)

        if self._banner:
            bar = QRectF(0, 0, W, max(42.0, H * 0.072))
            p.fillRect(bar, self._banner_col)
            f = QFont(self._label_font)
            f.setPointSize(int(self._label_font.pointSize() * 1.7))
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor(255, 255, 255))
            p.drawText(bar.adjusted(18, 0, -18, 0), Qt.AlignVCenter | Qt.AlignLeft,
                       self._banner)


# ══════════════════════════════════════════════════════════════
class StepProgress(QWidget):
    """步骤进度：N 个圆点 + 连线。已完成绿、当前黄、未到灰。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(96)
        self._step = 0
        self._total = 0
        self._font = QFont()
        self._font.setPointSize(10)

    def set_font_family(self, family: str, scale: float = 1.0) -> None:
        self._font = QFont(family)
        self._font.setPointSize(max(8, int(10 * scale)))

    def apply(self, step: int, total: int) -> None:
        if (step, total) != (self._step, self._total):
            self._step, self._total = int(step), int(total)
            self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        # ★ 裸 QWidget 子类不会自动画背景（样式表里的 background 需要
        #   WA_StyledBackground 才生效），不显式填一次就会露出系统浅灰底色，
        #   跟面板的深色对不上。
        p.fillRect(self.rect(), PANEL_BG)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(TEXT_DIM)
        p.setFont(self._font)
        p.drawText(QRectF(0, 4, self.width(), 22), Qt.AlignCenter,
                   "装配进度   %d / %d" % (self._step, self._total))

        n = max(1, self._total)
        pad = 22
        cy = self.height() * 0.68
        span = max(1, self.width() - 2 * pad)
        r = 11
        xs = [pad + span * i / max(1, n - 1) for i in range(n)] if n > 1 else [self.width() / 2.0]

        for i in range(n):
            if i > 0:
                p.setPen(QPen(LINE, 3))
                p.drawLine(QPointF(xs[i - 1], cy), QPointF(xs[i], cy))
        for i in range(n):
            if i < self._step:
                col = to_qcolor(UIState().color(OK))
            elif i == self._step:
                col = to_qcolor(UIState().color(WARN))
            else:
                col = QColor(84, 84, 84)
            p.setPen(QPen(QColor(24, 24, 24), 2))
            p.setBrush(QBrush(col))
            p.drawEllipse(QPointF(xs[i], cy), r, r)
            p.setPen(QColor(250, 250, 250))
            f = QFont(self._font)
            f.setBold(True)
            p.setFont(f)
            p.drawText(QRectF(xs[i] - r, cy - r, 2 * r, 2 * r), Qt.AlignCenter, str(i + 1))


# ══════════════════════════════════════════════════════════════
class StatusLamp(QWidget):
    """大圆形状态灯 —— 整个界面最醒目的一块。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setMaximumHeight(300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._col = to_qcolor(UIState().color(IDLE))
        self._glyph = LEVEL_GLYPH[IDLE]

    def apply(self, level: str, title: str) -> None:
        col = to_qcolor(UIState().color(level))
        glyph = LEVEL_GLYPH.get(level, "-")
        if (col.name(), glyph) != (self._col.name(), self._glyph):
            self._col, self._glyph = col, glyph
            self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), PANEL_BG)
        p.setRenderHint(QPainter.Antialiasing, True)
        side = max(60.0, min(self.width(), self.height()) - 14.0)
        r = QRectF((self.width() - side) / 2.0, (self.height() - side) / 2.0, side, side)
        p.setBrush(QBrush(self._col))
        p.setPen(QPen(LAMP_BORDER, 4))
        p.drawEllipse(r)
        f = QFont(self.font())
        f.setPointSizeF(max(20.0, side * 0.52))
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255))
        p.drawText(r, Qt.AlignCenter, self._glyph)


# ══════════════════════════════════════════════════════════════
class ReasonPanel(QWidget):
    """右侧面板：进度 + 状态灯 + 结论 + 原因明细。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(320)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)

        self.progress = StepProgress(self)
        self.title = QLabel("就绪", self)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setWordWrap(True)
        # ★ 给标题留够两行的高度：面板只有 20% 屏宽，长一点的告警会折行，
        #   不给高度的话会被下面的控件挤掉，看起来像被截断。
        self.title.setMinimumHeight(64)
        self.title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.lamp = StatusLamp(self)
        self.sep = QFrame(self)
        self.sep.setFixedHeight(1)
        self.sep.setStyleSheet("background: #484848; border: none;")
        self.detail_head = QLabel("诊断信息", self)
        self.details = QLabel("", self)
        self.details.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.details.setWordWrap(True)
        self.hint = QLabel("空格=重判   R=复位   F=全屏   Q=退出", self)
        self.hint.setAlignment(Qt.AlignCenter)

        lay.addWidget(self.progress)
        lay.addWidget(self.lamp, 3)
        lay.addWidget(self.title, 0)
        lay.addWidget(self.sep)
        lay.addWidget(self.detail_head)
        lay.addWidget(self.details, 2)
        lay.addWidget(self.hint, 0)

        self.setStyleSheet("QWidget { background: #262626; color: #e8e8e8; }")

    def set_fonts(self, family: str, scale: float = 1.0) -> None:
        self.progress.set_font_family(family, scale)
        big = QFont(family)
        big.setPointSize(max(12, int(17 * scale)))
        big.setBold(True)
        self.title.setFont(big)
        self.title.setStyleSheet("color: #ffffff;")
        mid = QFont(family); mid.setPointSize(max(10, int(13 * scale))); mid.setBold(True)
        self.detail_head.setFont(mid)
        self.detail_head.setStyleSheet("color: #9aa0a6;")
        body = QFont(family); body.setPointSize(max(10, int(13 * scale)))
        self.details.setFont(body)
        self.details.setStyleSheet("color: #d6d6d6;")
        small = QFont(family); small.setPointSize(max(8, int(10 * scale)))
        self.hint.setFont(small)
        self.hint.setStyleSheet("color: #7a7a7a;")

    def apply(self, state: UIState) -> None:
        self.progress.apply(state.step, state.total)
        self.lamp.apply(state.level, state.title)
        if self.title.text() != state.title:
            self.title.setText(state.title or "就绪")
        txt = "\n".join("• " + d for d in state.detail) if state.detail else "（暂无）"
        if self.details.text() != txt:
            self.details.setText(txt)
        col = state_color(state, state.level).name()
        self.title.setStyleSheet("color: %s;" % col)


# ══════════════════════════════════════════════════════════════
class BottomBar(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(38)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 14, 0)
        self.left = QLabel("", self)
        self.right = QLabel("", self)
        self.right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.left, 1)
        lay.addWidget(self.right, 1)

    def set_fonts(self, family: str, scale: float = 1.0) -> None:
        f = QFont(family); f.setPointSize(max(8, int(11 * scale)))
        self.left.setFont(f); self.right.setFont(f)
        self.left.setStyleSheet("color: #9aa0a6;")
        self.right.setStyleSheet("color: #7a7a7a;")

    def apply(self, state: UIState) -> None:
        self.left.setText("   |   ".join("%s：%s" % (k, v) for k, v in state.stats.items()))
        self.right.setText(state.source_name or "")
