# -*- coding: utf-8 -*-
"""界面层：只吃 UIState，不碰算法。

分层约定：核心算法不知道界面的存在；界面只从 UIState 里读数据。
换 UI 框架（OpenCV 窗口 → PyQt5）不需要改任何 core/ 代码。
"""
from __future__ import annotations

from .base import Renderer, UIState, find_cjk_font

__all__ = ["Renderer", "UIState", "find_cjk_font", "make_renderer"]


def make_renderer(kind: str = "opencv", **kw) -> Renderer:
    """工厂：按名字造渲染器。核心代码只依赖 Renderer 接口。"""
    kind = (kind or "opencv").lower()
    if kind == "opencv":
        from .opencv_renderer import OpenCVRenderer
        return OpenCVRenderer(**kw)
    if kind == "qt":
        from .qt_renderer import QtRenderer
        return QtRenderer(**kw)
    raise ValueError("未知的渲染器 %r（opencv | qt）" % kind)
