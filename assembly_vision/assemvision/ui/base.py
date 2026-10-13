# -*- coding: utf-8 -*-
"""UIState 与 Renderer 接口。"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..types import ERROR, IDLE, OK, WARN

# 颜色（BGR）——默认是黑底方案；实际值从 workpiece.yaml 的 ui.colors 读
DEFAULT_COLORS = {
    OK:    (50, 125, 46),
    WARN:  (37, 168, 249),
    ERROR: (40, 40, 198),
    IDLE:  (97, 97, 97),
}

# 状态灯里画的大符号。
# ★ 不要用 ✓ ✗ ○ —— 微软雅黑/思源黑体里可能没有这些码位，会渲染成方块。
#   下面这几个（√ × ! –）在中文字体里都有，跨平台最稳。
LEVEL_GLYPH = {
    OK:    "√",
    WARN:  "!",
    ERROR: "×",
    IDLE:  "–",
}

_CJK_CANDIDATES = [
    # Linux（树莓派 RaspiOS / Ubuntu）
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    # Windows
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
]


def find_cjk_font(explicit: str = "") -> Optional[str]:
    """找一个能画中文的字体文件。找不到就返回 None（界面会退化成英文标签）。"""
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for p in _CJK_CANDIDATES:
        if os.path.exists(p):
            return p
    # 兜底：在常见目录里扫一遍
    for root in ("/usr/share/fonts", r"C:\Windows\Fonts"):
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if fn.lower().endswith((".ttc", ".ttf", ".otf")) and \
                        any(k in fn.lower() for k in ("cjk", "wqy", "hei", "song", "yahei", "pingfang")):
                    return os.path.join(dirpath, fn)
    return None


@dataclass
class UIState:
    """渲染器唯一需要的输入。"""
    frame: Optional[np.ndarray] = None           # 主画面（BGR）
    boxes: List[Tuple[Tuple[int, int, int, int], str, str]] = field(default_factory=list)
    #        (bbox, 标签, 颜色键 ok/warn/error/idle)
    polygons: List[Tuple[np.ndarray, str, int]] = field(default_factory=list)
    #        (多边形像素, 颜色键, 线宽)

    level: str = IDLE
    title: str = ""
    detail: List[str] = field(default_factory=list)
    step: int = 0
    total: int = 0

    stats: Dict[str, Any] = field(default_factory=dict)
    banner: str = ""                              # 顶部横幅（严重告警时用）
    source_name: str = ""

    def color(self, key: str) -> Tuple[int, int, int]:
        return DEFAULT_COLORS.get(key, DEFAULT_COLORS[IDLE])


class Renderer(ABC):
    """渲染器接口。核心代码只认这个。

    run()     主循环（谁拥有事件循环谁实现它；Qt 会用 QTimer 覆盖）
    render()  拉一次状态并画出一帧（OpenCV 版返回画布；Qt 版返回窗口截图）
    show()    显示（OpenCV 版 imshow + waitKey；Qt 版由事件循环负责）
    """

    def __init__(self) -> None:
        self.frames_drawn = 0

    @abstractmethod
    def render(self, state: UIState) -> Optional[np.ndarray]:
        """把 UIState 画成一帧图像并返回（也可能直接在内部显示）。"""

    @abstractmethod
    def show(self, canvas: Optional[np.ndarray]) -> bool:
        """显示。返回 False 表示用户要求退出。"""

    def run(self, state_provider, fps: int = 30, save_video: str = "",
            max_frames: int = 0, quit_flag=None) -> int:
        """通用窗口循环：拉状态 → render → show。

        state_provider 是一个无参可调用对象，返回 UIState。
        quit_flag 可选：一个带 .is_set() 的对象（如 threading.Event），置位后退出。
        Qt 渲染器会覆盖本方法，用自己的事件循环。
        """
        import time as _time

        writer = None
        period = 1.0 / max(1.0, float(fps))
        try:
            while True:
                if quit_flag is not None and quit_flag.is_set():
                    break
                t0 = _time.time()
                st = state_provider()
                canvas = self.render(st)
                if save_video and canvas is not None:
                    if writer is None:
                        from ..imgio import videowriter_safe
                        h, w = canvas.shape[:2]
                        writer = videowriter_safe(save_video, "mp4v", float(fps), (w, h))
                        print("[ui] 界面录像 ->", save_video)
                    writer.write(canvas)
                if not self.show(canvas):
                    break
                self.frames_drawn += 1
                if max_frames and self.frames_drawn >= max_frames:
                    break
                dt = _time.time() - t0
                if dt < period:
                    _time.sleep(period - dt)
        except KeyboardInterrupt:
            print("\n[ui] 用户中断")
        finally:
            if writer is not None:
                writer.release()
                print("[ui] 界面录像已保存:", save_video)
        return 0

    def close(self) -> None:
        pass

    # 从配置加载颜色（配置里写的是 RGB，OpenCV 要 BGR，这里自动倒过来）
    def set_colors(self, colors: Dict[str, Sequence[int]]) -> None:
        for k, v in (colors or {}).items():
            if k in DEFAULT_COLORS and len(v) >= 3:
                DEFAULT_COLORS[k] = (int(v[2]), int(v[1]), int(v[0]))
