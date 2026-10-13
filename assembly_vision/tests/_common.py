# -*- coding: utf-8 -*-
"""测试公用工具：把配置、TrayMap、分割器、判定器一次装好。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from assemvision.config import load_config           # noqa: E402
from assemvision.core.checks import Qualifier        # noqa: E402
from assemvision.core.roi import TrayMap             # noqa: E402
from assemvision.core.segment import Segmenter       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))     # 让测试能 import make_synthetic


def wp_config():
    return load_config(os.path.join(ROOT, "config", "workpiece.yaml"))


def make_env(px_per_mm: float = 2.5, margin_mm: float = 12.0):
    """构造与 make_synthetic.render_scene 完全一致的坐标映射。"""
    wp = wp_config()
    tray = TrayMap(wp, homography=None, px_per_mm=px_per_mm,
                   origin_px=(margin_mm * px_per_mm, margin_mm * px_per_mm))
    seg = Segmenter(wp)
    q = Qualifier(wp, tray, None)
    return wp, tray, seg, q


def ok(msg: str) -> None:
    print("  [ok] %s" % msg)


def fail(msg: str) -> None:
    raise AssertionError(msg)


def check(cond: bool, msg: str) -> None:
    if cond:
        ok(msg)
    else:
        fail(msg)
