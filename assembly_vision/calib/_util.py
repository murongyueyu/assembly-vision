# -*- coding: utf-8 -*-
"""标定脚本公用工具。"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from assemvision.config import Config, load_config, merge   # noqa: E402
from assemvision.imgio import imread_safe, imwrite_safe      # noqa: E402


def paths():
    return dict(
        root=ROOT,
        camera=os.path.join(ROOT, "config", "camera.yaml"),
        workpiece=os.path.join(ROOT, "config", "workpiece.yaml"),
        calib=os.path.join(ROOT, "config", "calibration.yaml"),
        refs=os.path.join(ROOT, "data", "refs"),
    )


def load_all(args_camera: str = "", args_workpiece: str = "") -> tuple:
    p = paths()
    cam = load_config(args_camera or p["camera"])
    wp = load_config(args_workpiece or p["workpiece"])
    calib = load_config(p["calib"], required=False)
    if not calib.data:
        calib = Config({"meta": {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}}, p["calib"])
    return cam, wp, calib


def save_calib(calib: Config, out: str = "") -> str:
    p = paths()
    return calib.save(out or p["calib"])


def open_source(cam: Config, args):
    """按命令行参数打开图像来源（相机或离线文件）。"""
    from assemvision.sources.factory import make_source
    src = make_source(cam, getattr(args, "source", None), getattr(args, "path", None))
    if hasattr(src, "apply_locked_controls"):
        src.apply_locked_controls()      # type: ignore[attr-defined]
    src.open()
    return src


def grab_full(src, timeout_s: float = 5.0):
    """抓一帧（带全分辨率左眼）。返回 (left, right)。"""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        f = src.read(want_full=True)
        if f is not None and f.left_full is not None:
            return f.left_full, (f.right_full if f.right_full is not None else f.right)
        time.sleep(0.02)
    return None, None


def grab_small(src, timeout_s: float = 5.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        f = src.read(want_full=False)
        if f is not None and f.left is not None:
            return f.left, f
        time.sleep(0.02)
    return None, None


def ask(prompt: str, default: str = "") -> str:
    try:
        s = input(prompt).strip()
    except EOFError:
        s = ""
    return s or default


def show_and_pick(image: np.ndarray, n: int = 2, window: str = "pick") -> list:
    """在图上点 n 个点，返回像素坐标列表。左右键/回车确认，ESC 取消。"""
    pts = []
    disp = image.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x, y))
            cv2.circle(disp, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(disp, str(len(pts)), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, disp)
        k = cv2.waitKey(20) & 0xFF
        if len(pts) >= n and k in (13, 32):     # 回车 / 空格
            break
        if k == 27:                             # ESC
            pts = []
            break
        if k == ord("r"):
            pts = []
            disp = image.copy()
    cv2.destroyWindow(window)
    return pts
