# -*- coding: utf-8 -*-
"""列出所有相机索引，告诉你哪个才是双目相机。

★ 为什么需要：Windows 上可能有多个摄像头设备（笔记本自带、外接网络摄像头、双目相机…），
   而 OpenCV 只用**索引**寻址。索引指错设备是"打不开 / 画面不对 / 分辨率怪异"的常见原因。
   实测本机就同时存在 "HD Webcam"(Error) 与 "3D_SHOP" 两个 Camera 类设备。

判据（三个一起看）：
   1. 分辨率是否是 1280x720 这类"双目拼接"尺寸（单眼 640x720）
   2. 左右半幅是否有视差（真双目）——用 NCC，对左右眼亮度差不敏感
   3. 画面里有没有你的工位（绿卡纸 / 黑料盘）

用法
    python tools/list_cameras.py                 # 扫索引 0~4
    python tools/list_cameras.py --max 6         # 扫更多
    python tools/list_cameras.py --save          # 把每个能打开的画面存到 data/probe
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from assemvision.imgio import imwrite_safe                       # noqa: E402
from assemvision.sources.uvc_stereo import (                      # noqa: E402
    default_backend_name, resolve_backend)


def stereo_score(frame: np.ndarray):
    """左右半幅的 NCC 与最佳位移。返回 (最佳位移, 峰值NCC, 未对齐NCC)。"""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape[:2]
    half = w // 2
    a, b = g[:, :half], g[:, half:half * 2]
    if a.shape != b.shape or a.size == 0:
        return 0, 0.0, 0.0

    def ncc(d):
        x, y = a[:, d:], b[:, :half - d]
        if x.size == 0:
            return -1.0
        x = x - x.mean()
        y = y - y.mean()
        den = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
        return float((x * y).sum() / den) if den > 1e-9 else -1.0

    best = max(((ncc(d), d) for d in range(0, min(400, half - 5))), default=(-1.0, 0))
    return best[1], best[0], ncc(0)


def main() -> int:
    ap = argparse.ArgumentParser("list_cameras")
    ap.add_argument("--max", type=int, default=5, help="最多扫到第几个索引")
    ap.add_argument("--backend", default="", help="dshow | msmf | any，默认按平台")
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--save", action="store_true", help="把能打开的画面存到 data/probe")
    args = ap.parse_args()

    backend_name = args.backend or default_backend_name()
    backend = resolve_backend(backend_name)
    print("=" * 76)
    print(" 相机索引扫描    后端=%s  FOURCC=%s  请求 %dx%d"
          % (backend_name, args.fourcc, args.width, args.height))
    print("=" * 76)

    found = []
    for idx in range(args.max):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            print("  索引 %d : 打不开" % idx)
            continue
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            time.sleep(0.5)
            frame = None
            for _ in range(15):
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
            if frame is None:
                print("  索引 %d : 打开了但读不到帧" % idx)
                continue
            fh, fw = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_v = float(gray.mean())
            # 只有"宽明显大于高"且是偶数宽才可能是左右拼接
            d, ncc_peak, ncc0 = stereo_score(frame)
            is_stereo = (fw >= 2 * fh or fw == 1280) and d > 5 and ncc_peak > 0.3
            tag = "★ 疑似双目相机" if is_stereo else ""
            exp = "过曝" if mean_v > 200 else ("偏暗" if mean_v < 60 else "曝光正常")
            print("  索引 %d : %dx%d  mean=%5.1f (%s)  视差%+4dpx NCC=%.2f  %s"
                  % (idx, fw, fh, mean_v, exp, d, ncc_peak, tag))
            found.append((idx, fw, fh, is_stereo))
            if args.save:
                p = os.path.join(ROOT, "data", "probe", "cam%d_%dx%d.png" % (idx, fw, fh))
                if imwrite_safe(p, frame):
                    print("            已存样张: %s" % p)
        finally:
            cap.release()
        time.sleep(0.4)          # ★ 别连续猛开，UVC 相机反复开关会降级

    print("=" * 76)
    stereo = [f for f in found if f[3]]
    if stereo:
        idx = stereo[0][0]
        print(" ★ 结论：双目相机在**索引 %d**（%dx%d）" % (idx, stereo[0][1], stereo[0][2]))
        print("   请把 config/camera.yaml 的 device 改成 %d：\n      device: %d" % (idx, idx))
    elif found:
        print(" ⚠ 没有索引表现出「真双目」特征。可能：")
        print("   · 相机被别的程序占用（浏览器/会议软件/相机 App）")
        print("   · 设备处于 Error 状态（到设备管理器看有没有感叹号）")
        print("   · 需要拔插 USB 复位")
    else:
        print(" ✗ 一个相机都打不开。先做：拔插 USB → 换到主板后置口（不用 HUB）→ 关掉占用程序")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
