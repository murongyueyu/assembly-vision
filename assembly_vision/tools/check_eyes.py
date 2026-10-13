# -*- coding: utf-8 -*-
"""检查两个相机传感器是否真的独立 —— 遮住一个镜头，看哪半边变暗。

★ 为什么需要这个工具：
   双目相机如果只出"一个传感器的画面复制成两份"，或者两个镜头接反，
   整个方案就废了（没有视差 = 没有深度 = 立柱判据失效）。
   而肉眼看画面容易误判，用**实时的左右半幅亮度**判断最可靠。

用法
    python tools/check_eyes.py

然后按提示操作：
    1. 先不动，记下两半的基准亮度
    2. 用手指/纸片**完全遮住左边镜头** -> 看 LEFT 数值是否暴跌
    3. 松开，再**完全遮住右边镜头** -> 看 RIGHT 数值是否暴跌

判断
    · 遮左镜头 -> 只有 LEFT 跌  => 两个传感器独立，正常 ✓
    · 遮左镜头 -> 两半都跌      => 两半是同一个传感器，方案不可用 ✗
    · 遮左镜头 -> 只有 RIGHT 跌 => 左右接反了，需要把 capture.left_index 改成 1
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

from assemvision.sources.uvc_stereo import (                       # noqa: E402
    default_backend_name, resolve_backend)


def main() -> int:
    ap = argparse.ArgumentParser("check_eyes")
    ap.add_argument("--device", default="0")
    ap.add_argument("--backend", default="")
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--mode", default="1280x720")
    ap.add_argument("--exposure", type=float, default=-5.0)
    args = ap.parse_args()

    try:
        w, h = (int(v) for v in args.mode.lower().replace("x", "x").split("x"))
    except Exception:                                          # noqa: BLE001
        print("--mode 格式不对，应为 1280x720"); return 1

    backend_name = args.backend or default_backend_name()
    dev = int(args.device) if str(args.device).isdigit() else args.device
    cap = cv2.VideoCapture(dev, resolve_backend(backend_name))
    if not cap.isOpened():
        print("打不开相机"); return 1
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    # 锁手动曝光，否则自动曝光会"补回来"，让遮挡测试失效
    if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    if hasattr(cv2, "CAP_PROP_EXPOSURE"):
        cap.set(cv2.CAP_PROP_EXPOSURE, float(args.exposure))
    time.sleep(0.5)
    for _ in range(12):
        cap.read()

    win = "check eyes  (q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(win, min(1500, w), min(700, h + 70))
    except Exception:                                          # noqa: BLE001
        pass

    base_l = base_r = None
    n = 0
    t0 = time.time()
    fps = 0.0
    print("=" * 70)
    print(" 按提示遮挡镜头，观察 LEFT / RIGHT 的数值变化")
    print(" 按 q 退出")
    print("=" * 70)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue
        n += 1
        if time.time() - t0 >= 1.0:
            fps = n / (time.time() - t0)
            n, t0 = 0, time.time()

        H, W = frame.shape[:2]
        half = W // 2
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mL = float(g[:, :half].mean())
        mR = float(g[:, half:half * 2].mean())
        if base_l is None and mL > 5 and mR > 5:
            base_l, base_r = mL, mR

        vis = frame.copy()
        cv2.line(vis, (half, 0), (half, H), (0, 255, 255), 2)
        cv2.rectangle(vis, (0, 0), (W, 64), (0, 0, 0), -1)
        # LEFT 数值
        cv2.putText(vis, "LEFT half = %6.1f" % mL, (14, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if mL > 60 else (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(vis, "RIGHT half = %6.1f" % mR, (half + 14, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if mR > 60 else (0, 0, 255), 3, cv2.LINE_AA)
        if base_l:
            dl = mL / max(1e-6, base_l)
            dr = mR / max(1e-6, base_r)
            # 用"相对基准的比例"判断：< 0.5 视为被遮住
            mask_l = "DARK" if dl < 0.5 else "ok"
            mask_r = "DARK" if dr < 0.5 else "ok"
            cv2.putText(vis, "vs base: L %.2f (%s)   R %.2f (%s)" % (dl, mask_l, dr, mask_r),
                        (14, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, "%.1f fps" % fps, (W - 150, H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.imshow(win, vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print()
    print(" 怎么看结果：")
    print("   遮住左镜头 -> 只有 LEFT 变 DARK  => ✓ 两个传感器独立（正常）")
    print("   遮住左镜头 -> 两半都变 DARK      => ✗ 两半来自同一传感器，方案不可用")
    print("   遮住左镜头 -> 只有 RIGHT 变 DARK => 左右接反，把 camera.yaml 的")
    print("                                       capture.left_index 改成 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
