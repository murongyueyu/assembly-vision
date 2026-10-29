# -*- coding: utf-8 -*-
"""单档相机测试（供 probe_camera.py 以子进程方式调用）。

★ 为什么必须独立进程：
   UVC 相机在同一个进程里反复 VideoCapture 开关、来回切分辨率，
   容易陷入"降级状态" —— 实测踩过：连续开关十几次后，所有模式的帧率
   都从 48 fps 掉到 4.5 fps，拔插 USB 才恢复。
   所以每测一档就起一个新进程，进程退出时设备被干净释放。

也可单独使用：
    python tools/probe_one.py --mode 1280x720 --seconds 1.0 --out result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from assemvision.imgio import imwrite_safe                        # noqa: E402
from assemvision.sources.uvc_stereo import (                       # noqa: E402
    default_backend_name, resolve_backend)


def measure(cap: cv2.VideoCapture, seconds: float, warmup: int = 10):
    """抓一帧并测帧率。返回 (样本帧, 实测fps)。"""
    frame = None
    for _ in range(max(1, warmup)):          # 丢掉开头几帧，等自动曝光稳定
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    if frame is None:
        return None, 0.0
    t0 = time.time()
    n = 0
    while time.time() - t0 < seconds:
        ok, _ = cap.read()
        if ok:
            n += 1
    dt = time.time() - t0
    return frame, (n / dt if dt > 0 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser("probe_one")
    ap.add_argument("--device", default="0")
    ap.add_argument("--backend", default="")
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--mode", required=True, help="如 1280x720")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--keep", default="")
    ap.add_argument("--raw", action="store_true", help="以原始 JSON 打到 stdout")
    args = ap.parse_args()

    try:
        w, h = (int(v) for v in args.mode.lower().replace("*", "x").split("x"))
    except Exception:                                          # noqa: BLE001
        print(json.dumps({"ok": False, "note": "mode 格式错误"}))
        return 1

    backend_name = args.backend or default_backend_name()
    dev = int(args.device) if str(args.device).isdigit() else args.device

    res = {"req": [w, h], "ok": False, "got": None, "fps": 0.0, "note": ""}

    cap = cv2.VideoCapture(dev, resolve_backend(backend_name))
    if not cap.isOpened():
        res["note"] = "打不开相机"
    else:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, args.fps)
            time.sleep(0.4)
            frame, fps = measure(cap, seconds=args.seconds)
            if frame is None:
                res["note"] = "无帧"
            else:
                fh, fw = frame.shape[:2]
                res.update(ok=True, got=[fw, fh], fps=fps)
                if (fw, fh) != (w, h):
                    res["note"] = "回落到 %dx%d" % (fw, fh)
                if args.keep:
                    imwrite_safe(args.keep, frame)
        finally:
            cap.release()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f)
    if args.raw or args.out:
        print(json.dumps(res))
    else:
        print("  %-11s -> %-11s %6.1f fps  %s"
              % ("%dx%d" % (w, h),
                 ("%dx%d" % tuple(res["got"])) if res["got"] else "失败",
                 res["fps"], res["note"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
