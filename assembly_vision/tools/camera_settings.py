# -*- coding: utf-8 -*-
"""打开相机自带的驱动设置界面（用来手工调白平衡/曝光）。

★ 为什么需要这个工具：
   OpenCV 只能通过 CAP_PROP_* 控制相机，而这台相机的驱动**拒绝**了这些属性：
       实测 CAP_PROP_WB_TEMPERATURE  set() 返回 False（完全不生效）
       实测 CAP_PROP_AUTO_WB        set() 返回 True 但画面毫无变化
       实测 CAP_PROP_GAIN           set() 返回 False
   → 白平衡只能到**驱动自己的属性页**里去调，也就是 DirectShow 的
     "属性(Properties)" / "摄像头控制" 对话框。本工具就是把那个对话框弹出来。

★ 实测症状（调之前）：
   画面严重偏青，红通道只有蓝的 1/3 —— 连白纸都输出 R≈0.3×B
   后果：黄铜掩膜占比 0.00%，四类零件一个都检测不到

用法
    python tools/camera_settings.py            # 弹出属性页
    python tools/camera_settings.py --index 1  # 别的相机索引

弹出的对话框里找这些项（不同驱动叫法不同）：
    White Balance / 白平衡            -> 关掉自动，把色温调到画面不再偏青
    Auto White Balance / 自动白平衡    -> 取消勾选
    Exposure / 曝光                  -> 关掉自动（Auto），手工给一个值
    Brightness / 亮度、Gamma          -> 一般不用动
    Backlight Comp. / 背光补偿         -> 关掉（会破坏饱和度判据）
    PowerLine Frequency / 防闪烁       -> 设为 50Hz（中国），否则会有亮度条纹
调好后点确定，然后跑 tools/live_preview.py 复查颜色。
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

from assemvision.imgio import imwrite_safe                        # noqa: E402
from assemvision.sources.uvc_stereo import (                       # noqa: E402
    default_backend_name, resolve_backend)


def rgb_report(frame: np.ndarray) -> str:
    """报告三通道均衡性 —— 这是判断白平衡好不好的硬指标。"""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    b, gg, r = [float(frame[:, :, i].mean()) for i in range(3)]
    # 最亮 10% 当"白参考"：白纸应该是 R≈G≈B
    ys, xs = np.where(g >= np.percentile(g, 90))
    br = float(frame[:, :, 2][ys, xs].mean())
    bb = float(frame[:, :, 0][ys, xs].mean())
    bg = float(frame[:, :, 1][ys, xs].mean())
    ratio = br / max(1e-6, bb)
    ok = 0.85 <= ratio <= 1.18 and 0.85 <= (br / max(1e-6, bg)) <= 1.18
    return ("B=%.0f G=%.0f R=%.0f  R/B=%.2f  最亮区 R/B=%.2f  %s"
            % (b, gg, r, r / max(1e-6, b), ratio,
               "✓ 白平衡正常" if ok else "★ 仍然偏色（白参考 R/B 应接近 1.0）"))


def main() -> int:
    ap = argparse.ArgumentParser("camera_settings")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--backend", default="", help="dshow | msmf | any")
    ap.add_argument("--save", default="data/settings_check.png",
                    help="把当前画面存下来，方便对比调整前后")
    args = ap.parse_args()

    backend_name = args.backend or default_backend_name()
    cap = cv2.VideoCapture(args.index, resolve_backend(backend_name))
    if not cap.isOpened():
        print("打不开相机 index=%d（后端 %s）" % (args.index, backend_name))
        return 1
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    time.sleep(0.5)

    print("=" * 74)
    print(" 调整前颜色状态：")
    f = None
    for _ in range(15):
        ok, fr = cap.read()
        if ok and fr is not None:
            f = fr
    if f is not None:
        print("   " + rgb_report(f))
    print("=" * 74)

    print(" 正在弹出相机属性对话框…（在任务栏或窗口列表里找它，可能被主窗口挡住）")
    print(" 要改的项：白平衡(关自动)、曝光(关自动)、背光补偿(关)、防闪烁(50Hz)")
    ok = cap.set(cv2.CAP_PROP_SETTINGS, 0)
    print("   CAP_PROP_SETTINGS set -> %s" % ok)
    if not ok:
        print("   ★ 驱动不支持直接弹属性页。替代办法：")
        print("     1) 打开 Windows「相机」应用 → 右上角齿轮 → 找到白平衡/亮度")
        print("     2) 或到设备管理器 → 找到该相机 → 属性 → 可能有厂商页")
        print("     3) 或用厂商自带的 UVC 调试工具")
    input("  调好后回到这里按回车继续…")

    print()
    print("=" * 74)
    print(" 调整后颜色状态：")
    f2 = None
    for _ in range(20):
        ok, fr = cap.read()
        if ok and fr is not None:
            f2 = fr
    if f2 is not None:
        print("   " + rgb_report(f2))
        p = args.save if os.path.isabs(args.save) else os.path.join(ROOT, args.save)
        if imwrite_safe(p, f2):
            print("   已存图: %s" % p)
    else:
        print("   读帧失败")
    print("=" * 74)

    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
