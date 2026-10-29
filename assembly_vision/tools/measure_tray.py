# -*- coding: utf-8 -*-
"""量料盘槽位 —— 拍一张空料盘图，叠加 tray-mm 网格，用于确定槽位几何。

★ 为什么需要专用工具：
  料盘槽的形状/中心/尺寸必须**在 c3 标定的同一相机状态下**测量，
  否则单应不适用。而肉眼从普通照片估算很不准（实测踩过：同一料盘在两次
  拍摄里量出 0.613 和 0.877 mm/px，差 43%）。

做法：
  ① 用 c3 标定好的单应，把 tray-mm 网格（每 10mm 一条）画到实时画面上
  ② 保存一张带网格的原图 —— 你就能直接读出"某个槽的中心在 tray 多少 mm、
     半径多少 mm"
  ③ 同时输出每个网格交点的像素坐标，便于精确定位

用法
    python tools/measure_tray.py                    # 抓一帧存图
    python tools/measure_tray.py --out data/refs/tray_empty.png
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

from assemvision.config import load_config                     # noqa: E402
from assemvision.core.roi import TrayMap                        # noqa: E402
from assemvision.imgio import imwrite_safe                      # noqa: E402
from assemvision.sources.factory import make_source             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser("measure_tray")
    ap.add_argument("--out", default="data/refs/tray_empty.png")
    ap.add_argument("--grid-mm", type=float, default=10.0, help="网格间距 mm")
    ap.add_argument("--side-mm", type=float, default=160.0, help="料盘边长 mm")
    ap.add_argument("--exposure", type=float, default=-5.0)
    args = ap.parse_args()

    cam = load_config(os.path.join(ROOT, "config", "camera.yaml"))
    cal = load_config(os.path.join(ROOT, "config", "calibration.yaml"), required=False)
    if "tray_map" not in cal.data:
        print("★ 还没有料盘单应，请先跑 calib/c3_tray_homography.py")
        return 1
    H = np.asarray(cal.get("tray_map.homography"), dtype=np.float64)
    tray = TrayMap(load_config(os.path.join(ROOT, "config", "workpiece.yaml")),
                   homography=H, ref_scale=float(cal.get("tray_map.scale", 1.0)))
    mmpp = tray.mm_per_px(1.0)
    print("单应 GSD = %.4f mm/px（应与 c1 的实测值接近）" % mmpp)

    src = make_source(cam, "uvc", None)
    src.open()
    if hasattr(src, "apply_locked_controls"):
        src.apply_locked_controls()          # type: ignore[attr-defined]
    img = None
    for _ in range(30):
        f = src.read(want_full=True)
        if f is not None and f.left_full is not None:
            img = f.left_full
        time.sleep(0.03)
    src.close()
    if img is None:
        print("抓帧失败"); return 1

    vis = img.copy()
    S = float(args.side_mm)
    step = float(args.grid_mm)
    # 网格交点：tray-mm -> 像素
    gx = np.arange(0.0, S + 1e-6, step)
    gy = np.arange(0.0, S + 1e-6, step)
    mm = np.array([[x, y] for y in gy for x in gx], dtype=np.float64)
    px = tray.mm_to_px(mm, 1.0)
    # 网格线（横线按 y 分组，竖线按 x 分组）
    for yv in gy:
        line = tray.mm_to_px(np.array([[0.0, yv], [S, yv]]), 1.0)
        cv2.line(vis, tuple(np.round(line[0]).astype(int)),
                 tuple(np.round(line[1]).astype(int)), (0, 0, 255), 1)
    for xv in gx:
        line = tray.mm_to_px(np.array([[xv, 0.0], [xv, S]]), 1.0)
        cv2.line(vis, tuple(np.round(line[0]).astype(int)),
                 tuple(np.round(line[1]).astype(int)), (0, 0, 255), 1)
    # 每 20mm 标一次数字，避免太密
    for (x, y), p in zip(mm, px):
        if x % 20 == 0 and y % 20 == 0:
            cv2.putText(vis, "%d,%d" % (int(x), int(y)),
                        (int(p[0]) + 3, int(p[1]) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1, cv2.LINE_AA)

    # 把每个槽位现有 ROI 也画出来（供对比）
    for s in tray.slots:
        poly = np.round(tray.slot_polygon_px(s["id"], 1.0)).astype(np.int32)
        cv2.polylines(vis, [poly.reshape(-1, 1, 2)], True, (0, 255, 0), 1)

    p = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    ok1 = imwrite_safe(p, vis)
    ok2 = imwrite_safe(p.replace(".png", "_plain.png"), img)
    print("已存带网格图: %s  %s" % (p, "✓" if ok1 else "✗"))
    print("已存无网格图: %s  %s" % (p.replace(".png", "_plain.png"), "✓" if ok2 else "✗"))
    print()
    print("怎么看：图上的红色网格是 tray-mm（左上为原点 0,0，X向右、Y向下，每 %.0fmm 一格）"
          % step)
    print("  ① 读出每个槽的【中心】在 tray 的多少 mm")
    print("  ② 读出每个槽的【直径或长宽】多少 mm")
    print("  ③ 绿色框是当前代码假设的 ROI，对比一下就知道差在哪")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
