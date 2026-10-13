# -*- coding: utf-8 -*-
"""c1 实测 GSD 与真实焦距。

★★ 这是整个项目最该先做的一步（5 分钟）。
   厂家规格自相矛盾：EFL 3.7mm + 1/3" 传感器只能推出 78° 对角视场，
   规格书却写 FOV 100°。真实焦距差 1.5 倍，所有阈值都跟着差 1.5 倍。

做法
    1. 把相机架到最终高度（40-50 cm）
    2. 在料盘位置平放一把钢尺（或贴一条已知长度的纸条）
    3. 跑本脚本，在图上点出钢尺上两段已知距离的两个端点
    4. 输入这段的实际长度（mm）

用法
    python calib/c1_measure_fov.py                       # 用相机
    python calib/c1_measure_fov.py --path ruler.jpg      # 用一张图
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import (ask, grab_full, imread_safe, load_all, open_source,  # noqa: E402
                   save_calib, show_and_pick)

# 像元尺寸（µm）—— **仅用于"反推焦距"这个副产品**，不影响 GSD。
#
# ★ 重要：实测证明这个值无法用标称参数凑齐，别再纠结：
#     实测（用户确认料盘 160mm、工作距离 360mm）:
#         GSD = 0.8510 mm/px   ->  fov 覆盖 544.7 x 408.5 mm
#     若按 f=3.0mm + 5.0µm 反推:  算得 GSD 应为 0.595 mm/px（差 1.4 倍）
#     若按 f=3.0mm + 3.75µm 反推: 算得 GSD 应为 0.446 mm/px（差 1.9 倍）
#     若按 f=2.51mm + 3.75µm 反推: 算得 GSD 应为 0.534 mm/px（差 1.6 倍）
#   => 标称参数（焦距/像元）与实测对不上，但**这不影响 GSD**：
#      GSD 是从"点出的像素距离 ÷ 已知实际长度"直接算出来的，不依赖任何镜头假设。
#   => 所以：**以 GSD 为准，焦距反推值只作参考**，不要因为它偏差大就怀疑 GSD。
#   => 真要提精度，唯一的办法是拉近工作距离或换更长焦的镜头（见 README）。
PIXEL_UM = 5.0
# 厂家标称视场角/焦距（用于核对反推结果）。注意有两款镜头：
#   80° 无畸变 / 3.0 mm
#   130°        / 2.51 mm
NOMINAL_FOCAL_MM = 3.0
NOMINAL_FOV_DEG = 80.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--camera", default="")
    ap.add_argument("--workpiece", default="")
    ap.add_argument("--distance-mm", type=float, default=0.0,
                    help="相机到料盘的实测工作距离（mm），用于反推焦距")
    ap.add_argument("--known-mm", type=float, default=0.0,
                    help="点的两个点之间的实际长度（mm）；不给就交互输入")
    ap.add_argument("--px", default="", help="直接给两个像素点，如 '100,200;900,200'")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cam, wp, calib = load_all(args.camera, args.workpiece)

    if args.path:
        img = imread_safe(args.path)
        if img is None:
            print("读不到图:", args.path)
            return 1
        src = None
    else:
        src = open_source(cam, args)
        img, _ = grab_full(src)
        if img is None:
            print("抓不到帧")
            return 1

    # 取两个点
    if args.px:
        pts = [tuple(float(v) for v in seg.split(","))
               for seg in args.px.split(";") if seg.strip()]
    else:
        print("窗口里连续取流显示：左键点两个端点，r 重来，回车确认，ESC 取消")
        # ★ 传 src 进去 → 窗口里是**实时画面**，可以边看边调摆放/对焦
        pts = show_and_pick(img, 2, src=src)
    if len(pts) < 2:
        print("没有取到两个点")
        return 1

    px_dist = float(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
    known = args.known_mm or 0.0
    if known <= 0:
        known = float(ask("这两点的实际长度是多少 mm？", "100"))
    gsd = known / px_dist                      # mm / px
    dist = args.distance_mm
    if dist <= 0:
        dist = float(ask("相机到该平面的工作距离是多少 mm？（实测）", "400"))

    # 由 GSD 反推焦距：m = f/(d-f)，GSD = 像元/m
    m = (PIXEL_UM / 1000.0) / gsd              # 放大率
    f_mm = m * dist / (1.0 + m)                # 解 f = m(d-f) 里的 f

    h, w = img.shape[:2]
    print()
    print("=" * 66)
    print(" 两点像素距 = %.1f px，实际 = %.1f mm" % (px_dist, known))
    print(" 工作距离   = %.0f mm" % dist)
    print(" >>> GSD           = %.4f mm/px   （= 每像素 %.3f mm）" % (gsd, gsd))
    print(" >>> 每毫米多少像素 = %.3f px/mm" % (1.0 / gsd if gsd > 0 else 0.0))
    print(" >>> 反推焦距 f    = %.3f mm   （厂家标称 %.1f mm / %d°；按像元 %.2f µm 反推）"
          % (f_mm, NOMINAL_FOCAL_MM, int(NOMINAL_FOV_DEG), PIXEL_UM))
    dev = abs(f_mm - NOMINAL_FOCAL_MM) / max(1e-6, NOMINAL_FOCAL_MM)
    print("     与标称偏差   = %.0f%%   %s"
          % (dev * 100.0,
             "正常（说明像元常数 PIXEL_UM 用对了）" if dev < 0.15
             else "★ 偏大：核对 --distance-mm 是否量准、PIXEL_UM 是否与实际传感器相符"))
    print(" >>> 全画幅覆盖    = %.1f x %.1f mm  (%dx%d)" % (w * gsd, h * gsd, w, h))
    print()
    print(" 派生：料盘 160mm 占 %.0f px；装配位 Ø90mm 占 %.0f px"
          % (160.0 / gsd, 90.0 / gsd))
    print(" 可判定的尺寸下限约 %.2f mm（3-5 像素）" % (gsd * 4))
    print("=" * 66)

    calib.set("fov.work_distance_mm", dist)
    calib.set("fov.gsd_mm_per_px", round(gsd, 5))
    calib.set("fov.focal_mm_derived", round(f_mm, 4))
    calib.set("fov.frame_width_mm", round(w * gsd, 2))
    calib.set("fov.frame_height_mm", round(h * gsd, 2))
    calib.set("fov.pixel_um", PIXEL_UM)
    print("已写入:", save_calib(calib, args.out))

    if src is not None:
        src.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
