# -*- coding: utf-8 -*-
"""D435 帧率基准：USB3 下各模式的实测帧率，用来定正式跑哪个分辨率。

★ 为什么必须实测而不是看规格：
    规格给的 30fps 是理想值。真正决定能不能用的是「USB 带宽 + 主机解码 + Python 开销」，
    而且**工程里 part_track_demo 的瓶颈是每目标 KCF 约 8ms**（4 件 ≈ 32ms → 31fps 上限），
    相机帧率一旦低于这个数，追踪就会掉帧。所以要先知道相机能稳定给多少。

用法：
    & D:\\miniconda\\envs\\yolov11\\python.exe d435_fps_bench.py            # 默认扫常用模式
    ... --width 1280 --height 720 --fps 30            # 只测一个
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyrealsense2 as rs

ap = argparse.ArgumentParser("d435_fps_bench")
ap.add_argument("--seconds", type=float, default=4.0, help="每个模式测多少秒")
ap.add_argument("--width", type=int, default=0)
ap.add_argument("--height", type=int, default=0)
ap.add_argument("--fps", type=int, default=0)
args = ap.parse_args()

SCAN = [(640, 480, 30), (640, 480, 60), (848, 480, 30), (1280, 720, 30),
        (1280, 720, 60), (1920, 1080, 30)]
if args.width:
    SCAN = [(args.width, args.height, args.fps or 30)]


def enum(tries=6):
    for i in range(tries):
        try:
            return list(rs.context().query_devices())
        except RuntimeError as e:                                # noqa: BLE001
            print("  枚举重试 %d: %s" % (i + 1, str(e).strip()))
            time.sleep(1.5)
    return []


devs = enum()
if not devs:
    print("✗ 看不到相机（★ 先把 RealSense Viewer 关掉，它会独占设备）")
    raise SystemExit(1)
d0 = devs[0]
print("相机 %s SN=%s USB=%s"
      % (d0.get_info(rs.camera_info.name), d0.get_info(rs.camera_info.serial_number),
         d0.get_info(rs.camera_info.usb_type_descriptor)))
print("每模式测 %.1f 秒\n" % args.seconds)
print("  %-16s %-10s %-8s %s" % ("请求模式", "实测fps", "首帧耗时", "备注"))

best = None
for (w, h, f) in SCAN:
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
    tag = "%dx%d@%d" % (w, h, f)
    try:
        prof = pipe.start(cfg)
    except Exception as e:                                       # noqa: BLE001
        print("  %-16s %-10s %-8s %s" % (tag, "-", "-", "开流失败: %s" % str(e).strip()[:40]))
        try:
            pipe.stop()
        except Exception:                                        # noqa: BLE001
            pass
        time.sleep(0.6)
        continue

    t_first = None
    n = 0
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        fs = pipe.wait_for_frames(3000)
        if fs.get_color_frame():
            if t_first is None:
                t_first = time.time() - t0
            n += 1
    dt = time.time() - t0
    got = n / dt if dt else 0.0
    # 再验证一次形状（防止驱动悄悄换分辨率）
    shape = ""
    note = ""
    if got >= 0.9 * f:
        note = "✓ 达标"
        if best is None or (w * h > best[0] * best[1] and got >= 25):
            best = (w, h, f, got)
    elif got >= 25:
        note = "△ 帧率够用但没到标称"
        if best is None:
            best = (w, h, f, got)
    else:
        note = "✗ 太慢"
    print("  %-16s %-10.1f %-8s %s" % (tag, got,
                                       ("%.2fs" % t_first) if t_first else "-", note))
    pipe.stop()
    time.sleep(0.6)

print("\n" + "=" * 62)
if best:
    # 实测内参：1280x720 时 fx=905.74 -> 640x480 时 fx=452.87，比例固定
    fx = 905.74 * best[0] / 1280.0
    cover = best[0] * 360.0 / fx          # 360mm 处的水平覆盖
    gsd = 360.0 / fx                      # mm/px（同样在 360mm 处）
    print("★ 建议正式使用： %dx%d @ %d（实测 %.1f fps）" % (best[0], best[1], best[2], best[3]))
    print("  工程 part_track_demo 的真机基准是 28.9fps（相机上限 31）；")
    print("  相机只有 15fps 的话追踪会明显掉帧，建议优先选实测 >=25fps 的模式。")
    print("  按出厂内参算：360mm 处水平覆盖 %.0f mm，GSD ≈ %.3f mm/px" % (cover, gsd))
    print("  （HBV 当年的实测是 GSD 0.8623 mm/px、覆盖 555x417mm，可作对比）")
else:
    print("✗ 没有可用模式")
raise SystemExit(0)
