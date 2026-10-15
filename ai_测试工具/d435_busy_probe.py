# -*- coding: utf-8 -*-
"""D435 「Device or resource busy」排查：看传感器状态，必要时做一次硬件复位。

只读为主；只有加 --reset 才会真的复位相机（不改任何设置/文件）。
"""
from __future__ import annotations

import argparse
import time

import pyrealsense2 as rs

ap = argparse.ArgumentParser("d435_busy_probe")
ap.add_argument("--reset", action="store_true", help="硬件复位一次再试开流")
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--attempts", type=int, default=3, help="开流重试次数（间隔 3 秒）")
args = ap.parse_args()


def enum_devices(tries=6):
    for i in range(tries):
        try:
            return list(rs.context().query_devices())
        except RuntimeError as e:                                # noqa: BLE001
            print("  枚举第 %d 次失败: %s" % (i + 1, str(e).strip()))
            time.sleep(1.5)
    return []


print("=== ① 枚举 ===")
devs = enum_devices()
print("设备数 =", len(devs))
if not devs:
    raise SystemExit(1)

dev = devs[0]
print("  %s SN=%s USB=%s" % (dev.get_info(rs.camera_info.name),
                             dev.get_info(rs.camera_info.serial_number),
                             dev.get_info(rs.camera_info.usb_type_descriptor)))

print("\n=== ② 逐个传感器：是不是已经在流（busy 的直接原因）===")
for s in dev.sensors:
    nm = s.get_info(rs.camera_info.name)
    line = "  [%s]" % nm
    try:
        line += "  正在流=%s" % s.is_streaming
    except Exception as e:                                       # noqa: BLE001
        line += "  is_streaming 读不到(%s)" % e
    # 关键选项的当前值（顺便确认曝光/自动曝光现状）
    for opt, label in ((rs.option.enable_auto_exposure, "自动曝光"),
                       (rs.option.exposure, "曝光"),
                       (rs.option.gain, "增益"),
                       (rs.option.enable_auto_white_balance, "自动白平衡")):
        try:
            if s.supports(opt):
                line += "  %s=%s" % (label, s.get_option(opt))
        except Exception:                                        # noqa: BLE001
            pass
    print(line)
    try:
        for p in s.get_stream_profiles():
            pass
    except Exception:                                            # noqa: BLE001
        pass

if args.reset:
    print("\n=== ③ 硬件复位（--reset）===")
    try:
        dev.hardware_reset()
        print("  已发出复位；等 6 秒让设备重新枚举…")
        time.sleep(6)
        devs = enum_devices()
        print("  复位后设备数 =", len(devs))
    except Exception as e:                                       # noqa: BLE001
        print("  复位失败（可能不支持）:", e)

print("\n=== ④ 试开彩色流 ===")
ok = False
for i in range(args.attempts):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    try:
        pipe.start(cfg)
        print("  第 %d 次：开流成功 %dx%d@%d" % (i + 1, args.width, args.height, args.fps))
        t0 = time.time()
        n = 0
        for _ in range(20):
            fs = pipe.wait_for_frames(5000)
            if fs.get_color_frame():
                n += 1
        dt = time.time() - t0
        print("  读到 %d 帧，%.1f fps" % (n, n / dt if dt else 0))
        pipe.stop()
        ok = True
        break
    except Exception as e:                                       # noqa: BLE001
        print("  第 %d 次：%s" % (i + 1, str(e).strip()))
        try:
            pipe.stop()
        except Exception:                                        # noqa: BLE001
            pass
        time.sleep(3)

print("\n结论:", "✓ 能出图，之前是暂时性占用/状态不干净" if ok else "✗ 仍开不了流")
if not ok:
    print("  下一步建议（按顺序试）：")
    print("   1. 拔掉相机、等 10 秒、换**另一个 USB 3 口**再插（别用同一个口）")
    print("   2. 打开 RealSense Viewer 看能不能出图 —— 能的话说明是 Python 侧问题；")
    print("      不能的话说明相机被系统/驱动占着。")
    print("   3. 设备管理器里卸载 D435 的两个「USB Video Device」节点，再拔插一次。")
raise SystemExit(0 if ok else 1)
