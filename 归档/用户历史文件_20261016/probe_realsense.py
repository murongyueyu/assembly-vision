# -*- coding: utf-8 -*-
"""D435 专用探针（走 Intel RealSense SDK，不经过 OpenCV 的相机索引）。

★ 为什么不能用 cv2.VideoCapture：D435 在 Windows 上不会被 OpenCV 按「相机索引」
   枚举到（cv2.VideoCapture(0, CAP_DSHOW/CAP_MSMF) 全部 isOpened()=False，
   连索引 0~5 都扫不到），所以必须用 pyrealsense2 直接和硬件说话。

★ 本机实测（2026-10）：RealSense D435，序列号 262322070302，固件 5.15.1.55，
   **USB 2.1** —— 这决定了能跑哪些模式：USB 2 下 1280x720 只有 6/10/15 fps，
   1920x1080 只有 8 fps，**640x480 才有 30 fps**。想要 30fps 高分辨率就换 USB 3 口。

用法（必须用 yolo 环境的 python）：
    D:\\miniconda\\envs\\yolov11\\python.exe probe_realsense.py            # 只探测
    D:\\miniconda\\envs\\yolov11\\python.exe probe_realsense.py --show     # 开窗口看画面
    ... --show --width 640 --height 480 --fps 30                           # 指定模式
    ... --show --save out.jpg
"""
from __future__ import annotations

import argparse
import time


def list_color_profiles(rs, dev):
    """列出该设备所有彩色流模式（不重复、按分辨率/帧率排序）。"""
    out = []
    for s in dev.sensors:
        nm = s.get_info(rs.camera_info.name)
        if "RGB" not in nm and "Color" not in nm:
            continue
        for p in s.get_stream_profiles():
            if p.stream_type() != rs.stream.color:
                continue
            v = p.as_video_stream_profile()
            item = (v.width(), v.height(), v.fps(), str(p.format()))
            if item not in out:
                out.append(item)
    out.sort(key=lambda x: (-x[0] * x[1], -x[2], x[3]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser("probe_realsense")
    ap.add_argument("--show", action="store_true", help="打开实时窗口（按 q 退出，s 存图）")
    ap.add_argument("--save", default="", help="存一张彩色图到指定路径")
    ap.add_argument("--width", type=int, default=0, help="0=自动挑一个能用的模式")
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--fps", type=int, default=0, help="0=该分辨率下能跑的最高帧率")
    ap.add_argument("--serial", default="", help="多台设备时用序列号指定")
    ap.add_argument("--lock-ae", action="store_true",
                    help="关掉自动曝光/自动白平衡（★ 实测 USB2 下容易导致读不到帧，默认关）")
    args = ap.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError:
        print("✗ 没装 pyrealsense2。先执行：")
        print("    D:\\miniconda\\envs\\yolov11\\python.exe -m pip install pyrealsense2")
        return 2

    import numpy as np

    print("pyrealsense2 版本:", getattr(rs, "__version__", "未知"))

    devices = list(rs.context().query_devices())
    print("\n===== 第一步：SDK 看到几台设备 =====")
    print("  设备数 =", len(devices))
    if not devices:
        print("  ✗ SDK 一台都没看到。可能原因：")
        print("     · 相机没插好 / 线是坏线（D435 自带线，换一条 USB 3 数据线试）")
        print("     · 插在 USB 2.0 口或 HUB 上（要插主板后置 USB 3.0，蓝色口）")
        print("     · 被别的程序独占：RealSense Viewer / Windows 相机 App / Teams / Zoom")
        print("       —— 它们开着的时候 SDK 会看不到设备，先全关掉")
        print("     · 设备管理器里有感叹号：右键卸载设备再重新插拔")
        return 1

    for i, d in enumerate(devices):
        def info(key):
            try:
                return d.get_info(key)
            except Exception:  # noqa: BLE001
                return "?"
        usb = str(info(rs.camera_info.usb_type_descriptor))
        print("  [%d] %s" % (i, info(rs.camera_info.name)))
        print("       序列号 = %s" % info(rs.camera_info.serial_number))
        print("       固件   = %s" % info(rs.camera_info.firmware_version))
        print("       USB    = %s   %s"
              % (usb, "★ 3.x，可跑高分辨率高帧率"
                 if usb.startswith("3") else
                 "⚠ 2.x：带宽受限，1280x720 最高 15fps、1920x1080 只有 8fps；"
                 "要 30fps 高分辨率请换主板后置 USB 3 口（蓝色/带 SS 标记）"))
        print("       传感器: %s" % ", ".join(s.get_info(rs.camera_info.name) for s in d.sensors))

    dev = devices[0]
    serial = args.serial or dev.get_info(rs.camera_info.serial_number)

    print("\n===== 第二步：支持的彩色流模式 =====")
    profs = list_color_profiles(rs, dev)
    seen = set()
    for (w, h, f, fmt) in profs:
        key = (w, h)
        if key not in seen:
            seen.add(key)
            print("   %-10s -> fps %s"
                  % ("%dx%d" % (w, h), "/".join(str(p[2]) for p in profs if (p[0], p[1]) == key)))

    # ── 选模式：优先用用户指定的；指定不了（或 USB2 带宽不够）就自动降级 ──
    cands = []
    if args.width and args.height:
        cands.append((args.width, args.height, args.fps))
    cands += [
        (1280, 720, args.fps or 30), (1280, 720, 15), (1280, 720, 6),
        (640, 480, args.fps or 30), (640, 480, 15),
    ]
    # 兜底：设备实际支持的最高分辨率（按面积）
    if profs:
        w, h, f, _ = profs[0]
        cands.append((w, h, f))

    pipe = rs.pipeline()
    used = None
    for (w, h, f) in cands:
        cfg = rs.config()
        cfg.enable_device(serial)
        if f:
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
        else:
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8)
        try:
            profile = pipe.start(cfg)
            used = (w, h, f)
            break
        except Exception as e:  # noqa: BLE001
            print("   [试] %dx%d@%s -> 不行（%s）" % (w, h, f or "默认", str(e).strip()))
            try:
                pipe.stop()
            except Exception:  # noqa: BLE001
                pass
    if used is None:
        print("\n  ✗ 所有候选模式都开不了流。把 --width/--height/--fps 换成上面列出的组合再试。")
        return 1
    print("\n===== 第三步：开流成功，实际模式 = %dx%d @ %s =====" % (used[0], used[1], used[2] or "默认"))

    try:
        # 关掉自动曝光/自动白平衡，画面才稳定（后面做 HSV 阈值判据全靠它）
        # ★ 实测：USB 2.1 连接下刚开流就改曝光，容易导致 wait_for_frames 超时读不到帧，
        #   所以改成 --lock-ae 显式开启，默认不动相机设置、保证先能出图。
        if args.lock_ae:
            try:
                for s in profile.get_device().sensors:
                    if s.supports(rs.option.enable_auto_exposure):
                        s.set_option(rs.option.enable_auto_exposure, 0)
                    if s.supports(rs.option.enable_auto_white_balance):
                        s.set_option(rs.option.enable_auto_white_balance, 0)
                print("  已关闭自动曝光 / 自动白平衡（--lock-ae）")
            except Exception as e:  # noqa: BLE001
                print("  自动曝光设置跳过:", e)
        else:
            print("  保持相机自动曝光（要锁定请加 --lock-ae）")

        t0 = time.time()
        n = 0
        first = None
        try:
            for _ in range(30):                       # 预热，丢前几帧
                f = pipe.wait_for_frames(5000)
                c = f.get_color_frame()
                if c:
                    first = np.asanyarray(c.get_data())
                    n += 1
        except RuntimeError as e:
            print("  ✗ 读帧超时:", e)
            print("     → 常见于 USB 2.1 连接或不稳定端口。先试 **640x480@30**：")
            print("       --width 640 --height 480 --fps 30")
            print("     → 若还不行：换一条 USB 3 数据线 + 主板后置 USB 3 口（别用 HUB/延长线）")
            return 1
        dt = time.time() - t0
        if first is None:
            print("  ✗ 拿不到彩色帧")
            return 1
        print("  成功读取 %d 帧，实测 %.1f fps，frame.shape = %s"
              % (n, n / dt if dt else 0, first.shape))

        if args.save:
            import cv2
            ok = cv2.imwrite(args.save, first)     # 中文路径要用 cv2.imencode，这里路径是英文
            print("  已保存: %s  ok=%s" % (args.save, ok))

        if args.show:
            import cv2
            print("\n  打开窗口：q=退出  s=存图")
            k = 0
            while True:
                f = pipe.wait_for_frames()
                c = f.get_color_frame()
                if not c:
                    continue
                img = np.asanyarray(c.get_data())
                cv2.imshow("D435 color", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    k += 1
                    p = "d435_%02d.jpg" % k
                    cv2.imwrite(p, img)
                    print("  已保存:", p)
            cv2.destroyAllWindows()
    finally:
        pipe.stop()

    print("\n✓ 结论：D435 可以出图，走 pyrealsense2，不要再走 cv2.VideoCapture(索引)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
