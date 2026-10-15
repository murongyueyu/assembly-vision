# -*- coding: utf-8 -*-
"""深度 / 红外流的帧率上限 —— 逐项阻塞读帧，不用回调（上一版回调计数有问题）。

★ 为什么要单独测深度：彩色只是 1 个流；深度是 **2 个红外目在相机内部匹配后输出**，
   数据量与计算量都更大。**彩色能 60fps 不代表深度也能 60fps。**
   若「彩色 + 深度」能同时 60fps，整机才可能到 60fps；
   若深度只能 30，那 60fps 就只适用于纯 2D（不看深度）。

用法：
    & D:\\miniconda\\envs\\yolov11\\python.exe d435_depth_fps.py
"""
from __future__ import annotations

import time

import pyrealsense2 as rs

WARMUP_S = 2.0
MEASURE_S = 5.0


def enum(tries=6):
    for i in range(tries):
        try:
            return list(rs.context().query_devices())
        except RuntimeError as e:                                # noqa: BLE001
            print("  枚举重试 %d: %s" % (i + 1, str(e).strip()[:60]))
            time.sleep(1.5)
    return []


def cfg_of(specs):
    cfg = rs.config()
    for (st, w, h, f) in specs:
        if st == rs.stream.infrared:
            cfg.enable_stream(st, 1, w, h, rs.format.y8, f)      # 只开左目
        elif st == rs.stream.depth:
            cfg.enable_stream(st, w, h, rs.format.z16, f)
        else:
            cfg.enable_stream(st, w, h, rs.format.bgr8, f)
    return cfg


def measure(specs):
    """阻塞等待；返回 {流名: fps}。specs 里所有流都真的在跑。"""
    pipe = rs.pipeline()
    try:
        pipe.start(cfg_of(specs))
    except Exception as e:                                       # noqa: BLE001
        return None, str(e).strip()[:70]
    counts = {}
    try:
        # 预热
        t0 = time.time()
        while time.time() - t0 < WARMUP_S:
            pipe.wait_for_frames(3000)
        counts = {st: 0 for (st, _, _, _) in specs}

        def nm(st):
            return ("彩色" if st == rs.stream.color else
                    "深度" if st == rs.stream.depth else "红外")

        t0 = time.time()
        while True:
            fs = pipe.wait_for_frames(3000)
            for (st, _, _, _) in specs:
                if fs.first_or_default(st):
                    counts[st] += 1
            if time.time() - t0 >= MEASURE_S:
                break
        dt = time.time() - t0
        return {nm(st): counts[st] / dt for (st, _, _, _) in specs}, None
    finally:
        pipe.stop()


devs = enum()
if not devs:
    print("✗ 看不到相机（关掉 RealSense Viewer 再试）")
    raise SystemExit(1)
print("相机 %s SN=%s USB=%s"
      % (devs[0].get_info(rs.camera_info.name), devs[0].get_info(rs.camera_info.serial_number),
         devs[0].get_info(rs.camera_info.usb_type_descriptor)))
print("预热 %.0fs / 测量 %.0fs\n" % (WARMUP_S, MEASURE_S))
print("  %-34s %s" % ("要测的流", "实测"))

TESTS = [
    ("深度 640x480@30", [(rs.stream.depth, 640, 480, 30)]),
    ("深度 640x480@60", [(rs.stream.depth, 640, 480, 60)]),
    ("深度 480x270@60", [(rs.stream.depth, 480, 270, 60)]),
    ("深度 848x480@30", [(rs.stream.depth, 848, 480, 30)]),
    ("深度 848x480@60", [(rs.stream.depth, 848, 480, 60)]),
    ("红外左目 640x480@30", [(rs.stream.infrared, 640, 480, 30)]),
    ("红外左目 640x480@60", [(rs.stream.infrared, 640, 480, 60)]),
    ("红外左目 480x270@60", [(rs.stream.infrared, 480, 270, 60)]),
    ("★ 彩色+深度 640x480@30", [(rs.stream.color, 640, 480, 30),
                                (rs.stream.depth, 640, 480, 30)]),
    ("★ 彩色+深度 640x480@60", [(rs.stream.color, 640, 480, 60),
                                (rs.stream.depth, 640, 480, 60)]),
]
for title, specs in TESTS:
    try:
        res, err = measure(specs)
        if res is None:
            print("  %-34s %s" % (title, "开不了: %s" % err))
        else:
            print("  %-34s %s" % (title, "  ".join("%s %.1f fps" % (k, v) for k, v in res.items())))
    except Exception as e:                                       # noqa: BLE001
        print("  %-34s %s" % (title, "失败: %s" % str(e).strip()[:60]))
    time.sleep(0.8)

print("\n" + "=" * 74)
print(" 判读")
print("=" * 74)
print("  · 深度能到 ~60fps -> 「60fps 双目深度」可行，整机上限取决于算法耗时")
print("  · 深度只有 ~30fps -> 60fps 只能用于**彩色/2D**，深度得单独低频跑")
print("  ★ 60fps 下每帧只给算法 16.7ms（30fps 是 33.3ms），而工程实测")
print("    KCF 约 8ms/目标 —— 4 个目标就吃满 32ms。换 60fps 不会让算法变快，")
print("    反而更容易把算法逼到掉帧；真正受益的是**手工快速摆放/遮挡**这类场景。")
raise SystemExit(0)
