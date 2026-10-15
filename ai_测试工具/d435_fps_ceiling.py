# -*- coding: utf-8 -*-
"""D435 帧率上限精确测量（纯读帧，不存图、不 imshow、不跑算法）。

★ 为什么要重测：上一版 d435_fps_bench.py 给的是 26.4fps，而且
   「640x480@30 / 848x480@30 / 1280x720@30 / 1920x1080@30 全是 26.4」
   —— 三种带宽差 4 倍的流给出**完全相同**的数，这个形态本身就可疑：
   更像是**测量方法**卡住了（Python 循环 + wait_for_frames 的调度开销），
   而不是相机只跑 26.4。

本脚本的改进：
  1. **预热更长**（2 秒）并丢弃，避免把启动抖动算进平均
  2. 用**滑动窗口**报稳定段帧率，而不是全程平均
  3. 用**回调模式**对照 —— 让 SDK 的采集线程直接计数，绕开 Python 循环
  4. 只读不写：不 imwrite / 不 imshow / 不跑分割

用法：
    & D:\\miniconda\\envs\\yolov11\\python.exe d435_fps_ceiling.py
"""
from __future__ import annotations

import time

import numpy as np
import pyrealsense2 as rs

CASES = [
    (640, 480, 30),
    (640, 480, 60),      # 规格上不提供，看报什么错
    (424, 240, 60),
    (424, 240, 30),
    (1280, 720, 30),
    (1920, 1080, 30),
    (848, 480, 30),
]
WARMUP_S = 2.0
MEASURE_S = 6.0


def enum(tries=6):
    for i in range(tries):
        try:
            return list(rs.context().query_devices())
        except RuntimeError as e:                                # noqa: BLE001
            print("  枚举重试 %d: %s" % (i + 1, str(e).strip()[:60]))
            time.sleep(1.5)
    return []


def measure_blocking(w, h, f):
    """阻塞读帧，报全程平均 + 稳定段（后半段）帧率。"""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
    pipe.start(cfg)
    try:
        # 预热：丢掉启动抖动
        t0 = time.time()
        while time.time() - t0 < WARMUP_S:
            pipe.wait_for_frames(3000)

        t0 = time.time()
        marks = []           # (相对时间, 累计帧数)
        n = 0
        while True:
            fs = pipe.wait_for_frames(3000)
            if not fs.get_color_frame():
                continue
            n += 1
            now = time.time() - t0
            marks.append((now, n))
            if now >= MEASURE_S:
                break
        total = marks[-1][0]
        avg = n / total if total else 0.0
        # 稳定段：最后 3 秒
        tail = [(t, k) for (t, k) in marks if t >= total - 3.0]
        if len(tail) > 2:
            dt = tail[-1][0] - tail[0][0]
            dn = tail[-1][1] - tail[0][1]
            stable = dn / dt if dt > 0 else 0.0
        else:
            stable = avg
        return avg, stable, n
    finally:
        pipe.stop()


def measure_callback(w, h, f):
    """回调计数：让 SDK 采集线程自己数，绕开 Python 循环开销。"""
    cnt = {"n": 0}
    lock = {"stop": False}

    def cb(frame):
        if not lock["stop"]:
            cnt["n"] += 1

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
    try:
        pipe.start(cfg, cb)
    except Exception as e:                                       # noqa: BLE001
        return None, str(e).strip()[:50]
    try:
        time.sleep(WARMUP_S)
        cnt["n"] = 0                    # 预热后清零，只算测量段
        t0 = time.time()
        time.sleep(MEASURE_S)
        dt = time.time() - t0
        lock["stop"] = True
        return cnt["n"] / dt, None
    finally:
        lock["stop"] = True
        pipe.stop()


devs = enum()
if not devs:
    print("✗ 看不到相机（关掉 RealSense Viewer 再试）")
    raise SystemExit(1)
d0 = devs[0]
print("相机 %s SN=%s USB=%s 固件=%s"
      % (d0.get_info(rs.camera_info.name), d0.get_info(rs.camera_info.serial_number),
         d0.get_info(rs.camera_info.usb_type_descriptor),
         d0.get_info(rs.camera_info.firmware_version)))
print("预热 %.0fs / 测量 %.0fs\n" % (WARMUP_S, MEASURE_S))
print("  %-16s %-12s %-12s %s" % ("模式", "阻塞·全程", "阻塞·稳定段", "回调模式(绕开Python循环)"))

results = {}
for (w, h, f) in CASES:
    tag = "%dx%d@%d" % (w, h, f)
    try:
        avg, stable, n = measure_blocking(w, h, f)
        cb, cberr = measure_callback(w, h, f)
        cb_s = ("%.1f fps" % cb) if cb is not None else ("开不了: %s" % cberr)
        print("  %-16s %-12s %-12s %s" % (tag, "%.1f fps" % avg, "%.1f fps" % stable, cb_s))
        results[tag] = (avg, stable, cb)
    except Exception as e:                                       # noqa: BLE001
        print("  %-16s %s" % (tag, "开不了 / 失败: %s" % str(e).strip()[:50]))
    time.sleep(0.8)

print("\n" + "=" * 72)
print(" 判读")
print("=" * 72)
if "640x480@30" in results:
    a, s, cb = results["640x480@30"]
    print("  640x480@30：阻塞全程 %.1f / 稳定段 %.1f / 回调 %s" % (a, s, cb))
    ref = cb if cb else s
    if ref >= 28.0:
        print("  → ★ 相机本身能跑到 ~30fps。之前报的 26.4fps 是**测量方法**造成的，")
        print("     不是相机上限（三种分辨率给出同一个数就是证据）。")
    else:
        print("  → 稳定段也只有 %.1f，说明确实有东西在限速（下面查原因）。" % ref)
if "424x240@60" in results:
    a, s, cb = results["424x240@60"]
    print("  424x240@60：阻塞稳定段 %.1f / 回调 %s" % (s, cb))
    print("  → 这一档能明显高于 30 的话，说明「想要更高帧率」的路是**降彩色分辨率**。")
print("\n  ★ 注意：彩色流的帧率上限是**传感器 UVC 模式**决定的，与 USB 3 带宽无关 ——")
print("    640x480 RGB 一帧约 0.92MB，USB3 的 5Gbps 能塞几百 fps，但相机不给这个模式。")
raise SystemExit(0)
