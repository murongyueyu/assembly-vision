# -*- coding: utf-8 -*-
"""D435 自检探针 —— 走**工程自己的** `RealSenseSource`，不是另写一份取图代码。

★ 为什么要它、而不是直接用 用户存档/ 里的脚本：
  那些脚本是换机期的一次性排查工具，**不经过工程代码**。
  本工具特意去构造 `assemvision.sources.d435.RealSenseSource`，
  所以它验证的是"工程真能用 D435"这件事本身 —— 源码一改，它立刻跟着变。

用法
    python tools/probe_d435.py                       # 按 camera.yaml 的设置自检 + 存一帧
    python tools/probe_d435.py --modes 640x480,1280x720 --fps 30,60
    python tools/probe_d435.py --seconds 4 --distance-mm 360
    python tools/probe_d435.py --reset               # 设备 busy / 打不开时先硬件复位
    python tools/probe_d435.py --no-save             # 不写盘

判读（三条硬约束，见 docs/踩坑实录.md §2.30~§2.37）：
  * `USB` 必须是 **3.x**：2.1 链路上**一帧都拿不到**，别去调分辨率；
  * 帧率：640x480@30 实测 **30.0fps**，60fps 也真的能开（规格表没列全）；
  * 曝光：看 `过曝%`。近全白像素应 **≈0%**（工程里的报警阈值是 0.5%）——
    过曝会让整条判据链**静默失效**（实录 §2.14）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from assemvision.config import Config, load_config                  # noqa: E402
from assemvision.imgio import imwrite_safe                          # noqa: E402
from assemvision.sources.d435 import (RealSenseSource, import_rs,    # noqa: E402
                                      sdk_available, usb_is3)

# 本机实测过的模式（★ 不是猜的：见 docs/现状与交接.md 的 D435 基线）
DEFAULT_MODES: List[Tuple[int, int]] = [(640, 480), (848, 480), (1280, 720)]
DEFAULT_FPS: List[int] = [30]


def parse_modes(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in str(text or "").split(","):
        part = part.strip().lower().replace("*", "x")
        if "x" not in part:
            continue
        w, h = part.split("x")[:2]
        out.append((int(w), int(h)))
    return out


def parse_ints(text: str) -> List[int]:
    return [int(p) for p in str(text or "").split(",") if p.strip().isdigit()]


def hardware_reset(wait_s: float = 6.0) -> int:
    """硬件复位 —— `Device or resource busy` 的最后一招。

    ★ 复位能解决"**占用**"，解决不了"**链路速率**"（USB 2.1）——两件事别混（实录 §2.33）。
    ★ 固件只能降级 20 次，但复位不是刷固件，安全。
    """
    rs = import_rs()
    try:
        devs = list(rs.context().query_devices())
    except Exception as e:                                             # noqa: BLE001
        print("✗ 枚举设备失败：%s" % e)
        return 1
    if not devs:
        print("✗ 看不到任何 RealSense 相机（复位也需要先能看到它）")
        return 1
    for d in devs:
        try:
            sn = d.get_info(rs.camera_info.serial_number)
        except Exception:                                              # noqa: BLE001
            sn = "?"
        print("→ 复位 %s（SN=%s）…" % (d.get_info(rs.camera_info.name), sn))
        try:
            d.hardware_reset()
        except Exception as e:                                         # noqa: BLE001
            print("  ✗ 复位失败：%s" % e)
            return 1
    print("  等 %.0f 秒让它重新枚举…" % wait_s)
    time.sleep(wait_s)
    print("  复位完成。重新插拔 USB 也可以，效果一样。")
    return 0


def exposure_stats(bgr: np.ndarray) -> Tuple[float, float]:
    """(过曝比例, V 均值)。近全白 = 三通道都 ≥250，与 segment.py 的口径一致。"""
    if bgr is None or bgr.size == 0:
        return 0.0, 0.0
    near_white = np.all(bgr >= 250, axis=2)
    hsv = None
    try:
        import cv2
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    except Exception:                                                  # noqa: BLE001
        pass
    v_mean = float(hsv[:, :, 2].mean()) if hsv is not None else float(bgr.mean())
    return float(near_white.mean()), v_mean


def probe_mode(base: Config, w: int, h: int, fps: int,
               seconds: float, warmup: int) -> dict:
    """用工程的 RealSenseSource 开一个模式，测帧率并带回首帧。"""
    cam = Config(dict(base.data))          # 浅拷贝：只改自己要改的键
    cam.set("realsense.width", w)
    cam.set("realsense.height", h)
    cam.set("realsense.fps", fps)
    # ★ 自检工具里相机要"开一次就报一次错"，重试留 1 次就够，省得等太久
    cam.set("realsense.open_retries", 1)

    src = RealSenseSource(cam)
    t0 = time.time()
    try:
        src.open()
        src.apply_locked_controls()        # 与 pipeline/标定脚本同一条路
    except Exception as e:                                             # noqa: BLE001
        return {"ok": False, "err": str(e).strip(), "w": w, "h": h, "fps_req": fps}

    for _ in range(max(1, warmup)):        # 丢开头几帧（自动曝光要稳定）
        src.read()

    n, frame, t_start = 0, None, time.time()
    while time.time() - t_start < seconds:
        f = src.read()
        if f is not None and f.left is not None:
            n += 1
            frame = f.left
    dt = max(1e-6, time.time() - t_start)
    intr = src.color_intrinsics()
    info = {"ok": True, "w": w, "h": h, "fps_req": fps, "fps": n / dt,
            "frame": frame, "intr": intr, "usb": src.usb, "sn": src.dev_serial,
            "fw": src.fw, "name": src.dev_name, "shape": None if frame is None else frame.shape,
            "open_s": time.time() - t0}
    src.close()
    return info


def main() -> int:
    ap = argparse.ArgumentParser("probe_d435")
    ap.add_argument("--camera", default=os.path.join(ROOT, "config", "camera.yaml"))
    ap.add_argument("--modes", default="", help="如 640x480,1280x720（默认真测过的三档）")
    ap.add_argument("--fps", default="", help="如 30,60（默认 30）")
    ap.add_argument("--seconds", type=float, default=3.0, help="每档测帧率的时长")
    ap.add_argument("--warmup", type=int, default=15, help="丢弃的开头帧数")
    ap.add_argument("--distance-mm", type=float, default=360.0,
                    help="算 GSD 用的工作距离（与 c1 一致，默认 360）")
    ap.add_argument("--reset", action="store_true", help="先做硬件复位再自检")
    ap.add_argument("--no-save", action="store_true", help="不保存样张")
    args = ap.parse_args()

    print("=" * 78)
    print(" D435 自检探针（走工程的 assemvision.sources.d435.RealSenseSource）")
    print(" Python %s   %s" % (sys.version.split()[0], sys.platform))
    print("=" * 78)

    if not sdk_available():
        print("✗ 没装 pyrealsense2 —— D435 必须走 Intel SDK，cv2.VideoCapture 打不开它。")
        print("  装法：pip install pyrealsense2")
        return 1

    if args.reset and hardware_reset() != 0:
        return 1

    try:
        base = load_config(args.camera)
    except Exception as e:                                             # noqa: BLE001
        print("✗ 读 %s 失败：%s" % (args.camera, e))
        return 1

    modes = parse_modes(args.modes) or DEFAULT_MODES
    fps_list = parse_ints(args.fps) or DEFAULT_FPS
    # 相机自己说支持哪些模式 —— 列出来是为了**对照**（★ 它列不全，见实录 §2.37）
    list_supported()

    out_dir = os.path.join(ROOT, "data", "probe")
    if not args.no_save:
        os.makedirs(out_dir, exist_ok=True)

    results = []
    for (w, h) in modes:
        for fps in fps_list:
            tag = "%dx%d@%d" % (w, h, fps)
            r = probe_mode(base, w, h, fps, args.seconds, args.warmup)
            results.append(r)
            if not r["ok"]:
                print("  %-14s ✗ %s" % (tag, (r.get("err") or "").splitlines()[0]))
                continue
            blown, v_mean = exposure_stats(r["frame"])
            gsd = None
            if r["intr"] and r["intr"].get("fx"):
                gsd = args.distance_mm / float(r["intr"]["fx"])
            print("  %-14s fps %5.1f  画面 %s  过曝 %5.2f%%  V均值 %5.1f  GSD %s"
                  % (tag, r["fps"], r["shape"],
                     blown * 100.0, v_mean,
                     ("%.3f mm/px" % gsd) if gsd else "?"))
            if gsd:
                cover = r["shape"][1] * gsd, r["shape"][0] * gsd
                print("  %-14s 覆盖 %.0f x %.0f mm（@%.0fmm）  内参 fx=%.1f fy=%.1f"
                      % ("", cover[0], cover[1], args.distance_mm,
                         r["intr"]["fx"], r["intr"]["fy"]))
            if not args.no_save and r["frame"] is not None:
                p = os.path.join(out_dir, "d435_%s.png" % tag)
                try:
                    okw = bool(imwrite_safe(p, r["frame"]))
                except Exception as e:                                 # noqa: BLE001
                    okw, p = False, "%s（%s）" % (p, e)
                # ★ 打印必须是"成功/失败"，不能无脑说"已保存"（中文路径静默失败踩过，实录 §2.36）
                print("  %-14s 样张：%s %s" % ("", "已保存" if okw else "★ 保存失败", p))

    usable = [r for r in results if r["ok"]]
    if not usable:
        print("\n★ 一个模式都没开起来。检查顺序：")
        print("   ① 相机插好了没、线是数据线不是充电线")
        print("   ② **换 USB 3 口**（USB 2.1 链路下一帧都拿不到）")
        print("   ③ 关掉 RealSense Viewer（它会独占设备）")
        print("   ④ python tools/probe_d435.py --reset")
        return 1

    best = max(usable, key=lambda r: (r["fps"] >= 24.0, r["shape"][0] * r["shape"][1] * r["fps"]))
    usb = usable[0]["usb"]
    print("\n" + "=" * 78)
    print(" 设备：%s  SN=%s  固件=%s  USB=%s  开流耗时 %.1fs"
          % (usable[0]["name"], usable[0]["sn"], usable[0]["fw"], usb, usable[0]["open_s"]))
    if not usb_is3(usb):
        print(" ★★ USB 不是 3.x！实测这种链路上**一帧都拿不到** —— 先换 USB 3 口。")
    print(" 建议写入 config/camera.yaml 的值：")
    print("     realsense:")
    print("       width: %d" % best["w"])
    print("       height: %d" % best["h"])
    print("       fps: %d" % best["fps_req"])
    print("     # 实测 %.1f fps" % best["fps"])
    blown, v_mean = exposure_stats(best["frame"])
    print(" 曝光判读：过曝 %.2f%%（应 ≈0，报警阈值 0.5%%）、V 均值 %.1f" % (blown * 100.0, v_mean))
    if blown > 0.005:
        print("   ★ 过曝偏高：先在工位调 rs.option.exposure（**不是** DSHOW 的 -6.0！）")
        print("     —— 过曝会让判据链静默失效（实录 §2.14）。")
    print("=" * 78)
    return 0


def list_supported() -> None:
    """把相机自报的彩色模式列一遍（★ 只作对照：它列不全，60fps 实测能开）。"""
    try:
        rs = import_rs()
        devs = list(rs.context().query_devices())
        if not devs:
            print("★ SDK 看不到相机。（若确实插着：换 USB 3 口 / 关掉 Viewer / --reset）")
            return
        rows = []
        for s in devs[0].sensors:
            try:
                if not s.get_info(rs.camera_info.name).lower().startswith("rgb"):
                    continue
            except Exception:                                          # noqa: BLE001
                continue
            for p in s.get_stream_profiles():
                v = p.as_video_stream_profile()
                rows.append("%dx%d@%d" % (v.width(), v.height(), int(p.fps())))
        uniq = sorted(set(rows))
        print(" 相机自报的彩色模式（%d 个）：%s" % (len(uniq), "、".join(uniq) if uniq else "—"))
        print(" ★ 它**列不全**（实测 640x480@60 能开却没列）—— 想要某档就直接试一次。")
    except Exception as e:                                             # noqa: BLE001
        print(" （列模式失败，不影响自检：%s）" % str(e).strip())


if __name__ == "__main__":
    raise SystemExit(main())
