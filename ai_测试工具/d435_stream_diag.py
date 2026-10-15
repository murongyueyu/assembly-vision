# -*- coding: utf-8 -*-
"""D435 开流诊断：**逐传感器列出全部 profile，并逐个试开**（只读，不改工程）。

★ 为什么单独写这一个（2026-10 实测踩到）：
  用户终端里跑 `d435_intrinsics_check.py` 时：
    · 设备**枚举得到**（SN / 固件 / USB=3.2 都正常）；
    · `sensors[].get_stream_profiles()` **读得到**（说明设备句柄是通的）；
    · 但 **640x480@30 与 1280x720@30 都报 `Couldn't resolve requests`**；
    · 而且自报的彩色模式**只有 1920x1080 一组**（bgr8/bgra8/rgb8/rgba8/yuyv）。
  这三个事实合起来，指向的不是"权限"也不是"模式不存在"，而是
  **传感器的 profile 集合当前被限制住了**。
  所以本脚本要一刀切开两种可能：
    A. **设备被别的进程占着**（Windows「多应用相机模式」/ 相机 App / RealSense Viewer
       / 没退干净的 python）→ 表现是"所有 profile 都开不了"；
    B. **profile 真的只剩 1920x1080** → 表现是"1920x1080 能开、别的不能"。
  ★ 判据很简单：**看 1920x1080@30 能不能开**。
    能开 = B（那就在 1920x1080 上干活）；开不了 = A（去关占用）。

★ 顺带回答一个工程上的问题：`1920x1080` 与 `640x480` 的视野**是不是一回事**。
  工程文档（实录 §2.38）说 640×480 是**裁切模式**（FOV 55.9° vs 70.5°），
  本脚本会把每个能开的 profile 的 fx/fy/ppx/ppy 打出来，
  用 `2*atan(W/(2*fx))` 直接算出水平 FOV —— 一览便知哪个模式视野更宽。

用法（★ 在**不受限**的进程里跑；VSCode 里打开本文件按 F5）：
    & D:\\miniconda\\envs\\yolov11\\python.exe "ai_测试工具\\d435_stream_diag.py"
    & D:\\miniconda\\envs\\yolov11\\python.exe "ai_测试工具\\d435_stream_diag.py" --reset
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser("d435_stream_diag")
    ap.add_argument("--reset", action="store_true",
                    help="先硬件复位（清设备占用；★ 会冲掉手调的相机控件值）")
    ap.add_argument("--fps", type=int, default=30, help="逐个试开时用的帧率（默认 30）")
    ap.add_argument("--distance-mm", type=float, default=440.0,
                    help="工作距离，用于算 GSD（默认 440，按用户实测）")
    ap.add_argument("--limit", type=int, default=40, help="最多试开几个 profile（默认 40）")
    args = ap.parse_args()

    try:
        import pyrealsense2 as rs
    except Exception as e:                                              # noqa: BLE001
        print("没有 pyrealsense2：%s" % e)
        return 2

    print("═" * 84)
    print("  D435 开流诊断：逐传感器列 profile + 逐个试开")
    print("═" * 84)

    if args.reset:
        print("\n  --reset：硬件复位…")
        try:
            devs = list(rs.context().query_devices())
            if devs:
                devs[0].hardware_reset()
                print("  已发复位命令，等 6 秒…")
                time.sleep(6.0)
            else:
                print("  枚举不到设备，跳过")
        except Exception as e:                                          # noqa: BLE001
            print("  复位失败（继续尝试）：%s" % str(e).strip())

    try:
        devs = list(rs.context().query_devices())
    except Exception as e:                                              # noqa: BLE001
        print("枚举失败：%s" % str(e).strip())
        print("  ★ 受限（低完整性）进程的典型症状，实录 §2.39")
        return 3
    if not devs:
        print("SDK 看不到设备 —— ① USB 3 口 ② 关掉相机 App / Viewer ③ 拔插一次")
        return 3

    dev = devs[0]

    def dinfo(f):
        try:
            return str(dev.get_info(getattr(rs.camera_info, f)))
        except Exception:                                               # noqa: BLE001
            return "?"
    print("\n  设备: %s  SN=%s  固件=%s  USB=%s"
          % (dinfo("name"), dinfo("serial_number"),
             dinfo("firmware_version"), dinfo("usb_type_descriptor")))

    # ── ① 逐传感器列出全部 profile ──────────────────────────────
    print("\n① 各传感器自报的 profile")
    print("  " + "-" * 80)
    try:
        sensors = list(dev.sensors)
    except Exception as e:                                              # noqa: BLE001
        print("  ★ 读 sensors 失败：%s" % str(e).strip())
        print("    —— 这一条本身就是「设备被别人占着」的强信号")
        return 3
    print("  共 %d 个传感器" % len(sensors))

    all_profiles = []          # [(sensor_name, stream_name, fmt, w, h, fps)]
    for s in sensors:
        try:
            sname = str(s.get_info(rs.camera_info.name))
        except Exception:                                               # noqa: BLE001
            sname = "?"
        try:
            profs = list(s.get_stream_profiles())
        except Exception as e:                                          # noqa: BLE001
            print("  [%s] ★ 读 profile 失败：%s" % (sname, str(e).strip()))
            continue
        print("\n  [%s]  %d 个 profile" % (sname, len(profs)))
        by_res = {}
        for p in profs:
            try:
                if not p.is_video_stream_profile():
                    continue
                v = p.as_video_stream_profile()
                try:
                    st = rs.stream(v.stream_type()).name
                except Exception:                                       # noqa: BLE001
                    st = str(v.stream_type())
                try:
                    fmt = rs.format(v.format()).name
                except Exception:                                       # noqa: BLE001
                    fmt = str(v.format())
                key = (st, int(v.width()), int(v.height()), int(v.fps()))
                by_res.setdefault(key, []).append(fmt)
                all_profiles.append((sname, st, fmt, int(v.width()), int(v.height()),
                                     int(v.fps())))
            except Exception:                                           # noqa: BLE001
                continue
        for key in sorted(by_res, key=lambda k: (k[0], k[1], k[2], k[3])):
            st, w, h, fps = key
            print("     %-8s %4dx%-5d @ %-4d  格式: %s"
                  % (st, w, h, fps, "/".join(sorted(set(by_res[key])))))

    color_profs = [(sn, fmt, w, h, f) for (sn, st, fmt, w, h, f) in all_profiles
                   if st == "color"]
    resolutions = sorted(set((w, h) for (_sn, _f, w, h, _fp) in color_profs))
    print("\n  ★ 彩色流可用的**分辨率**一共 %d 种：%s"
          % (len(resolutions), "、".join("%dx%d" % r for r in resolutions) or "（无）"))

    # ── ② 逐个试开 ──────────────────────────────────────────────
    print("\n② 逐个试开（每个 profile 独立开一次流，读完就关）")
    print("  " + "-" * 80)
    print("  %-13s %-8s %-8s %-9s %-9s %-8s %-9s %s"
          % ("分辨率", "格式", "帧率", "fx", "fy", "FOVh°", "GSD", "结果"))
    ok_any = False
    ok_list = []
    tries = 0
    seen_mode = set()
    for (sname, fmt, w, h, fps) in sorted(color_profs, key=lambda t: (t[2] * t[3], t[4])):
        if (w, h) in seen_mode:
            continue                    # 每种分辨率只试第一个格式，避免刷屏
        if fps != int(args.fps):
            continue
        if tries >= int(args.limit):
            break
        tries += 1
        seen_mode.add((w, h))
        r = _try(rs, w, h, int(args.fps))
        if r is None:
            print("  %-13s %-8s %-8d %-9s %-9s %-8s %-9s 打不开（Couldn't resolve / 被占用）"
                  % ("%dx%d" % (w, h), fmt, fps, "-", "-", "-", "-"))
            continue
        intr = r
        fovh = 2.0 * math.degrees(math.atan(intr["width"] / (2.0 * intr["fx"])))
        gsd = float(args.distance_mm) / intr["fx"]
        print("  %-13s %-8s %-8d %-9.2f %-9.2f %-8.1f %-9.4f ✅ 能开"
              % ("%dx%d" % (w, h), fmt, fps, intr["fx"], intr["fy"], fovh, gsd))
        ok_any = True
        ok_list.append((w, h, fmt, fps, intr, fovh, gsd))

    # ── ③ 结论 ──────────────────────────────────────────────────
    print("\n" + "═" * 84)
    if ok_any:
        print("  ★★ 结论：**设备是好的、能开流** —— 不是权限问题，也不是设备被占。")
        print("     能开的分辨率：%s" % "、".join("%dx%d" % (w, h) for (w, h, *_r) in ok_list))
        print("     开不了的分辨率：%s"
              % ("、".join("%dx%d" % r for r in resolutions
                           if r not in set((w, h) for (w, h, *_r) in ok_list)) or "（无）"))
        print("\n     ⇒ 说明**当前相机只暴露了列出来的那几种分辨率**。")
        print("       下一步：① 用能开的那个分辨率干活（可能是 1920x1080）；")
        print("              ② 或者把传感器恢复成「默认 profile 集合」再试 640x480 ——")
        print("                 见下面 ④。")
        if any(w == 1920 for (w, _h, *_r) in ok_list):
            print("\n  ★ 注意：如果只有 1920x1080 能开，那**工程当前的 camera.yaml（640x480）**")
            print("    就必须改，或者先把传感器状态恢复。1920x1080 的视野最宽、GSD 最好，")
            print("    但每帧像素是 640x480 的 6.75 倍 —— 帧率与 KCF 成本都要重测。")
    else:
        print("  ★★ 结论：**一个都开不了** ⇒ 高度怀疑**设备被别的程序占着**。")
        print("     按顺序做：")
        print("       ① 关掉 Windows 设置里那个**相机预览页**（它通过 Frame Server 占着设备）")
        print("       ② 关掉 RealSense Viewer / Windows 相机 App")
        print("       ③ 任务管理器里把所有 `python.exe` 结束掉")
        print("       ④ 加 --reset 再跑一次本脚本")
        print("       ⑤ 拔插一次 USB（数据线 + USB 3 口）")

    # ── ④ 怎么把 sensor 恢复成默认 profile 集合 ──────────────────
    print("\n④ 若怀疑 profile 集合被限制住了，可以试这两条")
    print("   · 让传感器回到默认 profile（SDK 里是 `sensor.get_stream_profiles()` 的")
    print("     `is_default()` 那一组）：")
    print("       python -c \"import pyrealsense2 as rs; d=list(rs.context().query_devices())[0];"
          " [print(s.get_info(rs.camera_info.name),"
          " [str(p.as_video_stream_profile().width())+'x'+"
          "str(p.as_video_stream_profile().height()) for p in s.get_stream_profiles()"
          " if p.is_video_stream_profile() and p.as_video_stream_profile().is_default()])"
          " for s in d.sensors]\"")
    print("   · 或者直接**硬件复位**（会清掉你在 Windows 设置里手调的亮度/对比度/饱和度）：")
    print("       & D:\\miniconda\\envs\\yolov11\\python.exe \"ai_测试工具\\d435_stream_diag.py\" --reset")
    print("═" * 84)
    return 0 if ok_any else 4


def _try(rs, w, h, fps):
    """开一次彩色流并读内参。成功返回 dict，失败返回 None。"""
    pipe = rs.pipeline()
    cfg = rs.config()
    try:
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        prof = pipe.start(cfg)
    except Exception:                                                   # noqa: BLE001
        try:
            pipe.stop()
        except Exception:                                               # noqa: BLE001
            pass
        return None
    try:
        sp = prof.get_stream(rs.stream.color).as_video_stream_profile()
        i = sp.get_intrinsics()
        return {"fx": float(i.fx), "fy": float(i.fy), "ppx": float(i.ppx),
                "ppy": float(i.ppy), "width": int(i.width), "height": int(i.height)}
    except Exception:                                                   # noqa: BLE001
        return None
    finally:
        try:
            pipe.stop()
        except Exception:                                               # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
