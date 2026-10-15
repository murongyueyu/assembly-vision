# -*- coding: utf-8 -*-
"""D435 内参 + 相机控件 对账（只读，不改工程）。

★ 位置：`视觉/ai_测试工具/`（AI 自己的工具区），**不碰用户自己的 `用户存档/`**。
  本脚本是独立的：只用 `pyrealsense2`，**不 import 工程**，所以在哪跑都行。

回答两个具体问题：

**① D435 需要重新做棋盘格标定吗？—— 不需要**
  依据（见 `移植方案_零件分割识别追踪_到D435.md` §3.7）：
  · 彩色内参是**相机出厂标定**，`get_intrinsics()` 直接给；
  · 彩色流经 SDK 输出时是**已校正（rectified）**图像，`intrinsics.model` 就是
    畸变模型 —— **再叠一层 OpenCV 的畸变 = 把校正做两遍**，精度只会更差；
  · 深度也由**固件**用出厂标定算好 ⇒ 工程的 **c2 双目标定整条作废**
    （`camera.yaml` 里 `depth.enabled: false`、`right=None`）；
  · 真正要重跑的是 **c3（料盘单应）** —— 它才决定工程里的 `mm_per_px`。

**② 相机现在的控件到底是什么值？（本轮新增，优先级最高）**
  `camera.yaml:53` 现在是 `controls: {}` ⇒ 相机在跑**自己的自动策略**，
  用户在 Windows 设置里手调的亮度/对比度/饱和度**只写在驱动里**：
    · 工程不会固化它，重跑/换机就可能变回去；
    · ★★ `tools/probe_d435.py --reset` 会做**硬件复位**，把它一起冲掉。
  所以本脚本把 `enable_auto_exposure / exposure / gain /
  enable_auto_white_balance / white_balance / brightness / contrast /
  saturation / sharpness / gamma / …` 全部读出来，
  照 `camera.yaml:47-52` 的模板填进 `realsense.controls` 即可。

★ 已修正两处旧文档的错误（实录 §2.34/§2.38 与
  `docs/对话存档_2026_D435换机.md` 里的 0.795 / 509mm 都作废）：
    · 640x480@30 **不是** 509mm 覆盖，那是**拿 720p 内参等比外推**得来的；
      它是**裁切模式**，实测 fx=603.8 ⇒ 0.596mm/px、覆盖 382mm；
    · 所以**每换一个分辨率都要重读内参**，但这不是"重标"。
  ★ 工位实测补充：拿料盘 bbox 反推 GSD ≈ **0.478**，按 fx 反推工作距离约 **289mm**，
    而不是文档里一直写的 360mm —— **`--distance-mm` 必须实测**。

★ 用法（VSCode 里：打开本文件按 F5，见 `.vscode/launch.json`；
  终端里跑则**工作目录在哪都行**）：
    & D:\\miniconda\\envs\\yolov11\\python.exe "ai_测试工具\\d435_intrinsics_check.py" --distance-mm 289
    & D:\\miniconda\\envs\\yolov11\\python.exe "ai_测试工具\\d435_intrinsics_check.py" --modes 640x480,1280x720
    # 想把结果留档：
    ... --json-out "ai_测试工具\\工位帧\\内参与控件.json"

★ 必须在**不受限**的进程里跑（DSH 沙箱里会报 Mutex_TotalFailure，实录 §2.39）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _read_controls(rs, profile) -> dict:
    """读彩色传感器**当前**的曝光/增益/白平衡值（只读，不改）。

    ★ 为什么值得读：`camera.yaml:53` 现在是 `controls: {}`，
    也就是相机在跑**自己的自动策略**。用户反馈工位不过曝、白平衡也稳，
    所以这里的目的不是"修"，而是**把当前的好状态固化下来**——
    自动策略会随环境光变（白天/晚上、开灯/关灯、有人挡光），
    今天稳不等于明天稳。读出来填进 `realsense.controls` 即可。
    """
    names = ("enable_auto_exposure", "exposure", "gain",
             "enable_auto_white_balance", "white_balance", "backlight_compensation",
             "brightness", "contrast", "gamma", "saturation", "sharpness",
             "hue", "power_line_frequency", "enable_auto_exposure_priority",
             "auto_exposure_priority", "global_time_enabled")
    out = {}
    try:
        sensors = list(profile.get_device().sensors)
    except Exception:                                               # noqa: BLE001
        return out
    # 优先只读 RGB 传感器（exposure/gain 在深度传感器上也有，但改了跟彩色无关）
    target = None
    for s in sensors:
        try:
            nm = str(s.get_info(rs.camera_info.name)).lower()
        except Exception:                                           # noqa: BLE001
            nm = ""
        if "rgb" in nm:
            target = s
            break
    if target is None and sensors:
        target = sensors[0]
    if target is None:
        return out
    for n in names:
        opt = getattr(rs.option, n, None)
        if opt is None:
            continue
        try:
            if not target.supports(opt):
                continue
            v = target.get_option(opt)
            out[n] = int(v) if abs(v - round(v)) < 1e-6 else round(v, 4)
        except Exception:                                           # noqa: BLE001
            continue
    return out


def _list_color_profiles(rs, dev) -> list:
    """枚举相机**自报**的彩色流模式，返回 [(format名, w, h, fps), ...]。

    ★ 为什么要读它：用户终端里跑本脚本时 640x480@30 与 1280x720@30 都报
      `Couldn't resolve requests`（设备本身正常）。先看相机自报的清单，
      就能分清是"模式不存在/被占用"还是"别的原因"。
    ★ 实测踩过（实录 §2.37）：这张清单**列不全** —— 60fps 明明能开却没列出来。
      所以它只用来**参考**，不能拿它当"某模式一定不行"的依据。
    """
    out = []
    try:
        for s in dev.sensors:
            for p in s.get_stream_profiles():
                try:
                    if not p.is_video_stream_profile():
                        continue
                    if p.stream_type() != rs.stream.color:
                        continue
                    v = p.as_video_stream_profile()
                    try:
                        fmt = rs.format(v.format()).name
                    except Exception:                                   # noqa: BLE001
                        fmt = str(v.format())
                    out.append((fmt, int(v.width()), int(v.height()), int(v.fps())))
                except Exception:                                       # noqa: BLE001
                    continue
    except Exception as e:                                              # noqa: BLE001
        print("  （枚举模式清单失败：%s）" % str(e).strip())
        return []
    # 去重 + 按分辨率、帧率排序
    out = sorted(set(out), key=lambda t: (t[1], t[2], t[3], t[0]))
    return out


def _try_mode(rs, w, h, fps, distance_mm, retries: int = 2):
    """开一次流并读内参。成功返回 (intr, model, coeffs, gsd, cover, controls, 实际profile)。

    ★ 为什么要重试：D435 刚被别的进程占用过 / 刚拔插时，
      第一次 `pipe.start()` 会失败，隔一下重来就好（实录 §2.31 的同一类现象）。
    ★ 顺带把**实际生效的 profile** 打出来 —— 有时 `enable_stream` 请求的
      与真正协商到的不一样，只有读回来才知道（实录 §2.5 踩过）。
    """
    import time
    last = ""
    for k in range(max(1, retries)):
        pipe = rs.pipeline()
        cfg = rs.config()
        try:
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, int(fps))
            prof = pipe.start(cfg)
        except Exception as e:                                          # noqa: BLE001
            last = str(e).strip()
            try:
                pipe.stop()
            except Exception:                                           # noqa: BLE001
                pass
            if k + 1 < retries:
                time.sleep(1.0)
            continue
        try:
            sp = prof.get_stream(rs.stream.color).as_video_stream_profile()
            intr = sp.get_intrinsics()
            try:
                model = rs.distortion(intr.model).name
            except Exception:                                           # noqa: BLE001
                model = str(intr.model)
            coeffs = [float(c) for c in intr.coeffs]
            gsd = float(distance_mm) / float(intr.fx) if intr.fx else float("nan")
            cover = (int(intr.width) * gsd, int(intr.height) * gsd)
            try:
                fmt_name = rs.format(sp.format()).name
            except Exception:                                           # noqa: BLE001
                fmt_name = str(sp.format())
            desc = "%s %dx%d@%d" % (fmt_name, sp.width(), sp.height(), sp.fps())
            ctrls = _read_controls(rs, prof)
            return ({"fx": float(intr.fx), "fy": float(intr.fy),
                     "ppx": float(intr.ppx), "ppy": float(intr.ppy),
                     "width": int(intr.width), "height": int(intr.height)},
                    model, coeffs, gsd, cover, ctrls, desc)
        except Exception as e:                                          # noqa: BLE001
            last = str(e).strip()
        finally:
            try:
                pipe.stop()
            except Exception:                                           # noqa: BLE001
                pass
    print("  %-13s 打不开：%s" % ("%dx%d@%d" % (w, h, fps), last))
    return None


def _explain_open_failure(rs, supported: list) -> None:
    """一个模式都开不起来时，把**实测过的排查顺序**打出来。

    ★ 为什么值得单独写：`Couldn't resolve requests` 这个报错**看着像"模式不存在"**，
      但它同样会在"设备被占用 / 刚拔插 / 传感器没回到空闲状态"时出现，
      很容易查错方向（本项目在实录 §2.30~§2.33 连踩四次）。
    """
    print()
    print("  " + "!" * 74)
    print("  ★★ 一个模式都开不起来 —— 按这个顺序查（每条都是实测过的，实录 §2.30~§2.33）：")
    print("     ① **是不是别的程序占着相机？**")
    print("        RealSense Viewer / Windows【相机】App / 之前没退干净的 python 进程")
    print("        —— 都会让 `pipe.start()` 报 Couldn't resolve requests。")
    print("        查：任务管理器里搜 python.exe 与 RealSense，全关掉再跑。")
    print("     ② **硬件复位**（最快的一招，清占用不清链路速率）：")
    print("        & D:\\miniconda\\envs\\yolov11\\python.exe assembly_vision\\tools\\probe_d435.py --reset")
    print("     ③ **拔插一次 USB**（要用数据线、要插 USB 3 口）。")
    print("     ④ 上述都不行时，用**工程自己的源**再试一次（它带 3 次重试）：")
    print("        & D:\\miniconda\\envs\\yolov11\\python.exe ai_测试工具\\capture_site_frame.py")
    print("     ⑤ ★ 注意：`probe_d435.py --reset` 会把**你在 Windows 设置里手调的")
    print("        亮度/对比度/饱和度一起冲掉** —— 复位前先把值抄下来。")
    if supported:
        print("     ⑥ 相机自报的可用模式里有这些（对照一下你要的模式在不在）：")
        seen = []
        for (fmt, w, h, fps) in supported:
            tag = "%dx%d" % (w, h)
            if tag not in seen:
                seen.append(tag)
        print("        %s" % "、".join(seen))
    print("  " + "!" * 74)


def main() -> int:
    ap = argparse.ArgumentParser("d435_intrinsics_check")
    ap.add_argument("--distance-mm", type=float, default=360.0,
                    help="相机到**料盘平面**的实测工作距离（mm）。"
                         "★ 这个数决定了反算出来的 GSD，必须实测、别抄旧值")
    ap.add_argument("--modes", default="640x480,1280x720",
                    help="要读的彩色模式，逗号分隔（D435 彩色只有 1920x1080 / 1280x720 / "
                         "960x540 / 848x480 / 640x480 / 640x360 / 424x240）")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--json-out", default="", help="把结果写成 json（可选）")
    args = ap.parse_args()

    try:
        import pyrealsense2 as rs
    except Exception as e:                                          # noqa: BLE001
        print("没有 pyrealsense2：%s" % e)
        print("  装法：pip install pyrealsense2")
        return 2

    print("═" * 78)
    print("  D435 出厂内参对账   （工作距离 %.0f mm）" % args.distance_mm)
    print("═" * 78)

    modes = []
    for m in str(args.modes).split(","):
        m = m.strip().lower().replace("*", "x")
        if "x" not in m:
            continue
        w, h = m.split("x")[:2]
        modes.append((int(w), int(h)))
    if not modes:
        print("没有可解析的模式：%s" % args.modes)
        return 1

    ctx = rs.context()
    try:
        devs = list(ctx.query_devices())
    except Exception as e:                                          # noqa: BLE001
        print("枚举失败：%s" % e)
        print("  ★ 受限（低完整性）进程会报 Mutex_TotalFailure / No device connected，")
        print("    而普通终端同一命令正常 —— 那是**权限**不是相机（实录 §2.39）。")
        return 3
    if not devs:
        print("SDK 看不到任何 RealSense。检查：① USB 3 口 ② 关掉 RealSense Viewer")
        print("      ③ 拔插一次 ④ 还不行就 python tools/probe_d435.py --reset")
        return 3

    d = devs[0]
    def info(f):
        try:
            return str(d.get_info(getattr(rs.camera_info, f)))
        except Exception:                                           # noqa: BLE001
            return "?"
    print("  设备: %s  SN=%s  固件=%s  USB=%s"
          % (info("name"), info("serial_number"),
             info("firmware_version"), info("usb_type_descriptor")))
    if not info("usb_type_descriptor").startswith("3"):
        print("  " + "!" * 70)
        print("  ★★ USB = %s，不是 3.x —— 这种链路上**一帧都拿不到**（实录 §2.31）。"
              % info("usb_type_descriptor"))
        print("     换 USB 3 口再跑，别去调分辨率。")
        print("  " + "!" * 70)

    out = {"device": info("name"), "serial": info("serial_number"),
           "firmware": info("firmware_version"), "usb": info("usb_type_descriptor"),
           "distance_mm": float(args.distance_mm), "modes": {}}

    # ── ★★ 先枚举**相机自己说支持哪些彩色模式** ──────────────────────
    #   为什么必须这一步（2026-10 实测踩过）：
    #     用户终端里跑本脚本时报 `Couldn't resolve requests`（640x480@30 与 1280x720@30
    #     都打不开），而设备本身是好的（USB=3.2、固件正常、枚举得到）。
    #     `Couldn't resolve requests` 的语义就是"这个 profile 对不上" ——
    #     它**不一定**代表模式不存在（实录 §2.37：`rs-enumerate-devices` 的列表
    #     列不全，60fps 实际能开），所以**先按相机自报的清单试**，
    #     自报清单里没有的就不再徒劳 try（也就不会刷一屏莫名其妙的错）。
    supported = _list_color_profiles(rs, d)
    if supported:
        print("\n  相机自报的彩色模式（format / WxH / fps）：")
        for (fmt, w, h, fps) in supported:
            print("     %-8s %4dx%-4d @ %s" % (fmt, w, h, fps))
        if len(supported) <= 8:
            print("  ★ 只报了 %d 条 —— 实录 §2.37 记过：这张清单**列不全**，"
                  "所以下面仍会按你给的模式挨个试。" % len(supported))
    else:
        print("\n  ★ 读不到相机自报的模式清单（传感器正被占用时会这样）")

    print()
    print("  %-13s %-9s %-9s %-9s %-9s %-10s %-9s %s"
          % ("模式", "fx", "fy", "ppx", "ppy", "畸变模型", "GSD", "覆盖(mm)"))
    print("  " + "-" * 74)

    n_ok, n_fail = 0, []
    for (w, h) in modes:
        res = _try_mode(rs, w, h, int(args.fps), float(args.distance_mm))
        if res is None:
            n_fail.append("%dx%d@%d" % (w, h, args.fps))
            continue
        n_ok += 1
        intr, model, coeffs, gsd, cover, ctrls, prof_desc = res
        print("  %-13s %-9.2f %-9.2f %-9.2f %-9.2f %-10s %-9.4f %.0f x %.0f"
              % ("%dx%d@%d" % (w, h, args.fps), intr["fx"], intr["fy"],
                 intr["ppx"], intr["ppy"], model, gsd, cover[0], cover[1]))
        print("  %-13s 畸变系数 %s" % ("", [round(c, 6) for c in coeffs]))
        if prof_desc:
            print("  %-13s 实际生效的 profile：%s" % ("", prof_desc))
        # ── ★ 把**相机当前的光度状态**读出来（用于固化 realsense.controls）──
        if ctrls:
            print("  %-13s 当前相机控件（用于固化进 camera.yaml 的 realsense.controls）："
                  % "")
            for k, v in ctrls.items():
                print("  %-13s   %-28s = %s" % ("", k, v))
        out["modes"]["%dx%d" % (w, h)] = {
            "fx": intr["fx"], "fy": intr["fy"], "ppx": intr["ppx"], "ppy": intr["ppy"],
            "model": model, "coeffs": coeffs, "profile": prof_desc,
            "gsd_mm_per_px": gsd, "cover_mm": list(cover), "fps": int(args.fps),
            "controls": ctrls,
        }

    if n_fail:
        print("\n  ★ 打不开的模式：%s" % "、".join(n_fail))
        if n_ok == 0:
            _explain_open_failure(rs, supported)

    print()
    print("═" * 78)
    print("  怎么用这份对账")
    print("═" * 78)
    print("  ① **不需要做棋盘格标定**：fx/fy/ppx/ppy 是出厂值，SDK 直接给；")
    print("     彩色流已是校正图（畸变模型见上表），再叠 OpenCV 畸变会做两遍、反而更差。")
    print("  ② **不需要 c2 双目标定**：D435 的深度由固件算好，工程里 `right=None`。")
    print("  ③ ★ **必须重跑 c3（料盘单应）** —— 这才是决定 `TrayMap.mm_per_px()` 的那一步。")
    print("  ④ ★ 上表的 GSD 只作**交叉检验**：")
    print("       跑完 c3 后，用 `TrayMap.mm_per_px(1.0)` 推出来的 mm/px")
    print("       应与上表对应模式的 GSD **相差 < 1%%**；差得多说明单应或工作距离量错了。")
    print("  ⑤ 换分辨率（640x480 ⇄ 1280x720）必须重读本表 + 重跑 c3，")
    print("     但**不能**按像素数外推 —— 640x480 是裁切模式（实录 §2.38）。")
    print("  ⑥ 要做彩色↔深度对齐（`rs.align`）才需要出厂外参，那属 **B 模块**。")

    if args.json_out:
        p = os.path.abspath(args.json_out)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print("\n  已写入: %s" % p)
        except Exception as e:                                      # noqa: BLE001
            print("\n  ★ 写 json 失败: %s" % e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
