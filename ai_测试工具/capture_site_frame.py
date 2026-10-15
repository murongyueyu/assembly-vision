# -*- coding: utf-8 -*-
"""抓一帧**相机当前真实输出**，把该量的都量出来、该存的都存下来。

★ 位置：`视觉/ai_测试工具/`（AI 自己的工具区）。
  产物写到 `ai_测试工具/工位帧/`，**不碰用户自己的 `用户存档/`**。

为什么要有这个脚本（2026-10 实测踩的坑，见
`ai_测试工具/工位帧/工位实测记录.md`）：
  用户发的 `predict.py` 输出里，绿幕像素是 (B=137,G=139,R=0) 的**纯青绿**（H=90,S=255），
  零件是**接近白的粉**（S 只有 24）——
  用工程当前的 brass 区间 `H∈[10,45] S∈[50,180]` 只能切到 **15 px**（等于一个零件都没有）。
  而 Windows 设置窗口里的相机预览，零件是**铜红**的（H 中位 10、R 均值 180），
  同一个区间能切到 **8266 px**。
  ⇒ 两帧的色相/饱和度差得很远，**必须先确定「哪一帧代表相机现在的真实输出」**，
    再谈改阈值，否则改出来的阈值是错的。

本脚本做的事：
  ① 用工程自己的 `make_source(camera.yaml)` 取图（**走的就是 app.py 的取图路径**）
  ② 打印**实际帧尺寸**、内参、以及当前所有可读的相机控件值
  ③ 把这一帧原样存成 PNG（存到 `ai_测试工具\\工位帧\\`）
  ④ 当场跑一遍**当前的**分割并画出掩膜叠加图，一眼看出切没切到零件

★ 用法（VSCode 里：打开本文件按 F5，见 `.vscode/launch.json`；
  或在终端里跑 —— **工作目录在哪都行**，路径由脚本自己算）：
    & D:\\miniconda\\envs\\yolov11\\python.exe "ai_测试工具\\capture_site_frame.py"
    & D:\\miniconda\\envs\\yolov11\\python.exe "ai_测试工具\\capture_site_frame.py" --tag after_lock

★ 必须在**不受限**的进程里跑（DSH 沙箱里会报 Mutex_TotalFailure，实录 §2.39）。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 工作区根 = 视觉\
AV = os.path.join(ROOT, "assembly_vision")
sys.path.insert(0, AV)

OUT_DIR = os.path.join(ROOT, "ai_测试工具", "工位帧")


def imwrite_safe(p, img):
    """★ 中文路径必须走 imencode+tofile，裸 cv2.imwrite 会**静默失败**（实录 §2.36）。"""
    try:
        ok, buf = cv2.imencode(os.path.splitext(p)[1] or ".png", img)
        if not ok:
            return False
        buf.tofile(p)
        return os.path.exists(p) and os.path.getsize(p) > 0
    except Exception as e:                                              # noqa: BLE001
        print("   写盘异常：%s" % e)
        return False


def main() -> int:
    ap = argparse.ArgumentParser("capture_site_frame")
    ap.add_argument("--tag", default="now", help="这一帧的标签（用于文件名）")
    # ★★ 轮廓口径（2026-10 用户实测发现"底座不是圆、立柱只剩一条高光带"）：
    #   问题根因是**镜面高光把同一个零件切成了碎片** ——
    #     底座：brass 抓到「亮顶面」3041px、metal 抓到「暗边+阴影」3539px，两块不重合
    #           ⇒ brass 单独看是**月牙**（圆度 0.26），union 才是**实心圆**（圆度 0.88）
    #     立柱：brass 只抓到中间一条高光带（一样是碎片）
    #     轮盘：brass 本来就完整（用户确认过）—— 所以这个毛病只出在高光强的零件上
    #   ★ 实测这个改动**不引入新假阳性**：union 比 brass 多出的像素
    #     99.7% 落在料盘区内；区外那块（顶部木桌条 2088px）与 brass 抓到的是同一块，
    #     而且它靠 work_roi 挡（工程本来就这么做）。
    #   默认 **off**：工程主链路（`Segmenter.segment()`）还是用 brass，
    #   这里先只改本工具的显示与统计，不擅自改动判定链路。
    ap.add_argument("--parts-mask", choices=["brass", "union"], default="brass",
                    help="零件轮廓用哪个掩膜：brass（现状）| union = brass∪metal（形状更完整）")
    ap.add_argument("--camera", default=os.path.join(AV, "config", "camera.yaml"))
    ap.add_argument("--workpiece", default=os.path.join(AV, "config", "workpiece.yaml"))
    ap.add_argument("--warmup", type=int, default=15, help="丢掉开头几帧（自动曝光要稳定）")
    # ★ 临时试一个白平衡值（不改 camera.yaml）。
    #   为什么需要（2026-10 实测）：锁 AWB 之后色域**又变了一次** ——
    #     锁之前（AE+AWB 自动）：绿幕 H=81、零件暖色 H 中位 29、brass 掩膜 3994px ✅
    #     锁之后（exposure=156, WB=4600）：绿幕 H=87、零件 H=6、brass 掩膜 58px ❌
    #   原因：`4600` 是 AWB **关掉那一刻的读数**，不等于 AWB 真正在用的值，必须扫。
    #   用这个开关可以先逐个值试，找到合适的再写回 camera.yaml。
    ap.add_argument("--wb", type=int, default=None,
                    help="临时把白平衡设成这个色温再抓帧（不改配置文件）")
    ap.add_argument("--exposure", type=int, default=None,
                    help="临时把曝光设成这个值再抓帧（不改配置文件）")
    # ★ 为什么需要它：`pipe.start()` 报 `Couldn't resolve requests` /
    #   `Device or resource busy` 时，最常见的原因是**别的程序占着相机**
    #   （RealSense Viewer / Windows 相机 App / 没退干净的 python 进程）。
    #   `dev.hardware_reset()` 能清**占用**，但清不了**链路速率**——两件事别混（实录 §2.33）。
    #   ★★ 复位会**把你在 Windows 设置里手调的亮度/对比度/饱和度一起冲掉**，
    #      所以复位前先把值抄下来（用 d435_intrinsics_check.py 读）。
    ap.add_argument("--reset", action="store_true",
                    help="先做硬件复位再开流（清设备占用；★ 会冲掉手调的相机控件值）")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    from assemvision.config import load_config                        # noqa: E402
    from assemvision.sources.factory import make_source               # noqa: E402

    cam = load_config(args.camera)
    wp = load_config(args.workpiece)
    print("═" * 78)
    print("  抓工位帧（走工程自己的取图路径）")
    print("═" * 78)

    if args.reset:
        print("\n  --reset：先做硬件复位（清设备占用）…")
        try:
            import pyrealsense2 as rs                                    # noqa: PLC0415
            import time                                                  # noqa: PLC0415
            devs = list(rs.context().query_devices())
            if not devs:
                print("  ★ 枚举不到设备，复位跳过")
            else:
                devs[0].hardware_reset()
                print("  已发复位命令，等 5 秒让设备重新枚举…")
                time.sleep(5.0)
        except Exception as e:                                          # noqa: BLE001
            print("  ★ 复位失败（不影响继续尝试）：%s" % str(e).strip())

    try:
        src = make_source(cam)
        src.open()                       # ★ 必须显式开（见 part_track_demo 的说明）
        if hasattr(src, "apply_locked_controls"):
            src.apply_locked_controls()
        # ★ 临时覆盖（只影响本次抓帧，不写配置）—— 用来扫白平衡/曝光
        if args.wb is not None or args.exposure is not None:
            import pyrealsense2 as rs                                   # noqa: PLC0415
            import time as _t                                           # noqa: PLC0415
            _s = None
            for _x in src.profile.get_device().sensors:
                try:
                    if "rgb" in str(_x.get_info(rs.camera_info.name)).lower():
                        _s = _x
                        break
                except Exception:                                       # noqa: BLE001
                    continue
            if _s is None:
                _s = list(src.profile.get_device().sensors)[0]

            def _set(name, val):
                o = getattr(rs.option, name, None)
                if o is None or not _s.supports(o):
                    print("  ★ 传感器不支持 %s，跳过" % name)
                    return
                _s.set_option(o, float(val))
                print("  [临时] %s <- %s（回读 %.0f）" % (name, val, _s.get_option(o)))

            if args.wb is not None:
                _set("enable_auto_white_balance", 0)
                _set("white_balance", args.wb)
            if args.exposure is not None:
                _set("enable_auto_exposure", 0)
                _set("exposure", args.exposure)
            _t.sleep(0.6)                   # 让新控件生效几帧
            args.warmup = max(args.warmup, 20)
    except Exception as e:                                              # noqa: BLE001
        print("★ 相机打不开：%s: %s" % (type(e).__name__, e))
        print("  排查顺序（每条都是实测过的）：")
        print("   ① **是不是别的程序占着相机？** RealSense Viewer / Windows 相机 App /")
        print("      没退干净的 python 进程 —— 都会让开流失败（实录 §2.33）")
        print("   ② 加 --reset 重跑一次：本脚本 --reset，或")
        print("      & D:\\miniconda\\envs\\yolov11\\python.exe assembly_vision\\tools\\probe_d435.py --reset")
        print("      ★ 复位会**冲掉你在 Windows 设置里手调的相机控件值**，先抄下来")
        print("   ③ 是不是受限（低完整性）进程？实录 §2.39 —— 普通终端能跑就是权限")
        print("   ④ USB 必须是 3.x（实录 §2.31：2.1 链路上**一帧都拿不到**）")
        return 2

    print("  源：%s" % src.name)
    try:
        intr = src.color_intrinsics() if hasattr(src, "color_intrinsics") else None
        if intr:
            print("  内参：fx=%.2f fy=%.2f ppx=%.2f ppy=%.2f  流尺寸 %dx%d"
                  % (intr["fx"], intr["fy"], intr["ppx"], intr["ppy"],
                     int(intr["width"]), int(intr["height"])))
    except Exception:                                                   # noqa: BLE001
        pass

    frame = None
    for i in range(max(1, args.warmup)):
        f = src.read(want_full=False)
        if f is not None and getattr(f, "left", None) is not None:
            frame = f.left
    if frame is None:
        print("★ 一帧都没读到")
        src.close()
        return 3
    src.close()

    h, w = frame.shape[:2]
    print("\n  ★★ 实际帧尺寸 = **%dx%d**（这一步就能确认用的是哪个模式）" % (w, h))
    print("     长宽比 %.4f；D435 彩色模式：1920x1080 / 1280x720 / 960x540 / "
          "848x480 / 640x480 / 640x360 / 424x240" % (w / float(h)))

    # ── 色域体检（与 seg_gamut_report.py 同口径，但用**相机刚出的这一帧**）──
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    b, g, r = (frame[..., 0].astype(np.int32), frame[..., 1].astype(np.int32),
               frame[..., 2].astype(np.int32))
    print("\n  通道均值 B/G/R = %.1f / %.1f / %.1f" % (b.mean(), g.mean(), r.mean()))
    print("  亮度 V 均值 %.1f；近全白(minBGR>=250) **%.3f%%**（报警线 0.5%%），"
          "V>=250 占 %.3f%%"
          % (V.mean(), 100.0 * (frame.min(axis=2) >= 250).mean(),
             100.0 * (V >= 250).mean()))
    hs = S > 150
    if hs.any():
        hh = H[hs]
        print("  高饱和(S>150) %.1f%%；色相 p1/p50/p99 = %.0f / %.0f / %.0f"
              % (100.0 * hs.mean(), np.percentile(hh, 1), np.percentile(hh, 50),
                 np.percentile(hh, 99)))
    warm = (r > b + 25)
    print("  暖色像素(R>B+25) %.2f%%" % (100.0 * warm.mean()))
    if warm.any():
        print("     其 H p5/p50/p95 = %.0f / %.0f / %.0f ；S p50 = %.0f；"
              "R 均值 %.1f"
              % (np.percentile(H[warm], 5), np.percentile(H[warm], 50),
                 np.percentile(H[warm], 95), np.percentile(S[warm], 50), r[warm].mean()))

    # ── 当前分割跑一遍 ──
    from assemvision.core.segment import Segmenter                    # noqa: E402
    from assemvision.core.features import contour_features            # noqa: E402
    seg = Segmenter(wp)
    res = seg.segment(frame, scale=1.0)

    # ★★ 轮廓口径：union = brass ∪ metal（见 --parts-mask 的说明）
    #   为什么要给这个选项：镜面高光会把同一个零件切成碎片 ——
    #     brass 单独看，底座是「月牙」（圆度 0.26），union 才是「实心圆」（0.88）。
    #   用什么口径，后面的连通域统计与叠加图都跟着变（保持自洽）。
    if args.parts_mask == "union" and res.metal_mask is not None:
        parts_mask = cv2.bitwise_or(res.brass_mask or np.zeros_like(res.metal_mask),
                                    res.metal_mask)
        parts_blobs = seg._blobs(parts_mask,
                                 min_area_px=float(seg.brass.get("min_area_px", 500)))
        for _b in parts_blobs:
            _b.features = contour_features(_b.contour, parts_mask)
    else:
        parts_mask = res.brass_mask
        parts_blobs = res.brass_blobs

    n_brass = int(np.count_nonzero(parts_mask)) if parts_mask is not None else 0
    print("\n  当前分割（★ 未给 mm_per_px ⇒ 尺寸闸门静默关闭）")
    print("     零件口径 = **%s**%s"
          % (args.parts_mask,
             "" if args.parts_mask == "brass" else "  （brass ∪ metal，形状更完整）"))
    print("     掩膜 %6d px = %5.2f%%；连通域 %d 个（min_area_px=%s）"
          % (n_brass, 100.0 * n_brass / frame[..., 0].size, len(parts_blobs),
             seg.brass.get("min_area_px")))
    if parts_blobs:
        print("     各块面积：%s" % [int(x.area_px) for x in parts_blobs[:8]])
        for _i, _b in enumerate(parts_blobs[:5]):
            _f = _b.features or {}
            print("       #%d bbox=(%3d,%3d) %3dx%-3d  %5d px  圆度 %.2f  实度 %.2f  伸长 %.2f"
                  % (_i, _b.bbox[0], _b.bbox[1], _b.bbox[2], _b.bbox[3], _b.area_px,
                     _f.get("circularity", -1), _f.get("solidity", -1),
                     _f.get("elongation", -1)))
    print("     glove_bbox=%s  area=%.0f px" % (res.glove_bbox, res.glove_area_px))
    print("     blown_ratio=%.4f  v_mean=%.1f"
          % (res.detail.get("blown_ratio", -1), res.detail.get("v_mean", -1)))

    # ── 存图（原图 + 掩膜叠加）──
    base = os.path.join(OUT_DIR, "site_%s_%dx%d.png" % (args.tag, w, h))
    print("\n  原图：%s" % ("已保存 " + base if imwrite_safe(base, frame) else "★ 保存失败 " + base))

    def ov(mask, color, alpha=0.5):
        out = frame.copy()
        if mask is None:
            return out
        m = mask > 0
        out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        return out

    green = ((H >= 60) & (H <= 100) & (S > 120)).astype(np.uint8) * 255
    metal = ((V >= 150) & (S <= 100) & (green == 0)).astype(np.uint8) * 255
    # 第一栏画**实际用的零件口径**（brass 或 union），并描出真实轮廓
    left = ov(parts_mask, (0, 0, 255))
    for _b in parts_blobs:
        cv2.drawContours(left, [_b.contour], -1, (0, 255, 0), 2)
    vis = np.hstack([left, ov(metal, (0, 255, 255)),
                     ov(((r > b + 25) & (V >= 60)).astype(np.uint8) * 255, (255, 0, 255))])
    for i, t in enumerate(["零件口径=%s + 轮廓" % args.parts_mask, "metal", "warm(R>B+25)"]):
        cv2.putText(vis, t, (i * w + 6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, t, (i * w + 6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    ovp = os.path.join(OUT_DIR, "分析叠加图", "掩膜_%s.png" % args.tag)
    os.makedirs(os.path.dirname(ovp), exist_ok=True)
    print("  掩膜叠加：%s" % ("已保存 " + ovp if imwrite_safe(ovp, vis) else "★ 保存失败"))

    print("\n  判读：把上面这两个文件发给对方即可 ——")
    print("    · `site_%s_*.png` 是**相机现在的真实输出**（未经过任何 UI/截图链路）" % args.tag)
    print("    · `masks_%s.png` 左栏是当前 brass 区间切到的结果" % args.tag)
    print("      ★ **左栏应该把 4 个零件整块盖住**；只盖住零碎像素 = 阈值与当前相机设置不匹配")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
