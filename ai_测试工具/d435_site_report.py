# -*- coding: utf-8 -*-
"""换相机后的「数字对账单」—— 把必须重新实测的量一次量出来。

★ 为什么需要它：
    相机一换，**所有跟像素、色域、尺度有关的数字全部过期**：
      黄铜 HSV 区间（h∈[10,45] s∈[50,180]）、绿幕 H=83、手套 h∈[91,130]、
      GSD 0.8623mm/px、槽位覆盖率、零件特征模板(6) …
    而工程文档记着最惨的一条教训：**过曝时画面只是"看着亮一点"，
    判据链却会静默失效**（黄铜被推向青绿 -> 判成绿垫 -> 面积塌到 1/5，
    轮盘 3200px 掉到 626px，终端只报"扫优失败"）。
    所以换相机后的第一件事**不是**跑算法，而是把下面这些数重新量出来。

本工具就量这些（每个数都直接对应一条要改的配置/标定）：

    ① 分辨率 / 实测帧率 / USB 速率      -> camera.yaml 的 width/height/frame_rate_hint
    ② 过曝% (近全白像素) / V 均值        -> ★ 曝光必须调到过曝 ≈ 0%
    ③ 黄铜色域实测 (H/S/V 的范围)        -> workpiece.yaml 的 segment.brass.*
    ④ 绿幕色域实测                       -> 确认绿幕还落在 h=83 附近；测 s_max 的安全边界
    ⑤ 手套色域实测（把手放画面里）        -> segment.glove.h_min 等
    ⑥ 黄铜连通域最大块等效直径 + 面积     -> 确认"木桌 vs 真零件"的尺寸闸门还成不成立
    ⑦ 料盘 / 工位在画面里占多少像素       -> 决定新的 GSD 与 ROI（要重跑 c3）

用法：
    PY="D:\\miniconda\\envs\\yolov11\\python.exe"
    # ① 有相机：实时量（按 s 采样一次、q 结束并出报告）
    & $PY d435_site_report.py --width 640 --height 480 --fps 30

    # ② 没相机：拿一批图片/工程自带合成图先验证本工具本身
    & $PY d435_site_report.py --images "..\\assembly_vision\\data\\synthetic_seq"

    # ③ 录一段素材离线量（推荐：真机摆好件再量色域）
    & $PY predict.py --record real1.mp4 --seconds 20
    & $PY d435_site_report.py --video real1.mp4

输出：
    终端直接打印对账单；同时把采样帧与叠加图存到本目录 d435_report/
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # 视觉/
AV = os.path.join(ROOT, "assembly_vision")
if os.path.isdir(AV):
    sys.path.insert(0, AV)

OUT = os.path.join(HERE, "d435_report")


# ★ 中文路径：cv2.imwrite 在 Windows 上遇中文路径**静默失败**（只返回 False，不抛异常）。
#   本工作区路径是「视觉/ai_测试工具」，所以必须用工程的安全版本。
def _imwrite(path: str, img) -> bool:
    try:
        from assemvision.imgio import imwrite_safe
        return bool(imwrite_safe(path, img))
    except Exception:                                            # noqa: BLE001
        ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
        if ok:
            buf.tofile(path)
        return bool(ok)


# ── 单个掩膜的分析：面积、块数、最大块等效直径、HSV 分位数 ──────────────
def mask_stats(mask: np.ndarray, hsv: np.ndarray, label: str, min_area: int = 20) -> dict:
    n = int(np.count_nonzero(mask))
    out = {"label": label, "px": n, "blocks": 0, "max_diam_px": 0.0, "max_area_px": 0}
    if n == 0:
        return out
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in cnts if cv2.contourArea(c) >= min_area]
    out["blocks"] = len(big)
    if big:
        a = max(cv2.contourArea(c) for c in big)
        out["max_area_px"] = a
        out["max_diam_px"] = 2.0 * (a / np.pi) ** 0.5
    sel = hsv[mask > 0]
    if sel.size:
        out["H"] = tuple(float(v) for v in np.percentile(sel[:, 0], [5, 50, 95]))
        out["S"] = tuple(float(v) for v in np.percentile(sel[:, 1], [5, 50, 95]))
        out["V"] = tuple(float(v) for v in np.percentile(sel[:, 2], [5, 50, 95]))
    return out


def analyse(bgr: np.ndarray, wp=None) -> dict:
    """对一帧做全套测量，返回一个 dict。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    r = {}
    r["shape"] = bgr.shape
    # ① 过曝：HSV 的 V>=250（近全白）
    r["blown_ratio"] = float(np.count_nonzero(V >= 250)) / float(V.size)
    r["v_mean"] = float(V.mean())
    r["v_p95"] = float(np.percentile(V, 95))

    # ② 用「当前配置的阈值」切一遍，看还成不成立
    if wp is not None:
        def rng(sec, k, d=None):
            return wp.get("segment.%s.%s" % (sec, k), d)
        brass = ((H >= rng("brass", "h_min", 0)) & (H <= rng("brass", "h_max", 45))
                 & (S >= rng("brass", "s_min", 50)) & (S <= rng("brass", "s_max", 180))
                 & (V >= rng("brass", "v_min", 80)))
        glove = ((H >= rng("glove", "h_min", 91)) & (H <= rng("glove", "h_max", 130))
                 & (S >= rng("glove", "s_min", 80)) & (V >= rng("glove", "v_min", 130)))
        green = ((H >= 70) & (H <= 100) & (S >= 180))          # 绿幕附近
        r["brass"] = mask_stats(brass.astype(np.uint8) * 255, hsv, "当前黄铜阈值")
        r["glove"] = mask_stats(glove.astype(np.uint8) * 255, hsv, "当前手套阈值")
        r["green"] = mask_stats(green.astype(np.uint8) * 255, hsv, "绿幕(H70~100,S>=180)")

    # ③ 高饱和像素的色相直方图 —— 重新定 h_min/h_max 的直接依据
    sel = (S >= 60) & (V >= 40)
    if int(np.count_nonzero(sel)) > 100:
        hh = np.bincount(H[sel].ravel(), minlength=180)
        top = np.argsort(hh)[::-1][:5]
        r["hue_modes"] = [(int(t), float(H[sel].size and hh[t] / H[sel].size)) for t in top]
    else:
        r["hue_modes"] = []
    return r


def print_report(reports: list, wp=None) -> None:
    print("\n" + "=" * 78)
    print(" D435 换机对账单")
    print("=" * 78)
    if not reports:
        print("  没有样本")
        return
    shapes = {tuple(x["shape"]) for x in reports}
    blown = [x["blown_ratio"] for x in reports]
    vmean = [x["v_mean"] for x in reports]
    print("\n① 画面与曝光")
    print("   分辨率        : %s" % ", ".join("%dx%d" % (s[1], s[0]) for s in shapes))
    print("   过曝%%(V>=250) : 中位 %.3f%%   最大 %.3f%%   %s"
          % (100 * float(np.median(blown)), 100 * max(blown),
             "✓ 可用" if max(blown) < 0.005 else "★ 太高！黄铜会被判成绿垫，判据会静默失效 —— 必须先调曝光"))
    print("   V 均值        : 中位 %.1f（工程里 HBV 的实测是 -6.0 曝光下 216）" % float(np.median(vmean)))
    print("   目标          : 过曝 ≈ 0%，V 均值 100~160 之间较稳")

    if wp is None:
        return

    def med(section, field):
        vals = [x[section][field] for x in reports if section in x]
        return float(np.median(vals)) if vals else 0.0

    print("\n② 用【当前 workpiece.yaml 的阈值】切出来的结果")
    for sec, name in (("brass", "黄铜"), ("green", "绿幕"), ("glove", "手套")):
        px = med(sec, "px")
        d = med(sec, "max_diam_px")
        blk = med(sec, "blocks")
        print("   %-4s 像素中位 %8.0f   最大块 Ø%6.1fpx  块数中位 %.0f" % (name, px, d, blk))
    b = med("brass", "px")
    g = med("green", "px")
    total_px = float(reports[0]["shape"][0] * reports[0]["shape"][1])
    print("   ★ 判读：黄铜像素应该在 **零件大小量级**（HBV 上真零件最大 3300px、木桌 31795px）。")
    print("     当前黄铜占全画面 %.1f%%。" % (100.0 * b / max(1.0, total_px)))
    print("     若占比 >30% → 木桌/墙被误判了，这是换机后最常见的问题，")
    print("     靠调阈值挡不住（木桌 H/S 与黄铜本就重叠），必须靠 work_roi「位置即身份」。")
    if g < 100:
        print("   ⚠ 绿幕像素≈0：绿幕不在画面里，或者绿幕色相已经漂出 H70~100。")
        print("     绿幕 H 在 HBV 上实测是 **83**；D435 若漂到别处，s_max=180 这个边界要重定。")

    print("\n③ 高饱和像素的色相众数（重新定 h_min/h_max 的依据）")
    agg: dict = {}
    for x in reports:
        for h, f in x.get("hue_modes", []):
            agg[h] = agg.get(h, 0.0) + f
    for h, f in sorted(agg.items(), key=lambda kv: -kv[1])[:5]:
        print("   H = %3d   占比 %.1f%%" % (h, 100 * f / max(1, len(reports))))
    print("   ★ 黄铜在 HBV 上实测 H 上限 45、绿幕 H=83。若这里出现 45~75 之间的强峰，")
    print("     要小心：那正是「黄铜被过曝推向青绿」的区间。")

    print("\n④ 接下来必须做的（顺序不能换）")
    print("   1. 调曝光直到本报告 ① 的过曝 ≈ 0%（RealSense: rs.option.exposure）")
    print("   2. 按 ③ 的实测重定 segment.brass.h_min/h_max，按 ② 重定 s_max")
    print("   3. 重跑 c3 料盘单应 -> 拿到新的 GSD 与槽位像素坐标")
    print("   4. 重跑 c4 槽位基线 / c5 装配位基线 / c6 零件特征模板")
    print("   ★ 顺序不能换：先曝光（否则后面量的全是错的），再色域，再几何。")


def main() -> int:
    ap = argparse.ArgumentParser("d435_site_report")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--serial", default="")
    ap.add_argument("--samples", type=int, default=8, help="实时模式采样几帧")
    ap.add_argument("--lock-exposure", type=int, default=None, help="锁手动曝光后采样")
    ap.add_argument("--images", default="", help="离线：一个图片目录")
    ap.add_argument("--video", default="", help="离线：一个视频文件")
    ap.add_argument("--max-frames", type=int, default=60, help="离线最多读多少帧")
    ap.add_argument("--workpiece", default=os.path.join(AV, "config", "workpiece.yaml"))
    ap.add_argument("--no-save", action="store_true", help="不存采样帧")
    args = ap.parse_args()

    wp = None
    if os.path.exists(args.workpiece):
        try:
            from assemvision.config import load_config
            wp = load_config(args.workpiece)
            print("工件配置: %s" % wp.path)
        except Exception as e:                                   # noqa: BLE001
            print("（工件配置读不了，只做基础测量：%s）" % e)

    reports = []
    os.makedirs(OUT, exist_ok=True)

    # ── 离线：图片目录 ────────────────────────────────────────────
    if args.images:
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        files = sorted(os.path.join(args.images, f) for f in os.listdir(args.images)
                       if f.lower().endswith(exts))[:args.max_frames]
        print("离线图片 %d 张" % len(files))
        from assemvision.imgio import imread_safe
        for i, p in enumerate(files):
            img = imread_safe(p)
            if img is None:
                continue
            r = analyse(img, wp)
            r["file"] = p
            reports.append(r)
            if i < 3 and not args.no_save:
                _imwrite(os.path.join(OUT, "sample_%02d.png" % i), img)
        print_report(reports, wp)
        return 0

    # ── 离线：视频 ────────────────────────────────────────────────
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print("✗ 打不开视频 %s" % args.video)
            return 1
        i = 0
        while i < args.max_frames:
            ok, img = cap.read()
            if not ok:
                break
            r = analyse(img, wp)
            r["file"] = "%s#%d" % (args.video, i)
            reports.append(r)
            if i < 3 and not args.no_save:
                _imwrite(os.path.join(OUT, "sample_%02d.png" % i), img)
            i += 1
        cap.release()
        print("离线视频取 %d 帧" % i)
        print_report(reports, wp)
        return 0

    # ── 实时：D435 ───────────────────────────────────────────────
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("✗ 没装 pyrealsense2（或者用 --images / --video 走离线模式）")
        return 2

    devs = []
    last_err = None
    for attempt in range(6):
        try:
            devs = list(rs.context().query_devices())
            break
        except RuntimeError as e:                                # noqa: BLE001
            last_err = e
            # ★ 实测：相机刚插回来时 query_devices 会抛
            #   HResult 0x800703e3「由于线程退出或应用程序请求，已中止 I/O 操作」
            #   或 CreateNamedMutex returned Mutex_TotalFailure ——
            #   都是**等一两秒重试就好**，不是相机坏了。
            print("  设备枚举第 %d 次失败：%s（1.5 秒后重试）" % (attempt + 1, str(e).strip()))
            time.sleep(1.5)
    if not devs:
        print("✗ SDK 看不到相机（重试 6 次都失败）。最后一次错误：%s" % last_err)
        print("  检查：插好没 / 关掉 RealSense Viewer 与相机 App / 换 USB3 口 / 拔插一次复位")
        print("  （想先验证本工具：加 --images <图片目录>）")
        return 1
    serial = args.serial or devs[0].get_info(rs.camera_info.serial_number)
    try:
        usb = devs[0].get_info(rs.camera_info.usb_type_descriptor)
    except Exception:                                            # noqa: BLE001
        usb = "?"
    print("相机 %s SN=%s USB=%s" % (devs[0].get_info(rs.camera_info.name), serial, usb))

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    try:
        profile = pipe.start(cfg)
    except Exception as e:                                       # noqa: BLE001
        print("✗ 开流失败: %s（试 --width 640 --height 480 --fps 30）" % e)
        return 1

    if args.lock_exposure is not None:
        try:
            for s in profile.get_device().sensors:
                if s.supports(rs.option.enable_auto_exposure):
                    s.set_option(rs.option.enable_auto_exposure, 0)
                if s.supports(rs.option.exposure):
                    s.set_option(rs.option.exposure, float(args.lock_exposure))
            print("已锁手动曝光 = %s" % args.lock_exposure)
        except Exception as e:                                   # noqa: BLE001
            print("曝光设置失败:", e)

    print("\n采样 %d 帧…（把零件/手套按需要摆进画面，摆好后再按需要重跑本工具）" % args.samples)
    for _ in range(15):
        pipe.wait_for_frames(5000)

    t0 = time.time()
    for i in range(args.samples):
        fs = pipe.wait_for_frames(5000)
        cf = fs.get_color_frame()
        if not cf:
            continue
        img = np.asanyarray(cf.get_data())
        reports.append(analyse(img, wp))
        if i < 3 and not args.no_save:
            p = os.path.join(OUT, "live_%02d.png" % i)
            _imwrite(p, img)
    dt = time.time() - t0
    pipe.stop()
    print("采样 %d 帧用了 %.2f 秒（≈ %.1f fps）" % (len(reports), dt, len(reports) / max(1e-6, dt)))

    print_report(reports, wp)
    if not args.no_save:
        print("\n采样帧已存到: %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
