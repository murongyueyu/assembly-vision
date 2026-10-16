# -*- coding: utf-8 -*-
"""槽位可见性判定器 —— 一次抓帧，把「槽位检测该用什么机制」量成结论。

═══════════════════════════════════════════════════════════════════════
★ 这个工具存在的唯一理由（P0 闸门）
═══════════════════════════════════════════════════════════════════════
用户要求「基于之前调的相机的所有参数，检测料盘的槽位在哪」。

槽位能被看见的机制只有三种，而**它们谁是主力必须实测**：
  ① **亮度/色域** —— 黑槽 vs 黑盘面。已在用户特写上否掉（亮度差 0.5~12 灰阶、会反号）
  ② **纹理**     —— 槽底平滑 vs 盘面有打印层纹。特写(0.164mm/px)上是 7~12 倍，
                    ★★ 但**工位是 0.7287mm/px，层纹只有 0.27px（亚像素）**；
                    用正交化空盘图预演得 **1.27 倍** ⇒ **大概率不成立，必须实测**
  ③ **深度**     —— 槽是凹的（pocket_depth 2mm）。D435 在 440mm 处能分辨 ~1mm 级
                    ⇒ ★ **最可能的主力**，而且与颜色/纹理完全无关

本工具一次把 ①②③ 全量出来，并给出**明确的选型结论**。

═══════════════════════════════════════════════════════════════════════
用法（★ 必须在你自己的终端跑 —— 沙箱里打不开相机）
═══════════════════════════════════════════════════════════════════════
    Set-Location "D:\\university resource\\大三上\\视觉"
    & D:\\miniconda\\envs\\yolov11\\python.exe "ai_测试工具\\capture_pocket_frame.py"

★ 抓图要求：
    · **料盘上不放零件**（空盘）—— 零件会把槽盖住，量不到槽底
    · 四条边都进画面
    · 跑完会存 3 张图到 `ai_测试工具/工位帧/原图/`：
        `*_color.png`（彩色原图）
        `*_depth.png`（深度伪彩，肉眼看槽是不是凹的）
        `*_sample.png`（★ **采样点叠加图** —— 先看这张，确认采样点落在槽里）
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "assembly_vision"))

from assemvision.config import load_config                      # noqa: E402
from assemvision.core.roi import TrayMap                        # noqa: E402
from assemvision.core.tray_detect import detect_tray            # noqa: E402
from assemvision.imgio import imwrite_safe                      # noqa: E402

OUT_DIR = os.path.join(HERE, "工位帧", "原图")


# ──────────────────────────────────────────────────────────────
# 采样点：**必须由槽位几何算出来**，不能手挑（挑错过，见下）
#   ★ 教训（本工具 v1）：盘面参考点手挑了 (80,80)，而那落在轮盘槽里 ⇒ 量出"比值 0.99"
#     这种把结论搞反的数。所以现在：槽内点 = 槽心；盘面点 = **算法挑**，
#     要求"距所有槽心 > 槽半径 + 余量"，且在最靠近料盘中心的那块空地上。
# ──────────────────────────────────────────────────────────────
def _slot_radius_mm(s: dict) -> float:
    if s.get("type") == "circle":
        return float(s["diameter"]) / 2.0
    if s.get("type") == "capsule":
        # 胶囊用"半宽"当半径（保守：只在中间那条采样）
        return float(s["width"]) / 2.0
    return 5.0


def pick_surface_points(tray: TrayMap, n: int = 4) -> list:
    """自动挑「盘面」采样点：离所有槽足够远、且在料盘内。"""
    size = np.asarray(tray.cfg.get("tray.size", [160.0, 160.0]), float)
    slots = [(np.asarray(s["center"], float), _slot_radius_mm(s)) for s in tray.slots]
    cand = []
    step = 4.0
    for y in np.arange(step, size[1] - step + 1e-9, step):
        for x in np.arange(step, size[0] - step + 1e-9, step):
            p = np.array([x, y], float)
            dmin = min(float(np.linalg.norm(p - c)) - r for c, r in slots)
            if dmin >= 8.0:                      # 距所有槽至少 8mm 余量
                cand.append((dmin, p))
    if not cand:
        return []
    cand.sort(key=lambda t: -t[0])
    # 取最"空"的若干点，并要求彼此分散（贪心最远点）
    picked = [cand[0][1]]
    while len(picked) < n and len(picked) < len(cand):
        best, bd = None, -1.0
        for _d, p in cand:
            dd = min(float(np.linalg.norm(p - q)) for q in picked)
            if dd > bd:
                bd, best = dd, p
        if best is None or bd < 15.0:
            break
        picked.append(best)
    return picked


def build_provisional_traymap(frame_bgr: np.ndarray, tol: float = 0.45):
    """用**料盘轮廓**临时建一个 tray-mm ↔ 像素 映射（c3 过期时也能采样）。

    ★★ 必须**带旋转**（2026-10 rev.3r-10，真帧实测踩过）：
      第一版把料盘当**正放**（纯平移 + 缩放）⇒ 真帧里料盘转了 **−10.4°**，
      于是"盘面参考点"**四个全落到料盘外的绿幕上**，
      量出来的"相对盘面"是**绿幕的深度**（−11.5/−12.5/−3.5/−2.5mm），
      **整列不可用**。而"槽内"点因为靠近料盘中心、偏移小，反倒蒙对了。
      ⇒ 现在按检测到的**中心 + 朝向**做仿射（旋转 + 缩放 + 平移）。

    ★ 尺度用**标称 160mm / 实测 GSD** 定，**不用检测边长**：
      检测边长是 minAreaRect 的边长，形态学闭运算 + 阴影会把它撑大约 2%
      （真帧实测 224.2px × 0.7287 = 163.4mm vs 标称 160mm）。
      ★ 这条是"已知偏置"，`pose_from_tray_detect` 里也写明过同一件事。

    ★★ 本映射**只用于放采样点**（统计量对 5~10mm 偏移不敏感），
      **不是**判定链用的标定 —— 那仍然必须重跑 c3。
    """
    det = detect_tray(frame_bgr)
    if not det.ok or det.center_px is None:
        return None, det
    ang = float(det.angle_deg)
    # ★ minAreaRect 的角：θ=0 时两个框边分别沿 x/y 轴；料盘转了 −10.4° 时它报 79.6。
    #   屏幕系是 y 向下，所以实际旋转量取 **−θ**（与 `pose_from_tray_detect` 同一约定）。
    return (np.asarray(det.center_px, float), float(GSD_MM_PER_PX), -ang), det


def mm_to_px_prov(p, center_px, scale):
    """tray-mm → 像素（**带旋转**：绕料盘中心转，再按 GSD 缩放）。

    ★ 参数语义（与 `_pose_from_depth` 保持一致，别再自创）：
      · `center_px` = 料盘中心的像素坐标
      · `scale`      = **mm/px**（GSD），**不是** px/mm
      · `angle_deg`  = tray-mm 的 x 轴相对图像 x 轴的旋转（度，屏幕系 y 向下）
    """
    cx, cy, ang = center_px if len(np.asarray(center_px)) == 3 else (center_px[0], center_px[1], 0.0)
    mmpp = float(scale)
    u = (np.asarray(p, float) - np.array([80.0, 80.0])) / mmpp
    c, s = np.cos(np.radians(float(ang))), np.sin(np.radians(float(ang)))
    return (float(cx + u[0] * c - u[1] * s), float(cy + u[0] * s + u[1] * c))


def local_stats(gray: np.ndarray, mask_ok: np.ndarray, cx: float, cy: float,
                r_px: float, win: int):
    h, w = gray.shape[:2]
    x0, x1 = int(max(0, cx - r_px)), int(min(w, cx + r_px))
    y0, y1 = int(max(0, cy - r_px)), int(min(h, cy + r_px))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    sl = (slice(y0, y1), slice(x0, x1))
    roi = gray[sl].astype(np.float32)
    ok = mask_ok[sl]
    if int(ok.sum()) < 20:
        return None
    gf = gray.astype(np.float32)
    m = cv2.boxFilter(gf, -1, (win, win))
    m2 = cv2.boxFilter(gf * gf, -1, (win, win))
    ls = np.sqrt(np.maximum(0.0, m2 - m * m))[sl]
    lap = cv2.Laplacian(roi, cv2.CV_32F)
    return {"mean": float(roi[ok].mean()),
            "lstd": float(ls[ok].mean()),
            "lap": float(lap[ok].var()),
            "n": int(ok.sum())}


def rim_gradient(gray: np.ndarray, mask_ok: np.ndarray, cx: float, cy: float,
                 r_px: float) -> float:
    """沿过槽心的水平线，在槽口附近找最大 |dI/dx|（槽口可见性的判据）。"""
    h, w = gray.shape[:2]
    y = int(round(cy))
    x0, x1 = int(max(0, cx - 1.6 * r_px)), int(min(w, cx + 1.6 * r_px))
    if x1 - x0 < 11 or y < 0 or y >= h:
        return 0.0
    line = cv2.GaussianBlur(gray[y:y + 1, x0:x1], (5, 1), 0).astype(np.float32)
    grad = np.abs(np.diff(line, axis=1))[0]
    ok = mask_ok[y, x0:x1 - 1]
    if not np.any(ok):
        return 0.0
    return float(grad[ok].max())


def depth_stats(depth_mm: np.ndarray, cx: float, cy: float, r_px: float):
    """槽内/盘面的深度统计 + 局部噪声估计。"""
    h, w = depth_mm.shape[:2]
    x0, x1 = int(max(0, cx - r_px)), int(min(w, cx + r_px))
    y0, y1 = int(max(0, cy - r_px)), int(min(h, cy + r_px))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    roi = depth_mm[y0:y1, x0:x1]
    v = roi[roi > 0]
    if v.size < 20:
        return None
    med = float(np.median(v))
    # 噪声：残差对中值的 MAD -> sigma 估计
    mad = float(np.median(np.abs(v - med)))
    return {"median": med, "mad": mad, "sigma": 1.4826 * mad, "n": int(v.size),
            "valid": float(v.size) / float(roi.size)}


def report_color(color: np.ndarray, tray: TrayMap, center_px, scale: float) -> np.ndarray:
    """量 ①亮度 + ②纹理，返回画了采样点的叠加图。"""
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    # 掩膜：排除绿幕（高饱和绿）与过曝
    green = ((hsv[:, :, 0] >= 60) & (hsv[:, :, 0] <= 100) & (hsv[:, :, 1] >= 120))
    blown = gray >= 250
    mask_ok = ~(green | blown)

    mmpp = float(scale) if scale > 1e-9 else 0.7287     # ★ scale 现在是 **mm/px**
    win = max(3, int(round(2.0 / mmpp)) | 1)            # ★ 窗口按 mm/px 换算
    print("   （本帧 GSD ≈ %.4f mm/px；纹理窗口 %d px ≈ 2mm）" % (mmpp, win))
    print("\n" + "=" * 74)
    print("① + ② 彩色机制：亮度 / 纹理")
    print("=" * 74)
    print("%-14s %10s %12s %14s %12s" % ("采样点", "亮度mean", "局部std", "Laplacian", "槽口梯度"))
    ins, outs = [], []
    draw = color.copy()
    for s in tray.slots:
        r_mm = _slot_radius_mm(s)
        p = mm_to_px_prov(s["center"], center_px, scale)
        r_px = max(2.0, 0.45 * r_mm / mmpp)              # ★ mm/px ⇒ 半径按 mm÷GSD
        st = local_stats(gray, mask_ok, p[0], p[1], r_px, win)
        gr = rim_gradient(gray, mask_ok, p[0], p[1], r_mm / mmpp)
        if st:
            ins.append(st)
            print("%-14s %10.1f %12.2f %14.1f %12.1f"
                  % (s["id"], st["mean"], st["lstd"], st["lap"], gr))
        cv2.circle(draw, (int(p[0]), int(p[1])), int(max(3, r_px)), (0, 0, 255), 1)
        cv2.circle(draw, (int(p[0]), int(p[1])), 2, (255, 255, 0), -1)
    for i, q in enumerate(pick_surface_points(tray, 4), 1):
        p = mm_to_px_prov(q, center_px, scale)
        r_px = max(2.0, 6.0 / mmpp)                       # ★ 同上：6mm ÷ GSD
        st = local_stats(gray, mask_ok, p[0], p[1], r_px, win)
        if st:
            outs.append(st)
            print("%-14s %10.1f %12.2f %14.1f %12s"
                  % ("盘面#%d" % i, st["mean"], st["lstd"], st["lap"], "-"))
        cv2.circle(draw, (int(p[0]), int(p[1])), 4, (0, 255, 255), 2)

    if ins and outs:
        im_ = float(np.mean([s["mean"] for s in ins]))
        om_ = float(np.mean([s["mean"] for s in outs]))
        ist = float(np.mean([s["lstd"] for s in ins]))
        ost = float(np.mean([s["lstd"] for s in outs]))
        ilp = float(np.mean([s["lap"] for s in ins]))
        olp = float(np.mean([s["lap"] for s in outs]))
        print("\n  ⇒ 亮度差(盘面-槽内) = %+6.1f 灰阶" % (om_ - im_))
        print("  ⇒ 纹理比 局部std = %.2f×   Laplacian = %.2f×"
              % (ost / max(1e-6, ist), olp / max(1e-6, ilp)))
        print("  ★ 判据：|亮度差| > 15 灰阶 ⇒ 亮度可用；纹理比 > 3 ⇒ 纹理可用；")
        print("           两者都不满足 ⇒ **彩色机制不成立，必须用深度**")
    return draw


def analyze_offline(path: str, tag: str, save_sample: bool) -> int:
    """★ 离线模式：不开相机，直接分析一张已有的彩色图。"""
    from assemvision.imgio import imread_safe
    color = imread_safe(path)
    if color is None:
        print("★ 读不到：%s" % path)
        return 1
    if color.ndim == 2:
        color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
    print("=" * 74)
    print("离线分析: %s  %s" % (os.path.basename(path), "x".join(map(str, color.shape))))
    print("=" * 74)

    tray = TrayMap(load_config(os.path.join(ROOT, "assembly_vision", "config",
                                            "workpiece.yaml")))
    prov, det = build_provisional_traymap(color)
    if prov is None:
        print("★★ 没检测到料盘轮廓（%s）⇒ 无法定位采样点。" % det.reason)
        return 1
    center_px, scale = prov
    print("临时映射：料盘中心 px=(%.1f, %.1f)、边长 %.1f px ⇒ %.4f px/mm"
          % (center_px[0], center_px[1], float(np.mean(det.size_px)), scale))
    draw = report_color(color, tray, center_px, scale)
    if save_sample:
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, "%s_sample.png" % tag)
        print("\n采样点叠加图: %s（%s）" % (out, "成功" if imwrite_safe(out, draw) else "★ 失败"))
    print("\n（离线模式没有深度数据 ⇒ 跳过 ③）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser("capture_pocket_frame")
    ap.add_argument("--tag", default="pocketcheck", help="文件名标签")
    ap.add_argument("--warmup", type=int, default=15, help="丢掉开头几帧")
    ap.add_argument("--seconds", type=float, default=2.0, help="稳定多久再抓")
    ap.add_argument("--no-depth", action="store_true", help="只抓彩色（不开深度流）")
    ap.add_argument("--save-npy", action="store_true", default=True,
                    help="★ 存**原始深度**（float32 mm 的 .npy）—— 默认开。"
                         "没有它就没法定 step_mm：伪彩图是有损可视化的，量不准")
    ap.add_argument("--no-npy", dest="save_npy", action="store_false",
                    help="不存原始深度")
    ap.add_argument("--image", default="", help="★ 离线模式：直接分析一张已有的彩色图"
                                                "（不开相机；深度部分自动跳过）")
    ap.add_argument("--save-sample", action="store_true",
                    help="离线模式下也写采样点叠加图")
    args = ap.parse_args()

    if args.image:
        return analyze_offline(args.image, args.tag, args.save_sample)

    try:
        import pyrealsense2 as rs
    except Exception as e:                                   # noqa: BLE001
        print("★ 没有 pyrealsense2：%s" % e)
        return 1

    cam = load_config(os.path.join(ROOT, "assembly_vision", "config", "camera.yaml"))
    W = int(cam.get("realsense.width", 640))
    H = int(cam.get("realsense.height", 480))
    FPS = int(cam.get("realsense.fps", 30))
    print("=" * 74)
    print("槽位可见性判定（P0 闸门）  %dx%d@%d" % (W, H, FPS))
    print("=" * 74)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    use_depth = not args.no_depth
    if use_depth:
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    try:
        prof = pipe.start(cfg)
    except Exception as e:                                   # noqa: BLE001
        print("★ 开流失败：%s" % e)
        print("  · 先看是不是**设备被占用**（Windows 设置里的相机预览页会独占设备）")
        print("  · 再看是不是**受限进程**（要在普通终端跑）")
        return 1
    try:
        dev = prof.get_device()
        usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        print("设备: %s   USB %s" % (dev.get_info(rs.camera_info.name), usb))
        if str(usb) != "3.2":
            print("  ★★ USB 不是 3.2 —— 实测 USB 2.1 链路上**一帧都拿不到**")

        colorizer = rs.colorizer() if use_depth else None
        align = rs.align(rs.stream.color) if use_depth else None
        for _ in range(max(0, args.warmup)):
            pipe.wait_for_frames()
        t_end = time.time() + max(0.0, args.seconds)
        frames = None
        while time.time() < t_end:
            frames = pipe.wait_for_frames()
    finally:
        pass

    if frames is None:
        print("★ 没拿到帧")
        pipe.stop()
        return 1

    if use_depth:
        frames = align.process(frames)
    cf = frames.get_color_frame()
    if not cf:
        print("★ 没有彩色帧")
        pipe.stop()
        return 1
    color = np.asanyarray(cf.get_data())
    depth_mm = None
    depth_vis = None
    depth_scale = None
    if use_depth:
        df = frames.get_depth_frame()
        if df:
            raw = np.asanyarray(df.get_data())
            # ★★ 必须读 depth_scale 才算得出 mm（z16 的原始值**不是** mm）。
            #   实测踩过：把原始值当 mm / 按错 scale 倒推 / 推出 uint16 装不下的值，
            #   连错三次（实录 2.46.3 第 9~11 次）。
            try:
                depth_scale = float(prof.get_device().first_depth_sensor().get_depth_scale())
            except Exception:                                      # noqa: BLE001
                depth_scale = 1.0
            depth_mm = raw.astype(np.float32) * float(depth_scale)
            depth_mm[depth_mm <= 0] = 0.0          # ★ 0 = 无返回（全工程约定）
            depth_vis = np.asanyarray(colorizer.colorize(df).get_data())
            print("深度: 原始 dtype=%s  depth_scale=%.6f ⇒ 单位 mm；"
                  "有效 %.1f%%、中位 %.1fmm"
                  % (raw.dtype, depth_scale, 100.0 * (depth_mm > 0).mean(),
                     float(np.median(depth_mm[depth_mm > 0]))
                     if (depth_mm > 0).any() else 0.0))
    # 内参（★ 随分辨率变，必须现读）
    intr = cf.profile.as_video_stream_profile().get_intrinsics()
    print("彩色内参: fx=%.2f fy=%.2f ppx=%.2f ppy=%.2f  畸变=%s"
          % (intr.fx, intr.fy, intr.ppx, intr.ppy, intr.model))
    pipe.stop()

    os.makedirs(OUT_DIR, exist_ok=True)
    okc = imwrite_safe(os.path.join(OUT_DIR, "%s_color.png" % args.tag), color)
    print("\n彩色原图: %s（%s）" % (os.path.join(OUT_DIR, "%s_color.png" % args.tag),
                                   "成功" if okc else "★ 失败"))
    if depth_vis is not None:
        okd = imwrite_safe(os.path.join(OUT_DIR, "%s_depth.png" % args.tag), depth_vis)
        print("深度伪彩: %s（%s）" % (os.path.join(OUT_DIR, "%s_depth.png" % args.tag),
                                      "成功" if okd else "★ 失败"))
    # ★★ 原始深度必须单独存一份 —— 伪彩是**有损可视化**，
    #   量槽深 / 定 step_mm 只能靠原始 z16（本工具第一版只存了伪彩，
    #   结果真帧到手后无法定阈值，只能再抓一次）。
    if depth_mm is not None and args.save_npy:
        npy = os.path.join(OUT_DIR, "%s_depth_mm.npy" % args.tag)
        try:
            np.save(npy, depth_mm)
            print("★★ 原始深度: %s（float32 mm，0=无返回；**定阈值靠它**）" % npy)
            meta = os.path.join(OUT_DIR, "%s_depth_meta.txt" % args.tag)
            with open(meta, "w", encoding="utf-8") as fh:
                fh.write("depth_scale=%.8f\n" % (depth_scale or 1.0))
                fh.write("unit=mm\ninvalid=0\nshape=%s\n" % (depth_mm.shape,))
                fh.write("color_size=%s\n" % (color.shape,))
                fh.write("intr fx=%.4f fy=%.4f ppx=%.4f ppy=%.4f\n"
                         % (intr.fx, intr.fy, intr.ppx, intr.ppy))
            print("   元数据: %s" % meta)
        except Exception as e:                                     # noqa: BLE001
            print("   ★ 原始深度保存失败：%s" % e)

    # ── 建临时映射 + 挑采样点 ──
    tray = TrayMap(load_config(os.path.join(ROOT, "assembly_vision", "config",
                                            "workpiece.yaml")))
    prov, det = build_provisional_traymap(color)
    if prov is None:
        print("\n★★ 没检测到料盘轮廓（%s）⇒ 无法定位采样点。" % det.reason)
        print("   请确认：空盘、四条边都进画面、绿幕背景。")
        return 1
    center_px, scale = prov
    print("\n临时映射（★ 仅用于放采样点，不是标定）：料盘中心 px=(%.1f, %.1f)、"
          "边长 %.1f px ⇒ %.4f px/mm" % (center_px[0], center_px[1],
                                         float(np.mean(det.size_px)), scale))
    print("   ★ 判据：%.1f px × 0.7287 mm/px = **%.1f mm**（标称 160mm）"
          % (float(np.mean(det.size_px)), float(np.mean(det.size_px)) * 0.7287))

    # ── ① 亮度 + ② 纹理（与离线模式共用同一段代码，避免两处口径漂移）──
    draw = report_color(color, tray, center_px, scale)

    # ── ③ 深度机制 ──
    if depth_mm is not None:
        print("\n" + "=" * 74)
        print("③ 深度机制（槽是凹的）")
        print("=" * 74)
        print("%-14s %12s %12s %12s %10s" % ("采样点", "深度中位mm", "噪声σ mm", "有效占比", "相对盘面"))
        d_ins, d_outs = [], []
        for s in tray.slots:
            r_mm = _slot_radius_mm(s)
            p = mm_to_px_prov(s["center"], center_px, scale)
            ds = depth_stats(depth_mm, p[0], p[1], max(2.0, 0.45 * r_mm / mmpp))
            if ds:
                d_ins.append((s["id"], ds))
        for i, q in enumerate(pick_surface_points(tray, 4), 1):
            p = mm_to_px_prov(q, center_px, scale)
            ds = depth_stats(depth_mm, p[0], p[1], max(2.0, 6.0 / mmpp))
            if ds:
                d_outs.append(("盘面#%d" % i, ds))
        base = float(np.median([d["median"] for _n, d in d_outs])) if d_outs else None
        for name, ds in d_ins + d_outs:
            rel = ("%+7.2f mm" % (ds["median"] - base)) if base is not None else "-"
            print("%-14s %12.2f %12.3f %11.1f%% %10s"
                  % (name, ds["median"], ds["sigma"], 100.0 * ds["valid"], rel))
        if base is not None and d_ins:
            diffs = [ds["median"] - base for _n, ds in d_ins]
            sig = float(np.mean([ds["sigma"] for _n, ds in d_ins + d_outs]))
            print("\n  ⇒ 槽内相对盘面 = %s（应为**负**：槽是凹的）"
                  % ", ".join("%+.2f" % d for d in diffs))
            print("  ⇒ 深度噪声 σ ≈ %.3f mm" % sig)
            if sig > 1e-6:
                print("  ⇒ ★ 信噪比 = %.1f×（最深那个槽 / 噪声）"
                      % (max(abs(d) for d in diffs) / sig))
            print("  ★ 判据：槽深(pocket_depth=2mm) / 噪声σ > 3 ⇒ **深度可用**"
                  "（且空间平均后更强）")
    else:
        print("\n（没开深度流 ⇒ 跳过 ③）")

    ok = imwrite_safe(os.path.join(OUT_DIR, "%s_sample.png" % args.tag), draw)
    print("\n★ 采样点叠加图: %s（%s）"
          % (os.path.join(OUT_DIR, "%s_sample.png" % args.tag), "成功" if ok else "★ 失败"))
    print("   **先看这张图**：红圈应落在四个槽里、黄圈应落在槽外的盘面上。")
    print("   若不在 ⇒ 临时映射偏了，把图发我，我按图调整映射。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
