# -*- coding: utf-8 -*-
"""换机对账补充探针：**分割色域 + metal 掩膜余量**（只读，不改工程）。

★ 位置：`视觉/ai_测试工具/`（AI 自己的工具区）。
  与用户自己的 `用户存档/`（`predict.py` / `d435_site_report.py` 等）**分开**，
  互不覆盖。

与 `用户存档/d435_site_report.py` 的分工：
  · `d435_site_report.py`  = 相机侧对账单（曝光 / GSD / 帧率 / 尺寸）
  · **本脚本**             = **分割侧对账单**（黄铜 / 绿幕 / 手套的色域，
                             以及 `segment.metal` 那道「绿垫判据」的**余量**）

为什么需要它（2026-10 实测发现，详见
`移植方案_零件分割识别追踪_到D435.md` §3.2 与
`ai_测试工具/工位帧/工位实测记录.md`）：
  `Segmenter._metal_mask()` 的判据是
      green = (H ∈ [green_h_min, green_h_max]) & (S > green_s_min)
      metal = (V >= v_min) & (S <= s_max) & (~green)
  当前配置（HBV 口径）是 `green_h_min=60 / green_h_max=100`。
  ★ 工位实测：绿幕像素是 `(B=137,G=139,R=0)` ⇒ **H=90、R 通道为 0**，
  看起来"落在区间里"，但**离上界只有 10 个单位**，而零件的暖色像素
  H 中位只有 **3** —— 上下两头同时贴边。

★ 用法（VSCode 里：打开本文件按 F5，见 `.vscode/launch.json`；
  或在终端里跑 —— **工作目录在哪都行**，路径由脚本自己算）：
    # ① 不带参数：先量 ai_测试工具/工位帧/原图/ 下的工位帧，再量换机期样张与旧机对照
    & D:\\miniconda\\envs\\yolov11\\python.exe "ai_测试工具\\seg_gamut_report.py"
    # ② 指定图
    & D:\\miniconda\\envs\\yolov11\\python.exe "ai_测试工具\\seg_gamut_report.py" "图1.png" "图2.png"
    # ③ 想重定向到文件留档
    & D:\\miniconda\\envs\\yolov11\\python.exe "ai_测试工具\\seg_gamut_report.py" > "ai_测试工具\\工位帧\\色域对账.txt"

★ 判读（三件事，缺一不可）：
  ① **绿幕余量**：工位帧上绿幕的 H 分布 p2~p98 离 `[green_h_min, green_h_max]`
     两侧各留 **≥5** 个单位；
  ② **黄铜余量**：真零件（不是木桌）的 H/S 分布要完整落在 `segment.brass` 区间内，
     离上下界也各留 ≥5；
  ③ **metal 掩膜**：画面里"亮 + 不是绿垫"的像素占比要在**可解释的范围内** ——
     占比高不等于坏（木桌/白纸本来就亮），关键是**真零件那一块必须在 metal 里**，
     而**大面积假金属必须能被 work_roi / 尺寸闸门挡住**。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 工作区根 = 视觉\
AV = os.path.join(ROOT, "assembly_vision")
sys.path.insert(0, AV)

DATA_DIR = os.path.join(ROOT, "ai_测试工具", "工位帧")

from assemvision.config import load_config        # noqa: E402
from assemvision.core.segment import Segmenter    # noqa: E402

WP_PATH = os.path.join(AV, "config", "workpiece.yaml")


def imread(p):
    """★ 必须走 np.fromfile + imdecode：中文路径下裸 cv2.imread 会**静默失败**（实录 §2.36）。"""
    try:
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def pct(a, q):
    return float(np.percentile(a, q)) if a.size else float("nan")


def gamut_report(img, tag, wp):
    seg = Segmenter(wp)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    print("\n" + "=" * 78)
    print("  %s   (%dx%d)" % (tag, img.shape[1], img.shape[0]))
    print("=" * 78)
    print("  亮度：V 均值 %.1f   近全白(minBGR>=250) **%.3f%%**（报警线 0.5%%）"
          % (V.mean(), 100.0 * (img.min(axis=2) >= 250).mean()))

    # ── ① 高饱和像素的色相分布（最直观的一张体检表）──
    hi = S > 120
    n = int(np.count_nonzero(hi))
    if n:
        hh = H[hi]
        hist = np.bincount(hh.ravel(), minlength=180)
        top = np.argsort(-hist)[:8]
        print("  ① 高饱和(S>120) %d px (%.1f%%)   色相 p2/p50/p98 = %.0f / %.0f / %.0f"
              % (n, 100.0 * n / H.size, pct(hh, 2), pct(hh, 50), pct(hh, 98)))
        print("     众数 Top8：" + ", ".join("H=%d(%d)" % (t, hist[t]) for t in top))

    # ── ② 绿垫判据的余量（metal 掩膜的前提）──
    mt = dict(seg.metal or {})
    g_lo = int(mt.get("green_h_min", 60))
    g_hi = int(mt.get("green_h_max", 100))
    g_smin = int(mt.get("green_s_min", 120))
    green = (H >= g_lo) & (H <= g_hi) & (S > g_smin)
    in_band = (H >= g_hi - 5) & (H <= g_hi + 5) & (S > g_smin)
    print("  ② 绿垫判据  green = H∈[%d,%d] & S>%d" % (g_lo, g_hi, g_smin))
    print("     被判成绿垫：%6d px (%5.2f%%)" % (int(green.sum()), 100.0 * green.mean()))
    print("     ★ 上界 ±5 的邻域里另有 %d px —— **这部分必须趋近 0**（否则上界没有余量）"
          % int((in_band & ~green).sum()))
    if n:
        # 离两侧边界最近的"高饱和"色相落在哪
        print("     ★ 高饱和像素里最近的色相：下界侧 %d 个在 H=%s；上界侧 %d 个在 H=%s"
              % (int((hh < g_lo).sum()),
                 (int(hh[hh < g_lo].max()) if (hh < g_lo).any() else -1),
                 int((hh > g_hi).sum()),
                 (int(hh[hh > g_hi].min()) if (hh > g_hi).any() else -1)))

    # ── ③ 黄铜色域余量 ──
    br = dict(seg.brass or {})
    lo = np.array([br.get("h_min", 10), br.get("s_min", 50), br.get("v_min", 80)], np.uint8)
    hi2 = np.array([br.get("h_max", 45), br.get("s_max", 180), br.get("v_max", 255)], np.uint8)
    m_brass = cv2.inRange(hsv, lo, hi2)
    print("  ③ 黄铜判据  H∈[%d,%d] S∈[%d,%d] V≥%d"
          % (lo[0], hi2[0], lo[1], hi2[1], lo[2]))
    print("     命中 %6d px (%5.2f%% 全屏)" % (int(np.count_nonzero(m_brass)),
                                              100.0 * np.count_nonzero(m_brass) / m_brass.size))

    # ── ④ 走一遍工程真实分割路径 ──
    res = seg.segment(img, scale=1.0)
    print("  ④ Segmenter.segment()（★ 未给 mm_per_px ⇒ 尺寸闸门静默关闭）")
    print("     brass 连通域 %d 个；最大 %.0f px；min_area_px=%s"
          % (len(res.brass_blobs),
             (res.brass_blobs[0].area_px if res.brass_blobs else 0.0),
             br.get("min_area_px")))
    if res.metal_mask is not None:
        mm = int(np.count_nonzero(res.metal_mask))
        print("     metal 掩膜 %6d px (%5.2f%% 全屏) —— ★ 它只判「亮 + 不是绿垫」，"
              % (mm, 100.0 * mm / res.metal_mask.size))
        print("       所以木桌/白纸本来就会进来；挡它靠 work_roi + 尺寸闸门，不是靠颜色")
    print("     blown_ratio=%.4f  v_mean=%.1f" % (res.detail.get("blown_ratio", -1),
                                                 res.detail.get("v_mean", -1)))

    # ── ⑤ 手套色域（若画面里有手）──
    gl = dict(seg.glove or {})
    print("  ⑤ 手套判据  H∈[%d,%d] S≥%d V≥%d  ->  glove_bbox=%s  area=%.0f px"
          % (gl.get("h_min"), gl.get("h_max"), gl.get("s_min"), gl.get("v_min"),
             res.glove_bbox, res.glove_area_px))


def default_frames():
    """不带参数时要量哪些图：优先**工位帧**（那才是权威），再带上换机期样张做对照。"""
    out = []
    # ★ 工位实测帧（AI 自己抓的，见 ai_测试工具/capture_site_frame.py）
    site = os.path.join(DATA_DIR, "原图")
    if os.path.isdir(site):
        for n in sorted(os.listdir(site)):
            if n.lower().endswith((".png", ".jpg")):
                out.append((os.path.join(site, n), "工位 " + n))
    # 换机期样张（相机对着桌面时拍的，只作对照）
    probe = os.path.join(AV, "data", "probe")
    for n in ("d435_640x480@30.png", "d435_1280x720@30.png"):
        p = os.path.join(probe, n)
        if os.path.exists(p):
            out.append((p, "D435 桌面样张 " + n))
    # 旧相机（HBV）工位帧 —— 换机前的基准，用来看"差多少"
    refs = os.path.join(AV, "data", "refs")
    for n in ("asm_raw.png", "tray_empty.png", "now_raw.png"):
        p = os.path.join(refs, n)
        if os.path.exists(p):
            out.append((p, "HBV(旧机对照) " + n))
    return out


def main() -> int:
    wp = load_config(WP_PATH)
    if len(sys.argv) > 1:
        frames = [(a, os.path.basename(a)) for a in sys.argv[1:]]
    else:
        frames = default_frames()

    print("═" * 78)
    print("  分割色域 / metal 掩膜 对账单")
    print("  配置：%s" % WP_PATH)
    print("═" * 78)

    for p, tag in frames:
        img = imread(p)
        if img is None:
            print("\n  [%s] 读不出：%s（★ 中文路径要走 imdecode，见实录 §2.36）" % (tag, p))
            continue
        gamut_report(img, tag, wp)

    print("\n" + "═" * 78)
    print("判读（三条都要过）：")
    print("  ① 绿垫上界 ±5 的邻域里应**几乎没有**高饱和像素（现在 D435 桌面帧上只差 2）")
    print("  ② 工位帧上绿幕的 H p2~p98 离 [green_h_min, green_h_max] 两侧各留 ≥5")
    print("  ③ 真零件必须完整落在 segment.brass 区间内，且离上下界各留 ≥5")
    print("★ 这三条不通过就不要动 c6 —— 分割不稳时采出来的模板本身就是错的。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
