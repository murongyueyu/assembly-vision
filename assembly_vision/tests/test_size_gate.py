# -*- coding: utf-8 -*-
"""尺寸闸门测试 —— 用**等效直径**挡掉木桌那类"像黄铜的大块"。

★ 为什么必须有这一条（P1-1「开放位置」的前提）：
  木桌的色相/饱和度与黄铜**重叠**（实测木桌 H=19、S=122，落在黄铜的
  `h∈[10,45] / s∈[50,180]` 里），而且**换相机不解决** ——
  换 D435 后照样复现（实测黄铜 86048px = 全屏 28.0%）。
  调色相的代价是削掉真零件自己的高光（零件 S 上限 109、H 上限 45，与木桌本就重叠），
  所以只能靠**几何**：位置（work_roi）+ 尺寸（max_diam_mm）。
  实测数字（全图分割、不带 work_roi）：
    木桌 31795mm² = 等效直径 **201mm**  vs  真零件最大 3300mm² = **64.5mm**
  差 2.7 倍 -> 取 120mm 干净利落。

★★ 本测试真正在钉的三件事（只测"木桌被挡"是不够的）：
  1. **挡得住**：木桌尺寸的块必须整块消失（掩膜与 blobs 都不能剩）；
  2. **不误伤**：料盘里 4 个真零件一个都不能少、像素一个都不能掉；
  3. **顺序对**：被挡掉的块**也不能留在 metal 掩膜里** ——
     木桌属于「亮 + 不是绿」，`s_max=100` 只能把它的 metal 占比从 99.9% 压到 15.2%，
     压不到 0；顺序一反（先建 metal 再挡尺寸），工作 ROI 一放开就污染辐条扫优。

★ 还有一个"看着通过其实没测"的坑：**不传 mm_per_px 时闸门是静默关闭的**，
  于是"没有木桌"和"闸门没生效"在结果上完全一样。
  所以场景 1/2 是**对照实验**：同一张图，先证明不传时会漏进来，再证明传了就挡住。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check, make_env                              # noqa: E402
from make_synthetic import render_scene                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.core.part_id import features_from_blob          # noqa: E402

PPM = 4.0                 # 每毫米 4 像素（与 test_work_roi 同一口径）
MARGIN = 80.0             # 大边距画布：才有"料盘外"的区域放假木桌
# 合成图的黄铜色（BGR）。与 test_work_roi 用同一套，HSV≈(22,178,200)
BRASS_BGR = (60, 160, 200)
# ★ 木桌色（BGR）：按**实测**反算出来，不是随手取的。
#   H 的口径：本项目一律用 **OpenCV 的 0~180**（`workpiece.yaml` 的黄铜区间
#   `h∈[10,45]`、绿垫 `h∈[60,100]` 都是这个口径），实录记的 H=19 也是它。
#
#   ★★ 这里**故意取暗的那一半**（S=80 而不是 122），理由是个容易漏的坑：
#     实录 2.2 记的木桌 S 中值是 **67~87**（不是 122），而 `metal.s_max=100`
#     只挡得住 S>100 的像素 —— 实测"加了 s_max 后木桌区 metal 占比仍有 **15.2%**"
#     也印证了这一点：**木桌有相当一部分像素照样满足「亮 + 不是绿」**。
#     拿 S=122 去测，挡住它的是 `s_max`（颜色），尺寸闸门与 metal 裁剪根本没用上，
#     那就是"测了等于没测"。取 S=80 才能让场景 5 的 0 真的是**闸门在干活**。
WOOD_BGR = (144, 185, 210)     # BGR -> HSV(19, 80, 210)
WOOD_HSV = (19, 80, 210)       # OpenCV 口径（0~180）


def _hsv_of(bgr) -> tuple:
    """BGR -> (H, S, V)，**OpenCV 原样口径**（H 0~180）。

    ★ 口径统一很重要：本项目配置与实录里的 H 全是这个口径
      （黄铜 `h∈[10,45]`、绿垫 `h∈[60,100]`、木桌 H=19）。
      曾经把这里的 H 又乘了 2 当成"度"，结果木桌色被判成 38 而测试假失败 ——
      **测色域前先确认口径**，这是本项目反复踩的那类坑。
    """
    px = np.uint8([[list(bgr)]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
    return int(h), int(s), int(v)


def _diam_mm(area_px: float, mmpp: float) -> float:
    """等效直径（mm）= 2·√(面积/π)·mm_per_px —— 与 `Segmenter._size_gate` 同式。"""
    return 2.0 * float(np.sqrt(float(area_px) / np.pi)) * float(mmpp)


def _feats(b, mmpp: float, roi_area_px: float, bgr, mask):
    f = features_from_blob(b, roi_area_px=roi_area_px, mm_per_px=mmpp, bgr=bgr, mask=mask)
    return f


def main() -> int:
    wp, tray, seg, q = make_env(px_per_mm=PPM)
    img = render_scene(px_per_mm=PPM, margin_mm=MARGIN)
    H, W = img.shape[:2]
    mmpp = tray.mm_per_px(1.0)
    maxd = float(seg.max_diam_mm)
    print("   画布 %dx%d    mm_per_px %.3f    尺寸闸门上限 %.0fmm" % (W, H, mmpp, maxd))
    check(seg.work_roi_mode == "work", "配置里 work_roi_mode 当前是 work（ROI 仍在生效）")

    # ── 造一块"木桌"：尺寸与颜色都对准**实测**值 ─────────────────
    #   ★ 形状用**矩形**而不是正方形：画布只有 1720x1280，而"等效直径 201mm"
    #     在 0.25mm/px 下要 5.08e5 px²（正方形边长 712px），而工位 ROI 之外
    #     只剩"贴着下沿的一条"可用 —— 那些位置放不下 712 见方。
    #     用宽而扁的一条：面积够、等效直径够（实测 216mm，已超过实测木桌的 201mm），
    #     而且更像真实画面里贴着边缘的桌面。
    #   ★ 位置：**自动找一条与画面里任何黄铜像素都不相连的横条**。
    #     为什么要自动找，而不是写死坐标（这是本测试踩过的第二个坑）：
    #       ① 必须在工位 ROI 之外 —— 场景 4 才能证明"不靠位置、只靠尺寸也挡得住"；
    #       ② 必须与零件**不相邻** —— 摆到贴着料盘下沿时，木桌像素会与
    #          底座/顶盖**直接相连**，闸门量到的就是"木桌+零件"一整块，
    #          于是连真零件一起剔掉。
    #          实测踩过两次：y0=789（贴 ROI 边）剩 2 个零件；y0=940 仍剩 3 个
    #          （料盘的暗色边沿也落在黄铜色域里，比看上去低）。
    #          **手工试坐标是碰运气**，所以改成"扫到干净为止"。
    print("== 场景 0：假木桌的尺寸超过闸门、颜色与实测木桌一致 ==")
    hsv_clean = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # ★★ 摆位检查必须用**闭运算之后**的掩膜，不能用纯色域掩膜 ——
    #   踩过两次：纯色域看着"这一行没有黄铜"，但 `morph_close` 会把料盘
    #   底沿的暗色边沿往外扩 2~3px，正好和木桌连上，于是闸门量到
    #   "木桌 + 底座"一整块，连底座一起剔掉（实测 4 个零件只剩 3 个）。
    #   判据：**闸门量的是闭运算后的连通域，摆位就必须按闭运算后的掩膜留间隔。**
    brass_clean = seg._mask(hsv_clean, seg.brass, morph=True)
    roi_clean = q.work_roi_mask(img.shape, 1.0,
                                margin_mm=float(wp.get("segment.work_roi_margin_mm", 25.0)))
    free = (brass_clean == 0) & (roi_clean == 0)
    x0, x1 = 0, W - 1
    # ★ 留**3px 间隔**（`morph_close` 核是 5，半径 2）——
    #   "这一行是空的"不够：闭运算会把外扩 2px 内的东西连起来。
    pad = 3
    free_pad = free.copy()
    for d in range(1, pad + 1):
        free_pad[:-d, :] &= free[d:, :]
        free_pad[d:, :] &= free[:-d, :]
    free_pad[:, :-d] &= free[:, d:]
    free_pad[:, d:] &= free[:, :-d]
    y0 = None
    for cand in range(H - 1, 0, -1):
        if not free_pad[cand, :].all():      # 这一行 + 上方 3px 都要干净
            break
        y0 = cand
    y1 = H - 1
    check(y0 is not None, "找得到至少一行『与黄铜不相邻且在 ROI 外』的位置")
    print("   自动摆位：整条 x[%d,%d] y[%d,%d]（按**闭运算后**的掩膜 + %dpx 间隔找的）"
          % (x0, x1, y0, y1, pad))
    # 兜底：确认这一片确实与任何黄铜都不相连（不能只信行判断）
    probe = np.zeros_like(brass_clean)
    probe[y0:y1 + 1, x0:x1 + 1] = 255
    ring = cv2.dilate(probe, np.ones((5, 5), np.uint8))
    check(int(cv2.bitwise_and(ring, brass_clean).sum()) == 0,
          "这一片（含 2px 外扩邻域）里没有任何黄铜像素 —— 木桌不会与零件相连")
    rect_area = float((x1 - x0 + 1) * (y1 - y0 + 1))
    cen = ((x0 + x1) // 2, (y0 + y1) // 2)
    hh, ss, vv = _hsv_of(WOOD_BGR)
    print("   木桌色实测口径 HSV=(%d,%d,%d)，本测试用色实测 HSV=(%d,%d,%d)"
          % (WOOD_HSV + (hh, ss, vv)))
    check(abs(hh - WOOD_HSV[0]) <= 2 and abs(ss - WOOD_HSV[1]) <= 8
          and abs(vv - WOOD_HSV[2]) <= 8,
          "木桌色确实复现了实测的 H/S/V（否则这个测试是空转）")
    check(int(wp.get("segment.brass.h_min", 0)) <= hh <= int(wp.get("segment.brass.h_max", 179))
          and int(wp.get("segment.brass.s_min", 0)) <= ss
          <= int(wp.get("segment.brass.s_max", 255)),
          "★ 木桌色落在**黄铜色域内**（H=%d 在 [%d,%d]、S=%d 在 [%d,%d]）——"
          " 所以只能靠尺寸挡，不能靠色相"
          % (hh, int(wp.get("segment.brass.h_min", 0)), int(wp.get("segment.brass.h_max", 179)),
             ss, int(wp.get("segment.brass.s_min", 0)), int(wp.get("segment.brass.s_max", 255))))

    fake = img.copy()
    cv2.rectangle(fake, (x0, y0), (x1, y1), WOOD_BGR, thickness=-1)
    check(bool((fake[cen[1], cen[0]] != img[cen[1], cen[0]]).any()),
          "假木桌确实画上去了（对照点像素变了）")
    # ★ 面积要用**画上去之后**实测的，不要用 w*h 估 —— 这块贴着画面边缘，
    #   实际落进画面的像素数才是闸门看到的那一块。
    got_area = float(((fake[..., 0] == WOOD_BGR[0]) & (fake[..., 1] == WOOD_BGR[1])
                      & (fake[..., 2] == WOOD_BGR[2])).sum())
    d_rect = _diam_mm(got_area, mmpp)
    print("   块 x[%d,%d] y[%d,%d] = %d px² -> 等效 Ø%.0fmm（实测木桌是 Ø201mm，"
          "闸门上限 %.0fmm）" % (x0, x1, y0, y1, got_area, d_rect, maxd))
    check(got_area > 0.9 * rect_area, "画上去的面积与矩形面积一致（%.0f vs %.0f）"
          % (got_area, rect_area))
    check(d_rect > 1.1 * maxd, "这一块的等效直径**明确超过**闸门上限（%.0f > 1.1×%.0f）——"
          "留足余量，避免「刚好卡在阈值上」那种会随平台抖动的测试"
          % (d_rect, maxd))

    # ── 场景 1/2：对照实验 —— 不传尺度会漏，传了就挡 ──────────────
    print("== 场景 1：不传 mm_per_px 时闸门**静默关闭**（对照，证明测得出差别）==")
    r_off = seg.segment(fake, scale=1.0)
    gate_off = r_off.detail.get("size_gate", {})
    print("   detail.size_gate = %s" % (gate_off,))
    check(not gate_off.get("on", False), "不传尺度时闸门自报「没开」")
    check(r_off.brass_mask[cen[1], cen[0]] > 0, "假木桌中心被切成黄铜（否则后面测不出差别）")
    blob_off = max(r_off.brass_blobs, key=lambda b: b.area_px)
    d_off = _diam_mm(blob_off.area_px, mmpp)
    print("   最大块: %.0f px² -> 等效直径 %.0fmm，质心 %s"
          % (blob_off.area_px, d_off, tuple(round(v) for v in blob_off.centroid)))
    check(d_off > maxd, "对照组的最大块**超过**闸门上限（%.0f > %.0f）—— 它是应该被挡的" %
          (d_off, maxd))

    print("== 场景 2：传了 mm_per_px 后，木桌尺寸的块被整块剔除 ==")
    r_on = seg.segment(fake, scale=1.0, mm_per_px=mmpp)
    gate_on = r_on.detail.get("size_gate", {})
    print("   detail.size_gate = %s" % (gate_on,))
    check(gate_on.get("on", False), "闸门自报已启用")
    check(abs(float(gate_on.get("max_diam_mm", 0)) - maxd) < 1e-6,
          "闸门用的是配置里的上限 %.0fmm" % maxd)
    check(r_on.brass_mask[cen[1], cen[0]] == 0, "假木桌中心**不再**是黄铜了")
    check(float(gate_on.get("dropped_px", 0)) >= 0.9 * blob_off.area_px,
          "被剔除的像素数 ≈ 那块木桌的面积（%.0f / %.0f）"
          % (gate_on.get("dropped_px", 0), blob_off.area_px))
    d_on = max((_diam_mm(b.area_px, mmpp) for b in r_on.brass_blobs), default=0.0)
    print("   闸门后最大块的等效直径 %.0fmm（上限 %.0fmm）" % (d_on, maxd))
    check(d_on <= maxd + 1e-6, "闸门后**所有** brass blobs 都在上限以内")
    leak = sum(1 for b in r_on.brass_blobs
               if _diam_mm(b.area_px, mmpp) > maxd)
    check(leak == 0, "没有超尺寸的块泄漏到 blobs（泄漏 %d 个）" % leak)

    # ── 场景 3：不误伤真零件（这是最要紧的一条）──────────────────
    print("== 场景 3：料盘里的真零件一个都不能掉（闸门不能误伤）==")
    r_base = seg.segment(img, scale=1.0, mm_per_px=mmpp)
    # ★★ 参考值必须**重新渲染一张干净的图**再算 —— 不能拿上面那张 `fake`
    #     （它已经被画上木桌了）。第一版就是拿 `fake` 当参考，于是
    #     干净图 13.5 万像素 vs 带木桌图 64 万像素，测出来"差异 79%"
    #     —— **测试自己造假失败**，与闸门无关。这类"参考选错"比断言写错更隐蔽。
    clean = render_scene(px_per_mm=PPM, margin_mm=MARGIN)
    r_ref = seg.segment(clean, scale=1.0)
    print("   闸门剔除像素 = %d（干净料盘，应当为 0）"
          % r_base.detail.get("size_gate", {}).get("dropped_px", -1))
    check(int(r_base.detail.get("size_gate", {}).get("dropped_px", -1)) == 0,
          "干净场景下闸门**一个像素都没剔**（说明留的余量够）")
    check(len(r_base.brass_blobs) == len(r_ref.brass_blobs),
          "带闸门与不带闸门的块数一致（%d vs %d）"
          % (len(r_base.brass_blobs), len(r_ref.brass_blobs)))
    d_max_real = max(_diam_mm(b.area_px, mmpp) for b in r_base.brass_blobs)
    print("   真零件最大块等效直径 %.0fmm，上限 %.0fmm（余量 %.2f 倍）"
          % (d_max_real, maxd, maxd / max(1e-6, d_max_real)))
    # ★★ 判据从「< 0.8×上限」改成 **「≥1.25 倍余量」**（2026-10）：
    #   原来的 0.8 是个**没依据的系数**，它的效果是"上限必须比零件大 25%"——
    #   但写成 0.8×maxd 之后，**上限越大反而越容易通过**，而闸门本身又在变松，
    #   两边一起动，这条断言就说不清在守什么了。
    #   现在直接写成它的本意：**上限至少是实测最大零件的 1.25 倍**。
    #   依据：实测最大零件是**轮盘 Ø74.4mm**（`make_synthetic.py` 的 c3 正交化测量
    #   + 零件名义尺寸交叉吻合）。1.25 倍 ⇒ 下限 Ø93 —— 合成图的轮盘正好 Ø75，
    #   所以上限必须 ≥94 才过；`workpiece.yaml` 取 **100**。
    check(d_max_real * 1.25 <= maxd,
          "上限应≥实测最大零件的 1.25 倍（%.0f×1.25=%.0f ≤ %.0fmm）"
          % (d_max_real, d_max_real * 1.25, maxd))
    r_area = float((r_base.brass_mask > 0).sum())
    r_ref_area = float((r_ref.brass_mask > 0).sum())
    print("   料盘区前景像素: 不带闸门 %.0f   带闸门 %.0f" % (r_ref_area, r_area))
    check(abs(r_area - r_ref_area) <= 0.01 * max(1.0, r_ref_area),
          "料盘内的分割结果没被闸门改动（差异 ≤1%）")

    # ── 场景 4：work_roi_mode=full 时，挡木桌的只剩尺寸闸门 ───────
    #   这条是 P1-1「开放位置」的关键：位置放开了，尺寸必须顶上。
    print("== 场景 4：work_roi_mode=full（不用位置挡）也挡得住 ==")
    seg.work_roi_mode = "full"
    try:
        roi = q.work_roi_mask(fake.shape, 1.0, margin_mm=25.0)
        r_full = seg.segment(fake, scale=1.0, work_roi=roi, mm_per_px=mmpp)
        r_full_nogate = seg.segment(fake, scale=1.0, work_roi=roi)
        check(bool(roi[cen[1], cen[0]] == 0),
              "假木桌坐在工位 ROI **之外**（所以这条测的确实是尺寸，不是位置）")
        check(r_full_nogate.brass_mask[cen[1], cen[0]] > 0,
              "★ full 模式下**不传尺度**时木桌会漏进来（位置闸门已经不在了）")
        check(r_full.brass_mask[cen[1], cen[0]] == 0,
              "★ full 模式 + 尺寸闸门 -> 木桌照样被挡")
        check(r_full.detail.get("work_roi_mode") == "full", "detail 里记下了当前 ROI 模式")
        # 打开 full 后，料盘里的真零件必须一个不少 —— 这正是"开放位置"的目的：
        # 不再靠位置挡，靠尺寸挡，而真零件都在尺寸以内。
        check(len(r_full.brass_blobs) == 4,
              "full 模式下仍认出 4 个真零件块（实测 %d 个）" % len(r_full.brass_blobs))
        # 每个零件的 blobs 必须都还在，且规模不变 —— 这才是"没被误伤"。
        # ★ 不要用"质心处有黄铜像素"当判据：**轮盘的质心落在轮毂孔里**，
        #   那是背景不是黄铜（第一版就是这么写的，4 个只过 2 个，纯属判据选错）。
        areas = sorted(int(round(b.area_px)) for b in r_full.brass_blobs)
        print("   full 模式四块面积: %s" % (areas,))
        check(all(a >= 3000 for a in areas), "四块的面积都在零件量级（无碎块）")
        r_work = seg.segment(fake, scale=1.0, work_roi=roi, mm_per_px=mmpp)
        a_work = sorted(int(round(b.area_px)) for b in r_work.brass_blobs)
        check(len(a_work) == len(areas), "work 与 full 两种模式的块数一致（%d vs %d）"
              % (len(a_work), len(areas)))
    finally:
        seg.work_roi_mode = "work"

    # ── 场景 5：被挡掉的块也不能留在 metal 掩膜里（顺序铁律）──────
    print("== 场景 5：木桌不能进 metal 掩膜（尺寸闸门必须建在它之前）==")
    # ★★ 本场景必须**同时**满足两条，否则它是个空测试：
    #   ① 木桌色对 metal 判据**成立**（否则挡住它的是 s_max，不是闸门）；
    #   ② 画面里得有**真的金属**（否则 metal 掩膜本来就近空，"没泄漏"毫无意义）。
    #      —— 合成图的底色是黑的（V=28），"亮+不是绿"天然不成立，实测干净图 metal = 0。
    #      所以这里自己造几个亮球头（HV 低饱和高亮，和真实球头一个口径）。
    # ★ 用 HSV 造色（而不是手写 BGR）：手写 BGR 很难同时满足
    #   「V≥v_min」和「S≤s_max」，第一版写出 HSV(5,255,235)，S 直接顶到 255，
    #   于是球头自己就不算金属 —— 又是"测试自己把路堵死"。
    #   实测金属球头 V 中值 222、黄铜 S ≤109，所以取 HSV(20, 60, 235)。
    metal_bgr = tuple(int(c) for c in cv2.cvtColor(
        np.uint8([[[20, 60, 235]]]), cv2.COLOR_HSV2BGR)[0, 0])
    mh, ms, mv = _hsv_of(metal_bgr)
    mc0 = wp.get("segment.metal", {}) or {}
    print("   自造金属球头色 BGR=%s -> HSV=(%d,%d,%d)（判据需 V≥%d 且 S≤%d）"
          % (metal_bgr, mh, ms, mv,
             int(mc0.get("v_min", 150)), int(mc0.get("s_max", 100))))
    check(mv >= int(mc0.get("v_min", 150)) and ms <= int(mc0.get("s_max", 100)),
          "自造球头色满足 metal 判据（否则这条测不出 metal 通路）")

    def add_balls(frame):
        """在装配位画一圈亮球头（模拟轮盘辐条球头）。"""
        ac = np.asarray((wp.get("assembly.center") or [-65.0, 80.0]), dtype=np.float64)
        for k in range(8):
            t = 2.0 * np.pi * k / 8.0
            p = tray.mm_to_px(np.array([[ac[0] + 32.0 * np.cos(t),
                                         ac[1] + 32.0 * np.sin(t)]]), 1.0)[0]
            cv2.circle(frame, (int(round(p[0])), int(round(p[1]))), 5, metal_bgr, -1)
        return frame

    base_m = add_balls(img.copy())
    wood_m = add_balls(fake.copy())
    m_on = seg.segment(wood_m, scale=1.0, mm_per_px=mmpp)
    m_ref = seg.segment(base_m, scale=1.0, mm_per_px=mmpp)
    d_on = int((m_on.metal_mask > 0).sum())
    d_ref = int((m_ref.metal_mask > 0).sum())
    in_wood = int((m_on.metal_mask[y0:y1 + 1, x0:x1 + 1] > 0).sum())
    print("   metal 像素: 干净图+球头 %d   加木桌后 %d   **木桌区内 %d**"
          % (d_ref, d_on, in_wood))
    check(d_ref > 0, "对照图里确实有金属像素（%d）—— 否则这个场景是空转" % d_ref)
    check(in_wood == 0, "木桌区域内 metal 像素为 0（否则辐条扫优会被污染）")
    check(d_on == d_ref,
          "加了木桌后 metal 像素**一个都没多**（%d vs %d）—— 闸门把木桌从 metal 里也剔掉了"
          % (d_on, d_ref))
    # ★ 比较两张掩膜时必须**先二值化**：掩膜像素值是 255，
    #   直接 `bitwise_and` 求和得到的是 255×重叠数，不是重叠数
    #   （第一版就是把 255×648 与 648 比，必然不等）。
    b_on = (m_on.metal_mask > 0)
    b_ref = (m_ref.metal_mask > 0)
    inter = int((b_on & b_ref).sum())
    only_on = int((b_on & ~b_ref).sum())
    only_ref = int((b_ref & ~b_on).sum())
    print("   逐像素对比: 交集 %d   只在『加木桌图』有 %d   只在『对照图』有 %d"
          % (inter, only_on, only_ref))
    check(inter == d_ref,
          "球头在两张图里逐像素一致（交集 %d == 对照 %d）" % (inter, d_ref))
    check(only_on == 0 and only_ref == 0,
          "两张图的 metal 掩膜完全相同（多出 %d、少了 %d）" % (only_on, only_ref))
    # 反向对照：木桌色**本身**确实满足「亮 + 不是绿」-> 光靠 s_max 压不住
    hsv = cv2.cvtColor(fake, cv2.COLOR_BGR2HSV)
    mc = wp.get("segment.metal", {}) or {}
    hpx, spx, vpx = (int(v) for v in hsv[y0 + 4, x0 + 4])
    is_green = (int(mc.get("green_h_min", 60)) <= hpx
                <= int(mc.get("green_h_max", 100))
                and spx > int(mc.get("green_s_min", 120)))
    looks_metal = (vpx >= int(mc.get("v_min", 150))
                   and spx <= int(mc.get("s_max", 100)) and not is_green)
    print("   木桌像素 HSV=(%d,%d,%d)  is_green=%s  满足metal判据=%s"
          % (hpx, spx, vpx, is_green, looks_metal))
    check(looks_metal,
          "★ 本测试用的木桌色**确实满足** metal 判据（亮 + 不是绿、S=%d ≤ s_max=%d）"
          " —— 所以场景 5 的 0 是尺寸闸门挣来的，不是 s_max 顺手挡的"
          % (spx, int(mc.get("s_max", 100))))
    check(not is_green, "★ 木桌色也不落在绿垫区间（否则它会被当成绿垫而不是金属）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
