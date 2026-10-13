# -*- coding: utf-8 -*-
"""c5 装配位基线 + 辐条环带自动扫优。

产出（写进 calibration.yaml）
    assembly.wheel_height_mm        立柱装好时轮盘的高度基线（深度）
    assembly.base_height_mm         只装底座时中心的高度（用于深度判立柱）
    assembly.spoke_peak_count       辐条峰数（应等于 8）
    assembly.ring_coverage / hub_coverage
    assembly.radii_mm.spoke_band*   ★ 扫优出来的环带半径

为什么要扫优环带：数辐条必须在"有对比度"的半径上做 ——
    * 料盘里：取辐条"臂"，臂之间是黑底
    * 装配体上：臂之间看到的是下方底座（同为黄铜），只能用球头端伸出外圈的凸起
  真实轮盘的这几个半径我们并不知道（没有零件 STL），所以让基线标定自己找。

用法
    python calib/c5_assembly_baseline.py --frames 20          # 用完整装配体
    python calib/c5_assembly_baseline.py --no-depth           # 不标深度基线
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import (ask, grab_full, grab_small, load_all, open_source,  # noqa: E402
                   save_calib)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.core.checks import Qualifier                          # noqa: E402
from assemvision.core.depth import DepthEstimator                      # noqa: E402
from assemvision.core.polar import count_peaks, polar_profile           # noqa: E402
from assemvision.core.roi import circle_polygon, polygon_mask, TrayMap  # noqa: E402
from assemvision.core.segment import Segmenter                          # noqa: E402


def sweep_band(mask, center_px, mmpp: float, required: int = 8,
               r_lo_mm: float = 8.0, r_hi_mm: float = 40.0) -> dict:
    """扫优数辐条的环带：找一个「峰数 == required 且对比度最大」的半径区间。"""
    best = None
    for lo in np.arange(r_lo_mm, r_hi_mm - 4.0, 1.0):
        for span in (4.0, 6.0, 8.0, 12.0):
            hi = lo + span
            if hi > r_hi_mm:
                continue
            prof = polar_profile(mask, center_px, lo / mmpp, hi / mmpp,
                                 n_angles=720, n_radii=9)
            pk = count_peaks(prof, k_mad=3.0, min_width_deg=6.0)
            if len(pk) != required:
                continue
            med = float(np.median(prof))
            peak = float(np.percentile(prof, 95.0))
            contrast = peak - med
            # 峰宽也要像样（太窄说明是毛刺）
            minw = min(p[1] for p in pk)
            if minw < 8.0:
                continue
            score = contrast * min(1.0, minw / 15.0)
            if best is None or score > best["score"]:
                best = dict(score=score, r_min=float(lo), r_max=float(hi),
                            contrast=contrast, min_width=float(minw),
                            peaks=[round(p[0], 1) for p in pk])
    return best or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--camera", default="")
    ap.add_argument("--workpiece", default="")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cam, wp, calib = load_all(args.camera, args.workpiece)
    if "tray_map" not in calib.data:
        print("还没有料盘单应，请先跑 calib/c3_tray_homography.py")
        return 1
    H = np.asarray(calib.get("tray_map.homography"), np.float64)
    tray = TrayMap(wp, homography=H, ref_scale=float(calib.get("tray_map.scale", 1.0)))
    seg = Segmenter(wp)
    q = Qualifier(wp, tray, calib)

    src = open_source(cam, args)

    print("=== 第一步：料盘里的轮盘 —— 扫优「数辐条」用的环带 ===")
    print("请把【一个完整的轮盘组件】放在它的料位里，保持不动。")
    ask("准备好按回车…")
    mask = None
    center = None
    n = 0
    while n < 5:
        frame, f = grab_small(src)
        if frame is None:
            break
        res = seg.segment(frame, scale=f.scale)
        blob = q._blob_in(res, tray.slot_polygon_px("slot_wheel", f.scale))
        if blob is not None:
            mask, center = res.brass_mask, blob.centroid
            break
        n += 1
    if mask is None:
        print("在轮盘槽里找不到前景，检查 ROI 与光照")
        src.close()
        return 1

    mmpp = tray.mm_per_px(f.scale)
    best = sweep_band(mask, center, mmpp, required=8, r_lo_mm=8.0, r_hi_mm=34.0)
    if not best:
        print("扫优失败：整段半径里都数不出 8 个峰。检查轮盘是否完整、光照是否均匀。")
    else:
        print(" 槽内最优环带: r = %.1f ~ %.1f mm  对比度 %.2f  最窄峰 %.1f°"
              % (best["r_min"], best["r_max"], best["contrast"], best["min_width"]))
        print(" 峰角度:", best["peaks"])
        calib.set("assembly.radii_mm.spoke_band", [best["r_min"], best["r_max"]])
        calib.set("assembly.radii_mm.spoke_band_score", round(best["score"], 4))

    print()
    print("=== 第二步：装配位基线 ===")
    print("请把【完整装配体（底座+立柱+轮盘+顶盖）】放到装配位，保持不动。")
    ask("准备好按回车…")
    covs, rings, hubs, caps, spokes = [], [], [], [], []
    for i in range(args.frames):
        frame, f = grab_small(src)
        if frame is None:
            break
        res = seg.segment(frame, scale=f.scale)
        a = q.check_assembly(res, f.scale)
        if a.center_px is None:
            continue
        covs.append(a.ring_coverage); rings.append(a.hub_coverage)
        caps.append(a.cap_coverage)
        if a.spoke is not None:
            spokes.append(a.spoke.count)
    if not spokes:
        print("装配位采不到有效样本，检查装配位 ROI 是否对准")
        src.close()
        return 1
    spk_mode = int(np.bincount(np.array(spokes)).argmax())
    print(" 外圈环带覆盖率 %.3f   轮毂覆盖率 %.3f   顶盖覆盖率 %.3f"
          % (np.mean(rings), np.mean(hubs), np.mean(caps)))
    print(" 辐条峰数（众数） %d   样本 %s" % (spk_mode, spokes[:12]))
    calib.set("assembly.ring_coverage", round(float(np.mean(rings)), 4))
    calib.set("assembly.hub_coverage", round(float(np.mean(hubs)), 4))
    calib.set("assembly.cap_coverage", round(float(np.mean(caps)), 4))
    calib.set("assembly.spoke_peak_count", spk_mode)

    # 装配体上的辐条环带（球头端）
    frame, f = grab_small(src)
    if frame is not None:
        res = seg.segment(frame, scale=f.scale)
        blob = q._blob_in(res, tray.assembly_polygon_px(f.scale))
        if blob is not None:
            b2 = sweep_band(res.brass_mask, blob.centroid, mmpp, required=8,
                            r_lo_mm=26.0, r_hi_mm=42.0)
            if b2:
                print(" 装配体上最优环带: r = %.1f ~ %.1f mm  对比度 %.2f"
                      % (b2["r_min"], b2["r_max"], b2["contrast"]))
                calib.set("assembly.radii_mm.spoke_band_outer",
                          [b2["r_min"], b2["r_max"]])
            else:
                print(" 装配体上扫不出 8 峰（正常：臂之间是底座，只能用球头端）")

    # 深度基线
    if not args.no_depth:
        print()
        print("=== 第三步：深度基线（可选，用于判立柱） ===")
        depth = DepthEstimator(cam, calib)
        if depth.setup():
            L, R = grab_full(src)
            d = depth.compute(L, R) if L is not None else None
            if d is not None:
                plane = depth.fit_plane(d)
                if plane:
                    hm = depth.height_map(d, plane)
                    poly = tray.assembly_polygon_px(1.0)
                    c = poly.mean(axis=0) * depth.scale
                    r_px = q.r_outer / mmpp * depth.scale
                    h = depth.height_at(hm, (c[0], c[1]), r_px)
                    if h is not None:
                        print(" 轮盘区域高度基线 = %.1f mm" % h)
                        # ★ 两处都写：checks.py 读的是 assembly.baseline.wheel_height_mm，
                        #   而本脚本历史上只写了 assembly.wheel_height_mm，
                        #   结果运行时段检读不到基线、立柱判据静默失效。
                        calib.set("assembly.baseline.wheel_height_mm", round(float(h), 2))
                        calib.set("assembly.wheel_height_mm", round(float(h), 2))
        else:
            print(" 深度不可用（没有 c2 双目标定结果），跳过")

    src.close()
    print()
    print("已写入:", save_calib(calib, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
