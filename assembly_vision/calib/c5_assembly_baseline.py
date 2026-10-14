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
    """扫优数辐条的环带：找一个「峰数 == required **且角度均匀** 且对比度最大」的半径区间。

    ★★ 为什么必须判"角度均匀"（实测踩过，见 AGENTS.md 2.13）：
      只看"峰数 == 8"会选出**凑出来的 8 个峰**。实测：
        brass 掩膜 28mm 带 -> 8 峰，但间隔 38/37/43/91/23/10/22/96°（抖动 33°）
        metal 掩膜 30~35mm -> 8 峰，间隔 52/41/36/42/43/43/48/56°（抖动 5.9°）
      前者是噪声凑数，后者才是真的 8 辐条。**只看峰数会把两者当成一样好。**
      真 8 辐条的角度间隔应当均匀 = 360/8 = 45°。
    """
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
            # ★ 角度均匀度：间隔相对 360/required 的抖动
            angs = sorted(float(p[0]) for p in pk)
            diffs = np.diff(angs + [angs[0] + 360.0])
            jitter = float(np.std(diffs))
            # 抖动超过 12° 就认为"不是规整的 required 辐条"，直接排除。
            #   实测真辐条抖动 5.9°、凑数的 33° -> 12° 是很宽的判据。
            if jitter > 12.0:
                continue
            # 打分：对比度 × 峰宽 × 均匀度（抖动越小越高）
            score = contrast * min(1.0, minw / 15.0) * max(0.2, 1.0 - jitter / 45.0)
            if best is None or score > best["score"]:
                best = dict(score=score, r_min=float(lo), r_max=float(hi),
                            contrast=contrast, min_width=float(minw),
                            jitter=jitter,
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
    # ★ 单帧诊断：跳过采集，直接把装配位的**中间量**打出来后退出。
    #   为什么要它：c5 第二步的产出（ring/hub/cap 覆盖率、辐条峰数）
    #   全是"黑盒数字" —— 一旦是 0 或 nan，完全看不出是
    #   ①装配体没被检出 ②ROI 位置偏了 ③辐条环带不对 ④光照不行。
    #   实测踩过：辐条峰数众数 0、轮毂覆盖率 nan，只能靠猜。
    ap.add_argument("--diag", action="store_true",
                    help="单帧诊断装配位：打印检出情况/半径/各半径带峰值后退出")
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

    if args.diag:
        # ── 装配位单帧诊断 ────────────────────────────────────────
        frame = None
        for _ in range(40):
            fr, ff = grab_small(src)
            if fr is not None and float(fr.mean()) > 20:
                frame, f = fr, ff
                break
        if frame is None:
            print("★ 抓不到有效帧"); return 1
        scale = f.scale
        mmpp = tray.mm_per_px(scale)
        res, pose, psrc, _ident = q.segment_with_roi(seg, frame, scale=scale)
        poly = tray.assembly_polygon_px(scale)
        c_nom = poly.mean(axis=0)
        print()
        print("=" * 78)
        print(" 装配位单帧诊断   图 %s   mm/px=%.4f   位姿 src=%s"
              % (frame.shape[:2], mmpp, psrc))
        if pose is not None and getattr(pose, "ok", False):
            ac = pose.apply(np.array([wp.get("assembly.center", [-65.0, 80.0])], float))[0]
            c_cor = tray.mm_to_px(ac.reshape(1, 2), scale)[0]
            print("   θ=%+.1f° Δ=(%+.1f,%+.1f)mm  用件数=%d"
                  % (pose.theta_deg, pose.tx, pose.ty, pose.n_used))
            print("   装配位中心：标称 px(%.0f,%.0f)  位姿修正后 px(%.0f,%.0f)  差 %.1f px"
                  % (c_nom[0], c_nom[1], c_cor[0], c_cor[1],
                     float(np.hypot(c_cor[0] - c_nom[0], c_cor[1] - c_nom[1]))))
        print("=" * 78)
        # ① 装配位 ROI 内检出了什么
        blob = q._blob_in(res, poly)
        print(" ① 装配位 ROI 内的黄铜块：%s" % ("无" if blob is None else ""))
        if blob is not None:
            bb = blob.bbox
            print("      面积 %d px² = %.0f mm²   质心 px(%.1f,%.1f)  bbox %dx%d"
                  % (blob.area_px, blob.area_px * mmpp * mmpp,
                     blob.centroid[0], blob.centroid[1], bb[2], bb[3]))
            print("      等效直径 %.1f mm（轮盘外径应约 74mm）"
                  % (2.0 * np.sqrt(blob.area_px * mmpp * mmpp / np.pi)))
        # ② check_assembly 的产出
        a = q.check_assembly(res, scale)
        print()
        print(" ② check_assembly 产出：")
        print("      center_px=%s  ring=%.3f  hub=%.3f  cap=%.3f  spoke=%s"
              % (None if a.center_px is None else
                 "(%.0f,%.0f)" % (a.center_px[0], a.center_px[1]),
                 a.ring_coverage, a.hub_coverage, a.cap_coverage,
                 None if a.spoke is None else "count=%d ok=%s" % (a.spoke.count, a.spoke.ok)))
        # ③ 各半径带的峰值（看辐条到底在哪个半径上）
        if blob is not None:
            print()
            print(" ③ 以装配体质心为中心的**各半径带**峰数（找辐条在哪）：")
            print("      %-16s %-8s %-9s %s" % ("半径带 mm", "峰数", "对比度", "峰角度"))
            for lo in (8, 12, 16, 20, 24, 28, 32):
                hi = lo + 8.0
                prof = polar_profile(res.brass_mask, blob.centroid,
                                     lo / mmpp, hi / mmpp, n_angles=720, n_radii=9)
                pk = count_peaks(prof, k_mad=3.0, min_width_deg=6.0)
                med = float(np.median(prof)); peakv = float(np.percentile(prof, 95.0))
                angs = [round(p[0], 1) for p in pk][:10]
                mark = "  ← 8 峰" if len(pk) == 8 else ""
                print("      %-16s %-8d %-9.2f %s%s"
                      % ("%d~%d" % (lo, hi), len(pk), peakv - med, angs, mark))
        # ④ 画面内所有黄铜块
        print()
        print(" ④ 画面内所有黄铜连通域（含 ROI 外的）：")
        for b in (res.brass_blobs or []):
            mm = tray.px_to_mm(np.array([[b.centroid[0], b.centroid[1]]], float), scale)[0]
            inside = "装配位内" if q._blob_in(res, poly) is b else ""
            print("      %6.0fpx²=%6.0fmm²  质心tray-mm(%6.1f,%6.1f)  %s"
                  % (b.area_px, b.area_px * mmpp * mmpp, mm[0], mm[1], inside))

        # ⑤ 存一张标注图：ROI + 分割前景 + 检出块 + 各半径圈
        #   ★ 这些数字一旦自相矛盾（比如 hub 实心但 spoke=1），
        #     看一张图比继续推演快得多。实测踩过这个。
        try:
            import cv2
            from assemvision.imgio import imwrite_safe
            vis = frame.copy()
            ov = frame.copy()
            ov[res.brass_mask > 0] = (0, 255, 255)
            vis = cv2.addWeighted(vis, 0.6, ov, 0.4, 0)
            cv2.polylines(vis, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)],
                          True, (0, 0, 255), 2)
            if blob is not None:
                c = blob.centroid
                # 画各关键半径圈，直接看出"辐条/球头/外圈"各自在哪
                for r_mm, col in ((12.5, (255, 0, 0)),      # r_cap 顶盖
                                  (13.0, (255, 128, 0)),    # r_hub 轮毂
                                  (26.0, (0, 255, 0)),      # ring_band 内
                                  (30.0, (0, 255, 0)),      # ring_band 外
                                  (30.0, (255, 0, 255)),    # spoke_band_outer 内
                                  (35.0, (255, 0, 255)),    # spoke_band_outer 外
                                  (45.0, (0, 165, 255))):   # assembly.radius
                    rr = r_mm / mmpp
                    cv2.circle(vis, (int(round(c[0])), int(round(c[1]))),
                               int(round(rr)), col, 1)
                cv2.drawMarker(vis, (int(round(c[0])), int(round(c[1]))),
                               (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
            pc = np.round(c_nom).astype(int)
            cv2.drawMarker(vis, (int(pc[0]), int(pc[1])), (255, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 18, 2)
            cv2.putText(vis, "red=ROI  magenta+=nominal_center  circles=12.5/26/30/35/45mm",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(vis, "red=ROI  magenta+=nominal_center  circles=12.5/26/30/35/45mm",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            outp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "refs", "c5_assembly_diag.png")
            print()
            print(" ⑤ 已存标注图:", imwrite_safe(outp, vis), outp)
        except Exception as e:                                   # noqa: BLE001
            print(" ⑤ 存图失败:", e)
        src.close()
        return 0

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
        # ★ 用工位 ROI 分割（黄铜只在"料盘 + 装配位"外扩范围内检测）。
        #   实测踩过：木桌的 HSV 与黄铜重叠，不限制范围时整块木桌被判成黄铜，
        #   产生 13269mm² 的假连通域（真零件最大 3281mm²），
        #   会把"轮盘槽里有没有前景"这条判断带偏。见 AGENTS.md 2.12。
        res, _p, _s, _i = q.segment_with_roi(seg, frame, scale=f.scale)
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
        print(" 槽内最优环带: r = %.1f ~ %.1f mm  对比度 %.2f  最窄峰 %.1f°  角度抖动 %.1f°"
              % (best["r_min"], best["r_max"], best["contrast"],
                 best["min_width"], best.get("jitter", float("nan"))))
        print(" 峰角度:", best["peaks"])
        calib.set("assembly.radii_mm.spoke_band", [best["r_min"], best["r_max"]])
        calib.set("assembly.radii_mm.spoke_band_score", round(best["score"], 4))

    # ── 第一步补充：装配体上的辐条环带（球头端）也要在这里扫 ──────
    #   ★★ 为什么必须在**料盘里的轮盘**上扫，而不能等第二步在装配体上扫：
    #     装配体上的球头只有 6px 宽、边缘被绿垫混色，**brass 掩膜切不到**，
    #     而 metal 掩膜能切到。所以两次分别用对应的掩膜扫。
    print()
    print("=== 第一步补充：轮盘的球头带（用 metal 掩膜扫，供装配体数辐条用）===")
    f2, ff2 = grab_small(src)
    if f2 is not None:
        r2, _p, _s, _i = q.segment_with_roi(seg, f2, scale=ff2.scale)
        bl = q._blob_in(r2, tray.slot_polygon_px("slot_wheel", ff2.scale))
        if bl is not None and r2.metal_mask is not None:
            b_metal = sweep_band(r2.metal_mask, bl.centroid, tray.mm_per_px(ff2.scale),
                                 required=8, r_lo_mm=24.0, r_hi_mm=42.0)
            if b_metal:
                print(" 球头带（metal 掩膜）: r = %.1f ~ %.1f mm  对比度 %.2f  角度抖动 %.1f°"
                      % (b_metal["r_min"], b_metal["r_max"],
                         b_metal["contrast"], b_metal.get("jitter", float("nan"))))
                print(" 峰角度:", b_metal["peaks"])
                calib.set("assembly.radii_mm.spoke_band_outer",
                          [b_metal["r_min"], b_metal["r_max"]])
            else:
                print(" ★ metal 掩膜也扫不出规整的 8 峰，沿用配置里的现值")

    print()
    print("=== 第二步：装配位基线 ===")
    print("请把【完整装配体（底座+立柱+轮盘+顶盖）】放到装配位，保持不动。")
    ask("准备好按回车…")
    covs, rings, hubs, caps, spokes = [], [], [], [], []
    for i in range(args.frames):
        frame, f = grab_small(src)
        if frame is None:
            break
        # ★ 同样带工位 ROI：装配位在料盘左侧，木桌的假黄铜会污染
        #   ring_coverage / hub_coverage / cap_coverage 三个基线值。
        res, _p, _s, _i = q.segment_with_roi(seg, frame, scale=f.scale)
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
    # ★ nan 保护：hub/ring 在半径过小时 `_disc_coverage` 会返回 0，但若
    #   mask 为空数组求均值会得到 nan —— 写进 YAML 变成 `.nan`，
    #   之后读出来是 nan，比对全部静默失败。实测踩过（c5 首次跑）。
    def _safe_mean(xs):
        arr = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
        return float(arr.mean()) if arr.size else float("nan")
    print(" 外圈环带覆盖率 %.3f   轮毂覆盖率 %.3f   顶盖覆盖率 %.3f"
          % (_safe_mean(rings), _safe_mean(hubs), _safe_mean(caps)))
    print(" 辐条峰数（众数） %d   样本 %s" % (spk_mode, spokes[:12]))
    for key, val in (("ring_coverage", _safe_mean(rings)),
                     ("hub_coverage", _safe_mean(hubs)),
                     ("cap_coverage", _safe_mean(caps))):
        if np.isfinite(val):
            calib.set("assembly.%s" % key, round(val, 4))
        else:
            print(" ★ %s 全是 nan，**不写入**（避免污染配置）" % key)
    if spk_mode > 0:
        calib.set("assembly.spoke_peak_count", spk_mode)
    else:
        print(" ★ 辐条峰数众数为 0，**不写入**（0 峰说明判据没调好，写进去会污染基线）")

    # 装配体上的辐条环带（球头端）—— 用 metal 掩膜
    frame, f = grab_small(src)
    if frame is not None:
        res, _p, _s, _i = q.segment_with_roi(seg, frame, scale=f.scale)
        blob = q._blob_in(res, tray.assembly_polygon_px(f.scale))
        if blob is not None:
            b2 = sweep_band(res.metal_mask if res.metal_mask is not None else res.brass_mask,
                            blob.centroid, mmpp, required=8,
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
