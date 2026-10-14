# -*- coding: utf-8 -*-
"""`holes` 硬区间到底会不会翻车 —— 直接把边界条件测出来，不靠推断。

★ 上一轮的教训：
    我从"真机上 holes 数出 0/1/2/3"+"c6 区间宽度≈0"**推断**"这是主要漏检来源"，
    但端到端消融（A vs B）在真机帧上是 **12/12 完全相同** —— 推断没有被证实。
    所以这一轮改成**直接把边界条件构造出来**：
      「如果某个零件的 holes 恰好落在区间的空档里，会发生什么？」

★ 做法：拿真机帧里**真实检出**的那块零件的特征，只把 `holes` 依次设成
  0,1,2,3,4…，看每个零件的得分怎么变、什么时候掉到 min_score 以下。
  这是**受控实验**：其它特征一个都不动。

★★ 2026-10 更新（见 docs/踩坑实录.md §2.30）：
    本工具现在**同时打印修复前与修复后**两列：
      `修复前(tol=0)` = 老行为（c6 的宽度≈0 硬区间）
      `修复后(tol=N)` = 现行为（整数容忍带，底座单独 ±2）
    这样"空档在哪、容忍带补掉了哪几个"一屏就能对照。
    ★ 结论已经用真机 9 帧证实：底座实测 holes 跨 **0/1/2/3/4**（镜面过曝把
      掩膜打穿 + 中央露出一小块黄铜"岛"），而 c6 模板写的是 [1.985, 2.015] ——
      单这一项就丢 11% 的分（底座 0.77 -> 0.65，而**轮盘拿 0.70**，两者贴住）。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = r"D:\university resource\大三上\视觉\assembly_vision"
sys.path.insert(0, ROOT)

from assemvision.config import load_config                        # noqa: E402
from assemvision.core.checks import Qualifier                     # noqa: E402
from assemvision.core.part_id import (FEATURE_WEIGHTS,            # noqa: E402
                                      _band_score, _discrete_band)
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402


def load_c6(cal):
    """读模板：{part: (ranges, weights, 该零件的 holes_tol)}。"""
    out = {}
    stored = cal.get("part_templates", None)
    if isinstance(stored, dict):
        for part, v in stored.items():
            if isinstance(v, dict) and "ranges" in v:
                w = dict(v.get("weights") or {})
                per_tol = float(w.pop("holes_tol", 0.0))
                out[part] = ({k: tuple(x) for k, x in v["ranges"].items()},
                             {k: float(x) for k, x in w.items()}, per_tol)
    return out


def score_of(ranges, pp, feats, holes_tol: float = 0.0):
    """★ `holes_tol>0` 时对 holes 用**整数容忍带**（与 part_id 现行为一致）。

    ★ 本工具故意把两套都留着：`holes_tol=0` 就是**修复前**的行为，
      对照着看才能说清"空档在哪、容忍带补掉了几个"。
    """
    total = wsum = 0.0
    for key, (lo, hi) in ranges.items():
        v = feats.get(key)
        if v is None or not np.isfinite(v):
            continue
        w = float(pp.get(key, FEATURE_WEIGHTS.get(key, 1.0)))
        if w <= 0:
            continue
        if key == "holes" and holes_tol > 0:
            dlo, dhi = _discrete_band(lo, hi, holes_tol)
            s = _band_score(float(v), dlo, dhi)
        else:
            s = _band_score(float(v), float(lo), float(hi))
        total += w * s
        wsum += w
    return (total / wsum) if wsum > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser("measure_holes_gap")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="整数容忍带（默认 1 = 当前配置）；0 = 复现修复前的行为")
    args = ap.parse_args()

    wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
    cal = load_config(os.path.join(ROOT, "config", "calibration.yaml"))
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    q = Qualifier(wp, tray, cal)
    seg = Segmenter(wp)
    mmpp = tray.mm_per_px(1.0)
    c6 = load_c6(cal)
    min_score = float(wp.get("part_id.min_score", 0.55))

    # 取一帧里**真实检出**的零件特征（用它们当"标准样本"）
    img = imread_safe(os.path.join(ROOT, "data", "refs", "assembly_choice.png"))
    res = seg.segment(img, scale=1.0)
    samples = []
    for b in sorted(res.brass_blobs, key=lambda x: -x.area_px):
        a = b.area_px * mmpp * mmpp
        if a < 500 or a > 8000:
            continue
        f = dict(b.features or {})
        f["area_mm2"] = a
        rw, rh = float(f.get("rect_w", 0)), float(f.get("rect_h", 0))
        f["length_mm"] = max(rw, rh) * mmpp
        f["short_side_mm"] = min(rw, rh) * mmpp
        samples.append((a, f, b.centroid))

    print("=" * 104)
    print("  `holes` 边界条件受控实验：只改 holes，看四个零件的得分怎么掉")
    print("=" * 104)
    print("  样本取自 assembly_choice.png 里真实检出的 %d 块零件" % len(samples))
    print("  min_score = %.2f（低于它判「认不出」）" % min_score)
    print("  ★ 每格格式： 修复前(tol=0)  ->  修复后(tol=N)      ★ = 低于 min_score")
    print()

    print("  c6 的 holes 区间与其**整数容忍带**：")
    for p, (r, _w, per_tol) in c6.items():
        if "holes" not in r:
            continue
        lo, hi = r["holes"]
        t = per_tol if per_tol > 0 else args.tol
        band = _discrete_band(lo, hi, t)
        acc = [n for n in range(0, 10)
               if _band_score(float(n), band[0], band[1]) == 1.0]
        print("     %-10s 区间 [%9.5f, %8.5f]  tol=%g  整数带 [%g, %g]  满分接受 %s"
              % (p, lo, hi, t, band[0], band[1], acc))
    print()

    for idx, (a, feats, cen) in enumerate(samples):
        print("  -- 样本 %d：%6.0fmm2  质心(%3.0f,%3.0f) --"
              % (idx + 1, a, cen[0], cen[1]))
        print("     %-6s %s" % ("holes", "  ".join("%-24s" % p for p in c6)))
        for h in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0):
            f2 = dict(feats)
            f2["holes"] = h
            cells = []
            for _part, (ranges, pp, per_tol) in c6.items():
                s_old = score_of(ranges, pp, f2, 0.0)
                s_new = score_of(ranges, pp, f2, per_tol if per_tol > 0 else args.tol)
                cells.append("%-24s" % ("%.3f%s -> %.3f%s" % (
                    s_old, "*" if s_old < min_score else " ",
                    s_new, "*" if s_new < min_score else " ")))
            best_old = max(((p, score_of(r, w, f2, 0.0)) for p, (r, w, _t) in c6.items()),
                           key=lambda t: t[1])
            best_new = max(((p, score_of(r, w, f2, t)) for p, (r, w, t) in c6.items()),
                           key=lambda t: t[1])
            tag_old = best_old[0] if best_old[1] >= min_score else "认不出"
            tag_new = best_new[0] if best_new[1] >= min_score else "认不出"
            print("     %-6.0f %s   =>  %s(%.3f)  ->  %s(%.3f)"
                  % (h, "  ".join(cells), tag_old, best_old[1], tag_new, best_new[1]))
        print()

    print("=" * 104)
    print("  判读")
    print("=" * 104)
    print("  · 修复前（tol=0）：某些 holes 取值下**四个零件全部 < min_score** -> 那就是空档")
    print("  · 修复后：空档被整数容忍带补掉，同时保留区分度（轮盘的带 {0,1} 最窄）")
    print("  · 依据与实测数字（底座跨 0~4 的根因是镜面过曝）：docs/踩坑实录.md §2.30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
