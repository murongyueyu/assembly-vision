# -*- coding: utf-8 -*-
"""零件检测的**真机**体检：两套模板在真机帧上的实际表现 + 特征稳定性。

★ 为什么必须看真机帧：
  合成序列几何与真机差 5.8~10.8mm，c6 是在真机上标的 ——
  拿合成序列比两套模板是**不公平的对比**（c6 必然吃亏，实测召仅 4.8%）。
  但已从区间看出两个**真问题**，要在真机上确认：
    ① `holes` 区间被 c6 收成宽度≈0 的**硬约束**（轮盘 [0,0]、底座 [1.99,2.02]）
       而 holes 是权重最高的特征（2.0）—— 光照/角度/遮挡一变就翻车
    ② 逐零件权重只给了立柱，其余三件靠全局默认

★ 做法：在**真机帧**上把每块零件的**特征值**打出来（不判身份），
  对照两套区间看"落在里面还是外面"，从而知道区间定得合不合理。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = r"D:\university resource\大三上\视觉\assembly_vision"
sys.path.insert(0, ROOT)

from assemvision.config import load_config                        # noqa: E402
from assemvision.core.checks import Qualifier                     # noqa: E402
from assemvision.core.part_id import (FEATURE_WEIGHTS, _band_score,   # noqa: E402
                                      default_templates)
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402

WATCH = ("holes", "area_mm2", "length_mm", "short_side_mm", "elongation",
         "solidity", "circularity")


def main() -> int:
    wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
    cal = load_config(os.path.join(ROOT, "config", "calibration.yaml"))
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    q = Qualifier(wp, tray, cal)
    seg = Segmenter(wp)
    mmpp = tray.mm_per_px(1.0)

    c6 = {}
    stored = cal.get("part_templates", None)
    if isinstance(stored, dict):
        for part, v in stored.items():
            if isinstance(v, dict) and "ranges" in v:
                c6[part] = {k: tuple(x) for k, x in v["ranges"].items()}
    builtin = {t.part: dict(t.ranges) for t in default_templates()}

    print("═" * 88)
    print("  真机帧上的零件特征值 —— 对照两套区间看「落没落在里面」")
    print("═" * 88)
    print("  ★ 只看**完整的零件**（不被手遮、面积接近名义值），避免拿残缺件评区间")

    # 挑几帧零件较全的
    files = ["assembly_choice.png", "assembly_left_check.png", "assembly_positions.png",
             "asm_raw.png"]
    for name in files:
        img = imread_safe(os.path.join(ROOT, "data", "refs", name))
        if img is None:
            continue
        res = seg.segment(img, scale=1.0)
        blobs = sorted(res.brass_blobs, key=lambda b: -b.area_px)
        print()
        print("  ── %s ──" % name)
        for b in blobs[:4]:
            a = b.area_px * mmpp * mmpp
            if a < 400:
                continue
            f = dict(b.features or {})
            f["area_mm2"] = a
            rw, rh = float(f.get("rect_w", 0)), float(f.get("rect_h", 0))
            f["length_mm"] = max(rw, rh) * mmpp
            f["short_side_mm"] = min(rw, rh) * mmpp
            print("     %6.0fmm²  质心(%3.0f,%3.0f)" % (a, b.centroid[0], b.centroid[1]))
            head = "        %-16s %-10s %-14s %-14s %s" % ("特征", "实测", "内置区间", "c6 区间", "落在哪")
            print(head)
            for k in WATCH:
                v = f.get(k)
                if v is None:
                    continue
                bs = builtin.get("P3_wheel", {}).get(k) or builtin.get("P1_base", {}).get(k)
                # 两套区间都用"哪个零件"的不好定 —— 这里只在**能对上**时打印
                row_b = row_c = "-"
                for part in c6:
                    br = builtin.get(part, {}).get(k)
                    cr = c6[part].get(k)
                    if cr is None:
                        continue
                    inb = (br[0] <= float(v) <= br[1]) if br else None
                    inc = (cr[0] <= float(v) <= cr[1])
                    row_b = ("✓" if inb else "✗") if br else "-"
                    row_c = "✓" if inc else "✗"
                    print("        %-16s %-10.3f %-14s %-14s 内置%s c6(%s)%s"
                          % (k, float(v),
                             "[%.2f,%.2f]" % br if br else "(无)",
                             "[%.2f,%.2f]" % cr,
                             row_b, part, row_c))
                    break
            print()

    # ── holes 的稳定性：这是最关键的一条 ──────────────────────
    print("═" * 88)
    print("  ★ `holes`（权重最高=2.0）在真机帧上的实测值 —— 看它稳不稳")
    print("═" * 88)
    print("  %-26s %-10s %-10s %-10s" % ("帧", "面积mm²", "holes", "会被 c6 区间接受?"))
    print("  " + "-" * 60)
    for name in files + ["now_raw.png", "tray_empty.png"]:
        img = imread_safe(os.path.join(ROOT, "data", "refs", name))
        if img is None:
            continue
        res = seg.segment(img, scale=1.0)
        for b in sorted(res.brass_blobs, key=lambda x: -x.area_px)[:2]:
            a = b.area_px * mmpp * mmpp
            if a < 400 or a > 8000:
                continue
            h = (b.features or {}).get("holes")
            acc = []
            for part, rng in c6.items():
                if "holes" in rng:
                    lo, hi = rng["holes"]
                    acc.append("%s:%s" % (part, "✓" if lo <= float(h or 0) <= hi else "✗"))
            print("  %-26s %-10.0f %-10s %s" % (name, a, h, " ".join(acc)))
    print()
    print("  ★ 判读：若同一个零件在不同帧里 holes 不一致，而 c6 区间宽度≈0，")
    print("     那么在真机上**只要光照/角度/遮挡稍有变化就会判『认不出』**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
