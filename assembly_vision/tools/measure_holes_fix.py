# -*- coding: utf-8 -*-
"""修 `holes` 的硬约束：离散量必须容忍 ±1（先量出改法与收益，再写进配置）。

★ 已证实的根因（受控实验，见 tools/measure_holes_gap.py）：
    `count_holes` 在真机上对同一零件数出 **0/1/2/3**（帧间抖动），
    而 c6 把 `holes` 收成宽度≈0 的区间：
        P3_wheel [-0.00015, 0.00015]   P2_column [-0.00015, 0.00015]
        P4_cap   [ 1.985,   2.015  ]   P1_base   [ 1.985,   2.015  ]
    实测 **holes=1 时四个零件全部 < min_score = 认不出**（样本1 得 0.548）。

★ 为什么影响这么大：
    holes 是**权重最高**的特征（2.0 / wsum≈13.5）。越界后
    `_band_score` 衰减到 ~0，单这一项就把总分拉掉 **0.148**。
    而真机特征本身已经偏低，两者叠加就跌破 min_score。

★ 修法原则（离散量）：
    `holes` 是**整数计数**，不是连续量。噪声至少 ±1，
    所以区间必须覆盖「实测值 ±1」，否则一次数错就全盘否掉。
    轮盘/立柱真实 0 孔 -> 放宽到 **[-0.0002, 1.5]**（接受 0、1）
    顶盖/底座真实 2 孔 -> 放宽到 **[1.5, 3.5]**（接受 2、3；1 仍拒）

★ 本工具**只在内存里改**，先量出收益，再决定是否写进配置。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = r"D:\university resource\大三上\视觉\assembly_vision"
sys.path.insert(0, ROOT)

from assemvision.config import load_config                        # noqa: E402
from assemvision.core.checks import Qualifier                     # noqa: E402
from assemvision.core.part_id import FEATURE_WEIGHTS, _band_score  # noqa: E402
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402

GOOD_FRAMES = ["assembly_choice.png", "assembly_left_check.png", "assembly_positions.png"]
# 有 holes 真值可判的零件：轮盘/立柱 0 孔，顶盖/底座 2 孔
ZERO_HOLE = ("P3_wheel", "P2_column")
TWO_HOLE = ("P4_cap", "P1_base")


def load_c6(cal):
    out = {}
    stored = cal.get("part_templates", None)
    if isinstance(stored, dict):
        for part, v in stored.items():
            if isinstance(v, dict) and "ranges" in v:
                out[part] = ({k: tuple(x) for k, x in v["ranges"].items()},
                             {k: float(x) for k, x in (v.get("weights") or {}).items()})
    return out


def score_of(ranges, pp, feats):
    total = wsum = 0.0
    for key, (lo, hi) in ranges.items():
        v = feats.get(key)
        if v is None or not np.isfinite(v):
            continue
        w = float(pp.get(key, FEATURE_WEIGHTS.get(key, 1.0)))
        if w <= 0:
            continue
        total += w * _band_score(float(v), float(lo), float(hi))
        wsum += w
    return (total / wsum) if wsum > 0 else 0.0


def set_holes(tpls, zero_rng, two_rng):
    out = {}
    for part, (ranges, pp) in tpls.items():
        r = dict(ranges)
        if "holes" in r:
            r["holes"] = zero_rng if part in ZERO_HOLE else two_rng
        out[part] = (r, dict(pp))
    return out


def eval_real(tpls, min_score, q, seg, wp):
    tp = fn = fp = tn = 0
    for name in GOOD_FRAMES:
        img = imread_safe(os.path.join(ROOT, "data", "refs", name))
        if img is None:
            continue
        res, pose, _s, _i = q.segment_with_roi(seg, img, scale=1.0)
        for s in q.check_slots(res, 1.0, bgr=img, pose=pose):
            want = str(s.part or "")
            truth = float(s.coverage or 0.0) > 0.10
            top = max(((p, score_of(r, w, dict(s.features or {})))
                       for p, (r, w) in tpls.items()), key=lambda t: t[1],
                      default=(None, 0.0))
            got = top[0] if top[1] >= min_score else None
            if truth and got == want:
                tp += 1
            elif truth:
                fn += 1
            elif got is None:
                tn += 1
            else:
                fp += 1
    return dict(tp=tp, tn=tn, fn=fn, fp=fp)


def main() -> int:
    wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
    cal = load_config(os.path.join(ROOT, "config", "calibration.yaml"))
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    q = Qualifier(wp, tray, cal)
    seg = Segmenter(wp)
    mmpp = tray.mm_per_px(1.0)
    c6 = load_c6(cal)
    ms = 0.55

    print("═" * 88)
    print("  修 `holes` 硬约束：离散量容忍 ±1 —— 先量收益")
    print("═" * 88)
    print("  c6 原区间: ", "  ".join("%s=%s" % (p, [round(x, 4) for x in r.get("holes", ())])
                                  for p, (r, _w) in c6.items()))
    print()

    # ── ① 一个真实样本上的确定性对照（取样本1：底座 2993mm²）──
    img = imread_safe(os.path.join(ROOT, "data", "refs", "assembly_choice.png"))
    res = seg.segment(img, scale=1.0)
    sample = None
    for b in sorted(res.brass_blobs, key=lambda x: -x.area_px):
        a = b.area_px * mmpp * mmpp
        if 2000 < a < 4000:
            f = dict(b.features or {})
            f["area_mm2"] = a
            rw, rh = float(f.get("rect_w", 0)), float(f.get("rect_h", 0))
            f["length_mm"] = max(rw, rh) * mmpp
            f["short_side_mm"] = min(rw, rh) * mmpp
            sample = f
            break

    variants = [
        ("A 现状（宽度≈0）", c6),
        ("B 只容忍 ±1", set_holes(c6, (-0.0002, 1.5), (1.5, 3.5))),
        ("C 容忍 ±1 且放宽权重", set_holes(c6, (-0.0002, 2.5), (1.5, 4.5))),
    ]

    if sample is not None:
        print("  ── ① 受控对照（同一个真实样本，只改 holes 区间）──")
        print("     样本的特征: holes 真值待测；其它特征固定")
        print()
        print("     %-22s %s" % ("holes 实测值", "  ".join("%-16s" % v[0] for v in variants)))
        for h in (0.0, 1.0, 2.0, 3.0, 4.0):
            f2 = dict(sample)
            f2["holes"] = h
            cells = []
            for _label, tpls in variants:
                top = max(((p, score_of(r, w, f2)) for p, (r, w) in tpls.items()),
                          key=lambda t: t[1])
                cells.append("%-16s" % ("%s(%.3f)%s" % (top[0], top[1],
                                                        "★" if top[1] < ms else " ✓")))
            print("     %-22.0f %s" % (h, "  ".join(cells)))
        print("     （★ = 低于 min_score -> 判「认不出」）")
        print()

    # ── ② 真机帧端到端 ────────────────────────────────────────
    print("  ── ② 真机帧端到端（真值 = 零件落在哪个槽）──")
    print("  %-24s %-4s %-4s %-4s %-4s %s" % ("区间方案", "TP", "TN", "FN", "FP", "召回率"))
    print("  " + "-" * 62)
    for label, tpls in variants:
        st = eval_real(tpls, ms, q, seg, wp)
        rec = st["tp"] / max(1, st["tp"] + st["fn"])
        print("  %-24s %-4d %-4d %-4d %-4d %.1f%%"
              % (label, st["tp"], st["tn"], st["fn"], st["fp"], 100.0 * rec))

    print()
    print("  ── 判读 ──")
    print("     ① 若 A 在 holes=1 处出现 ★（全部认不出），而 B 不再出现 -> 修法有效")
    print("     ② 若 B 的召回率 >= A -> 放宽不损失真机准确率，可以放心写进配置")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
