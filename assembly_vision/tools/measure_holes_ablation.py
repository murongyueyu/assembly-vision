# -*- coding: utf-8 -*-
"""零件检测的核心缺陷定量：`holes` 区间宽度≈0 造成多少漏检？

★ 已证实的根因（真机实测）：
    `count_holes` 在真机上对**同一个零件**数出 0/1/2/3 个孔（帧间抖动），
    而 c6 标定把 `holes` 区间收成了宽度≈0 的硬约束：
        P3_wheel [-0.00, 0.00]   P2_column [-0.00, 0.00]
        P4_cap   [ 1.99, 2.02]   P1_base   [ 1.99, 2.02]
    而 `holes` 是**权重最高**的特征（2.0）—— 差 1 就整条不满足。

★ 本工具做**特征区间消融**（不改代码、只改内存里的区间），回答：
    A. 现状（c6 原区间）        在真机帧上能认对几个
    B. 只把 `holes` 放宽         能救回多少        <- 关键
    C. 再把权重最高的几个也放宽  上限在哪
    D. 内置初值区间             对照

★ 真值仍用「零件落在哪个槽」（一料位一件，与分类器无关）。
★ 只用**真机帧**里"零件确实在槽里"的那些（用手工挑过的干净帧）。
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
from assemvision.core.part_id import (FEATURE_WEIGHTS, _band_score,   # noqa: E402
                                      default_templates)
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402

# ★ 真机帧里**四个零件确实在各自槽里**的那些（其余是调试/半空帧，见 §2.19）
GOOD_FRAMES = ["assembly_choice.png", "assembly_left_check.png", "assembly_positions.png"]


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


def widen(tpls, keys, lo, hi):
    """把指定特征的区间放宽到 [lo, hi]（复制一份，不动原数据）。"""
    out = {}
    for part, (ranges, pp) in tpls.items():
        r2 = dict(ranges)
        for k in keys:
            if k in r2:
                r2[k] = (min(r2[k][0], lo), max(r2[k][1], hi))
        out[part] = (r2, dict(pp))
    return out


def eval_on_real(tpls, min_score, q, seg, tray, wp, mmpp):
    """真机帧上评：真值 = 零件落在哪个槽。"""
    slots = {s["id"]: s for s in (wp.get("slots") or [])}
    tp = fn = fp = tn = 0
    wrong = []
    for name in GOOD_FRAMES:
        img = imread_safe(os.path.join(ROOT, "data", "refs", name))
        if img is None:
            continue
        res, pose, _s, _i = q.segment_with_roi(seg, img, scale=1.0)
        for s in q.check_slots(res, 1.0, bgr=img, pose=pose):
            want = str(s.part or "")
            feats = dict(s.features or {})
            truth = float(s.coverage or 0.0) > 0.10
            top = max(((p, score_of(r, w, feats)) for p, (r, w) in tpls.items()),
                      key=lambda t: t[1], default=(None, 0.0))
            got = top[0] if top[1] >= min_score else None
            if truth and got == want:
                tp += 1
            elif truth:
                fn += 1
                wrong.append((name, s.slot_id, want, got, round(top[1], 3)))
            elif got is None:
                tn += 1
            else:
                fp += 1
    return dict(tp=tp, tn=tn, fn=fn, fp=fp), wrong


def main() -> int:
    ap = argparse.ArgumentParser("measure_holes_ablation")
    ap.add_argument("--min-score", type=float, default=0.55)
    args = ap.parse_args()

    wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
    cal = load_config(os.path.join(ROOT, "config", "calibration.yaml"))
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    q = Qualifier(wp, tray, cal)
    seg = Segmenter(wp)
    mmpp = tray.mm_per_px(1.0)

    c6 = load_c6(cal)
    builtin = {t.part: (dict(t.ranges), dict(t.weights)) for t in default_templates()}

    variants = [
        ("A 现状（c6 原区间）", c6),
        ("B 只放宽 holes", widen(c6, ("holes",), 0.0, 16.0)),
        ("C 再放宽 circularity/solidity",
         widen(widen(c6, ("holes",), 0.0, 16.0), ("circularity", "solidity"), 0.0, 1.0)),
        ("D 内置初值区间（对照）", builtin),
    ]

    print("═" * 84)
    print("  零件检测：`holes` 硬区间造成的漏检 —— 真机帧上的特征区间消融")
    print("═" * 84)
    print("  真值 = 「零件落在哪个槽」（一料位一件，与分类器无关）")
    print("  真机帧 = %s" % (GOOD_FRAMES,))
    print("  min_score = %.2f" % args.min_score)
    print()
    print("  %-32s %-4s %-4s %-4s %-4s %-9s %s"
          % ("区间方案", "TP", "TN", "FN", "FP", "召回率", "认对数/该有件数"))
    print("  " + "-" * 78)
    for label, tpls in variants:
        st, wrong = eval_on_real(tpls, args.min_score, q, seg, tray, wp, mmpp)
        rec = st["tp"] / max(1, st["tp"] + st["fn"])
        print("  %-32s %-4d %-4d %-4d %-4d %-8.1f%% %d/%d"
              % (label, st["tp"], st["tn"], st["fn"], st["fp"], 100.0 * rec,
                 st["tp"], st["tp"] + st["fn"]))
        if wrong and label.startswith("A"):
            print("      ↳ A 的漏/错明细:")
            for r in wrong:
                print("         %-24s %-14s 真值 %-10s -> %-10s 分 %.3f" % r)

    print()
    print("  ★ 判读：")
    print("     B 与 A 的差 = **放宽 holes 能救回多少**")
    print("     若 B 明显好于 A -> 现状的 `holes` 硬区间就是主要漏检来源，应放宽")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
