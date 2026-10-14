# -*- coding: utf-8 -*-
"""零件检测基础体检：现在到底用的是哪套模板？哪套更好？

★ 起因：前面两次端到端测量的**召回率对不上**
    - 用 `calibration.yaml`（含 c6 标定模板）     -> 召回率 75.0%
    - 用 `calibration.synthetic.yaml`（**无**模板 -> 走内置初值）-> 召回率 100%
  内置初值本该是"粗值"，却比 c6 标定的还好 —— 这不正常，必须查清。

★ 查三件事：
  ① 两套模板的**区间**逐特征对比（内置 vs c6 标定），看差在哪
  ② 在同一批帧上**分别用两套模板**测准确率/召回率（同帧同真值，只换模板）
  ③ 顺带看 c6 那套模板有没有"零件放错也照写"的痕迹
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = r"D:\university resource\大三上\视觉\assembly_vision"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from assemvision.config import load_config                        # noqa: E402
from assemvision.core.checks import Qualifier                     # noqa: E402
from assemvision.core.part_id import (FEATURE_WEIGHTS, _band_score,   # noqa: E402
                                      default_templates)
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402


def load_c6(cal):
    stored = cal.get("part_templates", None)
    if not isinstance(stored, dict):
        return {}
    out = {}
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


def eval_templates(tpls, min_score, q, seg, seq_dir, wp, cal):
    files = sorted(f for f in os.listdir(seq_dir) if f.endswith(".png"))
    tp = fn = fp = tn = 0
    wrong = []
    for name in files:
        img = imread_safe(os.path.join(seq_dir, name))
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
                wrong.append((name, s.slot_id, want, got, round(top[1], 2)))
            elif got is None:
                tn += 1
            else:
                fp += 1
    return dict(tp=tp, tn=tn, fn=fn, fp=fp), wrong


def main() -> int:
    wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
    cal = load_config(os.path.join(ROOT, "config", "calibration.yaml"))
    c6 = load_c6(cal)
    builtin = {t.part: (dict(t.ranges), dict(t.weights)) for t in default_templates()}

    print("═" * 82)
    print("  ① 两套模板的区间对比（内置初值 vs c6 标定）")
    print("═" * 82)
    for part in sorted(set(builtin) | set(c6)):
        print("  ── %s ──" % part)
        br = builtin.get(part, ({}, {}))[0]
        cr = c6.get(part, ({}, {}))[0]
        if not cr:
            print("      ★ c6 里没有这个零件 -> 运行时**走内置初值**")
            continue
        keys = sorted(set(br) | set(cr))
        print("      %-16s %-26s %-26s %s" % ("特征", "内置初值", "c6 标定", "差"))
        for k in keys:
            b = br.get(k)
            c = cr.get(k)
            bs = "[%.2f, %.2f]" % b if b else "(无)"
            cs = "[%.2f, %.2f]" % c if c else "(无)"
            d = ""
            if b and c:
                d = "宽 %.2f -> %.2f" % (b[1] - b[0], c[1] - c[0])
            print("      %-16s %-26s %-26s %s" % (k, bs, cs, d))
        # 权重差异
        bw = builtin.get(part, ({}, {}))[1]
        cw = c6.get(part, ({}, {}))[1]
        if cw:
            print("      逐零件权重(c6): %s" % (cw,))
    print()
    print("  内置初值零件数 %d ；c6 标定零件数 %d" % (len(builtin), len(c6)))

    # ── ② 同帧同真值，只换模板 ────────────────────────────────
    print()
    print("═" * 82)
    print("  ② 同一批帧、同一真值，只换模板（合成序列，精确真值）")
    print("═" * 82)
    tray = TrayMap(wp, homography=np.asarray(
        load_config(os.path.join(ROOT, "config", "calibration.synthetic.yaml"))
        .get("tray_map.homography"), float),
        ref_scale=float(load_config(os.path.join(
            ROOT, "config", "calibration.synthetic.yaml")).get("tray_map.scale", 1.0)))
    q = Qualifier(wp, tray, load_config(os.path.join(
        ROOT, "config", "calibration.synthetic.yaml")))
    seg = Segmenter(wp)
    seq = os.path.join(ROOT, "data", "synthetic_seq")

    print("  %-30s %-5s %-5s %-5s %-5s %-10s %s"
          % ("模板", "TP", "TN", "FN", "FP", "召回率", "误检"))
    print("  " + "-" * 76)
    for label, tpls in (("内置初值（default_templates）", builtin),
                        ("c6 标定（calibration.yaml）", c6)):
        if not tpls:
            print("  %-30s (没有模板，跳过)" % label)
            continue
        st, wrong = eval_templates(tpls, 0.55, q, seg, seq, wp, None)
        rec = st["tp"] / max(1, st["tp"] + st["fn"])
        print("  %-30s %-5d %-5d %-5d %-5d %-9.1f%% FP=%d"
              % (label, st["tp"], st["tn"], st["fn"], st["fp"], 100.0 * rec, st["fp"]))
        if wrong:
            print("      ↳ 漏/错 明细（前 6）:")
            for r in wrong[:6]:
                print("         %-14s %-14s 真值 %-10s -> %-10s 分 %.2f" % r)
    print()
    print("  ★ 判读：若内置初值明显更好 -> **c6 那套模板不能直接用**，")
    print("     要么重跑 c6（先修「零件放错也照写」），要么按实测重定区间。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
