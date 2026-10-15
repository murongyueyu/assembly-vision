# -*- coding: utf-8 -*-
"""色域特征到底有没有真的参与识别 —— **消融实验**（回答 docs/现状与交接.md 的 P2-1）。

★ 为什么"权重占比"不算证据：
  配置里色域 5 个特征占识别总分约 **31%** 权重（4.8 / 15.3），但那只是"写了多少"。
  四个零件的 `c_hue_mean` 区间几乎重合（都是黄铜：32.0~34.6，只差 1~2°），
  很可能这份权重**没有区分度、白占着**。
  要证明它有用只有一条路：**把它关掉，看准确率掉不掉。**

★ 分数的定义（`PartTemplate.score`，已核对源码）：
      w = 逐零件weights.get(k, FEATURE_WEIGHTS.get(k, 1.0))
      score = Σ(w·s) / Σw          <- **加权平均，归一化**
  所以"把色域权重置 0"= 它既不贡献也不占分母，**正是"关掉"的干净定义**。

★ 本工具**不动 Qualifier 的缓存分类器**（`_classifier()` 有 `_clf` 缓存，
  改了会污染后续调用）。改为直接拿 `ranges` + 改过的权重，
  用官方 `_band_score` 自己算分 —— 与运行时用同一套打分函数。

★ 三档：full / no_color / only_color
★ 真值仍用「零件落在哪个槽」（一料位一件，与分类器无关）。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from assemvision.config import load_config                        # noqa: E402
from assemvision.core.checks import Qualifier                     # noqa: E402
from assemvision.core.part_id import FEATURE_WEIGHTS, _band_score  # noqa: E402
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402

COLOR_KEYS = ("c_hue_mean", "c_hue_std", "c_sat_mean", "c_val_mean", "c_sat_hi_frac")


def eff_weight(key, per_part):
    """与 `PartTemplate.score` 完全一致的权重解析。"""
    return float(per_part.get(key, FEATURE_WEIGHTS.get(key, 1.0)))


def score_with_mode(ranges, per_part, feats, mode, min_score):
    """按 mode 算 (part, score)。mode: full | no_color | only_color"""
    total = wsum = 0.0
    for key, (lo, hi) in ranges.items():
        v = feats.get(key)
        if v is None or not np.isfinite(v):
            continue
        w = eff_weight(key, per_part)
        if mode == "no_color" and key in COLOR_KEYS:
            w = 0.0
        elif mode == "only_color" and key not in COLOR_KEYS:
            w = 0.0
        if w <= 0:
            continue
        total += w * _band_score(float(v), float(lo), float(hi))
        wsum += w
    if wsum <= 0:
        return None, 0.0
    s = total / wsum
    return (None if s < min_score else "HIT"), s


def load_templates(cal):
    """模板来源：优先标定文件的 `part_templates`，没有就用内置初值。

    ★★ 实测发现的重要事实：`calibration.synthetic.yaml` **没有 `part_templates`**
      （它是 make_synthetic 生成的，不含 c6 的产物），所以
      - 合成序列跑端到端时用的是 `part_id.py` 的**内置初值模板**；
      - 真机跑的是**c6 标定出来的模板**。
      两者**不是同一套模板** —— 所以"合成序列上的准确率"与"真机准确率"
      除了几何差异，还差着模板本身。这里把这条写清楚。
    """
    stored = None
    try:
        stored = cal.get("part_templates", None)
    except KeyError:
        stored = None
    if not isinstance(stored, dict):
        from assemvision.core.part_id import default_templates    # noqa: E402
        return [(t.part, dict(t.ranges), dict(t.weights)) for t in default_templates()], "内置初值"
    out = []
    for part, v in stored.items():
        if not isinstance(v, dict) or "ranges" not in v:
            continue
        out.append((part,
                    {k: tuple(x) for k, x in v["ranges"].items()},
                    {k: float(x) for k, x in (v.get("weights") or {}).items()}))
    return out, "calibration.part_templates"


def eval_mode(wp, cal, tray, q, seg, seq_dir, mode, tpls, min_score):
    """"该槽有没有件"由覆盖率（与分类器无关）当真值；身份由本档权重重算。"""
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
            truth = float(s.coverage or 0.0) > 0.10     # 与分类器无关的存在性真值
            # 取分数最高的模板；分数 < min_score 则判"认不出"
            top = max(((p, score_with_mode(r, w, feats, mode, 0.0)[1])
                       for p, r, w in tpls), key=lambda t: t[1], default=(None, 0.0))
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
    ap = argparse.ArgumentParser("measure_color_ablation")
    ap.add_argument("--workpiece", default=os.path.join(ROOT, "config", "workpiece.yaml"))
    ap.add_argument("--calibration",
                    default=os.path.join(ROOT, "config", "calibration.synthetic.yaml"))
    ap.add_argument("--seq", default=os.path.join(ROOT, "data", "synthetic_seq"))
    args = ap.parse_args()

    wp = load_config(args.workpiece)
    cal = load_config(args.calibration)
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    q = Qualifier(wp, tray, cal)
    seg = Segmenter(wp)

    tpls, src = load_templates(cal)
    try:
        min_score = float(cal.get("part_id.min_score", 0.55))
    except KeyError:
        min_score = 0.55

    print("═" * 80)
    print("  色域特征消融实验 —— 问的是「c_* 特征在**零件间**识别里有没有区分度」")
    print("═" * 80)
    print("  ★ 注意问题边界（用户澄清）：色域分割的**主要用途是")
    print("     ① 零件 vs 背景   ② 零件 vs 手套**，不是区分零件与零件。")
    print("     本节只测「零件间识别」这一层，测出「没贡献」**不等于色域没用** ——")
    print("     它在前两件事上是不可替代的（见分割层阈值与 §2.1/§2.3）。")
    print()
    print("  模板来源 = %s（%d 个）" % (src, len(tpls)))
    print("  打分函数 = Σ(w·s)/Σw（加权平均，归一化）—— 与 Runtime 的 PartTemplate.score 同源")
    print("  色域特征: %s" % (", ".join(COLOR_KEYS),))
    print()
    print("  %-12s %-5s %-5s %-5s %-5s %-10s %-9s"
          % ("档位", "TP", "TN", "FN", "FP", "准确率", "召回率"))
    print("  " + "-" * 68)

    results = {}
    for mode in ("full", "no_color", "only_color"):
        st, wrong = eval_mode(wp, cal, tray, q, seg, args.seq, mode, tpls, min_score)
        results[mode] = (st, wrong)
        tot = st["tp"] + st["tn"] + st["fn"] + st["fp"]
        rec = st["tp"] / max(1, st["tp"] + st["fn"])
        acc = 100.0 * (st["tp"] + st["tn"]) / max(1, tot)
        print("  %-12s %-5d %-5d %-5d %-5d %-9.1f%% %-8.1f%%"
              % (mode, st["tp"], st["tn"], st["fn"], st["fp"], acc, 100.0 * rec))

    f_, n_, o_ = results["full"][0], results["no_color"][0], results["only_color"][0]
    print()
    print("  ── 判读 ──")
    d = f_["tp"] - n_["tp"]
    print("  full -> no_color 认对数: %d -> %d（%+d）" % (f_["tp"], n_["tp"], -d))
    if d == 0:
        print("     ★★ **色域特征对识别没有可测贡献** —— 关掉它一个都不少认。")
        print("        结论：这份 ~31%% 的权重是白占的；真正区分零件的是形状/面积。")
    else:
        print("     ★ 色域有实际贡献：关掉后少认 %d 个。" % d)
    print("  only_color 认对 %d 个 -> %s"
          % (o_["tp"], "单靠颜色不足以区分（符合预期：都是黄铜）"
             if o_["tp"] < f_["tp"] else "单靠颜色也能认（意外，需复查）"))
    if d and results["no_color"][1]:
        print()
        print("  no_color 少认的明细（前 10 条）:")
        for r in results["no_color"][1][:10]:
            print("     %-14s %-14s 真值 %-10s -> %-10s 分 %.2f" % r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
