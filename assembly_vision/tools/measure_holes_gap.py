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


def main() -> int:
    wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
    cal = load_config(os.path.join(ROOT, "config", "calibration.yaml"))
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    q = Qualifier(wp, tray, cal)
    seg = Segmenter(wp)
    mmpp = tray.mm_per_px(1.0)
    c6 = load_c6(cal)
    min_score = 0.55

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

    print("═" * 86)
    print("  `holes` 边界条件受控实验：只改 holes，看四个零件的得分怎么掉")
    print("═" * 86)
    print("  样本取自 assembly_choice.png 里真实检出的 %d 块零件" % len(samples))
    print("  min_score = %.2f（低于它判「认不出」）" % min_score)
    print()

    # c6 的 holes 区间
    print("  c6 的 holes 区间：", end="")
    print("  ".join("%s=%s" % (p, list(r.get("holes", ()))[:2]) for p, (r, _w) in c6.items()))
    print()

    for idx, (a, feats, cen) in enumerate(samples):
        print("  ── 样本 %d：%6.0fmm²  质心(%3.0f,%3.0f) ──" % (idx + 1, a, cen[0], cen[1]))
        print("     %-6s %s" % ("holes", "  ".join("%-14s" % p for p in c6)))
        for h in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0):
            f2 = dict(feats)
            f2["holes"] = h
            row = []
            for part, (ranges, pp) in c6.items():
                s = score_of(ranges, pp, f2)
                row.append("%-14s" % ("%.3f%s" % (s, "  ★认不出" if s < min_score
                                                  else ("  →" + part if False else ""))))
            # 找出这一档谁最高、够不够 min_score
            best = max(((p, score_of(r, w, f2)) for p, (r, w) in c6.items()),
                       key=lambda t: t[1])
            tag = best[0] if best[1] >= min_score else "★认不出"
            print("     %-6.0f %s   => 判为 %s(%.3f)"
                  % (h, "  ".join(row), tag, best[1]))
        print()

    print("═" * 86)
    print("  判读")
    print("═" * 86)
    print("  · 若某些 holes 取值下**四个零件全部 < min_score** -> 那就是空档，真实翻车点")
    print("  · 空档有几个、落在哪，决定该把区间放宽成什么样")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
