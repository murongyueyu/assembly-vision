# -*- coding: utf-8 -*-
"""c4 槽位基线：放入一整套已知合格的标准件，逐槽采基线。

产出（写进 calibration.yaml）
    slots.<id>.coverage            标准件放入时的覆盖率
    slots.<id>.features.*          旋转不变特征的均值 ± 3σ
    slots.<id>.has_hole            该槽标准件是否有内孔

★ 阈值不写在代码里，全部从这份基线读 —— 换工件只要重跑本脚本。

用法
    python calib/c4_slot_baseline.py --frames 20
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import (ask, grab_small, load_all, open_source, paths,   # noqa: E402
                   save_calib)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.core.checks import Qualifier                       # noqa: E402
from assemvision.core.roi import TrayMap                            # noqa: E402
from assemvision.core.segment import Segmenter                      # noqa: E402

FEATS = ("area_px", "circularity", "solidity", "aspect", "elongation",
         "straightness", "end_width_ratio", "holes", "rect_w", "rect_h")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--camera", default="")
    ap.add_argument("--workpiece", default="")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cam, wp, calib = load_all(args.camera, args.workpiece)
    if "tray_map" not in calib.data:
        print("还没有料盘单应，请先跑 calib/c3_tray_homography.py")
        return 1

    H = np.asarray(calib.get("tray_map.homography"), np.float64)
    ref_scale = float(calib.get("tray_map.scale", 1.0))
    tray = TrayMap(wp, homography=H, ref_scale=ref_scale)
    seg = Segmenter(wp)
    q = Qualifier(wp, tray, calib)

    src = open_source(cam, args)
    acc = {s["id"]: [] for s in tray.slots}

    print("请把【一整套已知合格的标准件】按料位放好，保持不动。")
    ask("准备好了按回车开始采集…")
    n = 0
    while n < args.frames:
        frame, f = grab_small(src)
        if frame is None:
            print("抓帧失败"); break
        res = seg.segment(frame, scale=f.scale)
        slots = q.check_slots(res, f.scale)
        for sr in slots:
            if sr.present:
                acc[sr.slot_id].append((sr.coverage, sr.features))
        n += 1
        if n % 5 == 0:
            print("  已采 %d/%d 帧，各槽有效样本 %s"
                  % (n, args.frames, {k: len(v) for k, v in acc.items()}))
    src.close()

    print()
    print("=" * 70)
    ok_all = True
    for sid, samples in acc.items():
        if len(samples) < max(3, args.frames // 4):
            print(" !! %s 有效样本太少（%d），检查该槽是否放了件、ROI 是否正确"
                  % (sid, len(samples)))
            ok_all = False
            continue
        covs = np.array([s[0] for s in samples])
        print(" %s：" % sid)
        print("    覆盖率 均值 %.3f  σ %.3f  (min %.3f max %.3f)"
              % (covs.mean(), covs.std(), covs.min(), covs.max()))
        calib.set("slots.%s.coverage" % sid, round(float(covs.mean()), 4))
        calib.set("slots.%s.coverage_std" % sid, round(float(covs.std()), 4))
        calib.set("slots.%s.frames" % sid, len(samples))
        # 特征
        for fkey in FEATS:
            vals = np.array([s[1].get(fkey, np.nan) for s in samples], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            mu, sd = float(vals.mean()), float(vals.std())
            calib.set("slots.%s.features.%s.mean" % (sid, fkey), round(mu, 5))
            calib.set("slots.%s.features.%s.std" % (sid, fkey), round(max(sd, 1e-6), 5))
            print("      %-16s %.3f ± %.3f" % (fkey, mu, sd))
    print("=" * 70)
    if ok_all:
        print("已写入:", save_calib(calib, args.out))
    else:
        print("有槽位样本不足，未写入。请修正后重跑。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
