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
    reject = {s["id"]: 0 for s in tray.slots}
    while n < args.frames:
        frame, f = grab_small(src)
        if frame is None:
            print("抓帧失败"); break
        # ★ 用两遍分割（几何 ROI -> 定位姿 -> 位姿修正的 ROI 再分）：
        #   木桌的 HSV 与黄铜重叠，不限制范围会出现 13269mm² 的假连通域，
        #   把覆盖率与零件识别全带偏。见 config 的 work_roi_margin_mm。
        res, _pose, _src, _ident = q.segment_with_roi(seg, frame, scale=f.scale)
        # ★★ 必须传 bgr=frame：
        #   不传的话 check_slots 算不出**色域特征**，而且 `bgr=None` 时
        #   features_from_blob 也不会补 area_mm2/色域，于是零件识别会用
        #   "内置初值模板"去比对，跟刚标定好的真实模板对不上 ——
        #   实测症状：明明零件都放好了，slot_cap/slot_base 却一个样本都采不到。
        slots = q.check_slots(res, f.scale, bgr=frame)
        for sr in slots:
            # ★ 用 verdict == OK，不要用 sr.present：
            #   present 只反映覆盖率，**零件认错时（verdict=NG）它仍可能是 True**，
            #   那样就把"放错零件"的帧当作标准件基线采进去了。
            if sr.verdict == "OK":
                acc[sr.slot_id].append((sr.coverage, sr.features))
            else:
                reject[sr.slot_id] = reject.get(sr.slot_id, 0) + 1
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
            nr = reject.get(sid, 0)
            print(" !! %s 有效样本太少（%d），被拒 %d 帧" % (sid, len(samples), nr))
            print("    可能原因：① 该槽没放件  ② ROI 位置不对")
            print("              ③ 零件特征与 part_templates 不符（放错零件？）")
            print("    排查：跑 tools/tune_slots.py 看该槽的覆盖率与识别结果")
            ok_all = False
            continue
        covs = np.array([s[0] for s in samples])
        print(" %s：" % sid)
        print("    覆盖率 均值 %.3f  σ %.3f  (min %.3f max %.3f)  被拒 %d 帧"
              % (covs.mean(), covs.std(), covs.min(), covs.max(), reject.get(sid, 0)))
        calib.set("slots.%s.coverage" % sid, round(float(covs.mean()), 4))
        calib.set("slots.%s.coverage_std" % sid, round(float(covs.std()), 4))
        calib.set("slots.%s.frames" % sid, len(samples))
        # ★ has_hole：该槽标准件有没有内孔。
        #   文档字符串里一直写着会产出这个字段，但代码漏了 —— 补上。
        #   它给"齐套性/完整性"检查用：孔数对不上就说明零件不对或装反了。
        holes = [float(s[1].get("holes", np.nan)) for s in samples]
        holes = [h for h in holes if np.isfinite(h)]
        if holes:
            h_med = float(np.median(holes))
            calib.set("slots.%s.has_hole" % sid, bool(h_med >= 0.5))
            calib.set("slots.%s.holes_median" % sid, h_med)
            print("    内孔数 中值 %.0f  -> has_hole=%s"
                  % (h_med, "是" if h_med >= 0.5 else "否"))
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
