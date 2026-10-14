# -*- coding: utf-8 -*-
"""端到端零件识别测量 —— 在**有精确真值**的序列上跑完整流水线。

★ 为什么不在真机帧上做：
  `data/refs/` 里那些是**调试图**（带叠加框、有裁剪放大），
  实测「位姿 用0件」—— 一个零件都没认出来，拿它们算准确率是自欺欺人。
  真机准确率只能**在真机上现跑**（本工具带 `--live`）。

★ 真值从哪来（两条腿都踩实）：
  ① **合成序列**：布局是我们自己渲染的，per-frame 真值精确；
     但它**不是真机**，所以它测的是"判据正确性"，不是"现场准确率"。
  ② **`--live` 真机**：用「零件落在哪个槽」当身份真值 ——
     工件一料位一件，槽位坐标已实测标定，所以这是**与分类器无关的几何事实**。

★ 每帧的四状态（**别把"该缺的件"算成漏检**）：
      TP 该有且认对      TN 该缺且确实没认到
      FN 该有却没认到（漏检）  FP 该缺却认出来了（误检）
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
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402


def _frames(seq_dir):
    fs = sorted(f for f in os.listdir(seq_dir) if f.lower().endswith(".png"))
    return [os.path.join(seq_dir, f) for f in fs]


def measure_seq(seq_dir, wp, cal, tray, verbose=True):
    seg = Segmenter(wp)
    q = Qualifier(wp, tray, cal)
    clf = q._classifier()
    mmpp = tray.mm_per_px(1.0)

    tp = tn = fn = fp = 0
    rows = []
    for path in _frames(seq_dir):
        img = imread_safe(path)
        if img is None:
            continue
        res, pose, src, ident = q.segment_with_roi(seg, img, scale=1.0)
        slots = q.check_slots(res, 1.0, bgr=img, pose=pose)
        for s in slots:
            want = str(s.part or "")
            # ★ 认出的零件名在 `features["id_part"]` 里，不在 SlotResult 的属性上
            #   （SlotResult 只有 slot_id/part/present/coverage/features/verdict/reasons）。
            got = str((s.features or {}).get("id_part") or "")
            present = bool(s.present)
            # 真值：用**覆盖率**独立判断"这一帧这个槽里到底有没有件"，
            #       而不是用 `present`（那本身就是判定结果，拿它当真值是循环论证）。
            cov = float(s.coverage or 0.0)
            truth = cov > 0.10
            ok_id = (got == want) if present else False
            if truth and ok_id:
                tp += 1
            elif truth and not ok_id:
                fn += 1
            elif (not truth) and (not present):
                tn += 1
            else:
                fp += 1
            rows.append((os.path.basename(path), s.slot_id, truth, present, got, cov))
        if verbose:
            txt = " ".join("%s=%s" % (s.slot_id.replace("slot_", ""),
                                      ((s.features or {}).get("id_part") or "空")
                                      if s.present else "空")
                           for s in slots)
            print("     %-22s %s" % (os.path.basename(path), txt))

    return dict(tp=tp, tn=tn, fn=fn, fp=fp), rows


def report(stats, title):
    tp, tn, fn, fp = stats["tp"], stats["tn"], stats["fn"], stats["fp"]
    tot = tp + tn + fn + fp
    print()
    print("  ── %s ──" % title)
    print("     TP 该有且认对 = %-4d  TN 该缺且没认到 = %d" % (tp, tn))
    print("     FN 漏检        = %-4d  FP 误检          = %d" % (fn, fp))
    if tot:
        print("     ★ 准确率 (TP+TN)/全部 = %d/%d = %.1f%%"
              % (tp + tn, tot, 100.0 * (tp + tn) / tot))
    if tp + fn:
        print("     ★ 召回率（有件时认对）  = %.1f%%" % (100.0 * tp / (tp + fn)))
    if tp + fp:
        print("     ★ 精确率（认对/所有认出）= %.1f%%" % (100.0 * tp / (tp + fp)))


def measure_live(wp, cal, tray, n=40):
    """真机现跑：用「零件落在哪个槽」当身份真值。"""
    from assemvision.sources.factory import make_source               # noqa: E402
    cam = load_config(os.path.join(ROOT, "config", "camera.yaml"))
    src = make_source(cam)
    seg = Segmenter(wp)
    q = Qualifier(wp, tray, cal)
    clf = q._classifier()
    mmpp = tray.mm_per_px(1.0)

    slot_list = wp.get("slots", []) or []
    tp = fn = none_ = wrong = outside = 0
    print("  取 %d 帧（真值 = 零件落在哪个槽）。请把 4 个零件摆进各自槽里。" % n)
    for i in range(n):
        f = src.read()
        if f is None:
            continue
        res, pose, _s, _i = q.segment_with_roi(seg, f.left, scale=1.0)
        for b in sorted(res.brass_blobs, key=lambda x: -x.area_px)[:5]:
            feat = dict(b.features or {})
            feat["area_mm2"] = b.area_px * mmpp * mmpp
            rw, rh = float(feat.get("rect_w", 0)), float(feat.get("rect_h", 0))
            feat["length_mm"] = max(rw, rh) * mmpp
            feat["short_side_mm"] = min(rw, rh) * mmpp
            got, sc, _ = clf.classify(feat)
            cen = np.array([[b.centroid[0], b.centroid[1]]], float)
            mm = tray.px_to_mm(cen, 1.0)[0]
            if pose is not None and getattr(pose, "ok", False):
                mm = pose.inverse_apply(mm.reshape(1, 2))[0]
            hit = None
            for sl in slot_list:
                cx, cy = float(sl["center"][0]), float(sl["center"][1])
                if "diameter" in sl:
                    r = float(sl["diameter"]) / 2.0 + 8.0
                    if (mm[0] - cx) ** 2 + (mm[1] - cy) ** 2 <= r * r:
                        hit = sl
                        break
            if hit is None:
                outside += 1
                continue
            if got is None:
                none_ += 1
                print("     [%2d] 漏检 %-12s（该槽应放 %s，分 %.2f）"
                      % (i, hit["id"], hit.get("part"), sc))
            elif got == hit.get("part"):
                tp += 1
            else:
                wrong += 1
                print("     [%2d] 认错 %-12s 真值 %s -> 认成 %s（分 %.2f）"
                      % (i, hit["id"], hit.get("part"), got, sc))
    src.close()
    tot = tp + none_ + wrong
    print()
    print("  落在槽里的零件 : %d 个（另有 %d 个不在任何槽里，已排除）" % (tot, outside))
    print("  认对           : %d" % tp)
    print("  漏检（认不出） : %d" % none_)
    print("  认错           : %d" % wrong)
    if tot:
        print("  ★ 真机准确率 = %d/%d = %.1f%%" % (tp, tot, 100.0 * tp / tot))
    else:
        print("  ★ 没有可用样本 —— 请确认零件摆在槽里、曝光正常")


def main() -> int:
    ap = argparse.ArgumentParser("measure_part_id_e2e")
    ap.add_argument("--workpiece", default=os.path.join(ROOT, "config", "workpiece.yaml"))
    ap.add_argument("--calibration", default=os.path.join(ROOT, "config", "calibration.yaml"))
    ap.add_argument("--seq", default=os.path.join(ROOT, "data", "synthetic_seq"))
    ap.add_argument("--live", action="store_true", help="真机现跑（需要相机）")
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()

    wp = load_config(args.workpiece)
    cal = load_config(args.calibration)
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))

    print("═" * 70)
    print("  零件识别端到端测量")
    print("═" * 70)

    if not args.live:
        print("  合成序列（精确真值；测的是**判据正确性**，不是现场准确率）:")
        print("     dir = %s" % args.seq)
        stats, _rows = measure_seq(args.seq, wp, cal, tray)
        report(stats, "合成序列")
        print()
        print("  ★ 想要**真机准确率**请加 --live（用「零件在哪个槽」当真值）")
    else:
        measure_live(wp, cal, tray, n=args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
