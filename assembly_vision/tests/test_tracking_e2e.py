# -*- coding: utf-8 -*-
"""零件追踪的**离线自动验收**（不需要相机，进 `tests/run_all.py`）。

═══ 这个测试测什么、不测什么 ═══
★ **测**：追踪链路端到端能跑通，且在合成序列上指标**没有退化** ——
  覆盖 / 中心误差 / 幽灵 / 漏跟 / 身份切换 各有闸门。
★ **不测**：真机追踪质量。合成序列是渲染图 ——
  没有真实高光、没有噪声、零件不动、也没有手来遮挡。
  **真机验收要用 `tools/measure_tracking_e2e.py` + 真素材**（见它的文件头）。
  这条界线必须说清楚，否则"全绿"会被当成"真机没问题"。

═══ 为什么要这个测试（本模块的一个真 bug）═══
`tools/part_track_demo.py` 原来在**非检测帧**自己又写了一遍状态机，
而且与 `core/part_tracker.py` 的 `_adopt` **不一致**（没更新 `area_ref`、
直接赋 `part_id` 绕过 `apply_id`/`reid_ok`、没走形状闸、没重置若干计数器）。
后果是「同一个零件在检测帧与非检测帧上走两套身份逻辑」——
而 `test_tracker.py` 只测 core，`part_track_demo.py` 那条路**从来没有自动测试**。
本文件补上这一层：**断言 demo 与 core 共用同一套状态机**。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from _common import check, fail                         # noqa: E402

from assemvision.config import load_config            # noqa: E402
from assemvision.core.checks import Qualifier         # noqa: E402
from assemvision.core.part_id import classify_features  # noqa: E402
from assemvision.core.part_tracker import PartTracker   # noqa: E402
from assemvision.core.roi import TrayMap              # noqa: E402
from assemvision.core.segment import Segmenter        # noqa: E402
from assemvision.imgio import imread_safe             # noqa: E402


def _load_seq(seq_dir, limit=45):
    files = [os.path.join(seq_dir, f) for f in sorted(os.listdir(seq_dir))
             if f.lower().endswith(".png")]
    out = []
    for p in files[:limit]:
        img = imread_safe(p)
        if img is not None:
            out.append(img)
    return out


def main() -> int:
    wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
    cal = load_config(os.path.join(ROOT, "config", "calibration.synthetic.yaml"))
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    q = Qualifier(wp, tray, cal)
    seg = Segmenter(wp)
    clf = q._classifier()
    mmpp = tray.mm_per_px(1.0)
    tcfg = dict(wp.get("tracking", {}) or {})

    seq = os.path.join(ROOT, "data", "synthetic_seq")
    if not os.path.isdir(seq):
        fail("缺合成序列 %s（先跑 tools/make_synthetic.py）" % seq)
    frames = _load_seq(seq)
    check(len(frames) >= 20, "合成序列读入 %d 帧（>=20 才够验收）" % len(frames))

    lo = float(tcfg.get("min_area_px", 400))
    hi = float(tcfg.get("max_area_px", 10000))

    def run(every: int):
        cfg = dict(tcfg)
        cfg["detect_every_n"] = every
        tr = PartTracker(cfg)

        def classify(feats):
            pid, sc, _ = clf.classify(dict(feats))
            return pid, sc

        rows = []
        for i, fr in enumerate(frames):
            g = seg.segment(fr, scale=1.0, mm_per_px=mmpp)
            truth = [b for b in g.brass_blobs if lo <= b.area_px <= hi]
            detect = (i % every == 0)
            blobs = list(truth) if detect else []
            # ★ 交给追踪器之前先补特征（含色域）—— 与 part_track_demo 同一做法
            for b in blobs:
                b.features = classify_features(b, mm_per_px=mmpp, bgr=fr,
                                               mask=g.brass_mask)
            tracks = tr.update(fr, blobs, detect=detect,
                               classify=(classify if detect else None),
                               occluder=g.glove_mask)
            active = [t for t in tracks if t.ever_seen
                      and t.state in ("detected", "tracked")]
            # 逐帧配：轨迹质心 -> 最近的真值块质心
            used, errs = set(), []
            for t in active:
                best, bd = -1, 1e9
                for j, b in enumerate(truth):
                    if j in used:
                        continue
                    d = float(np.hypot(t.centroid[0] - b.centroid[0],
                                       t.centroid[1] - b.centroid[1]))
                    if d < bd:
                        best, bd = j, d
                if best >= 0 and bd <= 80.0:
                    used.add(best)
                    errs.append(bd)
            rows.append({"truth": len(truth), "active": len(active), "err": errs})
        return rows

    # ── 场景 1：每帧检测（基线）──
    print("== 场景 1：每帧检测（every=1）—— 误差下限、不该有幽灵 ==")
    r1 = run(1)[5:]                                   # 剔前 5 帧预热
    n_truth = sum(x["truth"] for x in r1)
    n_active = sum(x["active"] for x in r1)
    errs1 = [e for x in r1 for e in x["err"]]
    print("   真值块合计 %d，活跃轨迹合计 %d，配上的误差样本 %d"
          % (n_truth, n_active, len(errs1)))
    check(n_truth > 0, "合成序列里应该检测得到零件（实测 %d 块）" % n_truth)
    check(abs(n_active - n_truth) <= 2,
          "每帧检测时，活跃轨迹数应与真值块数一致（实测 %d vs %d）" % (n_active, n_truth))
    e_med = float(np.median(errs1)) if errs1 else float("inf")
    print("   中心误差 中位 %.2f px = %.2f mm" % (e_med, e_med * mmpp))
    check(e_med < 8.0, "每帧检测的中心误差应很小（实测中位 %.2f px）" % e_med)

    # ── 场景 2：抽帧 + 追踪 ──
    print("\n== 场景 2：抽帧检测 + 追踪（every=8）—— 误差不该爆、覆盖不该掉 ==")
    r8 = run(8)[5:]
    withp = [x for x in r8 if x["truth"] > 0]
    covered = [x for x in withp if x["active"] > 0]
    cov = 100.0 * len(covered) / max(1, len(withp))
    errs8 = [e for x in r8 for e in x["err"]]
    e_med8 = float(np.median(errs8)) if errs8 else float("inf")
    print("   有零件的帧 %d，其中有活跃轨迹的 %d ⇒ 覆盖 %.1f%%"
          % (len(withp), len(covered), cov))
    print("   中心误差 中位 %.2f px = %.2f mm（基线 %.2f px）"
          % (e_med8, e_med8 * mmpp, e_med))
    check(cov >= 95.0, "抽帧后覆盖应 >=95%%（实测 %.1f%%）" % cov)
    # ★ 判据：抽帧只该让误差**略微**变大，不该爆
    check(e_med8 <= max(8.0, e_med * 4.0),
          "抽帧后中心误差不该爆（实测 %.2f px，基线 %.2f px）" % (e_med8, e_med))
    # ★ 幽灵按"占总帧数比例"判（口径说明见 tools/measure_tracking_e2e.py 末尾）
    ghost = sum(max(0, x["active"] - x["truth"]) for x in r8)
    miss = sum(max(0, x["truth"] - x["active"]) for x in r8)
    print("   幽灵累计 %d 帧 / 漏跟累计 %d 帧（总 %d 帧）" % (ghost, miss, len(r8)))
    check(ghost <= 0.25 * len(r8),
          "幽灵不该超过总帧数的 25%%（实测 %d/%d）" % (ghost, len(r8)))
    check(miss <= 0.10 * len(r8),
          "漏跟不该超过总帧数的 10%%（实测 %d/%d）" % (miss, len(r8)))

    # ── 场景 3：★ demo 与 core 必须共用同一套状态机 ──
    print("\n== 场景 3：part_track_demo 的非检测帧路径必须走 absorb() ==")
    import inspect                                                   # noqa: PLC0415
    import tools.part_track_demo as demo_mod                         # noqa: PLC0415
    src = inspect.getsource(demo_mod)
    check("tracker.absorb(" in src,
          "demo 的非检测帧应调用 `tracker.absorb()`（共用 core 的状态机）")
    # 反向断言：不能再出现"自己改状态/直接赋 part_id"的老写法
    check("t.part_id, t.id_score = " not in src,
          "demo 不该再直接赋 `t.part_id, t.id_score`（那会绕过 apply_id/reid_ok）")
    check("t.visible_ratio = b.area_px" not in src,
          "demo 不该再自己算 visible_ratio（area_ref 的更新与状态机是一体的）")

    # ── 场景 4：实时路径与标定路径的特征必须一致 ──
    print("\n== 场景 4：classify_features() 是唯一入口 —— 两条路径算出的键要一致 ==")
    fr = frames[0]
    g = seg.segment(fr, scale=1.0, mm_per_px=mmpp)
    if not g.brass_blobs:
        fail("合成序列首帧没切到零件")
    b = g.brass_blobs[0]
    full = classify_features(b, mm_per_px=mmpp, bgr=fr, mask=g.brass_mask)
    check("area_mm2" in full and "length_mm" in full and "short_side_mm" in full,
          "带 mm_per_px 时应算出三个尺度特征")
    check("c_hue_mean" in full and "c_sat_mean" in full and "c_val_mean" in full,
          "★ 带 bgr/mask 时应算出**色域特征**（权重合计 3.2/18.1 —— 之前实时链路把它们扔了）")
    thin = classify_features(b, mm_per_px=mmpp)          # 不给图与掩膜
    check("area_mm2" in thin and "c_hue_mean" not in thin,
          "不给 bgr/mask 时应退回「只有形状+尺度」（老调用方行为不变）")

    print("\n零件追踪验收测试全部通过")
    print("★ 注意：这是**合成序列**上的回归 —— 真机追踪质量要用")
    print("   `tools/measure_tracking_e2e.py` + 真素材验（见该文件头）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
