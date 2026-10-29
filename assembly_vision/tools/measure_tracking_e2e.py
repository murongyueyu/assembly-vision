# -*- coding: utf-8 -*-
"""★★ 零件追踪的**端到端验收** —— 出可复现的数字，而不是"看着还行"。

═══ 为什么需要它（现状）═══
`tests/test_tracker.py` 的**离线单元覆盖其实很全**（遮挡三档 / 身份余量 /
形状失真冻结 / 判丢 / 防幽灵 / 全局最优认领 / 拿走又放回）。缺的不是"有没有测"，
而是**端到端、真素材、有指标**这一层：
  · 从没在真机素材上量化过"跟得准不准、会不会丢、身份会不会跳"；
  · 而这正是 P1-2（用户要求的追踪算法对比）的**必需前置** ——
    换算法之前得先有一把尺子。

═══ 真值从哪来（不依赖人工标注）═══
★ **用"每帧都全图检测"的结果当真值位置**：
    真值 = 每帧检测（`detect_every_n = 1`）出来的块的质心
    被测 = 抽帧检测 + 追踪（`detect_every_n = N`）出来的轨迹质心
两者跑**同一段素材**，逐帧把"轨迹质心"配到"最近的检测质心"上比。
★ 这个真值**有偏**（它也是算法算出来的，不是人工标的），但用来比
  "追踪 vs 检测"完全够 —— 检测与身份精度另用 `measure_part_dataset.py` 的人工真值基准评测。
  ★ 报告里会把这条偏置写明。

═══ 指标（每条都必须是数）═══
  ① **覆盖**：画面里有零件（真值 ≥1）的帧里，有活跃轨迹的占比；
  ② **中心误差**：轨迹质心 vs 真值质心的距离（均值 / 中位 / p95），同时给像素与毫米；
  ③ **身份稳定性**：`part_id` 在一段连续轨迹内的**切换次数**（越少越稳）；
  ④ **轨迹数**：每帧轨迹数 vs 真值块数 —— 多出来 = **幽灵**，少了 = **漏跟**；
  ⑤ **状态分布**：detected / tracked / occluded / lost 各占多少帧；
  ⑥ **耗时**：每帧均摊、分割摊薄、追踪器 update 的每目标成本；
  ⑦ **帧率**：中位数（剔前若干预热帧）。

═══ 用法 ═══
    # 真机实时（需要相机 + 不受限进程）
    & D:\\miniconda\\envs\\yolov11\\python.exe tools\\measure_tracking_e2e.py --live --frames 300
    # 离线素材（视频 / 图片序列）★ 推荐：同素材可反复跑，便于对比不同后端
    & ... tools\\measure_tracking_e2e.py --video data/videos/d435_track.mp4
    & ... tools\\measure_tracking_e2e.py --seq data/synthetic_seq
    # 只看某几档抽帧间隔（默认 1/3/5/8/12）
    & ... tools\\measure_tracking_e2e.py --video x.mp4 --every 1,4,8
    # 换后端对比（P1-2 要求的"要有数据"）
    & ... tools\\measure_tracking_e2e.py --video x.mp4 --backends kcf,csrt
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cv2                                                        # noqa: E402
from assemvision.config import load_config                        # noqa: E402
from assemvision.core.checks import Qualifier                     # noqa: E402
from assemvision.core.part_id import classify_features            # noqa: E402
from assemvision.core.part_tracker import PartTracker             # noqa: E402
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402

# 汇总时剔掉的开头帧数（追踪器 / 卡尔曼要几帧才稳）
WARMUP = 5


# ── 素材读取 ────────────────────────────────────────────────────
def make_reader(args, cam):
    """返回 (读一帧的函数, 描述, 清理函数)。"""
    if args.seq:
        files = [os.path.join(args.seq, f) for f in sorted(os.listdir(args.seq))
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        if not files:
            raise SystemExit("★ %s 里没有图片" % args.seq)
        state = {"i": 0}

        def rd():
            if state["i"] >= len(files):
                return None
            img = imread_safe(files[state["i"]])
            state["i"] += 1
            return img
        return rd, "图片序列 %s（%d 张）" % (args.seq, len(files)), lambda: None

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit("★ 打不开视频 %s" % args.video)

        def rd():
            ok, img = cap.read()
            return img if ok else None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        return (rd, "视频 %s（%d 帧 @%.1f fps）" % (args.video, n, fps),
                lambda: cap.release())

    # 真机
    from assemvision.sources.factory import make_source               # noqa: PLC0415
    src = make_source(cam)
    src.open()
    if hasattr(src, "apply_locked_controls"):
        src.apply_locked_controls()
    for _ in range(8):                     # 预热
        src.read(want_full=False)

    def rd():
        f = src.read(want_full=False)
        return getattr(f, "left", None) if f is not None else None
    return rd, "真机 %s" % src.name, lambda: src.close()


# ── 一次运行 ────────────────────────────────────────────────────
def run_once(frames, seg, clf, tracker_cfg, mmpp, backend, every, warmup=5):
    """用给定抽帧间隔跑一遍，返回逐帧记录。"""
    cfg = dict(tracker_cfg)
    cfg["backend"] = backend
    cfg["detect_every_n"] = int(every)
    tr = PartTracker(cfg)

    def classify(feats):
        """`PartTracker` 的回调：只吃特征字典（特征由 `prepare()` 预先挂好）。"""
        pid, sc, _ = clf.classify(dict(feats))
        return pid, sc

    def prepare(blob, img, mask):
        """把分类器要用的特征（含色域）算好挂到 `blob.features` 上。
        ★ 与 `tools/part_track_demo.py` 用**同一个** `part_id.classify_features()` ——
          这是"实时路径与标定路径特征一致"的落点。"""
        blob.features = classify_features(blob, mm_per_px=mmpp, bgr=img, mask=mask)
        return blob

    rows = []
    seg_ms_all = []
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        # ★★ 真值：**每帧都全图检测**（不受 every 影响）
        g = seg.segment(frame, scale=1.0, mm_per_px=mmpp)
        truth = [(float(b.centroid[0]), float(b.centroid[1]), float(b.area_px))
                 for b in g.brass_blobs
                 if tracker_cfg.get("min_area_px", 400) <= b.area_px
                 <= tracker_cfg.get("max_area_px", 10000)]
        # 被测：按 every 抽帧 + 追踪
        detect = (i % max(1, int(every)) == 0)
        t1 = time.perf_counter()
        blobs = [b for b in g.brass_blobs
                 if tracker_cfg.get("min_area_px", 400) <= b.area_px
                 <= tracker_cfg.get("max_area_px", 10000)] if detect else []
        # ★★ 交给追踪器之前先把特征补齐（含色域）—— 与 part_track_demo 同一做法，
        #   也是"实时路径与标定路径特征一致"的落点（见 classify_features 的说明）
        for _b in blobs:
            prepare(_b, frame, g.brass_mask)
        tracks = tr.update(frame, blobs, detect=detect,
                           classify=(classify if detect else None),
                           seen_ids=None, occluder=g.glove_mask)
        t2 = time.perf_counter()
        seg_ms_all.append((t1 - t0) * 1000.0)

        # 逐帧对比：把轨迹配到最近的检测质心
        errs = []
        used = set()
        for t in tracks:
            if not t.ever_seen:
                continue
            cx, cy = t.centroid
            best, bd = -1, 1e9
            for j, (tx_, ty_, _a) in enumerate(truth):
                if j in used:
                    continue
                d = math.hypot(cx - tx_, cy - ty_)
                if d < bd:
                    best, bd = j, d
            if best >= 0 and bd <= max(80.0, 3.0 * mmpp * 20):
                used.add(best)
                errs.append(bd)
        # ★★ "活跃"轨迹 = 当前**声称它在画面里**的（detected/tracked）——
        #   幽灵只能按这个口径数。
        #   ★ 我第一版把"所有 ever_seen 的轨迹"都算进去了，于是把
        #     **零件已被拿走、但按设计保留 60 秒的轨迹**（`gone_keep_seconds`，
        #     那是"拿走又放回要能恢复"的必要机制，见 `_prune()` 的说明）
        #     全算成幽灵 —— 判据选错，指标就没意义。
        active = [t for t in tracks if t.ever_seen and t.state in ("detected", "tracked")]
        rows.append({
            "i": i, "frame": frame,
            "n_truth": len(truth), "n_tracks": len(active),
            "n_all_tracks": len([t for t in tracks if t.ever_seen]),
            "n_matched": len(errs),
            "err": errs,
            "states": [t.state for t in active],
            "ids": [t.part_id for t in active],
            "frame_ms": (t2 - t1) * 1000.0,
            "detect": detect,
        })
    return rows, tr, seg_ms_all


def summarize(rows, tr, seg_ms_all, mmpp, every, backend, label, warmup=5):
    r = rows[warmup:]
    if not r:
        return None
    tot = len(r)
    with_part = [x for x in r if x["n_truth"] > 0]
    covered = [x for x in with_part if x["n_matched"] > 0]
    allerr = [e for x in r for e in x["err"]]
    # 身份切换：按"轨迹下标"追踪 part_id 变化
    ids_hist = [x["ids"] for x in r]
    switches = 0
    for k in range(1, len(ids_hist)):
        a, b = ids_hist[k - 1], ids_hist[k]
        if len(a) == len(b):
            switches += sum(1 for u, v in zip(a, b) if u and v and u != v)
    ghost = sum(max(0, x["n_tracks"] - x["n_truth"]) for x in r)
    miss = sum(max(0, x["n_truth"] - x["n_tracks"]) for x in r)
    st = {}
    for x in r:
        for s in x["states"]:
            st[s] = st.get(s, 0) + 1
    fms = [x["frame_ms"] for x in r]
    return {
        "label": label, "backend": backend, "every": every, "frames": tot,
        "cover_pct": 100.0 * len(covered) / max(1, len(with_part)),
        "err_mean": float(np.mean(allerr)) if allerr else float("nan"),
        "err_med": float(np.median(allerr)) if allerr else float("nan"),
        "err_p95": float(np.percentile(allerr, 95)) if allerr else float("nan"),
        "err_med_mm": (float(np.median(allerr)) * mmpp) if allerr else float("nan"),
        "switches": switches,
        "switch_per_100f": 100.0 * switches / max(1, tot),
        "ghost": ghost, "miss": miss,
        "states": st,
        "frame_ms_med": float(np.median(fms)),
        "seg_ms_med": float(np.median(seg_ms_all)),
        "fps": 1000.0 / max(1e-6, float(np.median(fms))),
        "api": getattr(tr, "api_name", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser("measure_tracking_e2e")
    ap.add_argument("--live", action="store_true", help="用真机（否则要 --video 或 --seq）")
    ap.add_argument("--video", default="")
    ap.add_argument("--seq", default="")
    ap.add_argument("--frames", type=int, default=300, help="最多读多少帧")
    ap.add_argument("--every", default="1,3,5,8,12",
                    help="抽帧间隔，逗号分隔（1 = 每帧检测，作为对照）")
    ap.add_argument("--backends", default="kcf", help="追踪后端，逗号分隔")
    ap.add_argument("--camera", default=os.path.join(ROOT, "config", "camera.yaml"))
    ap.add_argument("--workpiece", default=os.path.join(ROOT, "config", "workpiece.yaml"))
    ap.add_argument("--calibration", default=os.path.join(ROOT, "config", "calibration.yaml"))
    ap.add_argument("--save-overlay", default="", help="把对比画面录成视频（可选）")
    args = ap.parse_args()

    if not (args.live or args.video or args.seq):
        ap.error("要指定 --live 或 --video 或 --seq")

    wp = load_config(args.workpiece)
    cal = load_config(args.calibration)
    cam = load_config(args.camera)
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    q = Qualifier(wp, tray, cal)
    seg = Segmenter(wp)
    clf = q._classifier()
    mmpp = tray.mm_per_px(1.0)
    tcfg = dict(wp.get("tracking", {}) or {})

    rd, desc, close = make_reader(args, cam)
    print("═" * 86)
    print("  零件追踪端到端验收")
    print("═" * 86)
    print("  素材      : %s" % desc)
    print("  GSD       : %.4f mm/px" % mmpp)
    print("  面积闸门  : %.0f ~ %.0f px" % (tcfg.get("min_area_px", 400),
                                            tcfg.get("max_area_px", 10000)))
    print("  真值来源  : **每帧全图检测**的块质心（★ 有偏，见文件头说明）")

    frames = []
    t0 = time.perf_counter()
    while len(frames) < args.frames:
        f = rd()
        if f is None:
            break
        frames.append(f)
    close()
    print("  实际读入  : %d 帧（%.1f 秒）" % (len(frames), time.perf_counter() - t0))
    if len(frames) < 10:
        print("★ 帧太少，没法验收")
        return 1

    writer = None
    if args.save_overlay:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(args.save_overlay, cv2.VideoWriter_fourcc(*"mp4v"),
                                 30.0, (w, h))

    results = []
    everys = [int(x) for x in str(args.every).split(",") if x.strip()]
    backends = [x.strip() for x in str(args.backends).split(",") if x.strip()]
    for be in backends:
        for ev in everys:
            print("\n── 跑一遍：backend=%s  detect_every_n=%d ──" % (be, ev))
            rows, tr, sms = run_once(frames, seg, clf, tcfg, mmpp, be, ev)
            s = summarize(rows, tr, sms, mmpp, ev, be,
                          "backend=%s every=%d" % (be, ev))
            if s:
                results.append(s)
                print("   覆盖 %.1f%%   中心误差 中位 %.2fpx(%.2fmm) / p95 %.2fpx   "
                      "身份切换 %d 次(每百帧 %.1f)   幽灵 %d / 漏跟 %d   "
                      "每帧 %.1fms (%.1f fps)   分割 %.1fms"
                      % (s["cover_pct"], s["err_med"], s["err_med_mm"], s["err_p95"],
                         s["switches"], s["switch_per_100f"], s["ghost"], s["miss"],
                         s["frame_ms_med"], s["fps"], s["seg_ms_med"]))
            if writer is not None and be == backends[0] and ev == everys[-1]:
                for x in rows[WARMUP:]:
                    txt = "every=%d  tracks=%d  truth=%d" % (ev, x["n_tracks"],
                                                             x["n_truth"])
                    cv2.putText(x["frame"], txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 0), 3)
                    cv2.putText(x["frame"], txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 1)
                    writer.write(x["frame"])
    if writer is not None:
        writer.release()
        print("\n  对比画面已录：%s" % args.save_overlay)

    # ── 汇总表 ──────────────────────────────────────────────
    print("\n" + "═" * 86)
    print("  汇总（★ 抽帧间隔越小 → 检测越勤 → 误差下限越低、耗时越高）")
    print("═" * 86)
    print("  %-8s %-6s %-9s %-9s %-9s %-9s %-8s %-8s %s"
          % ("后端", "抽帧", "覆盖%", "中位px", "中位mm", "p95px",
             "切换/百帧", "幽灵/漏", "每帧ms"))
    for s in results:
        print("  %-8s %-6d %-9.1f %-9.2f %-9.2f %-9.2f %-10.1f %-8s %.1f"
              % (s["backend"], s["every"], s["cover_pct"], s["err_med"],
                 s["err_med_mm"], s["err_p95"], s["switch_per_100f"],
                 "%d/%d" % (s["ghost"], s["miss"]), s["frame_ms_med"]))

    print("\n  ★ 判读要点：")
    print("   · **误差下限**：拿 every=1 那一行当基线 —— 它是「每帧都检测」的误差，")
    print("     抽帧后的误差**只会比它大**，差值就是「追踪引入的代价」。")
    print("   · **覆盖**要接近 100%：画面里有零件却有帧没跟到 = 漏跟。")
    print("   · **幽灵 / 漏跟**都该是 0；非 0 说明轨迹数与实际零件数对不上。")
    print("   · **身份切换/百帧**越接近 0 越稳；非 0 且伴随误差尖峰 = 认错 / 抢块。")
    print("   · 真值本身也是算法算的（**有偏**）—— 它衡量的是「追踪 vs 检测」的差，")
    print("     不是「绝对位置精度」。绝对精度要靠 c3 标定 + 人工标注。")
    print("\n  ★★ 测量口径（免得把正常行为当 bug）：")
    print("   · `n_tracks` 只数**活跃**轨迹（state ∈ detected/tracked）——")
    print("     也就是「它声称此刻在画面里」的那些。")
    print("   · `occluded` / `lost` / `gone` 的轨迹**不算幽灵**：")
    print("     零件被手挡住、或被拿走后又放回来，工程都**故意保留轨迹**")
    print("     （`gone_keep_seconds: 60`，见 `PartTracker._prune()` 的说明）。")
    print("     把它们算成幽灵，等于把「设计」当「缺陷」——本工具第一版就这么错过。")
    print("   · **1 帧的瞬时差不算幽灵**：抽帧间隔 >1 时，检测帧之间新出现的零件")
    print("     要等到下一次检测才被建轨迹，中间那几帧天然对不上。")
    print("     ⇒ 判据是「幽灵占总帧数的比例」，不是「绝对 0」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
