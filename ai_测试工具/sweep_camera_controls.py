# -*- coding: utf-8 -*-
"""D435 白平衡 / 曝光扫描：找出「色域最像话」的那一组控件值。

★ 为什么需要它（2026-10 实测踩到）：
  把 `camera.yaml` 的 `realsense.controls` 填实之后，色域**又变了一次** ——
    锁之前（AE+AWB 自动）：绿幕 H=81、零件暖色 H 中位 29、`brass` 掩膜 3994 px ✅
    锁之后（exposure=156, white_balance=4600）：绿幕 H=**87**、零件 H=**6**、
                                              `brass` 掩膜 **58 px** ❌
  原因：`white_balance: 4600` 是 **AWB 关掉那一刻的读数**，它**不等于 AWB 真正在用的值**
  （AWB 是连续估计的）。WB 一改，色温映射就整体平移 —— 绿幕往青蓝走（H 变大）、
  零件往红走（H 变小），正好把 `brass` 的 [10,45] 从两头挤掉。
  ⇒ **不能沿用读数，必须扫**。

★ 判据（两个都要看，缺一不可）：
  ① **绿幕**的 H 应落在 `segment.metal.green_h_min/max`（默认 [60,100]）**中间偏下**，
     离两侧各留 ≥5；
  ② **零件**的暖色像素 H 中位应落在 `brass` 的 [10,45] **中间**，
     且 `brass` 掩膜能把零件整块盖住（覆盖率高）。
  ★ 只要 ① 满足、② 不满足，就说明 WB 选歪了 —— 因为绿幕和零件是**同一套光照下**
    的两个颜色，它们的 H 应该一起动。

用法（★ 在**不受限**的进程里跑；会临时改相机控件，跑完**恢复成配置里的值**）：
    & D:\\miniconda\\envs\\yolov11\\python.exe "ai_测试工具\\sweep_camera_controls.py"
    ... --wb 2800,3400,4000,4600,5200,6000,7000      # 只扫白平衡
    ... --exposure 120,156,200                        # 一起扫曝光
    ... --tag wb1                                     # 给本轮结果起名
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AV = os.path.join(ROOT, "assembly_vision")
sys.path.insert(0, AV)

OUT_DIR = os.path.join(ROOT, "ai_测试工具", "工位帧")


def stats(img, seg):
    """一张图的关键统计。"""
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
    B, G, R = (img[..., 0].astype(np.int32), img[..., 1].astype(np.int32),
               img[..., 2].astype(np.int32))

    # 绿幕：G 明显大于 R 与 B
    green = (G > R + 40) & (G > B + 20)
    # 零件候选：料盘区内、非绿幕、暖色
    h, w = img.shape[:2]
    roi = np.zeros((h, w), bool)
    roi[int(h * 0.10):int(h * 0.95), int(w * 0.30):] = True
    warm = roi & (~green) & (R > B + 20) & (V >= 60)

    res = seg.segment(img, scale=1.0)
    n_brass = int(np.count_nonzero(res.brass_mask)) if res.brass_mask is not None else 0
    # 暖色候选里被 brass 盖住的比例（≈ 零件覆盖率）
    cov = (100.0 * float((warm & (res.brass_mask > 0)).sum()) / max(1.0, float(warm.sum()))
           if res.brass_mask is not None else 0.0)
    return {
        "v_mean": float(V.mean()),
        "blown": 100.0 * float((img.min(axis=2) >= 250).mean()),
        "green_pct": 100.0 * float(green.mean()),
        "green_h50": float(np.median(H[green])) if green.any() else float("nan"),
        "green_h1": float(np.percentile(H[green], 1)) if green.any() else float("nan"),
        "green_h99": float(np.percentile(H[green], 99)) if green.any() else float("nan"),
        "warm_pct": 100.0 * float(warm.mean()),
        "warm_h50": float(np.median(H[warm])) if warm.any() else float("nan"),
        "warm_s50": float(np.median(S[warm])) if warm.any() else float("nan"),
        "brass_px": n_brass,
        "brass_cov": cov,
        "blobs": len(res.brass_blobs),
        "glove_bbox": res.glove_bbox,
    }


def main() -> int:
    ap = argparse.ArgumentParser("sweep_camera_controls")
    ap.add_argument("--wb", default="2800,3400,4000,4600,5200,6000,7000",
                    help="要扫的白平衡色温（逗号分隔）")
    # ★★ 对照组：**只锁曝光、AWB 保持自动**。
    #   为什么必须有这一条：`white_balance: 4600` 是 AWB **关掉那一刻的读数**，
    #   而 AWB 是连续估计的 —— **读数 ≠ 它真正在用的值**。
    #   实测证据：AWB 开着时绿幕 H=81；锁 4600 之后变 H=87（整体往青蓝偏），
    #   说明 4600 偏低（色温设低 ⇒ 红增益大 ⇒ 画面偏红 ⇒ 绿幕往青走）。
    #   ⇒ 除了扫固定值，还要量一次「AE 锁定 + AWB 自动」给出什么色域：
    #     若它色域好且稳定，那才是**最省事又鲁棒**的方案（AWB 本来就会追光照）。
    ap.add_argument("--auto-wb", action="store_true",
                    help="对照组：锁曝光但让 AWB 保持自动（额外测一行）")
    ap.add_argument("--exposure", default="",
                    help="要扫的曝光（逗号分隔，留空 = 只扫 WB）")
    ap.add_argument("--camera", default=os.path.join(AV, "config", "camera.yaml"))
    ap.add_argument("--workpiece", default=os.path.join(AV, "config", "workpiece.yaml"))
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--warmup", type=int, default=12, help="每次改完控件丢几帧再测")
    args = ap.parse_args()

    import cv2
    from assemvision.config import load_config
    from assemvision.core.segment import Segmenter
    from assemvision.sources.factory import make_source

    cam = load_config(args.camera)
    wp = load_config(args.workpiece)
    seg = Segmenter(wp)
    wbs = [int(x) for x in str(args.wb).split(",") if x.strip()]
    exps = [int(x) for x in str(args.exposure).split(",") if x.strip()] or [None]

    os.makedirs(OUT_DIR, exist_ok=True)
    src = make_source(cam)
    try:
        src.open()
    except Exception as e:                                              # noqa: BLE001
        print("★ 相机打不开：%s" % str(e).strip())
        return 2

    # 拿到彩色传感器的句柄（自己写选项，比走 apply_locked_controls 更可控）
    import pyrealsense2 as rs                                          # noqa: PLC0415
    sensor = None
    try:
        for s in src.profile.get_device().sensors:
            try:
                if "rgb" in str(s.get_info(rs.camera_info.name)).lower():
                    sensor = s
                    break
            except Exception:                                           # noqa: BLE001
                continue
        if sensor is None:
            sensor = list(src.profile.get_device().sensors)[0]
    except Exception as e:                                              # noqa: BLE001
        print("★ 拿不到传感器：%s" % str(e).strip())
        src.close()
        return 2

    def setopt(name, val):
        opt = getattr(rs.option, name, None)
        if opt is None or not sensor.supports(opt):
            return False
        try:
            sensor.set_option(opt, float(val))
            return True
        except Exception:                                               # noqa: BLE001
            return False

    def grab():
        img = None
        for _ in range(max(1, args.warmup)):
            f = src.read(want_full=False)
            if f is not None and getattr(f, "left", None) is not None:
                img = f.left
        return img

    # ★ 先关自动，否则下面的值白给
    setopt("enable_auto_exposure", 0)
    setopt("enable_auto_white_balance", 0)

    base = dict(cam.get("realsense.controls", {}) or {})
    print("═" * 96)
    print("  D435 白平衡 / 曝光扫描   （配置里的基准：WB=%s exposure=%s）"
          % (base.get("white_balance"), base.get("exposure")))
    print("  brass 区间 H∈[%s,%s] S∈[%s,%s] V≥%s   绿垫区间 H∈[%s,%s]"
          % (seg.brass.get("h_min"), seg.brass.get("h_max"), seg.brass.get("s_min"),
             seg.brass.get("s_max"), seg.brass.get("v_min"),
             seg.metal.get("green_h_min"), seg.metal.get("green_h_max")))
    print("═" * 96)

    rows = []

    # ★★ 先跑对照组：**AE 锁定 + AWB 自动**（这才是"相机会自己选"的色域）
    if args.auto_wb:
        setopt("enable_auto_exposure", 0)
        if base.get("exposure") is not None:
            setopt("exposure", base["exposure"])
        setopt("enable_auto_white_balance", 1)
        time.sleep(1.2)                     # AWB 要几帧才收敛
        img = grab()
        if img is not None:
            st = stats(img, seg)
            wb_now = None
            try:
                wb_now = int(sensor.get_option(rs.option.white_balance))
            except Exception:                                           # noqa: BLE001
                pass
            rows.append(("AWB", None, st))
            print("  ★ 对照组 AWB 自动（AE 锁定）| AWB 收敛到 %s | V %.1f 过曝 %.3f%% | "
                  "绿幕H %.1f (%.0f~%.0f) %.1f%% | 暖色H %.1f S %.0f | brass %d px 覆盖 %.1f%% 块 %d"
                  % (wb_now, st["v_mean"], st["blown"], st["green_h50"], st["green_h1"],
                     st["green_h99"], st["green_pct"], st["warm_h50"], st["warm_s50"],
                     st["brass_px"], st["brass_cov"], st["blobs"]))
            try:
                cv2.imencode(".png", img)[1].tofile(
                    os.path.join(OUT_DIR, "sweep_%s_AWB.png" % args.tag))
            except Exception:                                           # noqa: BLE001
                pass
        setopt("enable_auto_white_balance", 0)

    for exp in exps:
        if exp is not None:
            setopt("exposure", exp)
        for wb in wbs:
            setopt("white_balance", wb)
            time.sleep(0.25)
            img = grab()
            if img is None:
                print("  WB=%-5d exposure=%-5s  ✗ 读不到帧" % (wb, exp))
                continue
            st = stats(img, seg)
            rows.append((wb, exp, st))
            p = os.path.join(OUT_DIR, "sweep_%s_wb%d_exp%s.png"
                             % (args.tag, wb, exp if exp is not None else "cfg"))
            try:
                cv2.imencode(".png", img)[1].tofile(p)
            except Exception:                                           # noqa: BLE001
                pass
            print("  WB=%-5d exp=%-5s | V %5.1f 过曝 %5.3f%% | 绿幕H %5.1f (%3.0f~%3.0f) "
                  "%5.1f%% | 暖色H %5.1f S %3.0f | brass %5d px 覆盖 %5.1f%% 块 %d"
                  % (wb, exp if exp is not None else "cfg", st["v_mean"], st["blown"],
                     st["green_h50"], st["green_h1"], st["green_h99"], st["green_pct"],
                     st["warm_h50"], st["warm_s50"], st["brass_px"], st["brass_cov"],
                     st["blobs"]))

    # ── 打分排序 ──────────────────────────────────────────────
    #   ① 绿幕 H 要落在绿垫区间里、离两侧 ≥5；
    #   ② 暖色 H 要落在 brass 区间里、离两侧 ≥5；
    #   ③ brass 覆盖率越高越好；④ 过曝越低越好。
    g_lo = float(seg.metal.get("green_h_min", 60))
    g_hi = float(seg.metal.get("green_h_max", 100))
    b_lo = float(seg.brass.get("h_min", 10))
    b_hi = float(seg.brass.get("h_max", 45))

    def score(st):
        s = 0.0
        gh, wh = st["green_h50"], st["warm_h50"]
        if np.isfinite(gh):
            s += 30.0 if (g_lo + 5 <= gh <= g_hi - 5) else -20.0
        if np.isfinite(wh):
            s += 30.0 if (b_lo + 5 <= wh <= b_hi - 5) else -20.0
        s += min(30.0, st["brass_cov"] * 0.30)
        s -= min(20.0, st["blown"] * 4.0)
        return s

    print("\n" + "═" * 96)
    print("  排序（按「绿幕/零件都落在各自区间里 + 覆盖率高 + 过曝低」打分）")
    print("═" * 96)
    for (wb, exp, st) in sorted(rows, key=lambda r: -score(r[2]))[:10]:
        ok_g = (g_lo + 5 <= st["green_h50"] <= g_hi - 5) if np.isfinite(st["green_h50"]) else False
        ok_w = (b_lo + 5 <= st["warm_h50"] <= b_hi - 5) if np.isfinite(st["warm_h50"]) else False
        # ★ wb 可能是字符串 "AWB"（对照组），所以一律按字符串格式化 ——
        #   用 %d 会在对照组那一行抛 TypeError（第一版就是这么崩的）。
        print("  得分 %6.1f | WB=%-5s exp=%-5s | 绿幕H %5.1f %s | 零件H %5.1f %s | "
              "覆盖 %5.1f%% | 过曝 %.3f%% | 块 %d"
              % (score(st), str(wb), str(exp) if exp is not None else "cfg",
                 st["green_h50"], "OK " if ok_g else "❌",
                 st["warm_h50"], "OK " if ok_w else "❌",
                 st["brass_cov"], st["blown"], st["blobs"]))

    # ★ 额外点出"连通域个数 = 零件个数（4）"这一条 —— 它是**最硬**的判据：
    #   覆盖率再高，如果碎成十几块也没法用；正好 4 块说明四个零件各成一体。
    print("\n  ★ 连通域个数（画面里是 4 个零件 ⇒ 理想值是 4）：")
    for (wb, exp, st) in sorted(rows, key=lambda r: -score(r[2]))[:6]:
        mark = "✅ 正好 4 个" if st["blobs"] == 4 else ("%d 个" % st["blobs"])
        print("     WB=%-5s exp=%-5s -> %-12s （覆盖 %.1f%%）"
              % (str(wb), str(exp) if exp is not None else "cfg", mark, st["brass_cov"]))

    # ── 恢复成配置里的值 ──────────────────────────────────────
    print("\n  恢复配置里的控件值…")
    for k, v in base.items():
        if setopt(k, v):
            pass
    src.close()
    print("  已恢复 %d 项；原图都存到 %s\\sweep_%s_*.png" % (len(base), OUT_DIR, args.tag))
    print("\n★ 判读：挑那一行**两个 ✅ 都有、覆盖率高**的 WB（和 exposure），")
    print("  把值填回 camera.yaml 的 realsense.controls，然后重跑 capture_site_frame.py 定格。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
