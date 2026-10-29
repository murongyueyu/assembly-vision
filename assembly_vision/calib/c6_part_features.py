# -*- coding: utf-8 -*-
"""c6 零件特征标定 —— 抓取四类零件的真实特征向量，供特征识别使用。

═══ 为什么需要这一步 ═══

`assemvision/core/part_id.py` 内置的模板是**由零件几何推算的初值**，不是实测值。
特征识别必须用真实画面上的特征分布来定区间，否则容易认错。

跑法（把四类零件按正确槽位摆好）：
    python calib/c6_part_features.py

脚本会：
  ① 连拍若干帧，取每类零件特征的中值与分位区间（多帧取中值抗抖动）
  ② 打印特征表，让你对照"这个特征像不像这种零件"
  ③ 写入 config/part_templates.yaml

★ 注意：特征必须是**旋转不变**的。`rect_angle`（绝对角度）不参与，
  料盘转了多少度都不影响识别。
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from _util import load_all, open_source, save_calib               # noqa: E402
from assemvision.core.part_id import (FEATURE_WEIGHTS,           # noqa: E402
                                      features_from_blob,
                                      slot_area_px)
from assemvision.core.roi import TrayMap                        # noqa: E402
from assemvision.core.segment import Segmenter                  # noqa: E402

# 参与标定的特征。
# ★ 全部必须是**旋转不变**的：
#   形状类 —— 圆度/实度/伸长率/内孔数/端宽比/长宽比
#   尺度类 —— area_mm2（物理面积，由 mm_per_px 换算）
#   色域类 —— 色调均值/离散/饱和度/亮度（颜色天然与朝向无关）
USE_FEATS = ["circularity", "solidity", "elongation", "holes",
             "end_width_ratio", "aspect", "area_mm2", "area_frac",
             "c_hue_mean", "c_hue_std", "c_sat_mean", "c_val_mean",
             "c_sat_hi_frac"]
# 这些特征在合成图上四类零件完全相同（都是同一种黄铜色），
# 只有真实画面才有区分度 —— 标定时如果发现某特征四类取值几乎一样，
# 说明它对当前工件的区分没有贡献，可以按需在 FEATURE_WEIGHTS 里降权。


def main() -> int:
    ap = argparse.ArgumentParser("c6_part_features")
    ap.add_argument("--frames", type=int, default=25, help="每类零件采多少帧")
    ap.add_argument("--out", default="")
    ap.add_argument("--camera", default="")
    ap.add_argument("--workpiece", default="")
    ap.add_argument("--pctl", type=float, default=12.0,
                    help="区间取 [p, 100-p] 分位，默认 p=12（抗离群）")
    args = ap.parse_args()

    cam, wp, cal = load_all(args.camera, args.workpiece)
    if "tray_map" not in cal.data:
        print("★ 还没有料盘单应，请先跑 calib/c3_tray_homography.py")
        return 1
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    seg = Segmenter(wp)
    # ★ 要一个 Qualifier 才能用"两遍分割（含工位 ROI）"，
    #   否则木桌的假黄铜会污染特征值（它 H/S 与黄铜重叠，实测能到 13269mm²）。
    from assemvision.core.checks import Qualifier as _Q
    q = _Q(wp, tray, cal)

    print("=" * 72)
    print(" 零件特征标定")
    print("=" * 72)
    print("请确认：四类零件都放在【正确的槽位】里（不改槽位也没关系，")
    print("        下面的匹配是按下标来的，你只要保证每个槽里放的是对的那种零件）")
    print()
    for s in tray.slots:
        print("   %-14s 应该放：%-12s (%s)" % (s["id"], s["part"],
              (wp.get("parts.%s.name" % s["part"]) or "")))
    print()
    input("摆好后按回车开始采集...")

    src = open_source(cam, args)
    src.apply_locked_controls()
    print("采集中（%d 帧）..." % args.frames)

    samples = {s["id"]: [] for s in tray.slots}
    for i in range(args.frames):
        f = src.read(want_full=True)
        if f is None:
            time.sleep(0.03)
            continue
        img = f.left_full if getattr(f, "left_full", None) is not None else f.left
        if img is None:
            continue
        # ★ 用两遍分割（含工位 ROI）：木桌的 HSV 与黄铜重叠，不限制范围会
        #   产生假连通域污染特征值。见 config 的 work_roi_margin_mm。
        res, _pose, _src, _ident = q.segment_with_roi(seg, img, scale=1.0)
        # 把每个连通域归到它质心所在的槽
        for s in tray.slots:
            poly = tray.slot_polygon_px(s["id"], 1.0)
            cx = poly[:, 0].mean(); cy = poly[:, 1].mean()
            # 找离槽中心最近、且质心落在槽包围盒内的 blob
            best = None
            best_d = 1e18
            x0, y0 = poly[:, 0].min(), poly[:, 1].min()
            x1, y1 = poly[:, 0].max(), poly[:, 1].max()
            for b in res.brass_blobs:
                bx, by = b.centroid
                if not (x0 - 6 <= bx <= x1 + 6 and y0 - 6 <= by <= y1 + 6):
                    continue
                if b.area_px < 30:
                    continue
                d = (bx - cx) ** 2 + (by - cy) ** 2
                if d < best_d:
                    best_d, best = d, b
            if best is None:
                continue
            # ★ 用与运行时**完全相同**的取特征方式，保证标定与判定一致：
            #   area_mm2（物理面积）+ area_frac（相对该槽面积）+ 色域特征
            ft = features_from_blob(
                best,
                roi_area_px=slot_area_px(tray, s["id"], 1.0),
                mm_per_px=tray.mm_per_px(1.0),
                bgr=img, mask=res.brass_mask)
            samples[s["id"]].append(ft)
        if (i + 1) % 5 == 0:
            print("   已采 %d/%d 帧" % (i + 1, args.frames))
        time.sleep(0.02)
    src.close()

    # ── 统计 ──────────────────────────────────────────────
    print()
    print("=" * 72)
    print(" 特征统计（中值 / 分位区间）")
    print("=" * 72)
    header = "  %-14s %-12s" % ("槽位", "零件") + "".join("%-18s" % f for f in USE_FEATS)
    print(header)
    out_tpl = {}
    need = args.pctl
    for s in tray.slots:
        rows = samples[s["id"]]
        if len(rows) < 3:
            print("  %-14s %-12s ★ 样本太少(%d)，跳过" % (s["id"], s["part"], len(rows)))
            continue
        line = "  %-14s %-12s" % (s["id"], s["part"])
        rng = {}
        for f in USE_FEATS:
            vals = [r[f] for r in rows if f in r and np.isfinite(r[f])]
            if len(vals) < 3:
                line += "%-18s" % "-"
                continue
            med = float(statistics.median(vals))
            lo = float(np.percentile(vals, need))
            hi = float(np.percentile(vals, 100 - need))
            if lo > hi:
                lo, hi = hi, lo
            # 区间太窄时给一点余量，避免实测时刚好差一点就认不出
            pad = 0.15 * max(hi - lo, 0.05 * abs(med), 1e-3)
            rng[f] = [round(lo - pad, 5), round(hi + pad, 5)]
            line += "%-18s" % ("%.2f[%.2f,%.2f]" % (med, lo, hi))
        out_tpl[s["part"]] = {
            "slot": s["id"],
            "name": wp.get("parts.%s.name" % s["part"]) or "",
            "n_samples": len(rows),
            "ranges": rng,
        }
        print(line)

    # ══════════════════════════════════════════════════════════════════
    # ★★ 采集质量自检：判断"零件到底有没有放在正确的槽里"
    #
    # 这个环节是必需的。实测踩过：连续跑了三次，第 1 次零件摆对了，
    # 第 2/3 次零件位置全换了（立柱变成 elong=1.01 的圆件、底座变成 elong=6.3
    # 的细长件），而脚本**每次都照旧写入 calibration.yaml**，
    # 把好数据覆盖成了垃圾模板。
    #
    # ★★ 为什么不能靠"交叉验证正确率"来判断：
    #   那个数字是**循环验证** —— 用刚采的样本造模板，再用这个模板认同一批样本，
    #   必然 100%，**哪怕零件全放错也一样报 100%**。
    #   所以必须用**物理尺寸对账**：实测面积反推出的直径/长宽，
    #   要跟零件的名义尺寸对得上，否则就是槽里放错了东西。
    # ══════════════════════════════════════════════════════════════════
    # 零件的名义几何。
    # ★ 尺寸取自 workpiece.yaml 的 **slots**（槽位实测值），**不要用 radii_mm**：
    #   实测发现 radii_mm.cap=9.55（Ø19.1）与 slots.slot_cap.diameter=25 自相矛盾，
    #   而真实画面量出顶盖 Ø26.6mm —— 和槽的 Ø25 吻合，说明 radii_mm 那个值是错的。
    # ★ 容差给得宽（实测会因**透视投影 + 分割侵蚀**偏小 10~20%）：
    #   圆的直径容差 [0.55, 1.20] 倍，细长件面积容差 [0.5, 1.8] 倍。
    #   只要能把"轮盘 vs 顶盖"这种量级差异分出来就够了，不必精确。
    NOMINAL = {
        "P1_base":   dict(kind="圆", d=65.0,  note="底座圆盘（槽 Ø65）"),
        "P2_column": dict(kind="细长", l=85.0, w=15.0, note="立柱约 85x15 细杆"),
        "P3_wheel":  dict(kind="圆", d=75.0,  note="轮盘组件（槽 Ø75）"),
        "P4_cap":    dict(kind="圆", d=25.0,  note="顶盖铭牌（槽 Ø25）"),
    }

    def _implied(part: str, area_mm2: float, elong: float) -> str:
        """由实测面积与伸长率反推"这个槽里像什么形状"。"""
        nm = NOMINAL.get(part, {})
        if nm.get("kind") == "细长":
            w = (area_mm2 / max(elong, 1e-6)) ** 0.5
            return "细长约 %.0fx%.0fmm" % (w * elong, w)
        d = 2.0 * (area_mm2 / np.pi) ** 0.5 if area_mm2 > 0 else 0.0
        return "圆 Ø%.1fmm" % d

    print()
    print("=" * 72)
    print(" 采集质量自检（用物理尺寸对账，判断零件有没有放对槽）")
    print("=" * 72)
    print("  %-13s %-11s %-22s %-24s %s"
          % ("槽位", "零件", "应放什么(名义)", "实测是什么(反推)", "判定"))
    problems = []
    for s in tray.slots:
        v = out_tpl.get(s["part"])
        if not v:
            print("  %-13s %-11s ★ 该槽没有采到样本 —— 零件可能没放进去或者没检出"
                  % (s["id"], s["part"]))
            problems.append("%s 槽没采到样本（零件没放进去？）" % s["id"])
            continue
        r = v["ranges"]
        elong = (r["elongation"][0] + r["elongation"][1]) / 2.0 if "elongation" in r else 0.0
        area = (r["area_mm2"][0] + r["area_mm2"][1]) / 2.0 if "area_mm2" in r else 0.0
        want = NOMINAL.get(s["part"], {})
        want_kind = want.get("kind", "?")
        got_kind = "细长" if elong >= 1.8 else "圆"
        # 面积粗核：细长件与圆件分别用不同的相对容差（透视+分割侵蚀会让实测偏小）
        ok_area = True
        if want_kind == "圆" and want.get("d"):
            d_nom = want["d"]
            d_got = 2.0 * (area / np.pi) ** 0.5 if area > 0 else 0.0
            ok_area = 0.55 * d_nom <= d_got <= 1.20 * d_nom
        elif want_kind == "细长" and want.get("l"):
            a_nom = want["l"] * want.get("w", 15.0)
            ok_area = 0.5 * a_nom <= area <= 1.8 * a_nom
        ok = (got_kind == want_kind) and ok_area
        print("  %-13s %-11s %-22s %-24s %s"
              % (s["id"], s["part"], want.get("note", "?"),
                 _implied(s["part"], area, elong), "✓" if ok else "★ 不符"))
        if not ok:
            problems.append(
                "%s 槽：应放 %s（%s），实测却是 %s —— 槽里放的不是这种零件"
                % (s["id"], s["part"], want.get("note", "?"),
                   _implied(s["part"], area, elong)))

    if problems:
        print()
        print(" ★★ 自检不通过，**拒绝写入标定文件**（避免覆盖掉之前的好数据）：")
        for p in problems:
            print("     - %s" % p)
        print()
        print(" 请把四类零件按下面的对应关系摆好后再重跑：")
        for s in tray.slots:
            print("     %-13s <- %-11s %s" % (s["id"], s["part"],
                  NOMINAL.get(s["part"], {}).get("note", "")))
        print()
        print(" （注意：料盘要正放，不要整盘旋转 —— 槽位坐标是按正放标定的）")
        return 2
    print()
    print("   自检通过 ✓  四个槽里的零件都与名义尺寸相符")

    # ── 交叉验证：用刚采的模板回头认这些样本 ──
    # ★★ 注意：这个数字是**循环验证**，必然接近 100%，**不能当作标定可信的依据**。
    #   它的唯一作用是"检查特征区间有没有互相重叠到认不出来"。
    #   真正判断零件放对没有，靠上面的**物理尺寸对账**。
    print()
    print("=" * 72)
    print(" 交叉验证（★ 循环验证，必然 ~100%，只能看区间有没有重叠）")
    print("=" * 72)
    from assemvision.core.part_id import PartClassifier, PartTemplate
    tpls = [PartTemplate(part=p, name=v["name"],
                         ranges={k: tuple(x) for k, x in v["ranges"].items()})
            for p, v in out_tpl.items()]
    clf = PartClassifier(tpls, min_score=0.55)
    ok = 0; tot = 0
    for s in tray.slots:
        if s["part"] not in out_tpl:
            continue
        for r in samples[s["id"]]:
            got, sc, _ = clf.classify(r)
            tot += 1
            if got == s["part"]:
                ok += 1
    if tot:
        print("  正确 %d / %d  (%.1f%%)" % (ok, tot, 100.0 * ok / tot))
        if ok < tot:
            print("  ★ 有认错的 —— 说明某些特征区间重叠，需要加特征或提高分割质量")
        else:
            print("  （100% 是预期的：模板与样本同源。零件放对没有请看上面的物理对账）")

    if out_tpl:
        cal.set("part_templates", out_tpl)
        # ★ 权重快照写到**顶层** part_id 段，不要塞进 part_templates。
        #   塞进去它就成了 part_templates 的一个"键"，_classifier() 遍历时
        #   要靠"没有 ranges 就跳过"才不出错 —— 能跑，但语义很脏，
        #   排查时还会误以为多了一个模板。实测踩过。
        cal.set("part_id.feature_weights", FEATURE_WEIGHTS)
        print()
        print("已写入:", save_calib(cal, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
