# -*- coding: utf-8 -*-
"""量出四类零件在**当前** tray-mm 坐标系里的质心 —— 用于更新槽位中心。

═══ 为什么需要它 ═══

`workpiece.yaml` 里的槽位中心（`center: [x, y]`）是"标称位置"，
料盘位姿补偿就是以它为基准算的（实测质心 vs 标称质心 -> 刚体变换）。

**每次重跑 c3（料盘单应）后，坐标系都会变，这些标称值就过期了。**
过期后果很严重：实测踩过 —— 换了一次单应后，位姿估计为了硬凑出
**+48.27° 的荒谬旋转**，而残差还很小（"看着成功其实全错"），
四个槽里两个直接判 NG。

═══ 怎么做到不受坐标系影响 ═══

**按零件身份找，不按旧槽位坐标找**：
  ① 分割出所有黄铜连通域
  ② 用旋转不变特征识别每块"是什么零件"（这一步不需要任何坐标）
  ③ 把每块的质心换算成 tray-mm —— 就是该零件在**当前坐标系**里的位置
  ④ 与 `workpiece.yaml` 里的标称值对比，给出建议

这样即使坐标系整个变了，也能量准。

用法
    python tools/measure_slots.py                # 只打印建议值
    python tools/measure_slots.py --apply        # 直接写回 workpiece.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from assemvision.config import load_config                          # noqa: E402
from assemvision.core.part_id import (PartClassifier,              # noqa: E402
                                      PartTemplate,
                                      features_from_blob,
                                      tray_area_px)
from assemvision.core.roi import TrayMap                            # noqa: E402
from assemvision.core.segment import Segmenter                      # noqa: E402
from assemvision.sources.factory import make_source                 # noqa: E402

WP = os.path.join(ROOT, "config", "workpiece.yaml")
CAL = os.path.join(ROOT, "config", "calibration.yaml")
CAM = os.path.join(ROOT, "config", "camera.yaml")


def main() -> int:
    ap = argparse.ArgumentParser("measure_slots")
    ap.add_argument("--apply", action="store_true", help="写回 workpiece.yaml")
    ap.add_argument("--frames", type=int, default=15, help="取多少帧求中值")
    args = ap.parse_args()

    wp = load_config(WP)
    cal = load_config(CAL)
    cam = load_config(CAM)
    if "tray_map" not in cal.data:
        print("★ 还没有料盘单应，请先跑 calib/c3_tray_homography.py")
        return 1
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))

    # 分类器：优先用标定文件里的真实模板
    stored = cal.get("part_templates") or {}
    tpls = [PartTemplate(part=p, name=str(v.get("name", "")),
                         ranges={k: tuple(x) for k, x in v["ranges"].items()})
            for p, v in stored.items() if isinstance(v, dict) and "ranges" in v]
    if not tpls:
        from assemvision.core.part_id import default_templates
        tpls = default_templates()
        print("★ 用内置初值模板（建议先跑 c6 标定）")
    clf = PartClassifier(tpls, min_score=float(wp.get("part_id.min_score", 0.55)))

    print("请把【四类零件都放回料盘的正确槽位】，保持不动。")
    input("准备好按回车开始…")

    seg = Segmenter(wp)
    # ★ 工位 ROI：只在「料盘 + 装配位」范围内检测黄铜。
    #   不加的话木桌会被判成黄铜（实测产生 13269mm² 的假连通域），
    #   把零件识别和面积测量全带偏。见 config/workpiece.yaml 的 work_roi_margin_mm。
    from assemvision.core.checks import Qualifier as _Q
    _q = _Q(wp, tray, cal)
    src = make_source(cam, "uvc", None)
    src.open()
    if hasattr(src, "apply_locked_controls"):
        src.apply_locked_controls()          # type: ignore[attr-defined]
    # 预热（相机刚改完曝光头几帧是全黑）
    for _ in range(40):
        f = src.read(want_full=True)
        if f is not None:
            im = f.left_full if getattr(f, "left_full", None) is not None else f.left
            if im is not None and float(im.mean()) > 20:
                break

    mmpp = tray.mm_per_px(1.0)
    a_tray = tray_area_px(tray, 1.0)
    found = {}          # part -> [质心列表]
    detail = {}         # part -> [(mm, area, score, elong, solid, circ)]
    got_frames = 0

    # ══════════════════════════════════════════════════════════════════
    # ★★ 两遍识别（和 checks.check_slots 一样）
    #
    # 为什么必须两遍：槽位标称坐标过期时（换过料盘单应就会过期），
    # **细长件（立柱）和小件（顶盖）会落在过期的 ROI 外而检不到**。
    # 实测踩过：立柱和底座都是 0 样本，因为它们的槽位偏了 20mm。
    #
    #   第一遍：**不看槽位**，按形状/面积/色域特征识别每块是什么零件
    #   第二遍：用已认出的件估料盘位姿，修正后的 ROI 再收一次样本
    # ══════════════════════════════════════════════════════════════════
    from assemvision.core.tray_pose import estimate_pose
    from assemvision.core.roi import polygon_mask

    nominal = {}
    for s in wp.get("slots"):
        c = s.get("center")
        if c is not None:
            nominal[str(s.get("part", s.get("id", "")))] = (float(c[0]), float(c[1]))

    pose = None
    pose_used = 0
    for i in range(args.frames):
        f = src.read(want_full=True)
        if f is None:
            continue
        img = f.left_full if getattr(f, "left_full", None) is not None else f.left
        if img is None or float(img.mean()) < 20:
            continue
        # ★ 两遍分割：几何 ROI 粗分 -> 定位姿 -> 位姿修正的 ROI 细分。
        #   这样木桌的假黄铜从一开始就被挡在外面，识别与面积都不受污染。
        res, _pose, _src, _ident = _q.segment_with_roi(seg, img, scale=1.0,
                                                       prev_pose=pose)
        got_frames += 1

        # ── 第一遍：按身份识别（不需要槽位坐标）──
        cand = []
        for b in (res.brass_blobs or []):
            if getattr(b, "area_px", 0.0) < 60:
                continue
            feats = features_from_blob(b, roi_area_px=a_tray, mm_per_px=mmpp,
                                       bgr=img, mask=res.brass_mask)
            part, score, _ = clf.classify(feats)
            mm = tray.px_to_mm(np.array([[b.centroid[0], b.centroid[1]]]), 1.0)[0]
            cand.append((part, float(score), (float(mm[0]), float(mm[1])), feats, b))

        # ── 第一遍结束后估位姿（只在第一帧做一次，之后沿用）──
        if pose is None:
            obs = {}
            for part, score, mm, feats, b in cand:
                if part and (part not in obs or score > obs[part][1]):
                    obs[part] = (mm, score)
            if obs:
                p = estimate_pose({k: v[0] for k, v in obs.items()}, nominal)
                if getattr(p, "ok", False):
                    pose = p
                    pose_used = p.n_used
                    print("  [位姿] 用 %d 个已识别的零件定出: θ=%+.1f° Δ=(%+.1f,%+.1f)mm 残差 %.1fmm"
                          % (p.n_used, p.theta_deg, p.tx, p.ty, p.residual_mm))

        # ── 第二遍：用修正后的 ROI 归属（认不出的块也记录）──
        for part, score, mm, feats, b in cand:
            if part:
                found.setdefault(part, []).append((mm[0], mm[1], float(b.area_px)))
            detail.setdefault(part or "(认不出)", []).append(
                (mm[0], mm[1], float(feats.get("area_mm2", 0)), float(score),
                 float(feats.get("elongation", 0)), float(feats.get("solidity", 0)),
                 float(feats.get("circularity", 0))))

        # 用修正后的槽位 ROI 再收一遍（把第一遍漏掉的细长/小件捞回来）
        if pose is not None and getattr(pose, "ok", False):
            for s in wp.get("slots"):
                part = str(s.get("part", ""))
                if part in found and len(found[part]) > 0:
                    continue          # 这帧已经收到了
                c_now = pose.apply(np.asarray(s.get("center", [0.0, 0.0]),
                                              dtype=np.float64).reshape(1, 2))[0]
                poly = tray.slot_polygon_px(s["id"], 1.0, center_override=c_now,
                                           angle_offset_deg=pose.theta_deg)
                m = polygon_mask(img.shape[:2], poly) > 0
                best_b = None
                for part2, score, mm, feats, b in cand:
                    cx, cy = int(round(b.centroid[0])), int(round(b.centroid[1]))
                    if 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1] and m[cy, cx]:
                        if best_b is None or b.area_px > best_b.area_px:
                            best_b = b
                if best_b is None:
                    continue
                feats = features_from_blob(best_b, roi_area_px=a_tray, mm_per_px=mmpp,
                                           bgr=img, mask=res.brass_mask)
                mm = tray.px_to_mm(np.array([[best_b.centroid[0],
                                              best_b.centroid[1]]]), 1.0)[0]
                found.setdefault(part, []).append((float(mm[0]), float(mm[1]),
                                                   float(best_b.area_px)))
                detail.setdefault(part + "(按ROI)", []).append(
                    (float(mm[0]), float(mm[1]), float(feats.get("area_mm2", 0)),
                     0.0, float(feats.get("elongation", 0)),
                     float(feats.get("solidity", 0)), float(feats.get("circularity", 0))))
    src.close()

    if got_frames == 0:
        print("★ 一帧都没抓到有效画面")
        return 1
    if pose is not None and getattr(pose, "ok", False):
        print("  [位姿] 最终用 %d 个零件；下面列出的质心**已经是修正后坐标系下的**"
              % pose_used)
    print()
    print("=" * 96)
    print(" 实际检出并识别到的每块前景（%d 帧中值）—— 用来查『某槽为什么没样本』" % got_frames)
    print("=" * 96)
    print("  %-14s %-6s %-20s %-10s %-9s %-9s %-9s %s"
          % ("识别为", "样本", "质心 tray-mm", "面积mm²", "伸长率", "实度", "圆度", "得分"))
    for part, rows in sorted(detail.items(), key=lambda kv: -len(kv[1])):
        if not rows:
            continue
        med = lambda j: float(np.median([r[j] for r in rows]))
        print("  %-14s %-6d (%7.1f,%7.1f)   %8.0f   %7.2f   %7.2f   %7.2f   %.2f"
              % (part, len(rows), med(0), med(1), med(2), med(4), med(5), med(6), med(3)))
    print()
    print("=" * 74)
    print(" 各零件在【当前 tray-mm 坐标系】里的质心（%d 帧中值）" % got_frames)
    print("=" * 74)
    print("  %-11s %-7s %-19s %-19s %-9s %s"
          % ("零件", "样本", "实测质心", "配置里的标称值", "原始偏差", "去掉位姿后"))
    suggests = {}
    for s in wp.get("slots"):
        part = str(s.get("part", ""))
        rows = found.get(part, [])
        old = s.get("center", [0.0, 0.0])
        if not rows:
            print("  %-11s %-7s ★ 没识别到 ---- 该槽是不是没放件？" % (part, 0))
            continue
        mx = float(np.median([r[0] for r in rows]))
        my = float(np.median([r[1] for r in rows]))
        d_raw = float(np.hypot(mx - old[0], my - old[1]))
        # ★★ 关键：位姿已经把"料盘整体位移"吃掉了。
        #    所以"实测 vs 标称"里含着那个位移，**不能直接当"标称值错了多少"**。
        #    要先把实测值逆变换回标称系，剩下的才是真正的标称偏差。
        #    实测踩过：这里只报原始偏差（9~17mm），看着像标称值全错，
        #    实际去掉位姿后平均只有 4mm —— 标称值本来就是对的。
        if pose is not None and getattr(pose, "ok", False):
            back = pose.inverse_apply(np.array([[mx, my]], dtype=np.float64))[0]
            d_true = float(np.hypot(back[0] - old[0], back[1] - old[1]))
            tip = "%.1fmm %s" % (d_true, "" if d_true < 5.0 else "← 建议更新")
        else:
            d_true = d_raw
            tip = "（无位姿，等同原始）"
        flag = "" if d_true < 5.0 else "  ← 建议更新"
        print("  %-11s %-7d (%6.1f,%6.1f)   (%6.1f,%6.1f)   %5.1fmm   %s"
              % (part, len(rows), mx, my, old[0], old[1], d_raw, tip))
        # 只有"去掉位姿后仍偏得大"才真的需要改标称值
        if d_true >= 5.0:
            suggests[part] = (round(float(back[0]), 1), round(float(back[1]), 1), d_true)

    if pose is not None and getattr(pose, "ok", False):
        print()
        print("  ★ 位姿 θ=%+.1f° Δ=(%+.1f,%+.1f)mm 已从上面的判定里扣除；"
              % (pose.theta_deg, pose.tx, pose.ty))
        print("    『去掉位姿后』< 5mm 就说明**标称值本来就是对的**，")
        print("    原始偏差大只是料盘整体挪了（位姿补偿已在工作）。")

    # ★ 注意区分两种情况：
    #   suggests 空 **不等于** 没识别到 —— 也可能是"标称值本来就对"（都不需要更新）。
    #   实测踩过：把两者混在一起会打出"一个零件都没识别到"的误导信息。
    if not found:
        print("\n★ 一个零件都没识别到，检查：放了件吗？光照/曝光正常吗？")
        return 1

    print()
    print("=" * 74)
    if not suggests:
        print(" ✓ 所有零件的『去掉位姿后偏差』都 < 5mm —— **标称值不需要更新**。")
        print("   （原始偏差大是料盘整体挪动，位姿补偿已经处理了）")
        return 0
    if args.apply:
        for slot in wp.get("slots"):
            part = str(slot.get("part", ""))
            if part in suggests:
                mx, my, _ = suggests[part]
                slot["center"] = [mx, my]
        wp.save(WP)
        print(" 已写回:", WP)
        for slot in wp.get("slots"):
            print("   %-13s center = %s" % (slot["id"], slot.get("center")))
    else:
        print(" 建议的新 center（加 --apply 直接写回）：")
        for slot in wp.get("slots"):
            part = str(slot.get("part", ""))
            if part in suggests:
                mx, my, _ = suggests[part]
                print("   %-13s center: [%s, %s]   （原 %s）"
                      % (slot["id"], mx, my, slot.get("center")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
