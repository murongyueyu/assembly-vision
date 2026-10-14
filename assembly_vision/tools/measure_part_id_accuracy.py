# -*- coding: utf-8 -*-
"""零件识别准确率 + 料盘旋转鲁棒性 —— 可重复运行的量化工具。

★ 为什么必须有这个工具（在此之前本项目**没有**可信的准确率数字）：
  c6 里那段"交叉验证 100%"是**循环验证** ——
  「用刚采的样本造模板 -> 用这个模板认同一批样本」，必然 ~100%，
  哪怕零件全放错、哪怕槽是空的，也照样报 100%。
  所以那个数字**不能当作准确率**（AGENTS.md §2.11 已写明）。

★ 本工具怎么拿到"真值"：
  工件是**一料位一件**、槽位坐标已实测标定，所以
  **「零件落在哪个槽」就是它的身份真值** —— 与分类器无关，是几何事实。
  于是可以问一个有意义的问题：
      「用特征向量认出来的零件，和"它在哪个槽"一致吗？」

★ 怎么处理料盘旋转：
  真机帧里料盘可能是转过角度的。先用 `segment_with_roi` 解出位姿，
  再把每个 blob 的质心**逆变换回标称 tray-mm 系**，
  然后在标称系里判断它属于哪个槽 —— 这样"槽归属"不受料盘旋转影响。

用法：
    python tools/measure_part_id_accuracy.py                 # 默认测真机帧
    python tools/measure_part_id_accuracy.py --rotate         # 附加旋转鲁棒性
    python tools/measure_part_id_accuracy.py --files a.png b.png
"""
from __future__ import annotations

import argparse
import io
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


# ── 真值来源：真实画面（零件按 4 个槽摆好）────────────────────
#   ★★ 这一节我写错过，留个记录：原先把 `now_raw.png` / `slot_check.png` /
#      `layout_now.png` 等一起列进来当"零件在槽里"的帧，实测发现
#      **它们全是调试图或半空帧**（带叠加框、有裁剪放大）：
#         位姿一律「用 0 件」-> 一个零件都没识别出来 -> ok=False -> 退回标称坐标
#         `layout_now.png` 里只有轮盘走了、剩底座+立柱，
#         而底座被分割切小到 1542mm²（真值约 2970）后被认成了轮盘。
#      **拿这些帧算准确率是自欺欺人。**
#   ★ 所以现在**不预设帧表** —— 要么用 `--files` 明确指定，
#     要么去 `tools/measure_part_id_e2e.py --live` 在真机上现跑
#     （那才是唯一能给出**可信真机准确率**的路径）。
REAL_FRAMES: list = []          # 故意留空，见上面的说明


def _slot_centers(wp):
    """{slot_id: (cx, cy, part)}，tray-mm。"""
    out = {}
    for s in (wp.get("slots", []) or []):
        c = s.get("center")
        if c is None:
            continue
        out[s["id"]] = (float(c[0]), float(c[1]), str(s.get("part", "")))
    return out


def _inside(px, py, slot):
    """点是否落在槽的几何范围内（圆 / 旋转胶囊）。

    ★ 槽的字段名看 `workpiece.yaml`：
        圆槽   : diameter
        胶囊槽 : length / width / angle_deg
      容差给一点点（圆 +6mm、胶囊 +8mm）：分割出来的质心会被高光带带偏，
      而且贴边摆放时质心本来就可能压线。
    """
    cx, cy = float(slot["center"][0]), float(slot["center"][1])
    if "diameter" in slot:
        r = float(slot["diameter"]) / 2.0
        return (px - cx) ** 2 + (py - cy) ** 2 <= (r + 6.0) ** 2
    if "length" in slot:
        L, W = float(slot["length"]), float(slot["width"])
        ang = np.radians(float(slot.get("angle_deg", 0.0)))
        dx, dy = px - cx, py - cy
        # 反向旋转到槽的局部系
        u = dx * np.cos(ang) + dy * np.sin(ang)
        v = -dx * np.sin(ang) + dy * np.cos(ang)
        return abs(u) <= L / 2.0 + 8.0 and abs(v) <= W / 2.0 + 8.0
    return False


def run(files, wp_path, cal_path, scale=1.0):
    wp = load_config(wp_path)
    cal = load_config(cal_path)
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    q = Qualifier(wp, tray, cal)
    seg = Segmenter(wp)
    slots = _slot_centers(wp)
    mmpp = tray.mm_per_px(scale)

    print("  槽位真值（tray-mm）:")
    for sid, (cx, cy, part) in slots.items():
        print("     %-14s 中心(%6.1f,%6.1f)  应放 %s" % (sid, cx, cy, part))

    rows = []
    for name in files:
        path = name if os.path.isabs(name) else os.path.join(ROOT, "data", "refs", name)
        img = imread_safe(path)
        if img is None:
            print("  ★ 读不到 %s，跳过" % name)
            continue
        res, pose, src, _ident = q.segment_with_roi(seg, img, scale=scale)
        clf = q._classifier()
        print()
        print("  ── %s ──" % os.path.basename(path))
        print("     位姿: %s" % (pose.summary() if pose is not None else "(解不出)"))
        blobs = sorted(res.brass_blobs, key=lambda b: -b.area_px)[:6]
        for b in blobs:
            f = dict(b.features or {})
            a_mm2 = b.area_px * mmpp * mmpp
            f["area_mm2"] = a_mm2
            rw, rh = float(f.get("rect_w", 0)), float(f.get("rect_h", 0))
            f["length_mm"] = max(rw, rh) * mmpp
            f["short_side_mm"] = min(rw, rh) * mmpp
            got, sc, _det = clf.classify(f)
            # 质心 -> 标称 tray-mm（先按位姿逆变换）
            cen = np.array([[b.centroid[0], b.centroid[1]]], float)
            mm = tray.px_to_mm(cen, scale)[0]
            if pose is not None and getattr(pose, "ok", False):
                mm = pose.inverse_apply(mm.reshape(1, 2))[0]
            # 找它属于哪个槽
            hit = None
            for sid, slot in ((k, next(s for s in wp["slots"] if s["id"] == k))
                              for k in slots):
                if _inside(mm[0], mm[1], slot):
                    hit = sid
                    break
            truth = slots[hit][2] if hit else None
            mark = ("✓" if (truth and got == truth)
                    else ("✗" if truth else "·"))
            print("     %s %5.0fmm²  标称(%7.1f,%7.1f)  槽=%-14s 真值=%-10s "
                  "认出=%-10s 分%.2f"
                  % (mark, a_mm2, mm[0], mm[1], hit or "(无)",
                     truth or "(不在槽里)", got or "(认不出)", sc))
            rows.append(dict(frame=os.path.basename(path), area=a_mm2,
                             slot=hit, truth=truth, got=got, score=sc))
    return rows


def report(rows):
    lab = [r for r in rows if r["truth"]]
    ok = [r for r in lab if r["got"] == r["truth"]]
    none_ = [r for r in lab if r["got"] is None]
    wrong = [r for r in lab if r["got"] and r["got"] != r["truth"]]
    print()
    print("═" * 66)
    print("  结  论")
    print("═" * 66)
    print("  落在槽里的零件（有真值） : %d 个" % len(lab))
    print("  认出且正确               : %d" % len(ok))
    print("  认不出（低于 min_score） : %d" % len(none_))
    print("  认错                     : %d" % len(wrong))
    if lab:
        print("  ★ 准确率 = %d/%d = %.1f%%" % (len(ok), len(lab),
                                               100.0 * len(ok) / len(lab)))
        print("  ★ 漏检率（认不出）      = %.1f%%" % (100.0 * len(none_) / len(lab)))
        print("  ★ 误检率（认成别的零件）= %.1f%%" % (100.0 * len(wrong) / len(lab)))
        if wrong:
            print("  认错的明细：")
            for r in wrong:
                print("     %s  真值 %s -> 认成 %s（分 %.2f）"
                      % (r["frame"], r["truth"], r["got"], r["score"]))
        if none_:
            print("  认不出的明细：")
            for r in none_:
                print("     %s  真值 %s（最高分 %.2f < min_score）"
                      % (r["frame"], r["truth"], r["score"]))
    else:
        print("  ★ 没有可用样本 —— 这些帧里零件可能不在槽位里")
    return len(lab), len(ok)


def main() -> int:
    ap = argparse.ArgumentParser("measure_part_id_accuracy")
    ap.add_argument("--workpiece", default=os.path.join(ROOT, "config", "workpiece.yaml"))
    ap.add_argument("--calibration", default=os.path.join(ROOT, "config", "calibration.yaml"))
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--rotate", action="store_true",
                    help="附加：把真机帧转几个角度，看位姿能否复原、零件能否仍认出")
    args = ap.parse_args()

    files = args.files if args.files else REAL_FRAMES
    if not files:
        print()
        print("  ★ 没有指定帧。`data/refs/` 里的图大多是**调试图/半空帧**，")
        print("    拿它们算准确率会得出 0%（实测过：位姿「用 0 件」+ 底座被切小后认成轮盘）。")
        print()
        print("  要可信的真机准确率，请用：")
        print("      python tools/measure_part_id_e2e.py --live")
        print("    （真值 = 「零件落在哪个槽」，一料位一件，与分类器无关）")
        print()
        print("  想在指定图上测：")
        print("      python tools/measure_part_id_accuracy.py --files a.png b.png")
        return 2
    print("═" * 66)
    print("  零件识别准确率测量（真值 = 「零件在哪个槽」，与分类器无关）")
    print("═" * 66)
    rows = run(files, args.workpiece, args.calibration)
    report(rows)

    if args.rotate:
        print()
        print("═" * 66)
        print("  料盘旋转鲁棒性（把真机帧几何旋转，看位姿能否复原）")
        print("═" * 66)
        from measure_rotation import measure               # noqa: E402
        measure(args.workpiece, args.calibration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
