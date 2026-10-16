# -*- coding: utf-8 -*-
"""一次性：生成"这个零件是谁 / 该放哪个槽 / 现在测得是什么"的标注图。

为的是解决一个很实际的问题：文档里一直用 P1/P2/P3/P4 与 slot_xxx，
但用户手里没有一张"哪个零件是哪个"的对照图 —— 于是 c6 根本没法按说明摆件。

做法：
  ① 用料盘位姿修正后的 ROI 分割出四个零件
  ② 用**料盘自身坐标系**（tray-mm）把每个槽的中心算出来
  ③ 把每个块按"离哪个槽中心最近"归到槽上（不用像素坐标硬编码）
  ④ 画一张图：槽位多边形 + 该槽的名字 + 实测直径/伸长率，并把每个零件放大贴出来
"""
from __future__ import annotations

import os
import sys

import numpy as np
import cv2

ROOT = r"D:\university resource\大三上\视觉"
SRC = (r"C:\Users\murongyueyu\.dsh\attachments\v1\objects\22"
       r"\228e517febd7a88a92cd69485709b03773d7f623346927c737e4100cd14e1ca0")
OUT = os.path.join(ROOT, "ai_测试工具", "工位帧", "分析叠加图")
AV = os.path.join(ROOT, "assembly_vision")
sys.path.insert(0, AV)

from assemvision.config import load_config                        # noqa: E402
from assemvision.core.checks import Qualifier                     # noqa: E402
from assemvision.core.features import contour_features            # noqa: E402
from assemvision.core.part_id import classify_features            # noqa: E402
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.core.tray_detect import detect_tray              # noqa: E402

HUD_H = 100

# 每种零件"长什么样"的人话说明（给用户看的，不参与算法）
HUMAN = {
    "P3_wheel":  "8 根辐条的舵轮（最大，带孔）",
    "P2_column": "斜的细长杆（最细长）",
    "P4_cap":    "最小的小圆片",
    "P1_base":   "大圆盘（实心、中间有孔）",
}


def main() -> int:
    wp = load_config(os.path.join(AV, "config", "workpiece.yaml"))
    cal = load_config(os.path.join(AV, "config", "calibration.yaml"))
    seg = Segmenter(wp)
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    q = Qualifier(wp, tray, cal)
    clf = q._classifier()
    mmpp = tray.mm_per_px(1.0)

    img = cv2.imdecode(np.fromfile(SRC, dtype=np.uint8), cv2.IMREAD_COLOR)[HUD_H:]
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    roi, pose, src = q.tracking_roi(img, 1.0, detect=True)
    res = seg.segment(img, scale=1.0, work_roi=roi, mm_per_px=mmpp)
    t = wp.get("tracking", {})
    lo, hi = float(t.get("min_area_px", 400)), float(t.get("max_area_px", 10000))
    blobs = [b for b in res.brass_blobs if lo <= b.area_px <= hi]

    # ── 每个槽的中心（料盘 mm 坐标 -> 像素，带位姿修正）──
    slots = []
    for s in tray.slots:
        poly_mm = tray.slot_polygon_mm(s["id"]) if hasattr(tray, "slot_polygon_mm") \
            else None
        c_mm = np.asarray(s.get("center", [0.0, 0.0]), float).reshape(1, 2)
        if pose is not None and getattr(pose, "ok", False):
            c_mm_p = pose.apply(c_mm)
        else:
            c_mm_p = c_mm
        c_px = tray.mm_to_px(c_mm_p, 1.0)[0]
        try:
            poly_px = tray.slot_polygon_px(s["id"], 1.0)
        except Exception:                                            # noqa: BLE001
            poly_px = None
        slots.append({"id": s["id"], "part": s["part"],
                      "name": wp.get("parts.%s.name" % s["part"]) or "",
                      "center_mm": (float(s["center"][0]), float(s["center"][1])),
                      "center_px": (float(c_px[0]), float(c_px[1])),
                      "poly_px": poly_px})

    # ── 把块归到槽（用**料盘 mm 坐标**比，不用像素硬编码）──
    print("=" * 100)
    print("  这份图回答三件事：这个零件是谁 / 该放哪个槽 / 现在测得是什么")
    print("=" * 100)
    print("  料盘坐标系约定：原点(0,0)在料盘的**左上角**，x 向右、y 向下（0~160mm）")
    print()
    print("  %-13s %-11s %-22s %-16s %s"
          % ("槽位", "该放", "长什么样（人话）", "槽中心(mm)", "实测"))
    print("  " + "-" * 94)
    used = set()
    # ★★ 用**原始 mm 坐标**（单应反投）比，不要套 `pose.apply()`。
    #   为什么：`pose` 的平移是"料盘轮廓检测中心 − 标称中心"换算来的，
    #   而料盘轮廓检测在**反光/阴影**下会把中心估偏 —— 实测求出 tx=72.9mm，
    #   比真实的料盘位移（约 10mm）大了 7 倍，套上去反而把坐标推得更远。
    #   ★ 而"零件落在哪个槽"这件事**在料盘自身坐标系里就是自洽的**：
    #     料盘整体平移/微转时，四个块与四个槽的**相对关系不变**，
    #     所以直接用原始 mm 坐标匹配最稳。
    print()
    print("  ── 各块的料盘坐标（单应反投，单位 mm，料盘 0~160）──")
    print("  %-6s %-16s %-20s %s"
          % ("块", "像素质心", "料盘 mm", "最接近的槽"))
    for i, b in enumerate(blobs):
        mm0 = tray.px_to_mm(np.array([[b.centroid[0], b.centroid[1]]]), 1.0)[0]
        best, bd = "", 1e9
        for sl in slots:
            d = float(np.hypot(mm0[0] - sl["center_mm"][0],
                               mm0[1] - sl["center_mm"][1]))
            if d < bd:
                best, bd = "%s %s" % (sl["id"], sl["part"]), d
        print("  块%-4d (%.0f,%.0f)      (%.1f, %.1f)        %s  (差 %.0fmm)"
              % (i + 1, b.centroid[0], b.centroid[1], mm0[0], mm0[1], best, bd))
    print("  ★ 注：料盘整体平移/微转时这些 mm 值会整体偏移（因为单应是标定时的），")
    print("     但**"哪个块离哪个槽最近"这个相对关系不变** —— 匹配就是按它做的。")
    print()
    for sl in slots:
        best, bd = None, 1e9
        for i, b in enumerate(blobs):
            if i in used:
                continue
            mm0 = tray.px_to_mm(np.array([[b.centroid[0], b.centroid[1]]]), 1.0)[0]
            d = float(np.hypot(mm0[0] - sl["center_mm"][0],
                               mm0[1] - sl["center_mm"][1]))
            if d < bd:
                best, bd = i, d
        meas = "（这个槽是空的 / 没检出）"
        if best is not None and bd < 45.0:
            used.add(best)
            b = blobs[best]
            ft = contour_features(b.contour, res.brass_mask)
            d_mm = 2.0 * (b.area_px / np.pi) ** 0.5 * mmpp
            meas = ("Ø%.0fmm  伸长%.2f  内孔%d  离槽心%.0fmm"
                    % (d_mm, ft.get("elongation", 0.0), int(ft.get("holes", 0)), bd))
        print("  %-13s %-11s %-22s (%5.1f,%5.1f)    %s"
              % (sl["id"], sl["part"], HUMAN.get(sl["part"], ""),
                 sl["center_mm"][0], sl["center_mm"][1], meas))

    # ── 画图 ──
    vis = img.copy()
    for sl in slots:
        if sl["poly_px"] is not None:
            cv2.polylines(vis, [np.round(sl["poly_px"]).astype(np.int32)],
                          True, (0, 255, 255), 2)
        cx, cy = int(sl["center_px"][0]), int(sl["center_px"][1])
        txt = sl["part"]
        cv2.putText(vis, txt, (cx - 42, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 4)
        cv2.putText(vis, txt, (cx - 42, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
    for b in blobs:
        cv2.drawContours(vis, [b.contour], -1, (0, 0, 255), 2)
        cv2.drawMarker(vis, (int(b.centroid[0]), int(b.centroid[1])),
                       (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
    cv2.putText(vis, "YELLOW = slot region (from calibration)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
    cv2.putText(vis, "YELLOW = slot region (from calibration)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(vis, "RED = detected part", (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
    cv2.putText(vis, "RED = detected part", (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 右侧拼一张"放大对照"：每个槽的裁切
    tiles = []
    for sl in slots:
        cx, cy = int(sl["center_px"][0]), int(sl["center_px"][1])
        half = 55
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(img.shape[1], cx + half), min(img.shape[0], cy + half)
        if x1 <= x0 or y1 <= y0:
            continue
        tile = img[y0:y1, x0:x1].copy()
        tile = cv2.resize(tile, (140, 140), interpolation=cv2.INTER_NEAREST)
        cv2.putText(tile, sl["part"], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 4)
        cv2.putText(tile, sl["part"], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1)
        tiles.append(tile)
    if tiles:
        while len(tiles) < 4:
            tiles.append(np.zeros_like(tiles[0]))
        strip = np.hstack(tiles[:4])
        strip = cv2.resize(strip, (vis.shape[1], 150), interpolation=cv2.INTER_AREA)
        vis = np.vstack([vis, strip])

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "08_零件对照_谁是谁该放哪.png")
    okk, buf = cv2.imencode(".png", vis)
    if okk:
        buf.tofile(p)
        print("\n  对照图 -> %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
