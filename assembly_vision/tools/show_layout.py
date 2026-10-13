# -*- coding: utf-8 -*-
"""拍一张带槽位/装配位标注的图 —— 标定前用它确认"件该放哪、现在放对没有"。

★ 为什么需要：标定脚本（c4/c5/c6）都要求把零件放到**指定位置**再采集，
  但它们的提示只有文字，看不到 ROI 到底在哪。
  实测踩过：c5 第一步要求"把轮盘放回料盘槽"，用户却把成品放在了装配位，
  于是报"在轮盘槽里找不到前景"，来回一趟才发现是位置不对。

用法
    python tools/show_layout.py                 # 存一张标注图并打印各 ROI 的像素质心
    python tools/show_layout.py --live          # 开实时窗口（按 q 退出）
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from assemvision.config import load_config                     # noqa: E402
from assemvision.core.roi import TrayMap                        # noqa: E402
from assemvision.core.segment import Segmenter                  # noqa: E402
from assemvision.imgio import imwrite_safe                      # noqa: E402
from assemvision.sources.factory import make_source             # noqa: E402

# 每个 ROI 该放什么（提示用）
EXPECT = {
    "slot_wheel":  "轮盘组件（完整的一整个）",
    "slot_column": "立柱（细长杆）",
    "slot_cap":    "顶盖铭牌（小圆片）",
    "slot_base":   "底座圆盘（大圆盘）",
}


def main() -> int:
    ap = argparse.ArgumentParser("show_layout")
    ap.add_argument("--live", action="store_true", help="开实时窗口而不是只存一张图")
    ap.add_argument("--out", default="data/refs/layout.png")
    args = ap.parse_args()

    cam = load_config(os.path.join(ROOT, "config", "camera.yaml"))
    wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
    cal = load_config(os.path.join(ROOT, "config", "calibration.yaml"), required=False)
    if "tray_map" not in cal.data:
        print("★ 还没有料盘单应，请先跑 calib/c3_tray_homography.py")
        return 1
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    seg = Segmenter(wp)

    src = make_source(cam, "uvc", None)
    src.open()
    if hasattr(src, "apply_locked_controls"):
        src.apply_locked_controls()          # type: ignore[attr-defined]

    # ★★ 预热：相机刚改完曝光，头几帧是**全黑**的（项目里的老坑，见 calib/_util.warmup）。
    #   不预热就会存下一张纯黑图，然后误以为"ROI 不对"。
    warm = 0
    for _ in range(40):
        f = src.read(want_full=True)
        if f is None:
            time.sleep(0.05)
            continue
        im = f.left_full if getattr(f, "left_full", None) is not None else f.left
        if im is not None and float(im.mean()) > 5.0:
            warm += 1
            if warm >= 5:
                break
        time.sleep(0.05)
    if warm == 0:
        print("★ 预热失败：连续取不到有亮度的帧。检查曝光/遮挡/是否被别的窗口占用")

    print("=== 各 ROI 的位置（照着把零件放进去）===")
    for s in tray.slots:
        poly = tray.slot_polygon_px(s["id"], 1.0)
        c = poly.mean(axis=0)
        bb = (poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max())
        print("  %-13s 应放 %-22s 像素质心(%5.0f,%5.0f)  bbox x[%.0f,%.0f] y[%.0f,%.0f]"
              % (s["id"], EXPECT.get(s["id"], s.get("part", "")), c[0], c[1],
                 bb[0], bb[1], bb[2], bb[3]))
    ap_poly = tray.assembly_polygon_px(1.0)
    if ap_poly is not None:
        ac = ap_poly.mean(axis=0)
        print("  %-13s 应放 %-22s 像素质心(%5.0f,%5.0f)"
              % ("装配位", "完整装配体（装好的成品）", ac[0], ac[1]))
    print()

    win = "layout  (q=quit)"
    if args.live:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1000, 750)

    first = True
    bad = 0
    MAX_BAD = 120          # ★ 连续抓不到帧就退出，别死循环
    DARK = 5.0             # 均值低于此值视为黑帧，不当成有效画面
    while True:
        f = src.read(want_full=True)
        img = None
        if f is not None:
            img = f.left_full if getattr(f, "left_full", None) is not None else f.left
        if img is not None and float(img.mean()) < DARK:
            img = None         # 黑帧不算
        if img is None:
            bad += 1
            if bad >= MAX_BAD:
                print("★ 连续 %d 次抓不到有效帧（黑帧也算无效），退出。检查：" % MAX_BAD)
                print("  ① 相机是否被其它程序占用（浏览器/会议软件/别的标定窗口）")
                print("  ② 曝光是否过早（相机刚改完曝光要等几帧才有画面）")
                print("  ③ 直接跑 tools/live_preview.py 看能不能出画面")
                src.close()
                if args.live:
                    cv2.destroyWindow(win)
                return 1
            time.sleep(0.05)
            continue
        bad = 0
        vis = img.copy()
        # 前景染色：直观看出分割切到了什么
        r = seg.segment(img, scale=1.0)
        ov = img.copy()
        ov[r.brass_mask > 0] = (0, 255, 255)
        vis = cv2.addWeighted(vis, 0.55, ov, 0.45, 0)
        # 槽位
        for s in tray.slots:
            poly = np.round(tray.slot_polygon_px(s["id"], 1.0)).astype(np.int32)
            cv2.polylines(vis, [poly.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            c = poly.mean(axis=0).astype(int)
            lab = s["id"].replace("slot_", "")
            cv2.putText(vis, lab, (c[0] - 34, c[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, lab, (c[0] - 34, c[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # 装配位
        if ap_poly is not None:
            pp = np.round(ap_poly).astype(np.int32)
            cv2.polylines(vis, [pp.reshape(-1, 1, 2)], True, (255, 128, 0), 2)
            c = pp.mean(axis=0).astype(int)
            cv2.putText(vis, "ASSEMBLY", (c[0] - 48, c[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, "ASSEMBLY", (c[0] - 48, c[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1, cv2.LINE_AA)
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(vis, "green/yellow = slot ROI   orange = assembly   shading = brass detected",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if first and not args.live:
            p = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
            print("已存标注图:", imwrite_safe(p, vis), p)
            first = False
            break
        cv2.imshow(win, vis)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    src.close()
    if args.live:
        cv2.destroyWindow(win)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
