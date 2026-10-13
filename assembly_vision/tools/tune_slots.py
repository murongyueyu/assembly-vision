# -*- coding: utf-8 -*-
"""槽位覆盖率实时调参 —— 边摆零件边看每个槽判成什么。

★ 为什么需要这个工具：
  覆盖率 = ROI 内黄铜像素 / ROI 面积，它**同时**受三个因素影响：
    ① 槽位 ROI 的几何（位置/大小/形状）
    ② 分割阈值（h/s/v 范围）
    ③ 零件在槽里的实际摆放
  三者任一不对都会让"明明有件却判缺件"。光看静态截图无法区分是哪一项，
  实测踩过：在零件没摆好的截图上调参数，越调越偏。

  这个工具把三者放在**同一个实时画面**上，改一格立刻看到覆盖率变化。

按键
    [ / ]    s_min -/+ 5        （镜面高光去饱和时往下调）
    ; / '    s_max -/+ 5        （绿幕误判时往下调）
    - / =    h_max -/+ 2        （杆状件偏色时往上调；别超过 65，会误判）
    v / V    v_min -/+ 5
    g        切换显示前景 mask
    s        打印当前阈值与覆盖率
    q        退出

用法
    python tools/tune_slots.py
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
from assemvision.core.checks import Qualifier                   # noqa: E402
from assemvision.core.roi import TrayMap, coverage_ratio        # noqa: E402
from assemvision.core.segment import Segmenter                  # noqa: E402
from assemvision.sources.factory import make_source             # noqa: E402

WP = os.path.join(ROOT, "config", "workpiece.yaml")
CAL = os.path.join(ROOT, "config", "calibration.yaml")


def collide(win, img):
    cv2.imshow(win, img)


def main() -> int:
    ap = argparse.ArgumentParser("tune_slots")
    ap.add_argument("--present", type=float, default=0.45, help="判『有件』的覆盖率阈值")
    ap.add_argument("--absent", type=float, default=0.15, help="判『缺件』的覆盖率阈值")
    args = ap.parse_args()

    wp = load_config(WP)
    cal = load_config(CAL, required=False)
    if "tray_map" not in cal.data:
        print("★ 还没有料盘单应，请先跑 calib/c3_tray_homography.py")
        return 1
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    print("槽位（来自 workpiece.yaml）:")
    for s in tray.slots:
        if s.get("type") == "circle":
            print("   %-14s %-12s (%5.1f,%5.1f) Ø%.0f"
                  % (s["id"], s["part"], s["center"][0], s["center"][1], s["diameter"]))
        else:
            print("   %-14s %-12s (%5.1f,%5.1f) %.0fx%.0f @%.0f°"
                  % (s["id"], s["part"], s["center"][0], s["center"][1],
                     s["length"], s["width"], s["angle_deg"]))

    cam = load_config(os.path.join(ROOT, "config", "camera.yaml"))
    src = make_source(cam, "uvc", None)
    src.open()
    if hasattr(src, "apply_locked_controls"):
        src.apply_locked_controls()          # type: ignore[attr-defined]

    # 可调阈值（就地改 wp 的 data，Segmenter 每次重建）
    t = dict(
        s_min=int(wp.get("segment.brass.s_min", 50)),
        s_max=int(wp.get("segment.brass.s_max", 180)),
        h_max=int(wp.get("segment.brass.h_max", 45)),
        v_min=int(wp.get("segment.brass.v_min", 80)),
    )
    show_mask = False
    q = None                       # Qualifier：懒建，阈值变了才重建
    win = "tune_slots  [ ]s_min ; 's_max - =h_max v Vv_min g=mask s=print q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1100, 620)

    last = time.time()
    fps = 0.0
    while True:
        f = src.read(want_full=True)
        img = None
        if f is not None:
            img = f.left_full if getattr(f, "left_full", None) is not None else f.left
        if img is None:
            time.sleep(0.03)
            continue

        wp.data["segment"]["brass"].update(t)
        seg = Segmenter(wp)
        # ★ 用完整判据（Qualifier）而不是只算覆盖率：
        #   这样能同时看到「覆盖率」和「零件识别」，一眼区分
        #   "没放件"（覆盖率低）和"放错零件"（覆盖率够但识别成别的件）。
        #   Qualifier 只在阈值变化时重建（模板加载有开销，别每帧都建）。
        if q is None or q._smin != t["s_min"]:
            q = Qualifier(wp, tray, cal)
            q._smin = t["s_min"]
        # ★ 带工位 ROI 的两遍分割：木桌的 HSV 与黄铜重叠，不限制范围时
        #   画面顶部木桌会被判成黄铜，覆盖率与识别全被污染。见 config。
        r, _pose, _src, _ident = q.segment_with_roi(seg, img, scale=1.0)
        slot_res = q.check_slots(r, 1.0, bgr=img)

        vis = img.copy()
        if show_mask:
            ov = img.copy()
            ov[r.brass_mask > 0] = (0, 255, 0)
            vis = cv2.addWeighted(img, 0.4, ov, 0.6, 0)
        covs = {}
        verdicts = {}
        for sr in slot_res:
            sid = sr.slot_id
            poly = np.round(tray.slot_polygon_px(sid, 1.0)).astype(np.int32)
            cov = sr.coverage
            covs[sid] = cov
            verdicts[sid] = sr.verdict
            thr = sr.features.get("cov_thr_present", args.present)
            if sr.verdict == "OK":
                col = (0, 255, 0)
            elif sr.verdict == "NG":
                col = (0, 0, 255)
            else:
                col = (0, 165, 255)
            cv2.polylines(vis, [poly.reshape(-1, 1, 2)], True, col, 2)
            c = poly.mean(axis=0).astype(int)
            got = sr.features.get("id_part", "") or "?"
            # 第二行显示"识别成什么/该是什么"，放错零件时一眼可见
            txt1 = "%s %.2f>%.2f %s" % (sid.replace("slot_", ""), cov, thr, sr.verdict)
            txt2 = "id=%s want=%s" % (got.replace("P", ""), sr.part.replace("P", ""))
            for i, tx in enumerate((txt1, txt2)):
                y = c[1] + i * 15
                cv2.putText(vis, tx, (c[0] - 60, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.40, (0, 0, 0), 4, cv2.LINE_AA)
                cc = (0, 0, 255) if (i == 1 and got not in ("?", sr.part)) else col
                cv2.putText(vis, tx, (c[0] - 60, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.40, cc, 1, cv2.LINE_AA)

        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, 62), (0, 0, 0), -1)
        cv2.putText(vis, "s_min=%d  s_max=%d  h_max=%d  v_min=%d   present>%.2f absent<%.2f"
                    % (t["s_min"], t["s_max"], t["h_max"], t["v_min"], args.present, args.absent),
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        n_ok = sum(1 for v in verdicts.values() if v == "OK")
        n_ng = sum(1 for v in verdicts.values() if v == "NG")
        cv2.putText(vis, "OK %d / NG %d / ? %d    mask=%s   (%.0f fps)"
                    % (n_ok, n_ng, len(covs) - n_ok - n_ng, "on" if show_mask else "off", fps),
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if n_ng == 0 else (0, 165, 255), 2, cv2.LINE_AA)
        collide(win, vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or k == 27:
            break
        elif k == ord("["):
            t["s_min"] = max(0, t["s_min"] - 5)
        elif k == ord("]"):
            t["s_min"] = min(255, t["s_min"] + 5)
        elif k == ord(";"):
            t["s_max"] = max(50, t["s_max"] - 5)
        elif k == ord("'"):
            t["s_max"] = min(255, t["s_max"] + 5)
        elif k == ord("-"):
            t["h_max"] = max(10, t["h_max"] - 2)
        elif k == ord("="):
            t["h_max"] = min(179, t["h_max"] + 2)
        elif k == ord("v"):
            t["v_min"] = max(0, t["v_min"] - 5)
        elif k == ord("V"):
            t["v_min"] = min(255, t["v_min"] + 5)
        elif k == ord("g"):
            show_mask = not show_mask
        elif k == ord("s"):
            print("\n当前阈值: s_min=%d s_max=%d h_max=%d v_min=%d"
                  % (t["s_min"], t["s_max"], t["h_max"], t["v_min"]))
            for sid, c in covs.items():
                print("   %-14s %.3f  %s" % (sid, c,
                      "有件" if c >= args.present else ("缺件" if c <= args.absent else "存疑")))
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(1e-6, now - last)
        last = now

    src.close()
    cv2.destroyWindow(win)
    print("\n最终阈值（按 s 键时的值）: %s" % t)
    print("要固化请改 config/workpiece.yaml 的 segment.brass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
