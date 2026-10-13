# -*- coding: utf-8 -*-
"""c2 双目标定：基线 B、内外参、矫正映射表。

★ 相机基线是所有深度计算的根，厂家规格页没写，必须自己标出来。
★ 相机被移动过就必须重跑本脚本（以及 c3 / c5）。

用法
    python calib/c2_stereo.py --cols 9 --rows 6 --square 25
      cols/rows  棋盘格内角点数（注意是内角点，不是方格数）
      square     格子边长（mm）
    按空格抓一组（左右眼同时），建议 20-30 组，覆盖画面四角与不同倾斜；ESC 结束
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import grab_full, load_all, open_source, paths, save_calib   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--camera", default="")
    ap.add_argument("--workpiece", default="")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square", type=float, default=25.0)
    ap.add_argument("--min-pairs", type=int, default=12)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cam, wp, calib = load_all(args.camera, args.workpiece)
    src = open_source(cam, args)

    pattern = (args.cols, args.rows)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

    obj_pts, img_l, img_r = [], [], []
    size = None
    win = "stereo calib"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("空格=抓一组，r=撤销上一组，ESC=结束并标定")

    while True:
        L, R = grab_full(src)
        if L is None:
            print("抓帧失败")
            break
        size = L.shape[:2][::-1]
        gl = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
        okL, cL = cv2.findChessboardCorners(gl, pattern, None)
        okR, cR = cv2.findChessboardCorners(gr, pattern, None)

        view = np.hstack([L, R])
        for ok, c, off in ((okL, cL, 0), (okR, cR, L.shape[1])):
            if ok:
                cc = c.copy()
                cc[:, :, 0] += off
                cv2.drawChessboardCorners(view, pattern, cc, ok)
        cv2.putText(view, "pairs=%d  L=%s R=%s" % (len(obj_pts), okL, okR),
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        if k == ord("r") and obj_pts:
            obj_pts.pop(); img_l.pop(); img_r.pop()
            print("撤销一组，剩 %d" % len(obj_pts))
        if k == 32 and okL and okR:
            cl = cv2.cornerSubPix(gl, cL, (11, 11), (-1, -1), crit)
            cr = cv2.cornerSubPix(gr, cR, (11, 11), (-1, -1), crit)
            obj_pts.append(objp.copy())
            img_l.append(cl)
            img_r.append(cr)
            print("已抓 %d 组" % len(obj_pts))

    cv2.destroyWindow(win)
    src.close()

    if len(obj_pts) < args.min_pairs:
        print("有效组数 %d 少于 %d，不标定。多抓几组（覆盖四角与倾斜）再来。"
              % (len(obj_pts), args.min_pairs))
        return 1

    print("开始标定（%d 组，%s）…" % (len(obj_pts), size))
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_l, img_r, None, None, None, None, size,
        flags=cv2.CALIB_FIX_INTRINSIC if False else 0,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    # 用有理模型再标一次（大视场边缘残差更小）；失败就退回 5 参数
    try:
        rms_r, K1r, D1r, K2r, D2r, Rr, Tr, _, _ = cv2.stereoCalibrate(
            obj_pts, img_l, img_r, K1, D1, K2, D2, size,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        if rms_r < rms:
            rms, K1, D1, K2, D2, R, T = rms_r, K1r, D1r, K2r, D2r, Rr, Tr
            print("采用 8 参数有理畸变模型")
    except cv2.error:
        pass

    baseline = float(np.linalg.norm(T))
    print()
    print("=" * 66)
    print(" 重投影 RMS = %.4f px   %s" % (rms, "合格" if rms < 0.5 else "偏大，检查标定板"))
    print(" >>> 基线 B = %.2f mm" % baseline)
    print(" fx=%.2f fy=%.2f  cx=%.2f cy=%.2f" % (K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]))
    print("=" * 66)

    calib.set("stereo.baseline_mm", round(baseline, 3))
    calib.set("stereo.image_size", [int(size[0]), int(size[1])])
    calib.set("stereo.K_left", K1.tolist())
    calib.set("stereo.D_left", D1.ravel().tolist())
    calib.set("stereo.K_right", K2.tolist())
    calib.set("stereo.D_right", D2.ravel().tolist())
    calib.set("stereo.R", R.tolist())
    calib.set("stereo.T", T.ravel().tolist())
    calib.set("stereo.reprojection_rms_px", round(float(rms), 4))
    calib.set("meta.stereo_calibrated_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    print("已写入:", save_calib(calib, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
