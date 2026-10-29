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


# 棋盘格检测用到的 flags 组合。
# ★ 为什么要多试几组：默认参数在"边缘靠近画面边界 / 光照不均 / 略有反光"时经常漏检，
#   而立体标定要求**左右眼同时**检出，漏检率会被放大成平方。
#   CALIB_CB_ADAPTIVE_THRESH 对不均匀光照最有效，CALIB_CB_NORMALIZE_IMAGE 次之。
_BOARD_FLAGS = [
    0,
    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    cv2.CALIB_CB_ADAPTIVE_THRESH,
    cv2.CALIB_CB_NORMALIZE_IMAGE,
    cv2.CALIB_CB_FAST_CHECK if hasattr(cv2, "CALIB_CB_FAST_CHECK") else 0,
]


def find_board(gray, pattern):
    """在灰度图上找棋盘格，多种 flags 依次重试。返回 (ok, corners)。

    ★ 角点数组统一成 (N,1,2)：
      OpenCV 4.x 返回 (N,1,2)，**5.0 返回 (N,2)**。
      不统一的话，后面 `cc[:, :, 0]` 会报
      "too many indices for array: array is 2-dimensional, but 3 were indexed"。
    """
    for fl in _BOARD_FLAGS:
        try:
            ok, c = cv2.findChessboardCorners(gray, pattern, flags=fl)
        except Exception:                                      # noqa: BLE001
            continue
        if ok and c is not None:
            c = np.asarray(c, dtype=np.float32)
            if c.ndim == 2:                       # OpenCV 5.0 -> (N,2)
                c = c.reshape(-1, 1, 2)
            elif c.ndim == 3 and c.shape[1] != 1:
                c = c.reshape(-1, 1, 2)
            return True, c
    return False, None


def estimate_disp(gray_l, gray_r, max_disp: int = 360, step: int = 2) -> int:
    """粗估左右眼视差（像素）。★ 目前**未被 c2 使用**，保留供后续调试。

    实测结论（别急着用）：在本场景下逐行估计视差**不可靠** ——
      · 绿幕大面积无纹理，NCC/MAD 都没有明确峰
      · 各行内容的深度不同，视差本来就逐行变化（不是单一值）
      · 实测同一对图，全图 NCC 给 300+ px，逐行中值给 0~102 px，不可信
    如果以后要做，正路是用 StereoBM/SGBM 算出稀疏视差图，再取支撑区的中值。
    """
    h, w = gray_l.shape[:2]
    disp_max = max(10, min(max_disp, w - 20))
    xs = int(w * 0.08)
    xe = int(w * 0.92)
    if xe <= xs:
        return 0
    ests = []
    for frac in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        y = int(h * frac)
        y0, y1 = max(0, y - 4), min(h, y + 5)
        a = gray_l[y0:y1, xs:xe].astype(np.float32).ravel()
        if a.size < 40:
            continue
        am = a - a.mean()
        an = float(np.sqrt((am * am).sum()))
        if an < 1e-6:
            continue
        best_d, best_v = 0, -2.0
        for d in range(0, disp_max, step):
            b = gray_r[y0:y1, xs - d:xe - d].astype(np.float32).ravel()
            if b.shape != a.shape:
                break
            bm = b - b.mean()
            bn = float(np.sqrt((bm * bm).sum()))
            if bn < 1e-6:
                continue
            v = float((am * bm).sum() / (an * bn))
            if v > best_v:
                best_v, best_d = v, d
        if best_d > 0 and best_v > 0.3:
            ests.append(best_d)
    if not ests:
        return 0
    return int(np.median(ests))


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
    ap.add_argument("--save-pairs", default="",
                    help="把抓到的图像对存到这个目录（便于以后重标定，不用重抓）")
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
    win = "stereo calib  (SPACE=grab  r=undo  ESC=finish)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(win, 1500, 640)
    except Exception:                                          # noqa: BLE001
        pass
    print("空格=抓一组，r=撤销上一组，ESC=结束并标定")
    print("★ 必须 L 和 R 同时为 True 才能抓（两只眼都要看到完整棋盘格）")
    print("★ 画面中央的青色竖线标出「两只眼共同可见」区域，棋盘格中心放进去")

    while True:
        L, R = grab_full(src)
        if L is None:
            print("抓帧失败")
            break
        size = L.shape[:2][::-1]
        gl = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
        okL, cL = find_board(gl, pattern)
        okR, cR = find_board(gr, pattern)

        view = np.hstack([L, R])
        hh, ww = view.shape[:2]
        eye_w = L.shape[1]
        # 每只眼画自己的检出角点（find_board 已统一成 (N,1,2)，可直接用）
        if okL:
            cv2.drawChessboardCorners(view, pattern, cL, okL)
        if okR:
            cc = cR.copy()
            cc[:, :, 0] += eye_w
            cv2.drawChessboardCorners(view, pattern, cc, okR)

        # ── 共同可见区域的**参考**边界（静态，不是精确视差）──
        # ★ 为什么不动态估计视差：实测逐行 NCC / MAD 在本场景都不可靠 ——
        #   绿幕大面积无纹理、各行内容深度不同（视差本来就逐行变化），
        #   估出来的值从 0 到 300 乱跳，画出来的线反而误导。
        #   真正可靠的反馈是下面那两个大字 L / R：哪边没检测到一目了然。
        #   这条线只是提示"棋盘格别越过这里"，避免贴到左眼的左边。
        x0 = int(eye_w * 0.28)
        cv2.line(view, (x0, 90), (x0, hh), (255, 255, 0), 2)
        label = "keep the board to the RIGHT of this line"
        cv2.putText(view, label, (x0 + 8, hh - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(view, label, (x0 + 8, hh - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 0), 2, cv2.LINE_AA)

        # ── 双眼检测状态：大号、醒目 ──
        # ★ 注意别标反：LEFT 下面必须放 okL，RIGHT 下面必须放 okR。
        #   曾经写反过 —— 左眼明明画出了彩色角点，字母却显示 X，
        #   白白浪费一轮排查。改动这里务必对照清楚。
        cv2.rectangle(view, (0, 0), (ww, 86), (0, 0, 0), -1)
        cv2.putText(view, "LEFT feed", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(view, "detected" if okL else "NOT found", (14, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 255, 0) if okL else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(view, "RIGHT feed", (eye_w + 14, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(view, "detected" if okR else "NOT found", (eye_w + 14, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 255, 0) if okR else (0, 0, 255), 2, cv2.LINE_AA)
        # 中缝
        cv2.line(view, (eye_w, 0), (eye_w, hh), (0, 255, 255), 2)

        cap_txt = "pairs = %d" % len(obj_pts)
        cv2.putText(view, cap_txt, (ww - 260, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 255, 0), 3, cv2.LINE_AA)
        if okL and okR:
            hint = ">> both OK - press SPACE to grab"
            col = (0, 255, 0)
        elif not okL and not okR:
            hint = ">> neither eye sees the board - move it INTO the cyan region"
            col = (0, 0, 255)
        elif okL:
            hint = ">> only LEFT sees it - move the board RIGHT (into cyan region)"
            col = (0, 165, 255)
        else:
            hint = ">> only RIGHT sees it - move the board LEFT (into cyan region)"
            col = (0, 165, 255)
        cv2.putText(view, hint, (14, hh - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(view, hint, (14, hh - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, col, 2, cv2.LINE_AA)

        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        if k == ord("r") and obj_pts:
            obj_pts.pop(); img_l.pop(); img_r.pop()
            print("撤销一组，剩 %d" % len(obj_pts))
        if k == 32:
            if okL and okR:
                cl = cv2.cornerSubPix(gl, cL, (11, 11), (-1, -1), crit)
                cr = cv2.cornerSubPix(gr, cR, (11, 11), (-1, -1), crit)
                obj_pts.append(objp.copy())
                img_l.append(cl)
                img_r.append(cr)
                print("已抓 %d 组" % len(obj_pts))
                # ★ 存一份原始图像对：万一下次要重标（比如改了标定流程），
                #   可以直接用存下来的图重算，不用重新抓 52 组。
                if args.save_pairs:
                    os.makedirs(args.save_pairs, exist_ok=True)
                    cv2.imwrite(os.path.join(args.save_pairs, "L%03d.png" % len(obj_pts)),
                                L, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    cv2.imwrite(os.path.join(args.save_pairs, "R%03d.png" % len(obj_pts)),
                                R, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                print("★ 抓不了：L=%s R=%s —— 必须两只眼都看到完整棋盘格"
                      % (okL, okR))

    cv2.destroyWindow(win)
    src.close()

    if len(obj_pts) < args.min_pairs:
        print("有效组数 %d 少于 %d，不标定。多抓几组（覆盖四角与倾斜）再来。"
              % (len(obj_pts), args.min_pairs))
        return 1

    print("开始标定（%d 组，%s）…" % (len(obj_pts), size))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    # ══════════════════════════════════════════════════════════
    # ★★ 标定流程（这里曾经踩过大坑，务必按这个顺序）
    #
    # 旧写法是直接：
    #     stereoCalibrate(obj, img_l, img_r, None, None, None, None, size, flags=0)
    # 也就是**在没有内参初值的情况下，让 stereoCalibrate 同时求内参+外参**。
    # 实测后果（52 组、RMS 0.167 px 看起来很好）：
    #     fx=3703  fy=4093   cx=228  cy=155   基线 B=3626 mm
    # 单眼只有 640x480，fx 不可能 3703；相机也不可能离 3.6 米。
    # 原因：它跑到了**退化解** —— 用极端畸变把几何误差吸收掉，RMS 反而很小。
    # 而且后面那句"有理模型重标"会进一步放开自由度，更容易退化。
    #
    # 正确顺序：
    #   ① 先用 calibrateCamera **逐眼单独**标内参（标准做法，稳）
    #   ② 再 stereoCalibrate + CALIB_FIX_INTRINSIC，**只求外参 R/T**
    #   ③ 有理模型只在"内参已定 + 外参已定"的基础上小修，且必须先通过合理性检查
    # ══════════════════════════════════════════════════════════
    print("  ① 逐眼标定内参（calibrateCamera）…")
    rms_l, K1, D1, _, _ = cv2.calibrateCamera(obj_pts, img_l, size, None, None,
                                              criteria=crit)
    rms_r_, K2, D2, _, _ = cv2.calibrateCamera(obj_pts, img_r, size, None, None,
                                               criteria=crit)
    print("     左眼内参 RMS %.4f px   fx=%.1f fy=%.1f cx=%.1f cy=%.1f"
          % (rms_l, K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]))
    print("     右眼内参 RMS %.4f px   fx=%.1f fy=%.1f cx=%.1f cy=%.1f"
          % (rms_r_, K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]))

    print("  ② 固定内参，只求外参 R/T（stereoCalibrate + CALIB_FIX_INTRINSIC）…")
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_l, img_r, K1, D1, K2, D2, size,
        flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)

    # ── 合理性检查：不合格就直接用①的结果，不再让有理模型乱跑 ──
    def _sane(K, Tvec, w=size[0]):
        fx = float(K[0, 0])
        if not (0.3 * w < fx < 6.0 * w):          # 焦距应在合理量级
            return False, "fx=%.1f 超出合理范围(%.0f~%.0f)" % (fx, 0.3*w, 6.0*w)
        bl = float(np.linalg.norm(Tvec))
        if not (5.0 < bl < 600.0):                # 基线 5~600mm
            return False, "基线 %.1f mm 超出合理范围(5~600)" % bl
        return True, "fx=%.1f 基线=%.1f mm" % (fx, bl)

    ok_sane, why = _sane(K1, T)
    print("     合理性检查：%s  （%s）" % ("通过 ✓" if ok_sane else "★ 不通过", why))

    if ok_sane:
        try:
            rms_r2, K1r, D1r, K2r, D2r, Rr, Tr, _, _ = cv2.stereoCalibrate(
                obj_pts, img_l, img_r, K1, D1, K2, D2, size,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL,
                criteria=crit)
            ok2, why2 = _sane(K1r, Tr)
            if rms_r2 < rms * 0.8 and ok2:
                rms, K1, D1, K2, D2, R, T = rms_r2, K1r, D1r, K2r, D2r, Rr, Tr
                print("     采用 8 参数有理畸变模型（RMS %.4f，%s）" % (rms, why2))
            else:
                print("     放弃有理模型（RMS %.4f 未明显更优，或参数不合理：%s）"
                      % (rms_r2, why2))
        except cv2.error as e:
            print("     有理模型标定失败，保留 5 参数结果:", str(e)[:60])
    else:
        print("     ★ 跳过有理模型（基础外参已不合理，再放开自由度只会更糟）")
        print("     ★ 建议：多抓几组、覆盖画面四角与更大倾斜角，再重跑")

    baseline = float(np.linalg.norm(T))
    print()
    print("=" * 66)
    print(" 重投影 RMS = %.4f px   %s" % (rms, "合格" if rms < 0.5 else "偏大，检查标定板"))
    print(" >>> 基线 B = %.2f mm" % baseline)
    print(" fx=%.2f fy=%.2f  cx=%.2f cy=%.2f" % (K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]))
    if not _sane(K1, T)[0]:
        print(" ★★ 警告：结果仍不合理，请勿直接使用。多半是抓组姿态太单一，")
        print("     需要覆盖画面四角+大角度倾斜后重标。")
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
