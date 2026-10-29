# -*- coding: utf-8 -*-
"""c3 料盘单应 + 位姿标记参考位置。

两条路，任选：
  A) 用四个位姿标记（推荐）：料盘上已贴/已双色打印好标记，程序自动检测并算单应
  B) 手工点四点：在图上点出料盘上四个已知 tray mm 的点

★ 2D 主线用【未矫正】的原图：相机固定时，镜头畸变被单应一并吸收，
  反而省掉每帧 remap 的开销（省 4-10 ms）。所以 tray_map.on_rectified = false。

用法
    python calib/c3_tray_homography.py                 # 用相机 + 标记
    python calib/c3_tray_homography.py --manual        # 手工点四点
    python calib/c3_tray_homography.py --path tray.jpg # 用一张图
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import ask, grab_full, imread_safe, load_all, open_source, save_calib, show_and_pick  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.core.pose_check import PoseChecker       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--camera", default="")
    ap.add_argument("--workpiece", default="")
    ap.add_argument("--manual", action="store_true", help="手工点四点，不用标记")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="单应建立在哪个缩放比例上（1.0=全分辨率，0.5=实时半分辨率）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cam, wp, calib = load_all(args.camera, args.workpiece)
    src = None
    if args.path:
        img = imread_safe(args.path)
        if img is None:
            print("读不到图:", args.path); return 1
    else:
        src = open_source(cam, args)
        img, _ = grab_full(src)
        if img is None:
            print("抓不到帧"); return 1

    H = None
    refs = {}

    if not args.manual:
        pc = PoseChecker(cam, wp, calib)
        if not pc.enabled:
            print("aruco 不可用（需要 opencv-contrib-python），请改用 --manual")
            return 1
        centers = pc.marker_centers(img)
        print("检测到标记:", {k: (round(v[0], 1), round(v[1], 1)) for k, v in centers.items()})
        if len(centers) < 4:
            print("检测到的标记不足 4 个，检查标记尺寸/光照/是否被遮挡；或改用 --manual")
            return 1
        src_pts, dst_pts = [], []
        for idx, mid in enumerate(pc.ids_cfg):
            if mid in centers and idx < len(pc.positions):
                src_pts.append(pc.positions[idx])
                dst_pts.append(centers[mid])
        H, mask = cv2.findHomography(np.asarray(src_pts, np.float64),
                                     np.asarray(dst_pts, np.float64), 0)
        src_ok, dst_ok = src_pts, dst_pts
        # 参考位置（用于以后的开机位姿自检）
        pc.set_reference(img)
        refs = pc.reference_dict()
        print("位姿标记参考位置已记录：对角线 %.1f px" % (pc.ref_diag_px or 0))
    else:
        print("窗口里连续取流显示：点出料盘的四个【外角顶点】（顺序不限）")
        pts = show_and_pick(img, 4, src=src)
        if len(pts) < 4:
            print("没点够四点"); return 1

        # ══════════════════════════════════════════════════════════
        # ★★ 自动把"点到的 4 个像素点"对应到"料盘四角 mm 坐标"
        #
        # 为什么不能靠用户按顺序输入坐标：实测踩过 ——
        #   脚本要求 左上→右上→右下→左下，用户按"左上→右上→左下→右下"输入，
        #   第 3、4 点给反了，单应把轴翻转，之后所有槽位 ROI 全部错位。
        #   而**两个常用的校验指标都查不出这个错**：
        #     · 投回残差恒为 0.00 —— 4 点定单应有 8 自由度 8 约束，任何对应都能完美拟合
        #     · GSD 仍然正确     —— 翻转坐标轴不改变尺度
        #   => 所以改成自动识别角点顺序，用户只需"点到四个角"，顺序无所谓。
        #
        # 识别方法：料盘是平面上的正方形，透视变换下四点的几何关系仍保序。
        #   ① 用质心把四点分成上/下两组（各 2 点）
        #   ② 每组内按 x 排序 -> 左上/右上、左下/右下
        # ══════════════════════════════════════════════════════════
        P = np.asarray(pts, dtype=np.float64)
        c = P.mean(axis=0)
        order = np.argsort(P[:, 1])          # 先按 y 排
        top, bot = P[order[:2]], P[order[2:]]
        tl, tr = top[np.argsort(top[:, 0])]  # 上排按 x -> 左上、右上
        bl, br = bot[np.argsort(bot[:, 0])]  # 下排按 x -> 左下、右下
        side = float(np.linalg.norm(tr - tl))
        print("  自动识别角点顺序（你点的顺序无所谓）：")
        print("     左上 px=(%.0f,%.0f)  右上 px=(%.0f,%.0f)"
              % (tl[0], tl[1], tr[0], tr[1]))
        print("     右下 px=(%.0f,%.0f)  左下 px=(%.0f,%.0f)"
              % (br[0], br[1], bl[0], bl[1]))
        print("     上边长 %.0f px  下边长 %.0f px  左边长 %.0f px  右边长 %.0f px"
              % (np.linalg.norm(tr - tl), np.linalg.norm(br - bl),
                 np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        w_ask = ask("料盘边长 mm（回车用默认 160）：", "160")
        try:
            SIDE_MM = float(w_ask)
        except ValueError:
            SIDE_MM = 160.0
        # 料盘 mm 坐标系：左上为原点，X 向右、Y 向下
        mm = [[0.0, 0.0], [SIDE_MM, 0.0], [SIDE_MM, SIDE_MM], [0.0, SIDE_MM]]
        pts = [tuple(tl), tuple(tr), tuple(br), tuple(bl)]
        H, _ = cv2.findHomography(np.asarray(mm, np.float64),
                                  np.asarray(pts, np.float64), 0)
        src_ok, dst_ok = mm, pts

        # ★ 方向自检：X 向右、Y 向下的映射，其仿射部分的行列式应为【正】。
        #   若为负，说明角点被镜像识别（点到了意料之外的位置），必须拦下来。
        det = float(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])
        print("  方向自检：仿射行列式 = %.4f  -> %s"
              % (det, "✓ 方向正确（X右/Y下）" if det > 0 else "★ 方向异常（疑似镜像）"))
        if det <= 0:
            print("  ★ 角点顺序可能被识别反了，请重跑并确认四个角都点在料盘外角顶点上")

    if H is None:
        print("单应计算失败")
        return 1
    H = H / H[2, 2]

    # 复核：把源点投回去，看残差
    src_a = np.asarray(src_ok, np.float64).reshape(-1, 1, 2)
    pred = cv2.perspectiveTransform(src_a, H).reshape(-1, 2)
    err = np.linalg.norm(pred - np.asarray(dst_ok, np.float64), axis=1)
    print()
    print("=" * 66)
    print(" 单应 tray(mm) -> 像素：")
    print(np.array2string(H, precision=5, suppress_small=True))
    print(" 投回残差: 最大 %.2f px  均值 %.2f px  %s"
          % (err.max(), err.mean(), "合格" if err.max() < 3.0 else "偏大"))
    # 反算 GSD 复核
    p0 = cv2.perspectiveTransform(np.array([[[80.0, 80.0]]]), H).reshape(2)
    p1 = cv2.perspectiveTransform(np.array([[[81.0, 80.0]]]), H).reshape(2)
    print(" 该单应下 GSD ≈ %.4f mm/px" % (1.0 / np.hypot(*(p1 - p0))))
    print("=" * 66)

    calib.set("tray_map.homography", H.tolist())
    calib.set("tray_map.scale", float(args.scale))
    calib.set("tray_map.on_rectified", False)
    calib.set("tray_map.reprojection_rms_px", round(float(err.mean()), 4))
    if refs:
        for k, v in refs.items():
            calib.set("fiducials." + k, v)
    print("已写入:", save_calib(calib, args.out))

    if src is not None:
        src.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
