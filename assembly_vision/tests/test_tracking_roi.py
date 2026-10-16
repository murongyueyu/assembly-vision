# -*- coding: utf-8 -*-
"""料盘位姿 → 工位 ROI 的性质测试（不需要相机、不需要图片素材）。

═══ 为什么要有这个测试 ═══
真机实测踩到：**料盘挪了 (258, 58) px = (222, 50) mm**，
`work_roi` 却还停在标定时的位置（来自 `calibration.yaml` 的静态单应），
于是 **黄铜只覆盖 0.5%、4 个零件一个都不在 ROI 里** ⇒ 整条识别链全灭。
（详见 docs/踩坑实录.md 2.44）

修法是：ROI 由**料盘自身轮廓**定（`Qualifier.tracking_roi()`）。
而这条修法能不能成立，全看一个性质：

    pose.tx = (检测中心 - 标称中心) × mm_per_px      ← 含标称中心
    roi     = tray.mm_to_px(pose.apply(料盘四角))    ← 又用标称单应投回像素
    ⇒ **标称中心被抵消**，ROI 只由【检测到的料盘中心】决定。

也就是说：**就算 `calibration.yaml` 的单应平移早就过期，ROI 依然跟着料盘走。**

★ 这是纯几何性质，**用算术就能测**，不需要真图 —— 所以它适合当回归测试，
  而且比"拿一张真图比像素"稳定得多。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from _common import check, fail, make_env                 # noqa: E402

from assemvision.core.roi import TrayMap                  # noqa: E402
from assemvision.core.tray_detect import (TrayDetect,     # noqa: E402
                                          pose_from_tray_detect)
from assemvision.core.tray_pose import TrayPose           # noqa: E402


def _roi_center(tray: TrayMap, wp, pose, shape, scale=1.0):
    """按 `Qualifier.work_roi_mask` 的算法算 ROI 中心（只用料盘四角，省掉装配位圆）。"""
    from assemvision.core.checks import Qualifier          # noqa: PLC0415
    q = Qualifier(wp, tray, None)
    m = q.work_roi_mask(shape, scale, pose=pose, margin_mm=25.0)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def main() -> int:
    # ★★ 必须用**真实标定文件里的单应**（不是合成环境的仿射）。
    #   为什么：本测试要证明的性质正是"标定单应的平移过期也不影响 ROI"，
    #   而合成环境用的是 `px_per_mm` 仿射 —— 那是个**特殊情形**，
    #   在上面测会得到一个与真机不同的结论（第一版就掉进这个坑）。
    import yaml                                            # noqa: PLC0415
    cal_p = os.path.join(ROOT, "config", "calibration.yaml")
    wp_p = os.path.join(ROOT, "config", "workpiece.yaml")
    with open(wp_p, encoding="utf-8") as f:
        wp = yaml.safe_load(f)
    # ★ 用项目自己的 `load_config()`（它会把 "tray_map.homography" 这类点号键**摊平**，
    #   与工程其它地方一致）；直接 `yaml.safe_load` 得到的是**嵌套**结构，两种都支持。
    from assemvision.config import load_config              # noqa: PLC0415
    wp = load_config(wp_p)
    cal = load_config(cal_p)
    Hraw = cal.get("tray_map.homography")
    if Hraw is None:
        tm = cal.get("tray_map") or {}
        Hraw = tm.get("homography") if isinstance(tm, dict) else None
    if Hraw is None:
        fail("calibration.yaml 里找不到 tray_map.homography")
    H = np.asarray(Hraw, float)
    sc = cal.get("tray_map.scale", 1.0)
    tray = TrayMap(wp, homography=H, ref_scale=float(sc))
    # ★★ 画布必须**装得下挪动后的料盘**，否则 ROI 会被画面边界裁掉 ——
    #   裁掉之后"含住料盘中心"当然不成立，而那是**测试场景建错**，不是代码坏。
    #   （第一版用 640x480，料盘挪到 (740,340) 就出画面了，于是假失败。）
    shape = (900, 1100)
    mmpp = tray.mm_per_px(1.0)
    nom_c = tray.mm_to_px(np.array([[80.0, 80.0]]), 1.0)[0]
    print("真实标定单应：料盘中心(80,80)mm -> 像素 (%.1f, %.1f)，GSD %.4f mm/px"
          % (nom_c[0], nom_c[1], mmpp))

    print("\n== 场景 1：料盘没动时，ROI 中心应落在料盘中心（外扩对称，中心不动）==")
    det0 = TrayDetect(ok=True, center_px=(float(nom_c[0]), float(nom_c[1])),
                      size_px=(185.6, 185.6), angle_deg=90.0, area_px=34400.0,
                      fill=0.9)
    p0 = pose_from_tray_detect(det0, (nom_c[0], nom_c[1]), 185.6, mmpp,
                               nominal_angle_deg=90.0)
    check(getattr(p0, "ok", False), "位姿应有效（料盘没动 -> tx/ty 应约为 0）")
    print("   tx=%.3f ty=%.3f" % (p0.tx, p0.ty))
    check(abs(p0.tx) < 1e-6 and abs(p0.ty) < 1e-6,
          "料盘没动时平移应为 0（实测 tx=%.4f ty=%.4f）" % (p0.tx, p0.ty))
    c0 = _roi_center(tray, wp, p0, shape)
    if c0 is None:
        fail("料盘没动时 ROI 却是空的")
    from assemvision.core.checks import Qualifier as _Q0        # noqa: PLC0415
    m0 = _Q0(wp, tray, None).work_roi_mask(shape, 1.0, pose=p0, margin_mm=25.0)
    ix0, iy0 = int(round(nom_c[0])), int(round(nom_c[1]))
    inside0 = bool(m0[iy0, ix0] > 0)
    # ★ 断言用"**含住**"而不是"质心相等"：ROI = 料盘四边形（外扩 25mm）+ 装配位圆，
    #   单应本身还带轻微旋转与错切，所以**质心本来就不会等于料盘中心**
    #   （实测差 37px，完全正常）。第一版把断言写成"质心 < 2px"，那是错的判据。
    print("   ROI 质心 (%.1f,%.1f)（与料盘中心差 %.1f px，正常 —— 质心受装配位圆影响）"
          % (c0[0], c0[1], float(np.hypot(c0[0] - nom_c[0], c0[1] - nom_c[1]))))
    print("   ★ ROI 是否含住料盘中心 (%d,%d)：%s" % (ix0, iy0, "是" if inside0 else "否"))
    check(inside0, "★★ 料盘原位时 ROI 必须含住料盘中心")

    print("\n== 场景 2：★★ 核心性质 —— 料盘挪了，ROI 必须跟过去（含住料盘中心）==")
    # ★★ 这里断言的是"**弱**性质"，不是"位移量完全相等"（见文件末尾的说明）：
    #     ROI 的位移量会被"标定尺度的不自洽"放大/缩小（实测约 1.10 倍），
    #     因为 tx 用 `mm_per_px` 换算、而 mm_to_px 又用单应自身的尺度投回去。
    #   对"零件别被挡在 ROI 外面"这件事来说，**弱性质就够了**：
    #     只要 ROI 还盖着料盘、面积量级没崩，零件就在里面。
    c_nom = _roi_center(tray, wp, p0, shape)
    a0 = float(np.count_nonzero(
        __import__("assemvision.core.checks", fromlist=["Qualifier"])
        .Qualifier(wp, tray, None).work_roi_mask(shape, 1.0, pose=p0, margin_mm=25.0)))
    check(c_nom is not None, "料盘原位时应能算出 ROI")
    for dx, dy in ((258.0, 58.0), (-120.0, 90.0), (150.0, -100.0)):
        ex, ey = nom_c[0] + dx, nom_c[1] + dy
        det = TrayDetect(ok=True, center_px=(float(ex), float(ey)),
                         size_px=(185.6, 185.6), angle_deg=90.0, area_px=34400.0,
                         fill=0.9)
        p = pose_from_tray_detect(det, (nom_c[0], nom_c[1]), 185.6, mmpp,
                                  nominal_angle_deg=90.0)
        from assemvision.core.checks import Qualifier as _Q    # noqa: PLC0415
        m = _Q(wp, tray, None).work_roi_mask(shape, 1.0, pose=p, margin_mm=25.0)
        n = int(np.count_nonzero(m))
        if n == 0:
            fail("料盘挪 (%.0f,%.0f) 后 ROI 变空了" % (dx, dy))
        ys, xs = np.nonzero(m)
        # ★ 断言 1：ROI 必须**含住**检测到的料盘中心
        ix, iy = int(round(ex)), int(round(ey))
        inside = (0 <= iy < m.shape[0] and 0 <= ix < m.shape[1] and m[iy, ix] > 0)
        # ★ 断言 2：面积量级不能崩（0.5x ~ 2x 原位）
        ratio = n / max(1.0, a0)
        moved = float(np.hypot(xs.mean() - c_nom[0], ys.mean() - c_nom[1]))
        print("   挪 (%+.0f,%+.0f)：ROI %d px（原位 %d，比 %.2f），中心移动 %.0f px，"
              "含住料盘中心=%s" % (dx, dy, n, int(a0), ratio, moved, "是" if inside else "否"))
        check(inside, "★★ 料盘挪 (%+.0f,%+.0f) 后 ROI 必须含住料盘中心" % (dx, dy))
        check(0.5 <= ratio <= 2.0,
              "ROI 面积不该崩（比 %.2f，应在 0.5~2.0）" % ratio)
        check(moved > 0.3 * float(np.hypot(dx, dy)),
              "ROI 必须明显跟着料盘挪（移动 %.0f px，料盘挪了 %.0f px）"
              % (moved, float(np.hypot(dx, dy))))

    print("\n== 场景 3：★★ 标称单应平移过期（没重跑 c3）—— ROI 仍应跟料盘走 ==")
    H2 = H.copy()
    H2[0, 2] += 200.0                    # 人为把标称位置挪 200px，模拟过期
    tray2 = TrayMap(wp, homography=H2, ref_scale=float(cal["tray_map.scale"]))
    ncx2, ncy2 = tray2.mm_to_px(np.array([[80.0, 80.0]]), 1.0)[0]
    det = TrayDetect(ok=True, center_px=(float(nom_c[0]), float(nom_c[1])),
                     size_px=(185.6, 185.6), angle_deg=90.0, area_px=34400.0, fill=0.9)
    p = pose_from_tray_detect(det, (ncx2, ncy2), 185.6, tray2.mm_per_px(1.0),
                              nominal_angle_deg=90.0)
    c = _roi_center(tray2, wp, p, shape)
    if c is None:
        fail("标称单应过期时 ROI 变空了")
    from assemvision.core.checks import Qualifier as _Q2        # noqa: PLC0415
    m2 = _Q2(wp, tray2, None).work_roi_mask(shape, 1.0, pose=p, margin_mm=25.0)
    ixr, iyr = int(round(nom_c[0])), int(round(nom_c[1]))       # 真实料盘中心
    ixn, iyn = int(round(nom_c[0] + 200.0)), int(round(nom_c[1]))  # 过期标称位置
    ins_real = bool(m2[iyr, ixr] > 0)
    ins_nom = bool(m2[iyn, ixn] > 0)
    d_real = float(np.hypot(c[0] - nom_c[0], c[1] - nom_c[1]))
    d_nom = float(np.hypot(c[0] - (nom_c[0] + 200.0), c[1] - nom_c[1]))
    print("   标称单应平移 +200px（模拟没重跑 c3），检测到的料盘仍在 (%.1f,%.1f)"
          % (nom_c[0], nom_c[1]))
    print("      ROI 质心 = (%.1f,%.1f)，距真实料盘 %.1f px，距过期标称位置 %.1f px"
          % (c[0], c[1], d_real, d_nom))
    print("      ★ 含住真实料盘中心 = %s ；含住过期标称位置 = %s"
          % ("是" if ins_real else "否", "是" if ins_nom else "否"))
    check(ins_real,
          "★★ 标称单应平移过期时，ROI 仍必须含住【真实】料盘（说明平移被抵消了）")
    # ★ 用"质心距离"当判据只能要求"明显更靠近真实位置"，不能要求相等（见文件末尾说明）
    check(d_real < d_nom, "ROI 必须更贴近真实料盘而不是过期的标称位置")

    print("\n料盘位姿/ROI 性质测试全部通过")
    print("★ 注意：这里测的是**几何性质**（用算术推），不是真机识别率；")
    print("   真机还要重跑 c3（GSD）与 c6（色域模板），见 docs/现状与交接.md §3.4。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
