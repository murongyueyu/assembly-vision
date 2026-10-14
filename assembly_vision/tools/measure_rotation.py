# -*- coding: utf-8 -*-
"""料盘旋转/平移检测能力测量 —— 直接给 `estimate_pose` **已知真值的质心**。

★ 为什么绕开渲染与分割（这轮踩了两次，记清楚）：
  ① `render_scene(wheel_rotate_deg=...)` 转的是**辐条图案**，零件质心没动
     -> 位姿本来就该报 0°，拿它测整盘旋转是错的。
  ② 改用"整幅图几何旋转"之后，又发现 `make_synthetic.SLOTS` 的槽心和
     `workpiece.yaml` 的标称槽心**差了 5.8~10.8mm**，于是凭空多出一个
     ~2° / ~3mm 的固定偏差 —— 那是**合成数据自己的几何不一致**，
     与估计器无关（用渲染真值直接解也报 -2.02°，把它当标称再解就是 0.00°）。

  `estimate_pose` 是**纯几何算法**（输入:实测质心 + 标称质心；输出:θ,tx,ty）。
  所以测它就该**直接喂质心** —— 真值精确、无渲染/分割噪声干扰。
  渲染与分割那一层的误差另有工具测（见 `measure_part_id_e2e.py`）。

★ 本工具回答：**"料盘转了/挪了，系统能不能检测到、测得准不准"**
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from assemvision.config import load_config                        # noqa: E402
from assemvision.core.tray_pose import TrayPose, estimate_pose    # noqa: E402


def nominal_centers(wp):
    out = {}
    for s in (wp.get("slots") or []):
        c = s.get("center")
        if c is not None:
            out[str(s.get("part"))] = (float(c[0]), float(c[1]))
    return out


def make_observed(nominal, theta_deg, tx, ty, drop=()):
    """按已知刚体变换造"实测质心"。

    ★ 用 `TrayPose.apply` 自己来造 —— 这样**造数据的变换与解算的变换是同一套约定**，
      测的是"能不能解回来"，不会因为符号约定写反而得出假结论。
      （真正的符号正确性由 `test_tray_pose.py` 对着图像坐标单独验。）
    """
    pose = TrayPose(theta_deg=theta_deg, tx=tx, ty=ty, n_used=len(nominal), ok=True)
    keys = [k for k in nominal if k not in drop]
    pts = np.array([nominal[k] for k in keys], float)
    got = pose.apply(pts)
    return {k: (float(got[i, 0]), float(got[i, 1])) for i, k in enumerate(keys)}


def measure_rotation(nominal, angles, max_theta):
    print("  ── ① 纯旋转（tx=ty=0，4 件齐全）──")
    print("  %-9s %-10s %-10s %-9s %s" % ("真值θ°", "解出θ°", "误差°", "残差mm", "ok"))
    print("  " + "-" * 56)
    errs = []
    for th in angles:
        obs = make_observed(nominal, th, 0.0, 0.0)
        p = estimate_pose(obs, nominal, max_theta_deg=max_theta)
        if not getattr(p, "ok", False):
            print("  %-9.1f %-10s %-10s %-9s %s" % (th, "(解不出)", "-", "-", "False"))
            continue
        e = abs(((p.theta_deg - th) + 180.0) % 360.0 - 180.0)
        errs.append(e)
        print("  %-9.1f %-10.2f %-10.4f %-9.4f %s"
              % (th, p.theta_deg, e, p.residual_mm, "True"))
    if errs:
        print("     ★ 角度误差 最大 %.4f° / 平均 %.4f°" % (max(errs), float(np.mean(errs))))
    return errs


def measure_translation(nominal, offs, max_theta):
    print()
    print("  ── ② 纯平移（θ=0）──")
    print("  %-16s %-18s %-16s %s" % ("真值(mm)", "解出(mm)", "误差mm", "残差mm"))
    print("  " + "-" * 66)
    errs = []
    for (tx, ty) in offs:
        obs = make_observed(nominal, 0.0, tx, ty)
        p = estimate_pose(obs, nominal, max_theta_deg=max_theta)
        if not getattr(p, "ok", False):
            print("  %-16s %-18s %-16s %s" % ("(%+.1f,%+.1f)" % (tx, ty),
                                              "(解不出)", "-", "-"))
            continue
        e = float(np.hypot(p.tx - tx, p.ty - ty))
        errs.append(e)
        print("  %-16s %-18s %-16.4f %.4f"
              % ("(%+.1f,%+.1f)" % (tx, ty),
                 "(%+.2f,%+.2f)" % (p.tx, p.ty), e, p.residual_mm))
    if errs:
        print("     ★ 平移误差 最大 %.4f mm / 平均 %.4f mm" % (max(errs), float(np.mean(errs))))
    return errs


def measure_robust(nominal, angles, max_theta):
    """少零件、认错零件、以及超出闸门时的行为。"""
    print()
    print("  ── ③ 边界：零件不全 / 越界 / 认错 ──")
    cases = [
        ("3 件（少轮盘）",        make_observed(nominal, 8.0, 5.0, -3.0, drop=("P3_wheel",))),
        ("2 件（少轮盘+顶盖）",    make_observed(nominal, 8.0, 5.0, -3.0,
                                                 drop=("P3_wheel", "P4_cap"))),
        ("1 件（只剩底座）",      make_observed(nominal, 8.0, 5.0, -3.0,
                                                 drop=("P3_wheel", "P4_cap", "P2_column"))),
        ("0 件（全拿走）",        {}),
    ]
    for label, obs in cases:
        p = estimate_pose(obs, nominal, max_theta_deg=max_theta)
        print("  %-20s ok=%-6s %s" % (label, getattr(p, "ok", False), p.summary()))
    # 越界
    for th in (30.0, 40.0, 48.0):
        obs = make_observed(nominal, th, 0.0, 0.0)
        p = estimate_pose(obs, nominal, max_theta_deg=max_theta)
        print("  %-20s ok=%-6s %s  <- %s"
              % ("真转 %.0f°（超闸门 %.0f°）" % (th, max_theta), getattr(p, "ok", False),
                 p.summary(), "被闸门挡住=有意的保护" if not p.ok else "放行"))
    # 认错零件：把底座的质心换成顶盖的位置（模拟误识别）
    obs = make_observed(nominal, 3.0, 0.0, 0.0)
    obs["P1_base"] = nominal["P4_cap"]
    p = estimate_pose(obs, nominal, max_theta_deg=max_theta)
    print("  %-20s ok=%-6s %s  <- 残差过大应自动 ok=False"
          % ("底座认成顶盖的位置", getattr(p, "ok", False), p.summary()))


def main() -> int:
    ap = argparse.ArgumentParser("measure_rotation")
    ap.add_argument("--workpiece", default=os.path.join(ROOT, "config", "workpiece.yaml"))
    ap.add_argument("--angles",
                    default="-20,-15,-12,-10,-8,-5,-3,-1,0,1,3,5,8,10,12,15,20")
    ap.add_argument("--offsets",
                    default="0,0;5,0;-5,0;0,5;0,-5;20,0;-20,0;0,20;0,-20;30,25;-30,-25")
    args = ap.parse_args()

    wp = load_config(args.workpiece)
    nominal = nominal_centers(wp)
    gate = float((wp.get("tray_pose", {}) or {}).get("max_theta_deg", 25.0))

    print("═" * 72)
    print("  料盘旋转 / 平移 检测能力（直接喂已知真值的质心，无渲染/分割噪声）")
    print("═" * 72)
    print("  标称槽心（tray-mm）:")
    for k, v in nominal.items():
        print("     %-12s (%6.1f, %6.1f)" % (k, v[0], v[1]))
    print("  角度闸门 max_theta_deg = %.0f°" % gate)
    print()

    angles = [float(x) for x in args.angles.split(",")]
    offs = []
    for seg_ in args.offsets.split(";"):
        a, b = seg_.split(",")
        offs.append((float(a), float(b)))

    e_r = measure_rotation(nominal, angles, gate)
    e_t = measure_translation(nominal, offs, gate)
    measure_robust(nominal, angles, gate)

    print()
    print("═" * 72)
    print("  结  论")
    print("═" * 72)
    if e_r:
        print("  ① 旋转：在 ±%.0f° 内可检测，角度误差 最大 %.4f° / 平均 %.4f°"
              % (gate, max(e_r), float(np.mean(e_r))))
    if e_t:
        print("  ② 平移：可检测，误差 最大 %.4f mm / 平均 %.4f mm"
              % (max(e_t), float(np.mean(e_t))))
    print("  ③ 至少 2 件才能定旋转；1 件只能定平移；0 件 ok=False（绝不瞎猜）")
    print("  ④ 转角 > %.0f° 会被闸门挡住 —— 这是**有意的**保护" % gate)
    print("     （标称过期时会硬凑出 48° 荒谬旋转，且残差还很小，见 AGENTS.md §2.8）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
