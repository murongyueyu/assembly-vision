# -*- coding: utf-8 -*-
"""料盘位姿估计（用零件星座补偿料盘移动/旋转）离线测试。

测的是 assemvision/core/tray_pose.py。

★ 为什么这个功能必须有：
  原来整套判定建立在"料盘位置和朝向固定"上。实测踩过：料盘在 c3 标定之后
  被挪了约 18mm（**相机没动** —— 绿幕质心只差 3.8px，料盘暗区中心差了 18px），
  四个槽的 ROI 全偏 18~20mm，覆盖率从 0.6 掉到 0.2，明明有件全判"缺件"。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.core.tray_pose import (TrayPose, apply_pose_to_slot,   # noqa: E402
                                        estimate_pose)


# 与 workpiece.yaml 一致的标称位置
NOMINAL = {
    "P3_wheel":  (24.0, 53.0),
    "P2_column": (105.0, 54.0),
    "P4_cap":    (22.0, 132.0),
    "P1_base":   (105.0, 130.0),
}


def _apply_nominal(pose: TrayPose):
    """按给定位姿把标称点变换成"实测点"，模拟料盘被挪动。"""
    Q = np.array([NOMINAL[k] for k in NOMINAL], dtype=np.float64)
    return {k: tuple(v) for k, v in zip(NOMINAL, pose.apply(Q))}


def main() -> int:
    print("== 场景 1：料盘纯平移 ==")
    true = TrayPose(theta_deg=0.0, tx=18.0, ty=-11.0)
    obs = _apply_nominal(true)
    got = estimate_pose(obs, NOMINAL)
    print("   真实 Δ=(%.1f,%.1f)  估计 Δ=(%.1f,%.1f)  θ=%.2f  残差 %.2fmm 用%d件"
          % (true.tx, true.ty, got.tx, got.ty, got.theta_deg, got.residual_mm, got.n_used))
    check(got.ok, "位姿估计可用")
    check(abs(got.tx - true.tx) < 0.5 and abs(got.ty - true.ty) < 0.5,
          "平移估计误差 < 0.5mm")
    check(got.residual_mm < 0.5, "残差 < 0.5mm")

    print("== 场景 2：料盘纯旋转 ==")
    true = TrayPose(theta_deg=7.5, tx=0.0, ty=0.0)
    obs = _apply_nominal(true)
    got = estimate_pose(obs, NOMINAL)
    print("   真实 θ=%.2f  估计 θ=%.2f  残差 %.2fmm" % (true.theta_deg, got.theta_deg,
                                                   got.residual_mm))
    check(got.ok and abs(got.theta_deg - true.theta_deg) < 0.3, "旋转估计误差 < 0.3°")
    check(got.residual_mm < 0.5, "残差 < 0.5mm")

    print("== 场景 3：旋转 + 平移（真实工况）==")
    true = TrayPose(theta_deg=-1.5, tx=16.5, ty=-9.4)
    obs = _apply_nominal(true)
    got = estimate_pose(obs, NOMINAL)
    print("   真实 θ=%.2f Δ=(%.1f,%.1f)  估计 θ=%.2f Δ=(%.1f,%.1f)  残差 %.2fmm"
          % (true.theta_deg, true.tx, true.ty, got.theta_deg, got.tx, got.ty,
             got.residual_mm))
    check(got.ok and abs(got.theta_deg - true.theta_deg) < 0.3, "旋转估计正确")
    check(abs(got.tx - true.tx) < 1.0 and abs(got.ty - true.ty) < 1.0, "平移估计正确")

    print("== 场景 4：只有 2 个零件（其余被拿走）仍能估计 ==")
    true = TrayPose(theta_deg=4.0, tx=-8.0, ty=6.0)
    full = _apply_nominal(true)
    obs2 = {k: full[k] for k in ("P3_wheel", "P1_base")}
    got = estimate_pose(obs2, NOMINAL)
    print("   用 2 件: 估计 θ=%.2f Δ=(%.1f,%.1f) 残差 %.2fmm" % (got.theta_deg, got.tx,
                                                          got.ty, got.residual_mm))
    check(got.ok and got.n_used == 2, "2 个零件也能估计，n_used=2")
    check(abs(got.theta_deg - true.theta_deg) < 0.3, "2 件时旋转仍准")

    print("== 场景 5：只有 1 个零件只能定平移 ==")
    obs1 = {"P3_wheel": tuple(true.apply(np.array([NOMINAL["P3_wheel"]]))[0])}
    got = estimate_pose(obs1, NOMINAL)
    print("   用 1 件: θ=%.2f Δ=(%.1f,%.1f)  ok=%s" % (got.theta_deg, got.tx, got.ty, got.ok))
    check(got.n_used == 1 and abs(got.theta_deg) < 1e-6, "1 件时旋转取 0（定不了）")
    check(got.ok, "1 件时仍可用于平移补偿")

    print("== 场景 6：一个零件都没认出来 -> ok=False（不能瞎补）==")
    got = estimate_pose({}, NOMINAL)
    print("   空输入: ok=%s n_used=%d" % (got.ok, got.n_used))
    check(not got.ok, "没有依据时 ok=False，绝不瞎猜")

    print("== 场景 7：认错零件（观测与标称完全对不上）-> 应判不可信 ==")
    # 把 A 的观测配到 B 的标称上，残差会很大
    obs_bad = {"P3_wheel": (200.0, 200.0), "P1_base": (-50.0, -60.0)}
    got = estimate_pose(obs_bad, NOMINAL)
    print("   残差 %.1fmm  ok=%s" % (got.residual_mm, got.ok))
    check(not got.ok, "残差过大时 ok=False（多半是认错零件）")

    print("== 场景 7b：标称坐标过期时会出现荒谬旋转，必须被角度闸门拦下 ==")
    # ★ 实测踩过：换了一次料盘单应后，workpiece.yaml 里的标称槽位坐标是在**旧坐标系**
    #   里量的，于是估计器为了硬凑出 +48.27° 的荒谬旋转，而**残差还很小** ——
    #   这种"看着成功其实全错"最危险，所以必须卡角度。
    for bad_theta in (48.0, -40.0, 70.0):
        obs = _apply_nominal(TrayPose(theta_deg=bad_theta))
        got = estimate_pose(obs, NOMINAL)
        print("   模拟标称过期（等效转 %.0f°）: θ=%.1f 残差 %.2f  ok=%s"
              % (bad_theta, got.theta_deg, got.residual_mm, got.ok))
        check(not got.ok, "|θ|=%.0f° 超过闸门 -> ok=False（不用这个位姿）" % abs(bad_theta))
    for good_theta in (0.0, 5.0, 15.0, -20.0):
        obs = _apply_nominal(TrayPose(theta_deg=good_theta))
        got = estimate_pose(obs, NOMINAL)
        check(got.ok, "|θ|=%.0f° 在合理范围内 -> ok=True" % abs(good_theta))
        print("   合理旋转 %.0f° -> ok=True ✓" % good_theta)

    print("== 场景 8：apply_pose_to_slot 修正槽位（圆槽只移心，胶囊槽还要转朝向）==")
    pose = TrayPose(theta_deg=10.0, tx=5.0, ty=-3.0)
    circle = {"id": "slot_cap", "type": "circle", "center": [22.0, 132.0], "diameter": 25.0}
    caps = {"id": "slot_column", "type": "capsule", "center": [105.0, 54.0],
            "length": 91.0, "width": 28.0, "angle_deg": -44.0}
    c2 = apply_pose_to_slot(circle, pose)
    k2 = apply_pose_to_slot(caps, pose)
    print("   圆槽   center %s -> %s" % (circle["center"], [round(v, 1) for v in c2["center"]]))
    print("   胶囊槽 angle %.0f -> %.0f   center %s -> %s"
          % (caps["angle_deg"], k2["angle_deg"], caps["center"],
             [round(v, 1) for v in k2["center"]]))
    check(abs(k2["angle_deg"] - (-44.0 - 10.0)) < 1e-6, "胶囊槽朝向与TrayPose旋转矩阵一致")
    check("angle_deg" not in c2 or c2.get("angle_deg") is None or
          c2.get("angle_deg") == circle.get("angle_deg"), "圆槽不受朝向影响")
    check(circle["center"] == [22.0, 132.0], "不修改原槽位字典（避免污染配置）")

    print("\n料盘位姿估计测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
