# -*- coding: utf-8 -*-
"""料盘位姿自动估计 —— 用「零件星座」当基准，摆脱固定槽位坐标。

═══ 解决什么问题 ═══

原来整套判定建立在"料盘位置和朝向固定"这个前提上：c3 标一次料盘单应，
之后所有槽位 ROI 都按那次的位置算。**实测踩过**：料盘在 c3 之后被挪了约 18mm
（相机没动 —— 绿幕质心只差 3.8px，而料盘暗区中心差了 18px），
于是四个槽的 ROI 全部偏了 18~20mm，覆盖率和判定全错。

═══ 怎么做 ═══

**四类零件本身就是最好的基准点**：
  ① 它们的**相对布局是固定的**（料盘四个槽不会自己动）
  ② 形状/面积特征**与朝向无关**，所以能先认出"这块是什么零件"
  ③ 认出身份后，把"实测质心"与"标称质心"配对，就能解出料盘的刚体变换

流程：
    检出黄铜块 -> 按特征识别身份 -> 得到 (实测质心, 标称质心) 若干对
      -> 最小二乘拟合刚体变换 (旋转 R + 平移 t)
      -> 用 R,t 修正所有槽位 ROI

═══ 为什么用刚体变换而不是重新算单应 ═══

  · 只解 3 个自由度（θ, tx, ty），4 个零件给 8 个观测 -> 超定，稳
  · **不会把误识别放大成畸变**（单应 8 自由度、4 点恰好定解，一个点错就整体歪）
  · 料盘是刚体，本来就只有 3 个自由度

═══ 局限（要知道）═══

  · 至少要认出 **2 个**零件才能定旋转；只认出 1 个只能定平移
  · 零件被拿走时，星座会缺角 —— 用剩下的零件仍然能估计（超定或恰好）
  · **如果所有零件都被拿走**（比如装配完成），就没有基准了；
    这时沿用最后一次成功的位姿（`last_ok`）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TrayPose:
    """料盘相对标称位置的刚体变换。"""
    theta_deg: float = 0.0          # 旋转（度）
    tx: float = 0.0                 # 平移 x（tray mm）
    ty: float = 0.0                 # 平移 y（tray mm）
    n_used: int = 0                 # 参与拟合的零件数
    residual_mm: float = 0.0        # 拟合残差（mm）
    used: List[str] = field(default_factory=list)
    ok: bool = False                # 是否可用（至少 1 个零件）

    @property
    def R(self) -> np.ndarray:
        """旋转矩阵。

        ★ 符号约定（踩过坑，务必看清）：
          图像坐标系 **y 轴向下**，所以常规数学里的 [[c,-s],[s,c]] 在这个坐标系里
          对应的视觉旋转是 **-θ**。为了让 `theta_deg` 与"人眼看到的旋转方向"一致，
          这里用 [[c, s], [-s, c]]。

          实测症状（符号写反时）：模拟料盘转 +7.5°，估计器却报 -7.5°，
          而且残差高达 14.7mm（因为平移补偿也跟着算错了）。
        """
        a = np.radians(self.theta_deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, s], [-s, c]])

    def apply(self, pts_mm) -> np.ndarray:
        """把标称 tray-mm 点变换到当前料盘的 tray-mm 坐标系。

        ★ 注意方向：标称 -> 当前。槽位坐标是"标称"的，
          要拿它去匹配"当前"画面，就得正着变换。
        """
        p = np.asarray(pts_mm, dtype=np.float64).reshape(-1, 2)
        return (self.R @ p.T).T + np.array([self.tx, self.ty])

    def inverse_apply(self, pts_mm) -> np.ndarray:
        """当前 -> 标称（把实测点变回标称系，便于比较）。"""
        p = np.asarray(pts_mm, dtype=np.float64).reshape(-1, 2)
        return ((p - np.array([self.tx, self.ty])) @ self.R).reshape(-1, 2)

    def summary(self) -> str:
        return ("θ=%+.2f° Δ=(%+.1f,%+.1f)mm 用%d件 残差%.1fmm"
                % (self.theta_deg, self.tx, self.ty, self.n_used, self.residual_mm))


def estimate_pose(observed: Dict[str, Tuple[float, float]],
                  nominal: Dict[str, Tuple[float, float]],
                  min_residual_mm: float = 25.0,
                  max_theta_deg: float = 25.0) -> TrayPose:
    """由「识别出的零件 -> 实测质心」与「标称质心」解刚体变换。

    参数
      observed  {零件名: (x, y)} 当前实测质心（**未修正的 tray-mm**）
      nominal   {零件名: (x, y)} 标称质心（零件在正确槽位时的 tray-mm）
      min_residual_mm  残差超过它就认为拟合不可信（多半是认错了零件）
      max_theta_deg    ★ 旋转角超过它就判不可信。
        料盘是**放在桌上**的，实际不会转很多。实测踩过：标称槽位坐标过期
        （换了一次料盘单应、坐标系变了）时，估计器会为了硬凑出一个 **48°**
        的荒谬旋转，而残差还可能很小 —— 这种"看着成功其实全错"最危险，
        所以必须卡角度。

    返回 TrayPose；`ok=False` 表示没有足够依据（零件一个都没认出来）。
    """
    keys = [k for k in observed if k in nominal]
    if not keys:
        return TrayPose(ok=False)
    P = np.array([observed[k] for k in keys], dtype=np.float64)   # 实测
    Q = np.array([nominal[k] for k in keys], dtype=np.float64)    # 标称

    cO = P.mean(axis=0)
    cN = Q.mean(axis=0)
    A = P - cO          # 实测去中心
    B = Q - cN          # 标称去中心

    if len(keys) >= 2:
        # 最佳旋转：解 sum(a_x*b_y - a_y*b_x) 与 sum(a·b)
        num = float((A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]).sum())
        den = float((A[:, 0] * B[:, 0] + A[:, 1] * B[:, 1]).sum())
        theta = float(np.degrees(np.arctan2(num, den)))
    else:
        theta = 0.0      # 只有一个零件时定不了旋转

    pose = TrayPose(theta_deg=theta, n_used=len(keys), used=list(keys))
    # 平移：让"当前系中心"映射到"标称系中心"
    #   变换定义：标称 -> 当前，所以 当前中心 = R·标称中心 + t  => t = cO - R·cN
    pose.tx = float(cO[0] - (pose.R @ cN)[0])
    pose.ty = float(cO[1] - (pose.R @ cN)[1])

    # 残差：把实测点转回标称系，看与标称差多少
    back = pose.inverse_apply(P)
    res = np.linalg.norm(back - Q, axis=1)
    pose.residual_mm = float(res.mean())
    pose.ok = bool(pose.residual_mm <= min_residual_mm
                   and abs(pose.theta_deg) <= max_theta_deg)
    if not pose.ok and abs(pose.theta_deg) > max_theta_deg:
        pose.used.append("rejected:theta=%.1f" % pose.theta_deg)
    return pose


def apply_pose_to_slot(slot: dict, pose: TrayPose) -> dict:
    """把槽位定义按料盘位姿变换，返回**修正后**的槽位字典（不改原字典）。

    圆槽：只改中心（形状与朝向无关）
    胶囊槽：改中心 + 把 angle_deg 加上位姿旋转
    """
    out = dict(slot)
    c = np.asarray(slot.get("center", [0.0, 0.0]), dtype=np.float64)
    out["center"] = [float(v) for v in pose.apply(c.reshape(1, 2))[0]]
    if str(slot.get("type", "circle")) != "circle":
        out["angle_deg"] = float(slot.get("angle_deg", 0.0)) + pose.theta_deg
    out["_pose_corrected"] = True
    return out
