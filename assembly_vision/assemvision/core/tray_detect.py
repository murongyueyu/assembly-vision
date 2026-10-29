# -*- coding: utf-8 -*-
"""料盘位姿检测（用**料盘自身轮廓**，不依赖零件）。

═══ 为什么需要它（而不是只用零件星座）═══

`tray_pose.estimate_pose()` 用「零件星座」估位姿，很准，但有个硬前提：
**画面里得认得出零件**。实测踩过两种情况：

  ① 装配过程中零件被逐个拿走 —— 料盘慢慢变空，最后**一个零件都没有**，
     位姿估计 `ok=False`，只能退回过期的标称坐标，ROI 全错位。
  ② 料盘被挪得很远（本次实测往右挪了约 15cm）—— 同理。

而**料盘本身总是在画面里**（黑/深色、接近正方形、尺寸已知）。
所以再加一条更可靠的依据：**直接检测料盘的轮廓，用它的四个角定位置与朝向**。

═══ 怎么做 ═══

  ① 在 HSV 里取"暗色"掩膜（料盘是黑色 3D 打印件，V 低、S 低）
  ② 去掉绿幕与桌面 → 只留料盘候选
  ③ 找最大连通域，用 `minAreaRect` 得到一个带朝向的旋转矩形
  ④ 检查"像不像料盘"：长宽比接近 1、面积够大
  ⑤ 由矩形的**中心 + 角度**构造刚体变换（相对标称位置）

═══ 与零件星座的关系 ═══

**两者互补，运行时优先用零件星座**（能同时定位置和朝向，精度更高），
零件不够时退回料盘轮廓。料盘轮廓的精度差一些（轮廓受光照/遮挡影响，
边缘提取有几像素抖动），但**永远不会因为"零件被拿走"而失效**。

  优先级：零件星座（>=2 件，残差小） > 料盘轮廓 > 沿用上次位姿 > 标称坐标
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .tray_pose import TrayPose


@dataclass
class TrayDetect:
    """料盘轮廓检测结果。"""
    ok: bool = False
    center_px: Optional[Tuple[float, float]] = None
    size_px: Tuple[float, float] = (0.0, 0.0)      # minAreaRect 的 (w, h)
    angle_deg: float = 0.0                          # minAreaRect 角度
    area_px: float = 0.0
    fill: float = 0.0                               # 连通域面积 / 矩形面积
    reason: str = ""


def detect_tray(bgr: np.ndarray,
                v_max: int = 125,
                s_max: int = 140,
                min_area_ratio: float = 0.01,
                aspect_tol: float = 0.35,
                fill_min: float = 0.55) -> TrayDetect:
    """在画面里找料盘（深色、接近正方形的大连通域）。

    参数
      v_max, s_max      暗色阈值。料盘是黑色打印件，实测 V 中值约 60~90；
                        绿幕 V 高、桌面 S 高，都会被这两条挡掉。
      min_area_ratio    最小面积占全图比例（滤掉小暗块）
      aspect_tol        允许的长宽比偏差（料盘接近正方形 -> |w/h-1| 要小）
      fill_min          填充率下限：连通域面积 / 外接旋转矩形面积。
                        料盘中间有凹槽但整体是实心方块，填充率应较高；
                        细长阴影/边缘伪影填充率低。
    """
    d = TrayDetect()
    if bgr is None or bgr.size == 0:
        d.reason = "空图"
        return d
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 深色 + 低饱和：料盘是哑光黑，桌面/绿幕都不满足
    dark = cv2.inRange(hsv, np.array([0, 0, 0], np.uint8),
                       np.array([179, s_max, v_max], np.uint8))
    # 形态学：闭运算把凹槽/高光造成的洞补上，再开运算去毛刺
    k = max(3, int(round(min(H, W) * 0.02)) | 1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

    n, lab, stats, cent = cv2.connectedComponentsWithStats(dark, 8)
    if n <= 1:
        d.reason = "没找到深色连通域"
        return d
    i = 1 + int(np.argmax(stats[1:, 4]))
    area = float(stats[i, cv2.CC_STAT_AREA])
    if area < min_area_ratio * H * W:
        d.reason = "最大深色块太小（%.1f%%），可能被遮挡或在画面外" % (100 * area / (H * W))
        return d
    mask = (lab == i).astype(np.uint8) * 255
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not cnts:
        d.reason = "轮廓提取失败"
        return d
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    if w <= 1 or h <= 1:
        d.reason = "旋转矩形退化"
        return d
    d.center_px = (float(cx), float(cy))
    d.size_px = (float(w), float(h))
    d.angle_deg = float(ang)
    d.area_px = area
    d.fill = float(area / max(1.0, w * h))
    ratio = min(w, h) / max(w, h)
    if ratio < (1.0 - aspect_tol):
        d.reason = "长宽比 %.2f 不像方形料盘（可能只框到一部分）" % ratio
        return d
    if d.fill < fill_min:
        d.reason = "填充率 %.2f 太低（可能是不规则阴影）" % d.fill
        return d
    d.ok = True
    return d


def pose_from_tray_detect(det: TrayDetect, nominal_center_px, nominal_side_px: float,
                          mm_per_px: float, nominal_angle_deg: float = 90.0,
                          use_scale: bool = False,
                          pixel_to_mm: Optional[Callable] = None) -> TrayPose:
    """把"检测到的料盘"换算成 tray-mm 坐标系下的刚体变换。

    参数
      det                 detect_tray() 的结果
      nominal_center_px   标定时料盘中心的像素位置（由 c3 单应算出）
      nominal_side_px     标定时料盘边长（像素）
      mm_per_px           标定时的每像素多少毫米
      nominal_angle_deg   标定时料盘 minAreaRect 的角度（一般 90 或 0）
      use_scale           ★ 是否用"实测边长/标称边长"修正尺度。**默认 False**。
        原因：实测发现检测到的边长比标称大 6%（190 vs 180 px）——
        多半是形态学闭运算 + 阴影把轮廓撑大了。若拿这个比例去缩放坐标，
        会把 6% 的误差放大到整个视野（±15mm 量级）。
        **只取"中心 + 朝向"更稳**：中心由整个连通域的质心决定，
        对边缘膨胀不敏感（对称膨胀时质心不动）。
        只有在确认"相机距离变了"的场合才开 use_scale。
      pixel_to_mm         C标定逆映射（含当前scale）；有此输入时在同一平面
        解中心/方向，不再使用局部GSD或use_scale近似。外轮廓仍有90°歧义。

    返回 TrayPose（在 **tray-mm** 坐标系里表达，与零件星座的结果可直接互换）

    ★ 旋转角的换算：minAreaRect 的角是"图像里的方向"，要变成 tray-mm 里的
      旋转量，需要先取与标定角度的**差值**，再注意 y 轴向下的符号 ——
      与 TrayPose.R 的约定保持一致（见 tray_pose.TrayPose.R 的说明）。
    """
    if not det.ok or det.center_px is None:
        return TrayPose(ok=False)
    ncx, ncy = float(nominal_center_px[0]), float(nominal_center_px[1])
    cx, cy = det.center_px
    if pixel_to_mm is not None:
        # ★ 有单应时，先把中心和方向投回同一平面再解刚体；
        #   不能把全视野的像素位移乘盘中心一个GSD，远距离移动会累积偏差。
        ref_angle, now_angle = np.radians([nominal_angle_deg, det.angle_deg])
        points = np.array([[ncx, ncy], [cx, cy],
                           [ncx + np.cos(ref_angle), ncy + np.sin(ref_angle)],
                           [cx + np.cos(now_angle), cy + np.sin(now_angle)]], dtype=float)
        plane = np.asarray(pixel_to_mm(points), dtype=float).reshape(4, 2)
        ref_axis, now_axis = plane[2] - plane[0], plane[3] - plane[1]
        da = float(np.degrees(np.arctan2(now_axis[1], now_axis[0])
                             - np.arctan2(ref_axis[1], ref_axis[0])))
        da = (da + 45.0) % 90.0 - 45.0
        pose = TrayPose(theta_deg=-da, used=["tray_outline_homography"], ok=True)
        translation = plane[1] - pose.R @ plane[0]
        pose.tx, pose.ty = map(float, translation)
        return pose
    # 平移：像素差 -> mm，方向与 tray-mm 一致（x 右、y 下）
    k = float(mm_per_px)
    if use_scale and nominal_side_px > 1.0 and det.size_px[0] > 1.0:
        side = (det.size_px[0] + det.size_px[1]) / 2.0
        k = k * (side / float(nominal_side_px))
    tx = (cx - ncx) * k
    ty = (cy - ncy) * k
    # 旋转：minAreaRect 的角在 [0,90)，直接相减可能差 90 的整数倍，先归一化
    da = det.angle_deg - nominal_angle_deg
    while da > 45.0:
        da -= 90.0
    while da < -45.0:
        da += 90.0
    # ★ y 轴向下，图像里的"逆时针"在 tray-mm 里是"顺时针"，取负以匹配 TrayPose.R
    theta = -da
    pose = TrayPose(theta_deg=float(theta), n_used=0, residual_mm=0.0,
                    used=["tray_outline"], ok=True)
    # ★ 检测到的位移是【盘中心】的位移；TrayPose.apply却绕tray原点旋转。
    #   必须加上 c-Rc，才能表达绕自身中心转动，避免旋转时整盘ROI被甩开。
    #   本入口只有边长和单一GSD，参考中心以半边长换算，仍是局部尺度近似；
    #   ★ 透视较大/跨视野长距离移动时须用独立角点复核，不能称精确单应更新。
    half_mm = float(nominal_side_px) * float(mm_per_px) / 2.0
    center_mm = np.array([half_mm, half_mm], dtype=np.float64)
    offset = center_mm - pose.R @ center_mm
    pose.tx = float(tx + offset[0])
    pose.ty = float(ty + offset[1])
    return pose
