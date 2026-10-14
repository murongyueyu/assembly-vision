# -*- coding: utf-8 -*-
"""零件追踪离线测试 —— 卡尔曼 + 相关滤波 + 遮挡状态机。

★ 本模块专门钉住三个**实测踩过**的行为（都是真 bug，不是理论）：
  ① `visible_ratio` 必须正确下降
     第一版把 `area_ref` 更新写在算比值**之前**，参考值被遮挡后的小面积拉低
     -> `visible_ratio` 恒为 1.0 -> 状态永远 `detected`
     -> 每帧都拿"被切掉一半的残缺件"去重新识别 -> **认错**
       （实测：遮挡 50% 时 P3_wheel 被认成 P2_column）
  ② 遮挡时必须**冻结身份**（不拿残缺件去认）
  ③ 换标签必须有**分数余量**，否则一次误判就永久污染身份

★ 做法：**绕开分割**，直接喂"面积受控的假 Blob"。
  这样测的是状态机本身，不受分割/光照干扰（判据与像素解耦，见 AGENTS §6）。
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2                                                    # noqa: E402
from assemvision.core.part_tracker import (KalmanXY, PartTracker,   # noqa: E402
                                           _iou, make_tracker)
from assemvision.types import Blob                            # noqa: E402


def mk_blob(cx: float, cy: float, area_px: float, aspect: float = 1.0) -> Blob:
    """造一个面积受控的 Blob（矩形，可调长宽比）。"""
    h = float(np.sqrt(area_px / max(1e-6, aspect)))
    w = area_px / max(1e-6, h)
    pts = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                    [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]], np.float64)
    contour = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    kw = {}
    for f in dataclasses.fields(Blob):
        if f.name == "contour":
            kw[f.name] = contour
        elif f.name == "bbox":
            kw[f.name] = (int(cx - w / 2), int(cy - h / 2), int(w), int(h))
        elif f.name == "centroid":
            kw[f.name] = (float(cx), float(cy))
        elif f.name == "area_px":
            kw[f.name] = float(abs(cv2.contourArea(contour)))
        elif f.name == "features":
            kw[f.name] = {"area_px": float(abs(cv2.contourArea(contour))),
                          "rect_w": w, "rect_h": h, "aspect": aspect}
    return Blob(**kw)


def main() -> int:
    frame = np.full((480, 640, 3), 90, np.uint8)

    print("== 追踪器可用性（两套 API 都要能走，树莓派上常只有 4.x 的 create()）==")
    for kind in ("kcf", "mosse", "medianflow"):
        tr, api = make_tracker(kind)
        print("   %-12s -> %s" % (kind, api))
        check(tr is not None or "没有可用" in api,
              "%s 要么可用、要么给出明确原因（不能静默返回 None）" % kind)

    print("\n== 卡尔曼平滑：**轨迹误差**应下降（这才是它优化的目标）==")
    print("   ★ 我第一版测的是「逐帧位移的起伏」，那个指标是错的 ——")
    print("     正常工作的卡尔曼在跟匀速目标时位移本就接近真值，")
    print("     该看的是**离真值的误差**。扫参见 tools/_probe_kalman.py。")
    kf = KalmanXY(0.0, 0.0)
    rng = np.random.default_rng(0)
    truth, raw, smooth = [], [], []
    for i in range(60):
        tx = i * 1.0
        mx = tx + float(rng.normal(0, 3.0))
        truth.append(tx)
        raw.append(mx)
        kf.predict()
        sx, _sy = kf.correct(mx, 0.0)
        smooth.append(sx)
    truth = np.array(truth[5:])
    e_raw = float(np.mean(np.abs(np.array(raw[5:]) - truth)))
    e_sm = float(np.mean(np.abs(np.array(smooth[5:]) - truth)))
    print("   轨迹误差: 观测 %.3f px -> 滤波 %.3f px（降 %.0f%%）"
          % (e_raw, e_sm, 100 * (1 - e_sm / max(1e-9, e_raw))))
    check(e_sm < e_raw,
          "卡尔曼应把轨迹误差压小（%.3f < %.3f）" % (e_sm, e_raw))

    print("\n== 遮挡状态机：受控面积序列 ==")
    tr = PartTracker({"backend": "kcf", "detect_every_n": 1, "min_area_px": 100,
                      "max_area_px": 100000, "full_ratio": 0.80,
                      "occluded_ratio": 0.55, "area_ref_rate": 0.15})
    A0 = 4000.0
    # 前 3 帧满面积，建立参考并拿到身份
    for _ in range(3):
        tr.update(frame, [mk_blob(320, 240, A0)], detect=True,
                  classify=lambda f: ("P_REF", 0.90))
    check(len(tr.tracks) == 1, "应建立 1 条轨迹")
    t = tr.tracks[0]
    ref_after_full = t.area_ref
    print("   满面积建立: area_ref=%.0f part=%s" % (ref_after_full, t.part_id))
    check(t.part_id == "P_REF", "满面积时应拿到身份")

    # 逐帧递减：身份分类器**故意一直给错误答案**（且置信度很高）
    seen = []
    for frac in (0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20):
        tr.update(frame, [mk_blob(320, 240, A0 * frac)], detect=True,
                  classify=lambda f: ("P_WRONG", 0.99))
        if not tr.tracks:
            seen.append((frac, None, None, None))
            continue
        t = tr.tracks[0]
        seen.append((frac, t.visible_ratio, t.state, t.part_id))

    print("   %-8s %-10s %-11s %s" % ("面积比", "visible", "state", "part_id"))
    for frac, vis, st, pid in seen:
        print("   %-8.2f %-10s %-11s %s"
              % (frac, "%.2f" % vis if vis is not None else "-", st or "-", pid or "-"))

    # ① visible 必须跟着面积下降（这是第一版写错的那个 bug）
    vis_list = [v for _f, v, _s, _p in seen if v is not None]
    check(len(vis_list) >= 4 and vis_list[0] > 0.8 and vis_list[-1] < 0.45,
          "visible_ratio 必须随面积下降（实测 %s）—— 第一版恒为 1.0 是 bug"
          % ["%.2f" % v for v in vis_list[:4]])
    # ② area_ref 不能被遮挡拉低
    if tr.tracks:
        check(tr.tracks[0].area_ref >= ref_after_full * 0.9,
              "area_ref 不应被遮挡拉低（%.0f -> %.0f）"
              % (ref_after_full, tr.tracks[0].area_ref))
    # ③ 状态机：低可见度时必须 occluded
    occ = [s for _f, _v, s, _p in seen if s == "occluded"]
    check(len(occ) > 0, "可见度低于 occluded_ratio 时应进入 occluded 状态")
    # ④ 遮挡时身份必须冻结（不能被 P_WRONG 污染）
    polluted = [p for _f, _v, _s, p in seen if p == "P_WRONG"]
    check(not polluted,
          "遮挡时身份必须冻结，不能被错误标签污染（被污染 %d 次）" % len(polluted))

    print("\n== 换标签要有分数余量（防止一次误判永久污染）==")
    tr2 = PartTracker({"backend": "kcf", "detect_every_n": 1, "min_area_px": 100,
                       "max_area_px": 100000})
    tr2.update(frame, [mk_blob(320, 240, A0)], detect=True,
               classify=lambda f: ("P_A", 0.80))
    t2 = tr2.tracks[0]
    check(t2.part_id == "P_A", "第一次识别应被接受")
    tr2.update(frame, [mk_blob(320, 240, A0)], detect=True,
               classify=lambda f: ("P_B", 0.83))       # 只高一点点
    check(tr2.tracks[0].part_id == "P_A",
          "新标签只高 0.03（< id_margin）时应**保留旧标签**（否则会来回跳）")
    tr2.update(frame, [mk_blob(320, 240, A0)], detect=True,
               classify=lambda f: ("P_C", 0.99))       # 明显更高
    check(tr2.tracks[0].part_id == "P_C",
          "新标签明显更高（0.99 >> 0.80+0.12）时应该换")

    print("\n== 形状失真时也应冻结身份（被遮成细条 -> elongation 会失真）==")
    tr3 = PartTracker({"backend": "kcf", "detect_every_n": 1, "min_area_px": 100,
                       "max_area_px": 100000})
    tr3.update(frame, [mk_blob(320, 240, A0, aspect=1.0)], detect=True,
               classify=lambda f: ("P_ROUND", 0.80))
    # 喂一个细长条 + 明显更高的错误标签
    tr3.update(frame, [mk_blob(320, 240, A0 * 0.85, aspect=5.0)], detect=True,
               classify=lambda f: ("P_ROD", 0.99))
    check(tr3.tracks[0].part_id == "P_ROUND",
          "形状已失真（aspect=5，像细杆）时不该换成 P_ROD —— "
          "实测就发生过「轮盘被遮一半 -> 认出 P_column」")

    print("\n== 完全被遮住算「遮挡」，不算「丢失」（用户明确要求）==")
    print("   用户原话：被全部遮挡应算遮挡 ——「我没有动底座应该算作遮挡」；")
    print("   只有长时间什么都看不见，才算可能被拿走。")
    print("   ★ 实测踩过：原来 12 帧（0.4 秒）就判 lost，手搭上去半秒就报「丢失」。")
    tr4 = PartTracker({"backend": "kcf", "detect_every_n": 1, "min_area_px": 100,
                       "max_area_px": 100000, "max_occluded_frames": 20,
                       "max_lost_frames": 40})
    tr4.update(frame, [mk_blob(320, 240, 4000.0)], detect=True,
               classify=lambda f: ("P_KEEP", 0.9))
    check(len(tr4.tracks) == 1, "应先建立 1 条轨迹")
    # 喂 10 帧"什么都看不到"（blobs 为空 = 完全被手挡住）
    seen_states = []
    for i in range(10):
        tr4.update(frame, [], detect=True)
        seen_states.append(tr4.tracks[0].state if tr4.tracks else "PRUNED")
    print("   连续 %d 帧完全不可见 -> 状态: %s" % (len(seen_states),
                                                 sorted(set(seen_states))))
    check(all(s == "occluded" for s in seen_states),
          "完全被遮住（还不算久）必须报 occluded，不能报 lost（实测 %s）"
          % sorted(set(seen_states)))
    check(len(tr4.tracks) == 1, "occluded 状态绝不能把轨迹删掉（删了就找不回来了）")
    check(tr4.tracks[0].part_id == "P_KEEP", "被完全遮住时身份必须保留")

    # 继续遮到超过 max_occluded(20) -> 升级为 lost，但**仍然不删**
    for i in range(15):
        tr4.update(frame, [], detect=True)
    st = tr4.tracks[0].state if tr4.tracks else "PRUNED"
    print("   遮到 %d 帧 -> 状态 %s（应 lost，但仍保留）" % (10 + 15, st))
    check(st == "lost", "超过 max_occluded_frames 才升级为 lost（实测 %s）" % st)
    check(len(tr4.tracks) == 1, "lost 状态**仍不删** —— 手移开露出来要能恢复")

    # 手移开：零件重新可见 -> 应能恢复
    tr4.update(frame, [mk_blob(320, 240, 4000.0)], detect=True,
               classify=lambda f: ("P_KEEP", 0.9))
    check(tr4.tracks and tr4.tracks[0].state == "detected",
          "手移开后必须恢复为 detected（实测 %s）"
          % (tr4.tracks[0].state if tr4.tracks else "PRUNED"))
    check(tr4.tracks[0].part_id == "P_KEEP", "恢复后身份应仍在")

    # 再遮到超过 max_lost(40) -> gone，此时才允许清掉
    for i in range(45):
        tr4.update(frame, [], detect=True)
    gone_ok = (not tr4.tracks) or tr4.tracks[0].state in ("lost", "gone")
    check(gone_ok, "超过 max_lost_frames 之后才允许清理")

    print("\n== IoU 与 ROI 工具 ==")
    check(abs(_iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-6, "同框 IoU=1")
    check(_iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0, "不重叠 IoU=0")
    check(0.1 < _iou((0, 0, 10, 10), (5, 0, 10, 10)) < 0.4, "半重叠 IoU 在中间")
    box = tr.tracks[0].bbox if tr.tracks else (10, 10, 20, 20)
    tr3.tracks[0].bbox = box
    got = tr3.roi_of(tr3.tracks[0], pad=5)
    check(got is not None and got[0] == box[0] - 5,
          "roi_of 应按 pad 外扩（遮挡时拿它当 work_roi 重新分割用）")

    print("\n零件追踪测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
