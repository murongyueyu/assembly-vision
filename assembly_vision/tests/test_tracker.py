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
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2                                                    # noqa: E402
import assemvision.core.part_tracker as tracker_module
from assemvision.core.part_tracker import (KalmanXY, PartTracker,   # noqa: E402
                                           _iou, make_tracker, identity_relink_pairs)
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
    print("== 单件身份的远距离关联与逐帧检测模式 ==")
    refs = [("P1_base", 4000.0)]
    check(identity_relink_pairs(refs, [("P1_base", .9, 4000)], .85, (.8, 1.25)) == [(0, 0)],
          "唯一高分同身份、完整面积候选可重关联")
    for candidates in [[("P1_base", .6, 4000)], [("P1_base", .9, 1000)],
                       [("P1_base", .9, 4000), ("P1_base", .9, 4000)],
                       [("P1_base", .9, float("nan"))]]:
        check(not identity_relink_pairs(refs, candidates, .85, (.8, 1.25)),
              "低分、部分可见、多同类或非有限面积不能远距离猜测")
    check(not identity_relink_pairs(refs * 2, [("P1_base", .9, 4000)], .85, (.8, 1.25)),
          "多条同身份历史轨迹有歧义，不强行挑一条")
    cfg = dict(backend="detection", detect_every_n=1, min_area_px=100,
               identity_relink=True, identity_relink_min_score=.85,
               identity_relink_area_ratio=[.8, 1.25])
    detector = PartTracker(cfg)
    classifier = lambda features: ("P1_base", .95)
    detector.update(frame, [mk_blob(100, 120, 4000)], classify=classifier,
                    seen_ids={"P1_base"})
    original_track = detector.tracks[0]
    detector.update(frame, [mk_blob(450, 300, 4000)], classify=classifier,
                    seen_ids={"P1_base"})
    check(len(detector.tracks) == 1 and detector.tracks[0] is original_track
          and original_track.last_centroid == (450, 300) and original_track.lost_frames == 0,
          "跨越空间闸门后恢复同一对象，未新建重复轨迹且缺观测归零")
    covered = np.full(frame.shape[:2], 255, np.uint8)
    detector.update(frame, [], classify=classifier, seen_ids=set(), occluder=covered)
    check(original_track.state == "occluded" and original_track.covered,
          "逐帧检测缺观测但手仍覆盖时保持遮挡状态")
    detector.update(frame, [mk_blob(100, 120, 4000)], classify=classifier,
                    seen_ids={"P1_base"}, occluder=np.zeros(frame.shape[:2], np.uint8))
    check(len(detector.tracks) == 1 and detector.tracks[0] is original_track
          and original_track.state == "detected" and not original_track.covered,
          "遮挡后跨位置放回仍为同一轨迹并恢复检测状态")
    detector.update(frame, [mk_blob(100, 120, 4000), mk_blob(450, 300, 4000)],
                    classify=classifier, seen_ids={"P1_base"})
    check(len(detector.tracks) == 1 and detector.tracks[0] is original_track,
          "单件策略遇到两个同身份候选时保留原轨迹，不新建第二条")
    old_position = original_track.last_centroid
    detector.update(frame, [mk_blob(450, 300, 7000)], classify=classifier,
                    seen_ids={"P1_base"})
    check(len(detector.tracks) == 1 and original_track.last_centroid == old_position
          and original_track.state == "occluded" and original_track.visible_ratio == 0,
          "远处同身份但面积不符时不冒认位置，也不把历史轨迹标成当帧已检测")
    for call in [lambda: PartTracker(dict(cfg, detect_every_n=8)),
                 lambda: detector.update(frame, detect=False)]:
        try:
            call()
        except ValueError:
            check(True, "逐帧检测模式拒绝跳帧配置或调用")
        else:
            raise AssertionError("逐帧检测模式跳帧应报错")
    clock_value = [100.0]
    with patch.object(tracker_module, "time", SimpleNamespace(perf_counter=lambda: 500.0)):
        missing = PartTracker(dict(cfg, lost_grace_seconds=.5), clock=lambda: clock_value[0])
        missing.update(frame, [mk_blob(100, 120, 4000)], classify=classifier,
                       seen_ids={"P1_base"})
        missing_track = missing.tracks[0]
        clock_value[0] = 100.1
        missing.update(frame, [], classify=classifier, seen_ids=set())
        check(missing_track.state == "occluded", "手移开后缺观测先进入搜索宽限期")
        clock_value[0] = 100.8
        missing.update(frame, [], classify=classifier, seen_ids=set())
        check(missing_track.state == "lost", "逐帧检测模式按时间判丢，不因缺相关滤波器永远假定可见")
        missing.update(frame, [mk_blob(450, 300, 4000)], classify=classifier,
                       seen_ids={"P1_base"})
        check(len(missing.tracks) == 1 and missing.tracks[0] is missing_track
              and missing_track.state == "detected" and missing_track.clear_seconds == 0,
              "被判丢后完整零件出现可恢复同一轨迹并清掉搜索计时")

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

    print("\n== 丢失判据：遮挡物移开后 2 秒全图搜不到才算丢（用户明确定义）==")
    print("   用户原话：「不是持续很久才被怀疑，而是最后一次还检测到有零件的位置")
    print("   的时候，到**遮挡物已经完全消失了之后的一小段时间（暂定2秒）**还是")
    print("   没有检测到这个零件，算作这个零件丢失。而且找这个零件是**全图搜索**。』")
    print("   -> 三条必须成立：① 盖着时永不判丢 ② 计时从「移开」开始 ③ 找它是全图")

    def glove_mask_over(contour, shape=(480, 640)):
        """造一个"盖住这个轮廓"的遮挡物掩膜。"""
        m = np.zeros(shape, np.uint8)
        cv2.drawContours(m, [contour], -1, 255, -1)
        return m

    def no_glove(shape=(480, 640)):
        return np.zeros(shape, np.uint8)

    tr4 = PartTracker({"backend": "kcf", "detect_every_n": 1, "min_area_px": 100,
                       "max_area_px": 100000, "lost_grace_seconds": 0.15})
    b = mk_blob(320, 240, 4000.0)
    tr4.update(frame, [b], detect=True, classify=lambda f: ("P_KEEP", 0.9))
    check(len(tr4.tracks) == 1 and tr4.tracks[0].ever_seen,
          "应先建立 1 条轨迹且标记 ever_seen")

    # ① 遮挡物**一直盖着** -> 不管多少帧都必须是 occluded，不能 lost
    glove = glove_mask_over(b.contour)
    states_covered = []
    for _ in range(60):                     # 60 帧远超 lost_grace_frames(20)
        tr4.update(frame, [], detect=True, seen_ids=set(), occluder=glove)
        states_covered.append(tr4.tracks[0].state if tr4.tracks else "PRUNED")
    print("   遮挡物盖着 60 帧 -> 状态: %s" % sorted(set(states_covered)))
    check(all(s == "occluded" for s in states_covered),
          "① 遮挡物**还盖着**时永不判丢（实测 %s）" % sorted(set(states_covered)))
    check(len(tr4.tracks) == 1 and tr4.tracks[0].part_id == "P_KEEP",
          "盖着时轨迹与身份都必须保留")

    # ② 遮挡物**移开**但零件**重新出现**（全图搜到了）-> 一直 occluded/detected，不判丢
    ok_again = []
    for _ in range(30):
        tr4.update(frame, [], detect=True, seen_ids={"P_KEEP"}, occluder=no_glove())
        ok_again.append(tr4.tracks[0].state if tr4.tracks else "PRUNED")
    print("   移开后全图搜到 P_KEEP 30 帧 -> 状态: %s" % sorted(set(ok_again)))
    check("lost" not in ok_again,
          "② 全图能搜到就绝不能判丢（实测 %s）" % sorted(set(ok_again)))

    # ③ 遮挡物**移开**且全图**搜不到** -> 超过 lost_grace_frames 才判 lost
    # ★ 判据改成**按秒**了，所以测试必须让**真实时间过去**
    #   （第一版未加 sleep，瞬间跑 40 帧 -> clear_seconds 恒为 ~0 -> 永远不判丢）。
    seen = []
    t0 = time.perf_counter()
    for i in range(1, 41):
        tr4.update(frame, [], detect=True, seen_ids=set(), occluder=no_glove())
        st = tr4.tracks[0].state if tr4.tracks else "PRUNED"
        seen.append((time.perf_counter() - t0, st))
        time.sleep(0.02)          # 每帧 20ms -> 40 帧共 0.8s，跨过 0.15s 窗口
    early = [st for el, st in seen if el <= 0.15]
    late = [st for el, st in seen if el > 0.15]
    print("   移开后全图搜不到: 0.15s 内 %s -> 之后 %s"
          % (sorted(set(early)), sorted(set(late))))
    check(all(s == "occluded" for s in early),
          "③ 宽限窗口内应是 occluded（正在全图搜），实测 %s" % sorted(set(early)))
    check("lost" in late,
          "③ 宽限窗口过后还没搜到 -> lost（实测 %s）" % sorted(set(late)))

    # ④ 恢复：全图又搜到了 -> 立刻回来
    tr4.update(frame, [b], detect=True, classify=lambda f: ("P_KEEP", 0.9),
               seen_ids={"P_KEEP"}, occluder=no_glove())
    check(tr4.tracks and tr4.tracks[0].state == "detected",
          "④ 搜到就立刻恢复 detected（实测 %s）"
          % (tr4.tracks[0].state if tr4.tracks else "PRUNED"))
    check(tr4.tracks[0].part_id == "P_KEEP", "恢复后身份应仍在")

    print("\n== 防重复轨迹：块不能被「远处的轨迹」抢走（实测出过幽灵轮廓）==")
    print("   用户截图现象：界面上出现形状扭曲、位置在料盘外的橙色「全图搜」影子。")
    print("   根因：匹配用的是**卡尔曼预测位置**（追踪失败后会漂移），")
    print("         配上宽松的 dist_match_px=40 -> 轨迹抢到别处的块 -> 轨迹错配/虚增。")
    print("   修法：匹配改用 last_centroid（最后一次真正检测到的位置）+ 全局最优认领。")
    tr5 = PartTracker({"backend": "kcf", "detect_every_n": 1, "min_area_px": 100,
                       "max_area_px": 100000, "iou_match": 0.15, "dist_match_px": 40.0})
    # 在 A 点建一条轨迹
    tr5.update(frame, [mk_blob(200, 200, 4000.0)], detect=True,
               classify=lambda f: ("P_A", 0.9))
    check(len(tr5.tracks) == 1, "A 点应建立 1 条轨迹")
    # 之后 A 点**一直空着**，同时在**很远**的 B 点放一块 -> 必须是**新建**轨迹
    for _ in range(5):
        tr5.update(frame, [mk_blob(560, 400, 4000.0)], detect=True,
                   classify=lambda f: ("P_B", 0.9))
    ids = sorted(t.part_id for t in tr5.tracks if t.part_id)
    print("   A 点空着 + B 点有块 -> 轨迹身份: %s（应为 P_A 与 P_B 两条）"
          % ids)
    check(len(tr5.tracks) == 2 and ids == ["P_A", "P_B"],
          "远处的新块必须**新建**轨迹，不能被 A 的轨迹抢走（实测 %s）" % ids)
    # A 的轨迹位置不应被 B 的块带跑
    ta = next((t for t in tr5.tracks if t.part_id == "P_A"), None)
    check(ta is not None and abs(ta.last_centroid[0] - 200) < 30,
          "A 的轨迹不该被 B 的块拉走（实测 last_centroid=%s）"
          % (str(None if ta is None else tuple(round(v) for v in ta.last_centroid)),))

    print("\n== 全局最优认领：一个块只能被一条轨迹拿走 ==")
    print("   ★ 构造要点：两条轨迹必须**互相靠近到都能匹配同一个块**，")
    print("     否则测不出「只能被一条认领」（第一版把两条隔了 60px，")
    print("     结果它们自己被合成一条了）。")
    tr6 = PartTracker({"backend": "kcf", "detect_every_n": 1, "min_area_px": 100,
                       "max_area_px": 100000, "iou_match": 0.1, "dist_match_px": 200.0})
    tr6.update(frame, [mk_blob(300, 200, 4000.0)], detect=True,
               classify=lambda f: ("P_1", 0.9))
    # 第二块放在 100px 外：与第一条的 last_bbox 不重叠，但 dist(100) < dist_match(200)
    # -> 会被当成"同一条轨迹移动了"，所以这里要**先让它成为独立轨迹**：
    #    直接改第一条的 last_bbox 让它"看起来没动过"是不行的，
    #    改用较小 dist_match 建两条，再放大 dist_match 测认领。
    tr6.dist_match = 40.0
    tr6.update(frame, [mk_blob(300, 200, 4000.0), mk_blob(400, 200, 4000.0)],
               detect=True, classify=lambda f: ("P_2", 0.9))
    check(len(tr6.tracks) == 2, "应建立 2 条轨迹（实测 %d）" % len(tr6.tracks))
    # 现在放宽 dist_match，让两条轨迹**都**够得着下面这块
    tr6.dist_match = 200.0
    before = [(t.part_id, tuple(round(v) for v in t.last_centroid))
              for t in sorted(tr6.tracks, key=lambda z: z.part_id or "")]
    tr6.update(frame, [mk_blob(340, 200, 4000.0)], detect=True)
    after = [(t.part_id, tuple(round(v) for v in t.last_centroid))
             for t in sorted(tr6.tracks, key=lambda z: z.part_id or "")]
    changed = [i for i in range(len(before)) if before[i] != after[i]]
    print("   一块喂进去 -> 被更新的轨迹数 = %d（应为 1）" % len(changed))
    check(len(changed) == 1,
          "一个块只能被一条轨迹认领（实测被 %d 条同时认领）" % len(changed))
    check(len(tr6.tracks) == 2, "不该因为这个块多出轨迹（实测 %d 条）" % len(tr6.tracks))

    print("\n== “认不出” ≠ “不在”：有像零件但认不出的块时不能判丢 ==")
    print("   实测踩过：底座明明在那儿，只是分类器没给出 id，")
    print("   却被报成「丢失」——这是最误导的一种错。")
    tr7 = PartTracker({"backend": "kcf", "detect_every_n": 1, "min_area_px": 100,
                       "max_area_px": 100000, "lost_grace_seconds": 0.05})
    b7 = mk_blob(320, 240, 4000.0)
    tr7.update(frame, [b7], detect=True, classify=lambda f: ("P_HERE", 0.9))
    t0 = time.perf_counter()
    states = []
    for _ in range(30):
        # ★ 关键：seen_ids 空（没认出 id）但 unidentified=True（有像零件的块）
        tr7.update(frame, [], detect=True, seen_ids=set(),
                   occluder=no_glove(), unidentified=True)
        states.append(tr7.tracks[0].state if tr7.tracks else "PRUNED")
        time.sleep(0.01)
    print("   有未知块、全图没认出 id，跑 %.2fs -> 状态: %s"
          % (time.perf_counter() - t0, sorted(set(states))))
    check(all(sv == "occluded" for sv in states),
          "有‘像零件但认不出’的块时不能判丢（实测 %s）" % sorted(set(states)))

    print("\n== 防幽灵：新轨迹只能建在允许的区域内 ==")
    tr8 = PartTracker({"backend": "kcf", "detect_every_n": 1, "min_area_px": 100,
                       "max_area_px": 100000})
    far = mk_blob(620, 460, 4000.0)          # 在“允许区基”之外
    tr8.update(frame, [far], detect=True, classify=lambda f: ("P_NOISE", 0.9),
               allow_new=lambda b: b.centroid[0] < 400)
    print("   区外的块 + allow_new 拒绝 -> 轨迹数 = %d（应为 0）" % len(tr8.tracks))
    check(len(tr8.tracks) == 0,
          "allow_new 拒绝的块不能建轨迹（实测 %d 条）" % len(tr8.tracks))
    near = mk_blob(200, 200, 4000.0)
    tr8.update(frame, [near], detect=True, classify=lambda f: ("P_OK", 0.9),
               allow_new=lambda b: b.centroid[0] < 400)
    print("   区内的块 -> 轨迹数 = %d（应为 1）" % len(tr8.tracks))
    check(len(tr8.tracks) == 1, "allow_new 放行的块应正常建轨迹")

    print("\n== 拿走又放回：轨迹不能早删，回来要能恢复 ==")
    tr9 = PartTracker({"backend": "kcf", "detect_every_n": 1, "min_area_px": 100,
                       "max_area_px": 100000, "lost_grace_seconds": 0.05,
                       "gone_keep_seconds": 30.0})
    b9 = mk_blob(320, 240, 4000.0)
    tr9.update(frame, [b9], detect=True, classify=lambda f: ("P_BACK", 0.9))
    for _ in range(30):                      # 拿走（全图搜不到）
        tr9.update(frame, [], detect=True, seen_ids=set(), occluder=no_glove())
        time.sleep(0.01)
    st_gone = tr9.tracks[0].state if tr9.tracks else "PRUNED"
    print("   拿走后：轨迹数=%d 状态=%s（应仍在，不是 PRUNED）"
          % (len(tr9.tracks), st_gone))
    check(len(tr9.tracks) == 1, "拿走后轨迹不能立刻删（否则放回来就成新轨迹）")
    # 放回原位
    tr9.update(frame, [mk_blob(320, 240, 4000.0)], detect=True,
               classify=lambda f: ("P_BACK", 0.9))
    print("   放回原位：轨迹数=%d 状态=%s 身份=%s"
          % (len(tr9.tracks), tr9.tracks[0].state if tr9.tracks else "-",
             tr9.tracks[0].part_id if tr9.tracks else "-"))
    check(len(tr9.tracks) == 1 and tr9.tracks[0].state == "detected",
          "放回原位应**恢复同一条轨迹**（而不是新建），否则就是幽灵")

    print("\n== 幻影轨迹：从未真正跟住过的轨迹要快速清掉 ==")
    print("   用户现象：黑色料盘外出现橙色「全图搜 0 帧」标记。")
    print("   「0 帧」=刚建就没跟住过 -> 从噪声里生的轨迹。")
    print("   修：这类轨迹 i) 画面上不画（tracked_once=False）ii) 活过短限就删")
    trA = PartTracker({"backend": "kcf", "detect_every_n": 1, "min_area_px": 100,
                       "max_area_px": 100000, "lost_grace_seconds": 0.05,
                       "phantom_miss_frames": 20})
    # 只喂**一帧**：建出轨迹，但之后再也不给它任何块
    trA.update(frame, [mk_blob(620, 460, 4000.0)], detect=True,
               classify=lambda f: ("P_GHOST", 0.9))
    n0 = len(trA.tracks)
    alive = []
    for i in range(45):
        trA.update(frame, [], detect=True, seen_ids=set(), occluder=no_glove())
        alive.append(len(trA.tracks))
        time.sleep(0.002)
    print("   只喂一帧后：建了 %d 条；之后轨迹数 1->%d（前 5 帧 %s）"
          % (n0, alive[-1], alive[:5]))
    check(n0 == 1, "应先建出 1 条轨迹")
    check(alive[-1] == 0,
          "长时间没拿到检测结果的孤灵轨迹应被清掉"
          "（实测剩 %d 条）" % alive[-1])
    # ★ 但**遮挡期间绝不能删**（否则“手盖住真零件”会丢轨迹）
    trC = PartTracker({"backend": "kcf", "detect_every_n": 1, "min_area_px": 100,
                       "max_area_px": 100000, "lost_grace_seconds": 0.05,
                       "phantom_miss_frames": 20})
    bc = mk_blob(320, 240, 4000.0)
    trC.update(frame, [bc], detect=True, classify=lambda f: ("P_OCC", 0.9))
    glove_c = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(glove_c, [bc.contour], -1, 255, -1)
    for _ in range(45):
        trC.update(frame, [], detect=True, seen_ids=set(), occluder=glove_c)
        time.sleep(0.002)
    print("   遮挡中 45 帧（超过幻影限 20）：轨迹数 = %d（应为 1）" % len(trC.tracks))
    check(len(trC.tracks) == 1,
          "遮挡期间绝不能删轨迹（实测剩 %d 条）" % len(trC.tracks))
    check(trC.tracks[0].state == "occluded",
          "遮挡期间应为 occluded（实测 %s）" % trC.tracks[0].state)
    # 真零件不能被误删：每帧都有检测
    trB = PartTracker({"backend": "kcf", "detect_every_n": 1, "min_area_px": 100,
                       "max_area_px": 100000, "never_tracked_frames": 3})
    for _ in range(10):
        trB.update(frame, [mk_blob(320, 240, 4000.0)], detect=True,
                   classify=lambda f: ("P_REAL", 0.9))
    print("   每帧都有检测的真零件：轨迹数 = %d（应为 1）" % len(trB.tracks))
    check(len(trB.tracks) == 1, "真零件绝不能被当幻影删掉")

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
