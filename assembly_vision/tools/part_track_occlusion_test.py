# -*- coding: utf-8 -*-
"""验证"只描可见部分的轮廓"+ 遮挡下身份冻结 —— 出一张可看的对照图。

用户要求：「如果被手遮挡只需要描出没被遮挡部分的轮廓就行。」

本工具要证明三件事（每个都出图）：
  ① 画的轮廓 = **可见黄铜区域的真实轮廓**（`findContours` 的结果），
     不补全、不外推被挡住的形状
  ② 遮挡不重（可见 >= full_ratio）时 -> **仍然重新识别身份**
  ③ 遮挡重（可见 <= occluded_ratio）时 -> 状态变 `occluded`，
     **身份沿用上次**（残缺件拿去认必然认错 —— 这是刻意的取舍）
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = r"D:\university resource\大三上\视觉\assembly_vision"
sys.path.insert(0, ROOT)

import cv2                                                        # noqa: E402
from assemvision.config import load_config                        # noqa: E402
from assemvision.core.checks import Qualifier                     # noqa: E402
from assemvision.core.part_tracker import PartTracker             # noqa: E402
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe, imwrite_safe           # noqa: E402

GLOVE_BGR = (200, 140, 90)          # 蓝手套近似色（BGR）

wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
cal = load_config(os.path.join(ROOT, "config", "calibration.yaml"))
tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
               ref_scale=float(cal.get("tray_map.scale", 1.0)))
q = Qualifier(wp, tray, cal)
seg = Segmenter(wp)
mmpp = tray.mm_per_px(1.0)
clf = q._classifier()

base = imread_safe(os.path.join(ROOT, "data", "refs", "assembly_choice.png"))
res = seg.segment(base, scale=1.0)
targets = [b for b in res.brass_blobs if 2000 < b.area_px < 8000]
targets.sort(key=lambda b: -b.area_px)
print("  用于试验的零件: %s"
      % ["%dx%d@(%.0f,%.0f)" % (cv2.boundingRect(b.contour)[2],
                                cv2.boundingRect(b.contour)[3],
                                b.centroid[0], b.centroid[1]) for b in targets])


def classify(feats):
    f = dict(feats)
    f["area_mm2"] = float(f.get("area_px", 0)) * mmpp * mmpp
    rw, rh = float(f.get("rect_w", 0)), float(f.get("rect_h", 0))
    f["length_mm"] = max(rw, rh) * mmpp
    f["short_side_mm"] = min(rw, rh) * mmpp
    pid, sc, _ = clf.classify(f)
    return pid, sc


outs = []
for frac in (0.0, 0.3, 0.5, 0.7, 0.85):
    tr = PartTracker({"backend": "kcf", "min_area_px": 400, "max_area_px": 8000})
    frame = base.copy()
    # 只保留一块零件，其余涂掉，画面更干净好判读
    keep = targets[0]
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(mask, [keep.contour], -1, 255, -1)
    keeponly = cv2.bitwise_and(frame, frame, mask=mask)
    bg = np.full_like(frame, 90)
    frame = np.where(mask[..., None] > 0, keeponly, bg)

    # ★★ 时序很重要：**先喂满面积帧建立参考与身份**，再喂遮挡帧。
    #   第一版对每个遮挡度都新建追踪器、直接喂遮挡帧 —— 那是"新目标"，
    #   参考面积 = 当前面积，于是 visible_ratio 恒为 1.0，测出来的结论是假的。
    #   真实场景恰恰是"先看见完整零件，然后手才遮上来"。
    r0 = seg.segment(frame, scale=1.0)
    b0 = [b for b in r0.brass_blobs if 100 <= b.area_px <= 8000]
    for _ in range(3):                       # 3 帧满面积，稳住参考
        tr.update(frame, b0, detect=True, classify=classify)

    x, y, w, h = cv2.boundingRect(keep.contour)
    if frac > 0:
        # ★ 用**零件轮廓自身的宽度**（不是 bbox 宽）来算遮挡宽度。
        #   第一版用 `min(w*frac, w-12)` 夹取 —— 85% 那档被夹成 12px，
        #   结果**根本没遮住零件**，于是又回到"满可见/detected"，看着像 bug 其实是测试构造错。
        xs = keep.contour.reshape(-1, 2)[:, 0]
        part_w = float(xs.max() - xs.min())
        ow = int(round(part_w * frac))
        cv2.rectangle(frame, (int(xs.max()) - ow, y - 8),
                      (int(xs.max()) + 6, y + h + 8), GLOVE_BGR, -1)

    r = seg.segment(frame, scale=1.0)
    blobs = [b for b in r.brass_blobs if 100 <= b.area_px <= 8000]
    # ★ 只保留**离目标最近**的那块 —— 第一版把画面里所有块都喂进去，
    #   遮挡到 85% 时目标块面积最小，被别的块顶替（质心距 87px），
    #   于是 visible 反而回到 100%，看着像 bug 其实是选块策略错。
    if blobs:
        tgt = np.array(keep.centroid)
        blobs = [min(blobs, key=lambda z: float(np.hypot(z.centroid[0] - tgt[0],
                                                          z.centroid[1] - tgt[1])))]
    # 遮挡帧只跑**追踪 + 用追踪框重分割**（和 demo 里的路径一致），不再全图重检测，
    # 否则又会走"新目标"分支。这里用 detect=True 让它吸收一次检测（模拟 demo 的检测帧）。
    tracks = tr.update(frame, blobs, detect=True, classify=classify)

    canvas = frame.copy()
    tr.draw(canvas, show_bbox=False)          # ★ 只画形状轮廓
    # 标注可读信息
    for t in tracks:
        info = ("%s 可见 %.0f%% 面积 %.0fpx² 状态 %s"
                % (t.part_id or "(未识别)", 100 * t.visible_ratio, t.area_px, t.state))
        cv2.putText(canvas, info, (10, 22 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (250, 250, 250), 1, cv2.LINE_AA)
    cv2.putText(canvas, "遮挡 %.0f%%" % (frac * 100), (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 2, cv2.LINE_AA)

    out = os.path.join(ROOT, "data", "refs", "track_occ_%02d.png" % int(frac * 100))
    imwrite_safe(out, canvas)
    outs.append(out)
    n_cont = sum(1 for t in tracks if t.contour is not None)
    print("  遮挡 %2.0f%%  ->  轨迹 %d 条（带轮廓 %d）  %s"
          % (frac * 100, len(tracks), n_cont,
             " ".join("%s/%.0f%%/%s" % (t.part_id or "-",
                                        100 * t.visible_ratio, t.state)
                      for t in tracks)))

# 拼成一张对照图，便于一眼看全
imgs = [imread_safe(p) for p in outs]
h = min(i.shape[0] for i in imgs)
imgs = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in imgs]
strip = np.concatenate(imgs, axis=1)
out_all = os.path.join(ROOT, "data", "refs", "track_occlusion_compare.png")
imwrite_safe(out_all, strip)
print()
print("  已保存对照图 -> %s" % out_all)
print("  ★ 判读：轮廓应**只覆盖未被手套挡住的黄铜部分**（可见部分），")
print("     遮挡越重轮廓越小、但**形状仍然完整**（不补全、不外推）。")
print("     遮挡 <= 45% 时 state 应为 detected/tracked 且**带身份**；")
print("     遮挡 >= 55% 时 state 应为 occluded 且**身份沿用**（不拿残缺件去认）。")
