# -*- coding: utf-8 -*-
"""工位全局 ROI 测试 —— 只在「料盘 + 装配位」范围内检测黄铜。

★ 为什么需要（本项目的真实痛点）：
  木桌的 HSV 与黄铜重叠（实测桌面 S 中值 67~87，黄铜 S 区间 52~109）。
  实测踩过一次：**画面顶部的木桌整块被判成黄铜**，产生 13269mm² 的假连通域
  （真零件最大才 3281mm²），把零件识别和面积测量全带偏。

★ 为什么不用「按固定比例裁掉上下 15%」：
  那是可行的快速办法，但**余量太薄** —— 实测料盘底沿到 376px，
  而「中间 70%」只到 408px，料盘稍微往下挪一点就被切。
  所以改成**几何锚定**：料盘外扩 margin + 装配位外扩 margin，
  既能挡住桌面，又留足缓冲，而且跟着料盘位姿一起动。

★★ 写这个测试踩的两个坑（都是测试本身的问题，不是 ROI 的问题）：
  1. 合成图画布默认只留 12mm 边距 —— 「画布内、ROI 外」的区域几乎不存在，
     于是"放在 ROI 外"的位置其实落在料盘 ROI 里，测不出效果。
     -> 必须**用大边距画布**（--margin 80mm）。
  2. 只判断"在 ROI 外"还不够 —— 还要确认该处在**不带 ROI 时真的会被切成黄铜**。
     否则放了个空位置，前后都是 0，看着像"ROI 没生效"。
     -> 候选点必须**用不带 ROI 的结果筛一遍**。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check, make_env                              # noqa: E402
from make_synthetic import ASM_CENTER, ASM_RADIUS, render_scene  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.core.checks import Qualifier                    # noqa: E402
from assemvision.core.roi import TrayMap                         # noqa: E402
from assemvision.core.segment import Segmenter                   # noqa: E402

PPM = 4.0
MARGIN = 80.0            # ★ 画布边距要够大，才有"画布内、ROI 外"的区域可测
ROI_MARGIN = 25.0
BRASS_BGR = (60, 160, 200)   # 与合成图同一套黄铜色（HSV≈22,178,200）


def main() -> int:
    wp = make_env(px_per_mm=PPM)[0]
    import make_synthetic as ms
    _, _, org = ms.canvas_size(PPM, MARGIN, True)
    tray = TrayMap(wp, homography=None, px_per_mm=PPM,
                   origin_px=(float(org[0]), float(org[1])))
    seg = Segmenter(wp)
    q = Qualifier(wp, tray, None)
    img = render_scene(px_per_mm=PPM, margin_mm=MARGIN)
    H, W = img.shape[:2]
    roi = q.work_roi_mask(img.shape, 1.0, margin_mm=ROI_MARGIN)
    check(roi is not None, "ROI 生成成功")
    check(roi.shape[:2] == (H, W), "ROI 与图同尺寸")

    def inside(x_mm, y_mm) -> bool:
        px = np.round(tray.mm_to_px(np.array([[x_mm, y_mm]], float), 1.0)[0]).astype(int)
        if not (0 <= px[0] < W and 0 <= px[1] < H):
            return False
        return bool(roi[px[1], px[0]] > 0)

    print("== 场景 1：ROI 必须覆盖料盘 + 装配位 ==")
    for x, y in ((0, 0), (160, 0), (160, 160), (0, 160), (80, 80)):
        check(inside(x, y), "料盘 (%.0f,%.0f) 在 ROI 内" % (x, y))
    check(inside(ASM_CENTER[0], ASM_CENTER[1]), "装配位中心在 ROI 内")
    r = ASM_RADIUS - 2.0
    for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        check(inside(ASM_CENTER[0] + r * np.cos(t), ASM_CENTER[1] + r * np.sin(t)),
              "装配位边缘在 ROI 内")

    print("== 场景 2：ROI 要比料盘本身**外扩**（留缓冲，料盘挪动不会切到）==")
    rows_, cols_ = np.where(roi > 0)          # ★ np.where 给的是 (row, col)=(y, x)
    bx0, bx1 = int(cols_.min()), int(cols_.max())
    by0, by1 = int(rows_.min()), int(rows_.max())
    qpx = tray.mm_to_px(np.array([[0.0, 0.0], [160.0, 0.0],
                                  [160.0, 160.0], [0.0, 160.0]]), 1.0)
    tx0, tx1 = int(np.floor(qpx[:, 0].min())), int(np.ceil(qpx[:, 0].max()))
    ty0, ty1 = int(np.floor(qpx[:, 1].min())), int(np.ceil(qpx[:, 1].max()))
    print("   料盘像素框 x[%d,%d] y[%d,%d]   ROI 外接框 x[%d,%d] y[%d,%d]"
          % (tx0, tx1, ty0, ty1, bx0, bx1, by0, by1))
    check(bx0 <= tx0 and bx1 >= tx1 and by0 <= ty0 and by1 >= ty1,
          "ROI 外接框严格包含料盘（左%d 右%d 上%d 下%d px 的缓冲）"
          % (tx0 - bx0, bx1 - tx1, ty0 - by0, by1 - ty1))
    check((bx1 - bx0) > (tx1 - tx0), "ROI 横向确实比料盘宽（不是刚好贴着）")

    print("== 场景 3：远离工位的区域必须在 ROI 外（这才是它存在的意义）==")
    n_checked = n_out = 0
    for x, y in ((-60.0, -60.0), (220.0, -60.0), (220.0, 220.0), (-60.0, 220.0),
                 (80.0, -60.0), (80.0, 225.0)):
        px = np.round(tray.mm_to_px(np.array([[x, y]], float), 1.0)[0]).astype(int)
        if not (0 <= px[0] < W and 0 <= px[1] < H):
            continue
        n_checked += 1
        if roi[px[1], px[0]] == 0:
            n_out += 1
    check(n_checked > 0 and n_out == n_checked,
          "料盘外 60mm 的四周都在 ROI 外（%d/%d）" % (n_out, n_checked))

    print("== 场景 4：ROI 真的能挡掉画面别处的假黄铜 ==")
    # ★ mm_per_px 现在必须传（2026-10 新增尺寸闸门）：尺度唯一来源 = TrayMap。
    #   不传的话闸门**静默关闭** —— 场景 4 就会测成"ROI 与闸门都没生效"。
    mmpp = tray.mm_per_px(1.0)
    # ★★ 候选点必须同时满足两条，否则测不出效果：
    #     ① 在 ROI 外
    #     ② **不带 ROI 时那里真的会被切成黄铜**（否则前后都是 0，看着像没生效）
    base = seg.segment(img, scale=1.0, mm_per_px=mmpp)
    out_zone = (roi == 0)
    k = 30
    cands = []
    for cy in range(k, H - k, 6):
        for cx in range(k, W - k, 6):
            if out_zone[cy, cx] and base.brass_mask[cy, cx] == 0:
                cands.append((cx, cy))
    # 优先挑远离工位的（画面四角附近）
    cands.sort(key=lambda c: -((c[0] - W / 2) ** 2 + (c[1] - H / 2) ** 2))
    pat = []
    for c in cands:
        if all(abs(c[0] - p[0]) > 3 * k or abs(c[1] - p[1]) > 3 * k for p in pat):
            pat.append(c)
        if len(pat) >= 6:
            break
    check(len(pat) >= 2, "找到 %d 个『ROI 外且原本非黄铜』的位置" % len(pat))

    fake = img.copy()
    for (cx, cy) in pat:
        fake[cy - k // 2:cy + k // 2, cx - k // 2:cx + k // 2] = BRASS_BGR
    res_no = seg.segment(fake, scale=1.0, mm_per_px=mmpp)
    res_roi = seg.segment(fake, scale=1.0, work_roi=roi, mm_per_px=mmpp)
    # 先确认假黄铜"不带 ROI 时确实被切出来了"，否则这个测试是空转
    n_seen = sum(1 for (cx, cy) in pat if res_no.brass_mask[cy, cx] > 0)
    check(n_seen == len(pat),
          "假黄铜在不带 ROI 时全部被切成前景（%d/%d）——否则测试无意义"
          % (n_seen, len(pat)))
    n_leak = sum(1 for (cx, cy) in pat if res_roi.brass_mask[cy, cx] > 0)
    s_no = float((res_no.brass_mask > 0).sum())
    s_roi = float((res_roi.brass_mask > 0).sum())
    print("   %d 块假黄铜 @ %s" % (len(pat), pat))
    print("   前景总像素: 不带 ROI %.0f   带 ROI %.0f   挡掉 %.0f（%.0f%%）"
          % (s_no, s_roi, s_no - s_roi, 100 * (s_no - s_roi) / max(1.0, s_no)))
    check(n_leak == 0, "所有 %d 块假黄铜都被挡住（泄漏 %d 块）" % (len(pat), n_leak))
    check(s_roi < s_no, "带 ROI 的前景总面积减少")

    print("== 场景 5：ROI 不能误伤料盘内的零件 ==")
    # 料盘内的黄铜必须**不受 ROI 影响**（这是"不误伤"的核心保证）
    a_in_no = int(base.brass_mask.sum())
    a_in_roi = int(seg.segment(img, scale=1.0, work_roi=roi, mm_per_px=mmpp).brass_mask.sum())
    print("   原图前景: 不带 ROI %d   带 ROI %d" % (a_in_no, a_in_roi))
    check(abs(a_in_no - a_in_roi) <= 0.02 * max(1, a_in_no),
          "料盘内的零件没被 ROI 削掉（差异 ≤2%）")

    print("== 场景 6：margin=0 时 ROI 仍覆盖料盘本身 ==")
    roi0 = q.work_roi_mask(img.shape, 1.0, margin_mm=0.0)
    px = np.round(tray.mm_to_px(np.array([[80.0, 80.0]], float), 1.0)[0]).astype(int)
    check(roi0 is not None and roi0[px[1], px[0]] > 0, "margin=0 时料盘中心仍在 ROI 内")

    print("\n工位 ROI 测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
