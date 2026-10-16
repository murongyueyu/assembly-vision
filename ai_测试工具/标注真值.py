# -*- coding: utf-8 -*-
"""★★ 真值标注工具 —— 你点四下，我就有了"标准答案"。

═══ 为什么需要它 ═══
实测（2026-10 rev.3r-11）：这张真帧上**槽沿的自动检测做不到**：
  · 霍夫变换找到的"最强环"全挤在料盘**中心**（~95,~95），那是中央十字线 + 层纹，
    **不是槽沿**；
  · 逐点穷举的最优解在最优附近是**平台** ⇒ 同一顶盖一会儿"偏高"、一会儿"偏右下"
    （用户实测反馈过三种不同偏差）。
★ 根因：**打印出来的槽沿信噪比太低**，任何"自动找边"的判据都会被纹理/分隔线打败。
★ 结论：**这一帧上，"槽在哪"该由人给**。人点记下坐标 ⇒
  · 我可以用它**标定/校准**（比自动检测可靠得多）；
  · 也可以把它当**回归基准**，以后每次改算法都拿它量偏差。

═══ 用法 ═══
  1) 打开图（默认用 21_槽位描边_正视图.png，也可以 --image 指定）
  2) **在四个槽里各点一下**（顺序无所谓，会按最近的设计位置自动归属）
     · 左键点 = 落一个点；**右键点 = 撤销**
     · 滚轮/`+`/`-` 缩放，按住中键或方向键平移（大图更准）
  3) 按 **s** 存结果（写到 工位帧/真值_*.json + 打印到控制台）
  4) 按 **r** 清空重来，**q** 退出

★ 标注结果会**同时打印成"可以直接粘给我的文本"**，你复制过来我就能直接用。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "assembly_vision"))

from assemvision.imgio import imread_safe, imwrite_safe      # noqa: E402

OUT_DIR = os.path.join(HERE, "工位帧")

# ★ 设计几何（料盘 mm，左上原点）—— 用于把标注点**自动归属**到最近的槽，
#   并算出"标注 vs 设计"的偏差（这就是我需要的校准量）
SLOTS = [("slot_wheel", 49.4, 49.1, 44.0, "circle"),
         ("slot_column", 124.9, 41.2, 14.0, "capsule"),
         ("slot_cap", 53.4, 130.6, 12.5, "circle"),
         ("slot_base", 119.6, 123.0, 32.5, "circle")]
COL = {"slot_wheel": (0, 0, 255), "slot_column": (0, 255, 255),
       "slot_cap": (255, 0, 255), "slot_base": (0, 255, 0)}
ZH = {"slot_wheel": "轮盘 wheel", "slot_column": "立柱 column",
      "slot_cap": "顶盖 cap", "slot_base": "底座 base"}


def main() -> int:
    ap = argparse.ArgumentParser(description="真值标注：在四个槽里各点一下")
    ap.add_argument("--image", default=os.path.join(
        HERE, "工位帧", "分析叠加图", "21_槽位描边_正视图.png"),
        help="要标注的图（默认用拉正视图）")
    ap.add_argument("--side", type=int, default=640,
        help="★ 这张图对应的料盘边长像素（拉正视图是 640；原图请用 --side 0 表示按原图像素）")
    a = ap.parse_args()

    img0 = imread_safe(a.image)
    if img0 is None:
        print("读不到 %s" % a.image)
        return 1
    H0, W0 = img0.shape[:2]
    print("标注图：%s  %dx%d" % (a.image, W0, H0))
    if a.side and a.side > 0:
        ppm = a.side / 160.0
        print("★ 换算：%.4f px/mm（料盘 %d px = 160mm）" % (ppm, a.side))
    else:
        ppm = 1.0
        print("★ 未给 --side：坐标只记**像素**，不换算 mm")

    # 缩放显示（大图点得准）
    scale = 1.0
    win = "标注真值：四个槽各点一下 | s=存 r=清空 +/-=缩放 q=退出"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1100, W0), min(1000, H0))

    pts = []          # 屏幕坐标（原图坐标）

    def on_mouse(event, x, y, flags, _p):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((float(x), float(y)))
            print("  点 %d: (%.1f, %.1f) px%s"
                  % (len(pts), x, y,
                     "  = (%.1f, %.1f) mm" % (x / ppm, y / ppm) if ppm != 1.0 else ""))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            g = pts.pop()
            print("  撤销 (%.1f, %.1f)" % g)

    cv2.setMouseCallback(win, on_mouse)

    while True:
        view = img0.copy()
        # 设计位置画蓝圈（对照用）
        if a.side and a.side > 0:
            for cid, cx, cy, r, kind in SLOTS:
                cv2.circle(view, (int(cx * ppm), int(cy * ppm)), int(r * ppm),
                           (235, 160, 60), 2, cv2.LINE_AA)
                cv2.putText(view, ZH[cid].split()[0],
                            (int(cx * ppm) - 20, int(cy * ppm)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 160, 60), 1, cv2.LINE_AA)
        # 已点
        for i, (x, y) in enumerate(pts):
            cv2.drawMarker(view, (int(x), int(y)), (255, 255, 255),
                           cv2.MARKER_CROSS, 18, 2)
            cv2.putText(view, str(i + 1), (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        bar = np.zeros((32, view.shape[1], 3), np.uint8)
        cv2.putText(bar, "已点 %d 个（四个槽各一个）| 右键撤销 | s 存 r 清空 | q 退出"
                    % len(pts), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, np.vstack([view, bar]))
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            break
        if k == ord('r'):
            pts.clear()
            print("  已清空")
        if k in (ord('+'), ord('=')):
            scale = min(4.0, scale * 1.25)
            cv2.resizeWindow(win, int(W0 * scale), int(H0 * scale))
        if k in (ord('-'), ord('_')):
            scale = max(0.3, scale / 1.25)
            cv2.resizeWindow(win, int(W0 * scale), int(H0 * scale))
        if k == ord('s'):
            if len(pts) < 2:
                print("★ 至少点 2 个再存")
                continue
            res = {}
            for i, (x, y) in enumerate(pts):
                # ★ 自动归属：离哪个设计位置最近（拉正视图里直接比，最直观）
                if a.side and a.side > 0:
                    mm = np.array([x / ppm, y / ppm])
                    best = min(SLOTS, key=lambda s: np.hypot(mm[0] - s[1],
                                                             mm[1] - s[2]))
                    res[best[0]] = dict(px=[round(x, 2), round(y, 2)],
                                        mm=[round(x / ppm, 2), round(y / ppm, 2)],
                                        design_mm=[best[1], best[2]],
                                        dev_mm=[round(x / ppm - best[1], 2),
                                                round(y / ppm - best[2], 2)])
                else:
                    res["pt%d" % (i + 1)] = dict(px=[round(x, 2), round(y, 2)])
            out = dict(image=os.path.basename(a.image), side_px=a.side,
                       ppm=ppm, points=res, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
            tag = time.strftime("%m%d_%H%M%S")
            js = os.path.join(OUT_DIR, "真值_%s.json" % tag)
            try:
                with open(js, "w", encoding="utf-8") as fh:
                    json.dump(out, fh, ensure_ascii=False, indent=2)
                print("★ 已存：%s" % js)
            except Exception as e:                                 # noqa: BLE001
                print("★ 写 json 失败：%s" % e)
            # ★ 把"可以直接粘给我"的文本打出来
            print("\n" + "=" * 62)
            print("▼ 把下面这几行复制给我 ▼")
            print("=" * 62)
            print("图: %s   side_px=%d   ppm=%.4f" % (os.path.basename(a.image),
                                                      a.side, ppm))
            for cid, v in res.items():
                if "mm" in v:
                    print("%-13s 标注 mm = (%7.2f, %7.2f)   设计 mm = (%7.2f, %7.2f)"
                          "   偏差 = (%+6.2f, %+6.2f)"
                          % (cid, v["mm"][0], v["mm"][1], v["design_mm"][0],
                             v["design_mm"][1], v["dev_mm"][0], v["dev_mm"][1]))
                else:
                    print("%-13s 像素 = %s" % (cid, v["px"]))
            print("=" * 62 + "\n")
            im2 = view.copy()
            p2 = os.path.join(OUT_DIR, "分析叠加图", "30_真值标注_%s.png" % tag)
            os.makedirs(os.path.dirname(p2), exist_ok=True)
            print("标注图：%s（%s）" % (p2, "成功" if imwrite_safe(p2, im2) else "★失败"))
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
