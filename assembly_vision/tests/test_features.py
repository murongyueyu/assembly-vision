# -*- coding: utf-8 -*-
"""形状特征测试：圆度、内孔数、细长比、弯曲、两端宽度比。"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.core.features import (contour_features,     # noqa: E402
                                       end_width_ratio, straightness)


def _largest_contour(mask):
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(found[-2], key=cv2.contourArea)


def main() -> int:
    print("== 形状特征 ==")
    canvas = np.zeros((300, 300), np.uint8)

    # 正圆（注意：栅格化圆的周长偏大，圆度会低于 1，实测约 0.89 属正常）
    m = canvas.copy()
    cv2.circle(m, (150, 150), 80, 255, -1)
    f = contour_features(_largest_contour(m))
    print("   正圆:", {k: round(v, 3) for k, v in f.items()
                    if k in ("circularity", "solidity", "aspect", "holes")})
    check(f["circularity"] > 0.85, "正圆圆度 > 0.85（栅格化损失后可接受）")
    check(f["holes"] == 0, "实心圆内孔数=0")

    # 正方形：圆度应明显低于圆（用于验证判据有区分度）
    m = canvas.copy()
    cv2.rectangle(m, (70, 70), (230, 230), 255, -1)
    f_sq = contour_features(_largest_contour(m))
    print("   正方形: circularity=%.3f solidity=%.3f" % (f_sq["circularity"], f_sq["solidity"]))
    check(f_sq["circularity"] < f["circularity"] - 0.05,
          "正方形的圆度明显低于正圆（判据有区分度）")

    # 带一个孔的环
    # ★ 注意：孔洞必须靠掩膜来数 —— 用 RETR_EXTERNAL 取到的外轮廓本身不含内孔，
    #   所以 contour_features 要同时传掩膜。生产代码里 Segmenter 就是这么传的。
    m = canvas.copy()
    cv2.circle(m, (150, 150), 80, 255, -1)
    cv2.circle(m, (150, 150), 30, 0, -1)
    f = contour_features(_largest_contour(m), m)
    print("   圆环（传掩膜）: holes =", f["holes"])
    check(f["holes"] == 1, "圆环内孔数=1")

    # 带缺口（模拟破损）：咬掉右上角一大块
    m = canvas.copy()
    cv2.circle(m, (150, 150), 80, 255, -1)
    cv2.rectangle(m, (150, 20), (250, 120), 0, -1)
    f = contour_features(_largest_contour(m))
    print("   缺口圆: circularity=%.3f solidity=%.3f" % (f["circularity"], f["solidity"]))
    check(f["solidity"] < 0.90, "有缺口时凸包实度明显下降")

    # 细长条（立柱）：直 vs 弯
    m = canvas.copy()
    cv2.line(m, (40, 150), (260, 150), 255, 14)
    fc = contour_features(_largest_contour(m))
    print("   直条: elongation=%.2f straightness=%.4f" % (fc["elongation"], fc["straightness"]))
    check(fc["elongation"] > 5.0, "直条细长比 > 5")
    check(fc["straightness"] < 0.02, "直条直线度很小")

    m = canvas.copy()
    pts = np.array([[40, 160], [150, 148], [260, 120]], np.int32)
    cv2.polylines(m, [pts], False, 255, 14)
    fb = contour_features(_largest_contour(m))
    print("   弯条: elongation=%.2f straightness=%.4f" % (fb["elongation"], fb["straightness"]))
    check(fb["straightness"] > fc["straightness"] * 2.0,
          "弯条的直线度明显大于直条（可判弯曲）")

    # 两端宽度比：一端粗一端细
    m = canvas.copy()
    cv2.rectangle(m, (40, 140), (200, 160), 255, -1)     # 杆
    cv2.rectangle(m, (40, 120), (70, 180), 255, -1)      # 一端球头（更宽）
    ratio = end_width_ratio(_largest_contour(m))
    print("   一端粗: end_width_ratio = %.2f" % ratio)
    check(ratio > 1.5, "一端球头一端细杆的宽度比 > 1.5")

    m = canvas.copy()
    cv2.rectangle(m, (40, 140), (240, 160), 255, -1)     # 对称长条
    ratio_s = end_width_ratio(_largest_contour(m))
    print("   对称条: end_width_ratio = %.2f" % ratio_s)
    check(ratio_s < 1.3, "对称零件宽度比接近 1")

    print("\n形状特征测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
