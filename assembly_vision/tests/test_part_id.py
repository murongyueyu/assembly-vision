# -*- coding: utf-8 -*-
"""零件特征识别（旋转不变）离线测试。

测的是 assemvision/core/part_id.py 的分类器：用**旋转不变**的特征向量
判定"这是什么零件"，从而让料盘旋转/零件放歪时仍然认得出来。

★ 为什么这些断言是"多尺度"的：特征里最关键的是 area_mm2（物理面积），
  它 = 像素面积 × mm_per_px²。如果哪天有人把归一化改回"像素面积"或用
  bbox 比例，就会在某个尺度下崩掉 —— 多尺度跑一遍能立刻发现。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check, wp_config                          # noqa: E402
from make_synthetic import render_scene                        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.core.checks import Qualifier                  # noqa: E402
from assemvision.core.part_id import (PartClassifier,          # noqa: E402
                                      default_templates,
                                      features_from_blob,
                                      slot_area_px)
from assemvision.core.roi import TrayMap                       # noqa: E402
from assemvision.core.segment import Segmenter                 # noqa: E402


def main() -> int:
    print("== 场景 1：多尺度下都能正确识别四类零件 ==")
    for ppm in (2.5, 4.0, 6.0):
        wp = wp_config()
        tray = TrayMap(wp, homography=None, px_per_mm=ppm,
                       origin_px=(12.0 * ppm, 12.0 * ppm))
        seg = Segmenter(wp)
        q = Qualifier(wp, tray, None)
        res = seg.segment(render_scene(px_per_mm=ppm), scale=1.0)
        slots = q.check_slots(res)
        got = {s.slot_id: s.features.get("id_part", "") for s in slots}
        want = {s["id"]: s["part"] for s in tray.slots}
        print("   %.1f px/mm 识别: %s" % (ppm, got))
        bad = {k: (want[k], v) for k, v in got.items() if v != want[k]}
        check(not bad, "%.1f px/mm 下四类零件全部认对" % ppm)
        check(all(s.verdict == "OK" for s in slots),
              "%.1f px/mm 下四个槽判定都是 OK" % ppm)

    print("== 场景 2：area_mm2 是物理面积，跨尺度稳定 ==")
    areas = {}
    for ppm in (2.5, 4.0):
        wp = wp_config()
        tray = TrayMap(wp, homography=None, px_per_mm=ppm,
                       origin_px=(12.0 * ppm, 12.0 * ppm))
        seg = Segmenter(wp)
        q = Qualifier(wp, tray, None)
        res = seg.segment(render_scene(px_per_mm=ppm), scale=1.0)
        for s in q.check_slots(res):
            areas.setdefault(s.slot_id, []).append(s.features.get("area_mm2", 0.0))
    for sid, vals in areas.items():
        lo, hi = min(vals), max(vals)
        rel = abs(hi - lo) / max(1e-6, (hi + lo) / 2.0)
        print("   %-13s %.0f / %.0f mm²  相对差 %.2f%%" % (sid, vals[0], vals[1], 100 * rel))
        check(rel < 0.05, "%s 的 area_mm2 跨尺度相对差 < 5%%" % sid)

    print("== 场景 3：把零件拿掉后不应再认成『有件』 ==")
    wp = wp_config()
    tray = TrayMap(wp, homography=None, px_per_mm=2.5, origin_px=(30.0, 30.0))
    seg = Segmenter(wp)
    q = Qualifier(wp, tray, None)
    res = seg.segment(render_scene(px_per_mm=2.5, missing={"slot_cap"}), scale=1.0)
    m = {s.slot_id: s.verdict for s in q.check_slots(res)}
    print("   判定:", m)
    check(m["slot_cap"] == "NG", "顶盖被拿掉 -> 判 NG")
    check(m["slot_wheel"] == "OK" and m["slot_base"] == "OK"
          and m["slot_column"] == "OK", "其余三槽仍判 OK")

    print("== 场景 4：轮盘朝向改变后仍能识别（几何旋转，无插值伪影）==")
    # ★ 为什么只转轮盘、不转立柱：
    #   槽位是**固定的**（立柱槽是 91x28 @-44° 的胶囊，不动）。把零件转出槽口范围，
    #   零件会被槽壁截断，覆盖率必然下降 —— 那是几何事实，不是"识别不了朝向"。
    #   实测踩过：把所有零件都转，立柱在 45° 时覆盖率从 0.40 掉到 0.25，
    #   于是被判存疑；但真实工位里槽不会跟着零件转。
    #   轮盘的辐条图案是**唯一朝向敏感**的零件特征，所以只测它。
    #
    # ★ 用几何旋转而不是 warpAffine 转掩膜：位图旋转会在离散网格上产生
    #   孔洞合并/锯齿等伪影，测出来的"失败"是伪影造成的（实测踩过）。
    wp = wp_config()
    tray = TrayMap(wp, homography=None, px_per_mm=4.0, origin_px=(48.0, 48.0))
    seg = Segmenter(wp)
    q = Qualifier(wp, tray, None)
    # 8 根辐条下 45° 是一个周期，所以跨度只需覆盖 0~45°
    angles = (0, 5, 11, 17, 23, 29, 34, 40, 45)
    n_ok = 0
    for ang in angles:
        img = render_scene(px_per_mm=4.0, wheel_rotate_deg=ang, noise=0.0)
        res = seg.segment(img, scale=1.0)
        slots = {s.slot_id: s for s in q.check_slots(res, bgr=img)}
        got = slots["slot_wheel"].features.get("id_part", "")
        am = slots["slot_wheel"].features.get("area_mm2", 0.0)
        good = (got == "P3_wheel")
        n_ok += 1 if good else 0
        print("   轮盘辐条转 %2d° -> %-11s 分=%.2f 面积=%.0fmm²  %s"
              % (ang, got or "(认不出)", slots["slot_wheel"].features.get("id_score", 0),
                 am, "✓" if good else "★"))
    check(n_ok == len(angles), "轮盘辐条转 %d 个角度后仍识别为轮盘" % len(angles))

    print("\n零件特征识别测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
