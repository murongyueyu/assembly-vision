# -*- coding: utf-8 -*-
"""分割 + 槽位覆盖率（要求①的核心判据）离线测试。"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check, make_env                     # noqa: E402
from make_synthetic import SLOTS, render_scene           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.core.segment import glove_hits, overexposed_ratio  # noqa: E402


def main() -> int:
    # 与旧定义逐像素对照，覆盖混合通道、阈值边界、步长及非标准输入。
    rng = np.random.default_rng(20261017)
    image = rng.integers(240, 256, (37, 43, 3), dtype=np.uint8)
    image[0, :6] = [[249,255,255], [250,250,250], [255,255,255],
                    [255,249,255], [255,255,249], [0,0,0]]
    for array in (image, image[::2, ::2], image.astype(np.float32),
                  rng.integers(0, 256, (9, 11, 4), dtype=np.uint8)):
        for threshold in (-1, 0, 249, 249.5, 250, 255, 256, float("nan"), float("inf")):
            expected = float((array.min(axis=2) >= threshold).mean())
            check(overexposed_ratio(array, threshold) == expected,
                  "过曝比例与原定义一致：dtype=%s shape=%s 阈值=%s" % (
                      array.dtype, array.shape, threshold))
    wp, tray, seg, q = make_env()
    print("== 场景 1：全部就位 ==")
    img = render_scene()
    res = seg.segment(img)
    slots = q.check_slots(res, bgr=img)
    covs = {s.slot_id: s.coverage for s in slots}
    print("   覆盖率:", {k: round(v, 3) for k, v in covs.items()})
    # ★ 覆盖率阈值是**按槽缩放**的（见 checks.check_slots 的 cov_scale）：
    #   立柱是 85x15 的细杆、槽是 91x28 的宽胶囊，覆盖率天生只有约 0.40，
    #   用统一的 0.45 卡它必然误判。所以这里比的是**每槽自己的阈值**，
    #   不是写死的 0.45（早期写死过，改几何后立刻挂）。
    below = {s.slot_id: (round(s.coverage, 3),
                         round(s.features.get("cov_thr_present", 0.45), 3))
             for s in slots if s.coverage < s.features.get("cov_thr_present", 0.45)}
    check(not below, "四个槽的覆盖率都超过各自阈值 %s" % (below or ""))
    check(all(s.verdict == "OK" for s in slots), "四个槽判定都是 OK")

    print("== 场景 2：全部拿掉 ==")
    img0 = render_scene(missing=set(SLOTS.keys()))
    res0 = seg.segment(img0)
    slots0 = q.check_slots(res0, bgr=img0)
    covs0 = {s.slot_id: s.coverage for s in slots0}
    print("   覆盖率:", {k: round(v, 3) for k, v in covs0.items()})
    above = {s.slot_id: (round(s.coverage, 3),
                         round(s.features.get("cov_thr_present", 0.15), 3))
             for s in slots0 if s.coverage > s.features.get("cov_thr_present", 0.15)}
    check(not above, "四个槽的覆盖率都低于各自阈值 %s" % (above or ""))
    check(all(s.verdict == "NG" for s in slots0), "四个槽判定都是 NG")

    print("== 场景 3：只缺顶盖 ==")
    img1 = render_scene(missing={"slot_cap"})
    res1 = seg.segment(img1)
    m = {s.slot_id: s.verdict for s in q.check_slots(res1)}
    print("   判定:", m)
    check(m["slot_cap"] == "NG", "顶盖槽判 NG")
    check(m["slot_wheel"] == "OK" and m["slot_base"] == "OK"
          and m["slot_column"] == "OK", "其余三槽仍判 OK")

    print("== 场景 4：蓝手套压在轮盘上 ==")
    # ★ 轮盘槽位置从合成用的 SLOTS 取，别写死 —— 实测改为左上 (42,43) 后
    #   写死的 (120,40) 就压到立柱槽上了。
    img2 = render_scene(glove_at=SLOTS["slot_wheel"]["center"])
    res2 = seg.segment(img2)
    check(res2.glove_bbox is not None, "检测到手套")
    hits = glove_hits(res2.glove_mask, tray.slot_polygons_px(1.0))
    print("   hand_on:", hits)
    check(hits.get("slot_wheel", False), "hand_on(slot_wheel) 为真")

    print("\n分割与覆盖率测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
