# -*- coding: utf-8 -*-
"""辐条计数离线测试 —— 本项目唯一的重点指标。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check, make_env                      # noqa: E402
from make_synthetic import render_scene                  # noqa: E402


def main() -> int:
    wp, tray, seg, q = make_env()
    print("== 辐条计数（多半径投票 + 自适应阈值）==")
    cases = [(0, 8), (1, 7), (2, 6), (4, 4)]
    for missing_n, expect in cases:
        img = render_scene(missing_spokes=missing_n)
        res = seg.segment(img)
        spk = q.check_spokes(res)
        print("   少画 %d 根 -> 数出 %d 根（期望 %d），峰角度 %s"
              % (missing_n, spk.count, expect, spk.peaks_deg[:9]))
        check(spk.count == expect, "少画 %d 根时数出 %d 根" % (missing_n, expect))
        if expect < 8:
            check(len(spk.missing_deg) > 0 or spk.count < 8,
                  "缺辐条时能给出缺失信息")

    img = render_scene()
    res = seg.segment(img)
    spk = q.check_spokes(res)
    check(spk.ok, "完整轮盘判为 8 根（ok=True）")

    # ── ★ 半径带选择策略（真机假 NG 的回归）─────────────────────────
    # 见 polar.count_spokes 的 docstring：三个子带各自数峰 -> 众数并列取小
    # 会把装好的轮盘数成 5 根。这里钉住「同一份输入，uniform 不能比 vote 差」。
    #
    # ★ 用 `q.r_spoke`（12~24mm）**不是** `q.r_spoke_out`（30~35mm）：
    #   后者是**装配体上球头**的带，而这里是**合成图料盘里的轮盘**（外半径仅 22.4mm），
    #   用 30~35mm 会整圈落在轮盘外面 -> 两边都数 0，测试变成假通过。
    #   （本测试自己踩过一次，写下来免得再犯。）
    print("\n== 半径带选择策略（uniform 是默认，别改回 vote）==")
    wp2, tray2, seg2, q2 = make_env()
    cfg_sel = (wp2.get("check", {}) or {}).get("spokes", {}) or {}
    check(str(cfg_sel.get("select", "uniform")).lower() == "uniform",
          "workpiece.yaml 的 check.spokes.select 应为 uniform（默认）")
    for missing_n, expect in [(0, 8), (1, 7), (2, 6)]:
        img = render_scene(missing_spokes=missing_n)
        res = seg2.segment(img)
        blob = None
        for b in res.brass_blobs:
            if blob is None or b.area_px > blob.area_px:
                blob = b
        check(blob is not None, "少画 %d 根时能找到轮盘连通域" % missing_n)
        if blob is None:
            continue
        mmpp = tray2.mm_per_px(1.0)
        r0, r1 = q2.r_spoke[0] / mmpp, q2.r_spoke[1] / mmpp
        from assemvision.core.polar import count_spokes          # noqa: E402
        kw = dict(required=8, n_radii_bands=3, k_mad=3.0, min_width_deg=6.0,
                  select="uniform")
        uni = count_spokes(res.brass_mask, blob.centroid, r0, r1, **kw)
        vot = count_spokes(res.brass_mask, blob.centroid, r0, r1,
                           **dict(kw, select="vote"))
        print("   少画 %d 根 -> uniform %d 根 / vote %d 根（期望 %d）"
              % (missing_n, uni.count, vot.count, expect))
        check(uni.count >= vot.count,
              "少画 %d 根时 uniform(%d) 不应比 vote(%d) 数得少"
              % (missing_n, uni.count, vot.count))
        check(uni.count == expect,
              "少画 %d 根时 uniform 应数出 %d 根（得到 %d）"
              % (missing_n, expect, uni.count))

    # spoke_jitter_deg 本身的正确性
    from assemvision.core.polar import spoke_jitter_deg        # noqa: E402
    ideal8 = [(i * 45.0, 10.0, 1.0) for i in range(8)]
    check(spoke_jitter_deg(ideal8, 8) < 1e-6, "理想均布 8 峰抖动为 0")
    skew8 = [(i * 45.0 + (12.0 if i == 0 else 0.0), 10.0, 1.0) for i in range(8)]
    check(spoke_jitter_deg(skew8, 8) > 5.0, "歪掉一个峰后抖动明显变大")
    check(not (spoke_jitter_deg(ideal8, 6) < 1e9),
          "峰数与 required 不符时抖动为 inf（算不出来就不猜）")

    print("\n辐条计数测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
