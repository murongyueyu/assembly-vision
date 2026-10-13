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

    print("\n辐条计数测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
