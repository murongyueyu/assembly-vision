# -*- coding: utf-8 -*-
"""装配位完整性判定（把 c5 基线接进 checks.py）离线测试。

测的是 assemvision/core/checks.py 的 `spoke_vs_baseline()`。

★ 为什么把判据抽成纯函数再测：
  `check_assembly()` 的输入是像素分割结果，要造出"正好少 2 根辐条"的合成图
  很脆（分割阈值一变就假失败）。抽成纯函数后判据与像素解耦，测试才稳。

★ 测试数据全部来自**真机实测**，不是编的：
  c5 实跑 12 帧辐条数 = [8,8,8,8,6,6,7,8,8,6,8,8]
  -> 完整在位就是 6~8，散布 ±1。容差据此取 2。

★ 为什么不比 ring/hub/cap 覆盖率：
  实测（合成图各种缺件组合）ring 0.994、hub 0.880~0.998、cap 0.874~1.000 ——
  **连缺顶盖都还有 0.874，几乎没有区分度**；真机上 c5 实测三个数全是 ~1.0。
  拿它们当阈值等于没判。**不要为了"用上基线"而加没有区分度的判据。**
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assemvision.core.checks import spoke_vs_baseline         # noqa: E402


def main() -> int:
    print("== 装配位完整性：辐条数 vs c5 基线 ==")

    # ① c5 真机实测的 12 帧，一根都不能误报
    c5_measured = [8, 8, 8, 8, 6, 6, 7, 8, 8, 6, 8, 8]
    false_alarms = [c for c in c5_measured if spoke_vs_baseline(c, 8, slack=2)[0]]
    print("   c5 实测 12 帧 %s -> 误报 %d 个" % (c5_measured, len(false_alarms)))
    check(not false_alarms,
          "c5 实测的完整在位帧一个都不能被误报（误报的是 %s）" % false_alarms)

    # ② 明显缺件要报出来（基线 8、容差 2 -> 5 及以下）
    for c in (1, 2, 3, 4, 5):
        hit, _ = spoke_vs_baseline(c, 8, slack=2)
        check(hit, "%d 根应判异常（基线 8、容差 2）" % c)
    print("   缺件侧 1~5 根全部报出 ✓")

    # ③ 边界：恰好 base-slack 不报（判据用 `<` 不是 `<=`）
    hit, _ = spoke_vs_baseline(6, 8, slack=2)
    check(not hit,
          "恰好 6 根（= 8-2）不报 —— 边界若用 <= 会把实测出现过的 6 误报")

    # ④ 容差可调：收到 1 时 6 根要报
    hit, _ = spoke_vs_baseline(6, 8, slack=1)
    check(hit, "容差收到 1 时 6 根应报异常")

    # ⑤ 0 根不在这里判（避免与 `spk.count < required` 重复报同一件事）
    hit, why = spoke_vs_baseline(0, 8, slack=2)
    check(not hit, "0 根不由本条判（整个轮盘没装时已有基础判据）")
    print("   0 根的说明: %s" % why)

    # ⑥ 基线无效时不判（c5 没跑过 / 被清空过）
    for baseline in (None, 0, "", "abc"):
        hit, _ = spoke_vs_baseline(5, baseline, slack=2)
        check(not hit,
              "基线 %r 无效时应不判（不能拿 0 当基线 -> 会把所有帧都报异常）" % (baseline,))
    print("   基线无效时全部不判 ✓")

    # ⑦ 基线不是 8 也要对（换工件/换流程时基线会不同）
    hit, _ = spoke_vs_baseline(10, 12, slack=2)
    check(not hit, "基线 12、实测 10（差 2）在容差内")
    hit, _ = spoke_vs_baseline(9, 12, slack=2)
    check(hit, "基线 12、实测 9（差 3）应报")

    print("\n装配位完整性测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
