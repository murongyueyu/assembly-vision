# -*- coding: utf-8 -*-
"""离散量（`holes`）的**整数容忍带**测试。

═══ 为什么单独开一个模块 ═══

`holes` 是**整数计数**，而真机上同一个零件会数出 0/1/2/3/4 ——
根因是金属镜面过曝把掩膜打穿、中央又露出一小块黄铜"岛"。
而 c6 标定把区间收成宽度≈0（`[-0.00015, 0.00015]` / `[1.985, 2.015]`），
于是观测值 1 时**四个零件全部 < min_score（认不出）**；
底座实测 1 时单这一项就丢 `2.0/18.1 = 11%` 的分，从 0.77 掉到 0.65，
而**轮盘拿 0.70** —— 两者贴住，`tracking.id_margin=0.12` 就不安全了。

修的思路：**离散量必须容忍 ±1**（底座过曝最严重，单独给 ±2）。
本模块把这条性质钉住，因为它靠眼睛看不出来 ——
得分只掉 0.11，界面上"还是认出来了"，只有真去比数字才发现。

依据与实测数字：docs/踩坑实录.md §2.30；配置见
`workpiece.yaml` 的 `part_id.holes_tol` 与 `calibration.yaml` 的
`part_templates.P1_base.weights.holes_tol`。

═══ 三层断言 ═══

① 纯函数 `_band_score_discrete`：端点向外取整 + 各放 tol
② **向后兼容**：`tol=0` 必须与老的 `_band_score` **逐值完全相同**
   （否则"加了容忍带"就变成"偷偷改了所有特征的行为"）
③ 分类器端到端：真机实测的四个零件特征字典 × holes 扫描，
   **标注零件永远赢、且永远不跌破 min_score**
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.core.part_id import (DISCRETE_FEATURES,     # noqa: E402
                                      PartClassifier, PartTemplate,
                                      _band_score, _band_score_discrete)

# ── 真机实测的特征字典（assembly_choice.png，pose_src=parts）──────────
# ★ 这是**真机**值，不是合成图值：轮盘 solidity=0.746、circ=0.397（分割缺边），
#   与合成图的 0.995 / 0.902 差很远。测试必须用真机值才有意义。
# ★ 必须是**全部 15 项**（含 length_mm/short_side_mm 与 5 项色域）：
#   `PartTemplate.score` 的分母 wsum 只统计"自己 ranges 里有、且 feats 里也有"
#   的特征，少一项分母就变小、所有得分被抬高 —— 那样测出来的就不是运行时的数。
#   本 fixture 与运行时对账过：轮盘 0.706 / 立柱 0.778 / 顶盖 0.948 / 底座 0.813。
REAL_FEATS = {
    "P3_wheel": {
        "holes": 0.0, "area_mm2": 2993.26499, "length_mm": 70.26152,
        "short_side_mm": 70.14680, "elongation": 1.00164,
        "solidity": 0.74638, "circularity": 0.39692,
        "end_width_ratio": 1.06421, "aspect": 1.00164, "area_frac": 0.59626,
        "c_hue_mean": 32.35536, "c_hue_std": 5.78128, "c_sat_mean": 81.38885,
        "c_val_mean": 155.98827, "c_sat_hi_frac": 0.14932,
    },
    "P2_column": {
        "holes": 0.0, "area_mm2": 889.94987, "length_mm": 68.27850,
        "short_side_mm": 16.41161, "elongation": 4.16038,
        "solidity": 0.92575, "circularity": 0.43656,
        "end_width_ratio": 1.00930, "aspect": 4.16038, "area_frac": 0.27129,
        "c_hue_mean": 33.61480, "c_hue_std": 6.59855, "c_sat_mean": 80.76119,
        "c_val_mean": 153.50118, "c_sat_hi_frac": 0.22938,
    },
    "P4_cap": {
        "holes": 2.0, "area_mm2": 497.01879, "length_mm": 26.82708,
        "short_side_mm": 24.38825, "elongation": 1.10000,
        "solidity": 0.95842, "circularity": 0.83461,
        "end_width_ratio": 1.05988, "aspect": 1.10000, "area_frac": 0.67093,
        "c_hue_mean": 31.76007, "c_hue_std": 6.48365, "c_sat_mean": 70.39907,
        "c_val_mean": 180.09938, "c_sat_hi_frac": 0.05435,
    },
    "P1_base": {
        "holes": 1.0, "area_mm2": 2972.07571, "length_mm": 62.96228,
        "short_side_mm": 61.55057, "elongation": 1.02294,
        "solidity": 0.96395, "circularity": 0.78973,
        "end_width_ratio": 1.05861, "aspect": 1.02294, "area_frac": 0.74472,
        "c_hue_mean": 34.73186, "c_hue_std": 5.67176, "c_sat_mean": 64.43784,
        "c_val_mean": 182.86920, "c_sat_hi_frac": 0.01947,
    },
}

# ── 运行时模板的 `holes` 区间（c6 实测；与 calibration.yaml 一致）──────
# ★ 故意用**实测的原值**（宽度≈0 / 1.985~2.015），不要在这里"先修好" ——
#   本模块要证明的正是"容忍带能把这种窄区间救回来"。
C6_HOLES = {
    "P3_wheel": (-0.00015, 0.00015),
    "P2_column": (-0.00015, 0.00015),
    "P4_cap": (1.985, 2.015),
    "P1_base": (1.985, 2.015),
}

# ── 运行时模板的其余区间（c6 实测，与 calibration.yaml 一致）──────────
# ★ 含 `length_mm` / `short_side_mm` —— 它们是权重最高的形状特征，漏了就不是运行时路径。
C6_OTHER = {
    "P3_wheel": {"circularity": (0.3525, 0.5875), "solidity": (0.711, 0.869),
                 "elongation": (0.6565, 1.3635), "end_width_ratio": (0.9095, 1.2305),
                 "aspect": (0.6565, 1.3635), "area_mm2": (2285.5, 4244.5),
                 "area_frac": (0.63513, 0.66487),
                 "length_mm": (52.58, 61.72), "short_side_mm": (51.84, 60.86)},
    "P2_column": {"circularity": (0.345, 0.575), "solidity": (0.81, 0.99),
                  "elongation": (2.184, 5.0), "end_width_ratio": (0.9605, 1.2995),
                  "aspect": (2.184, 5.0), "area_mm2": (751.8, 1396.2),
                  "area_frac": (0.317, 0.343),
                  "length_mm": (54.79, 64.31), "short_side_mm": (4.4, 26.0)},
    "P4_cap": {"circularity": (0.6225, 1.0), "solidity": (0.864, 1.0),
               "elongation": (0.702, 1.458), "end_width_ratio": (0.884, 1.196),
               "aspect": (0.702, 1.458), "area_mm2": (388.5, 721.5),
               "area_frac": (0.7046, 0.7354),
               "length_mm": (22.4, 26.3), "short_side_mm": (20.84, 24.46)},
    "P1_base": {"circularity": (0.5925, 0.9875), "solidity": (0.864, 1.0),
                "elongation": (0.7215, 1.4985), "end_width_ratio": (0.918, 1.242),
                "aspect": (0.7215, 1.4985), "area_mm2": (2208.5, 4101.5),
                "area_frac": (0.75423, 0.77577),
                "length_mm": (53.41, 62.69), "short_side_mm": (48.05, 56.45)},
}

# ★ 底座过曝最严重（实测 0~4），运行时单独给它 ±2；其余 ±1。
PER_PART_TOL = {"P3_wheel": 1.0, "P2_column": 1.0, "P4_cap": 1.0, "P1_base": 2.0}

MIN_SCORE = 0.55


def _runtime_templates() -> list:
    """按**运行时**的方式建模板：c6 窄区间 + 逐零件 holes_tol。"""
    out = []
    for part, (lo, hi) in C6_HOLES.items():
        ranges = {"holes": (lo, hi)}
        ranges.update(C6_OTHER[part])
        out.append(PartTemplate(
            part=part, ranges=ranges,
            discrete_tol={"holes": PER_PART_TOL[part]}))
    return out


def scene_disc_tol() -> None:
    print("== 场景 1：`_band_score_discrete` 的语义 ==")
    print("   规则：c6 区间**先取整成一个整数 n**，再各放 tol")
    print("        （c6 的 [1.985,2.015] 跨 {2,3} 是 pad 太小的产物，")
    print("          物理上只有一个真值；不收掉会多放格 —— 见函数注释）")

    # ★★ 这张表是本条修法的**唯一事实来源**，逐格钉住。
    #    我在 _discrete_band 里连错三次，都是被这里抓住的。
    #    (区间, tol, 满分接受集合)
    cases = [
        ((-0.00015, 0.00015), 1.0, {0, 1}, "轮盘/立柱 真值0 ±1"),
        ((-0.00015, 0.00015), 2.0, {0, 1, 2}, "轮盘/立柱 真值0 ±2"),
        ((1.985, 2.015), 1.0, {1, 2, 3}, "顶盖 真值2 ±1"),
        ((1.985, 2.015), 2.0, {0, 1, 2, 3, 4}, "底座 真值2 ±2"),
    ]
    for (lo, hi), tol, want, label in cases:
        got = {n for n in range(-3, 10) if _band_score_discrete(n, lo, hi, tol) == 1.0}
        check(got == want, "%s -> 满分集合应为 %s（实测 %s）"
              % (label, sorted(want), sorted(got)))

    # 带外必须**至少有一格衰减**（差 1 个比"完全不可能"要高分），
    # ★ 但衰减**有硬截断**：`_band_score` 的距离超过带宽就直接归 0
    #   （带宽 2 时，带外第 2 格起就是 0）。这是老函数的行为，不是本次引入的。
    check(_band_score_discrete(4.0, 1.985, 2.015, 1.0) > 0.0,
          "带上界之上一格应仍有分（实测 %.3f）"
          % _band_score_discrete(4.0, 1.985, 2.015, 1.0))
    check(_band_score_discrete(4.0, 1.985, 2.015, 1.0)
          > _band_score_discrete(6.0, 1.985, 2.015, 1.0),
          "越远应越低或持平（差1=%.3f 差3=%.3f）"
          % (_band_score_discrete(4.0, 1.985, 2.015, 1.0),
             _band_score_discrete(6.0, 1.985, 2.015, 1.0)))
    check(_band_score_discrete(2.0, -0.00015, 0.00015, 1.0) == 0.0,
          "零孔零件带是 {0,1}：2 恰好落在截断点 -> 0（实测 %.3f）"
          % _band_score_discrete(2.0, -0.00015, 0.00015, 1.0))

    # tol 必须被夹住，防止配置里写 999 让判据"永远满分"
    check(_band_score_discrete(1e6, 0.0, 0.0, 999.0) == 0.0,
          "荒唐的 tol 必须被夹到上限（否则判据等于没有）")

    # ★★ 真机实测分布必须**逐类全部被覆盖** —— 这是这条修法的目的本身。
    #   实测（9 帧）：轮盘 {0}、立柱 {0}、顶盖 {2,3}、底座 {0,1,2,3,4}
    observed = {
        (-0.00015, 0.00015): {0},                 # 轮盘
        (1.985, 2.015): {2, 3},                   # 顶盖
    }
    tol_of = {(-0.00015, 0.00015): 1.0, (1.985, 2.015): 1.0}
    for rng, vals in observed.items():
        got = {n for n in range(0, 10)
               if _band_score_discrete(n, rng[0], rng[1], tol_of[rng]) == 1.0}
        check(vals <= got, "实测分布 %s 必须被 %s ±%g 覆盖（实测覆盖 %s）"
              % (sorted(vals), rng, tol_of[rng], sorted(got)))
    # 底座单独 ±2 —— 它跨 0~4 五个值，是唯一需要 ±2 的零件
    got_base = {n for n in range(0, 10)
                if _band_score_discrete(n, 1.985, 2.015, 2.0) == 1.0}
    check({0, 1, 2, 3, 4} <= got_base,
          "底座 ±2 必须覆盖实测 {0,1,2,3,4}（实测覆盖 %s）" % sorted(got_base))


def scene_backward_compat() -> None:
    print("== 场景 2：`tol=0` 必须与老行为**逐值完全相同** ==")
    print("   理由：容忍带只该影响离散量；一旦 tol=0 还改了行为，")
    print("         就是「偷偷改了所有特征」，回归测试会失去意义")
    diff = 0
    for lo, hi in [(0.0, 0.0), (1.985, 2.015), (0.45, 1.0), (751.8, 1396.2)]:
        for v in np.linspace(lo - 4.0, hi + 4.0, 41):
            if abs(_band_score_discrete(v, lo, hi, 0.0) - _band_score(v, lo, hi)) > 1e-12:
                diff += 1
    check(diff == 0, "164 组取值下 tol=0 与 `_band_score` 完全一致（不一致 %d 组）" % diff)

    # ★ 但 tol=0 时端点**仍会向外取整** —— 这是刻意的：
    #   `holes` 是整数计数，观测值 2.0 与标定区间 [1.985, 2.015] 是同一件事。
    check(_band_score_discrete(2.0, 1.985, 2.015, 0.0) == 1.0,
          "离散量取整后 [1.985,2.015] 应涵盖整数 2")


def scene_unrecognizable_gap() -> None:
    print("== 场景 3：★ 修复前的翻车点 —— holes=1 时四个零件全部认不出 ==")
    print("   做法：直接调 `PartTemplate.score`，**显式**给 tol（不走分类器的合并逻辑），")
    print("         这样「tol=0」真的就是改动前的行为")
    feats = dict(REAL_FEATS["P1_base"])
    feats["holes"] = 1.0

    # tol=0 = 改动前（c6 的 [1.985, 2.015] 硬区间）
    old = _runtime_templates()[3]          # P1_base
    old.discrete_tol = {"holes": 0.0}
    s_old, det_old = old.score(feats)
    print("     tol=0（改动前）：holes 项得分 %.3f  ->  总分 %.3f（门槛 %.2f）"
          % (det_old.get("holes", -1), s_old, MIN_SCORE))

    # tol=2 = 改动后（底座单独 ±2）
    new = _runtime_templates()[3]
    new.discrete_tol = {"holes": 2.0}
    s_new, det_new = new.score(feats)
    print("     tol=2（改动后）：holes 项得分 %.3f  ->  总分 %.3f"
          % (det_new.get("holes", -1), s_new))

    check(det_old.get("holes") == 0.0,
          "改动前 holes=1 时该项应为 0（实测 %.3f）" % det_old.get("holes", -1))
    check(det_new.get("holes") == 1.0,
          "改动后 holes=1 时该项应为 1（实测 %.3f）" % det_new.get("holes", -1))

    # ★ 这一项在总分里的分量 = 权重 / wsum。实测真机上底座总分
    #   0.77 -> 0.65（掉 0.12），而**轮盘拿 0.70** —— 贴住 id_margin=0.12。
    gain = s_new - s_old
    print("     单项修复带来的总分提升 = %.3f（真机实测 0.77-0.65=0.12）" % gain)
    check(gain > 0.10, "提升应 > 0.10，否则说明该特征权重被改小了（实测 %.3f）" % gain)

    # 真机实测：底座 0.77（holes 命中）、0.65（holes 未命中）。
    # 修复后 holes=1 也必须落在"命中"那一档。
    check(s_new > 0.70, "修复后总分应回到 0.70 以上（实测 %.3f）" % s_new)


def scene_score_matrix() -> None:
    print("== 场景 4：真机四个零件 × holes 扫描 —— 各自实测取值必须认对 ==")
    print("   ★ 只扫**每类零件真机上真的出现过**的取值（下表），不是全扫 0~4：")
    print("     轮盘 {0}  立柱 {0}  顶盖 {2,3}  底座 {0,1,2,3,4}")
    print("     （拿「真机上不会出现的值」去断言是假的严格 —— 会失败，但失败没有意义）")
    tpls = _runtime_templates()
    clf = PartClassifier(tpls, min_score=MIN_SCORE, holes_tol=1.0)

    observed = {"P3_wheel": (0,), "P2_column": (0,), "P4_cap": (2, 3),
                "P1_base": (0, 1, 2, 3, 4)}
    print()
    print("     %-10s %-18s %s" % ("零件", "实测 holes 取值", "结果"))
    ok_all = True
    for part, vals in observed.items():
        cells = []
        for h in vals:
            feats = dict(REAL_FEATS[part])
            feats["holes"] = float(h)
            pid, sc, _ = clf.classify(feats)
            good = (pid == part) and (sc > MIN_SCORE)
            ok_all = ok_all and good
            cells.append("%d->%s(%.2f)%s" % (h, pid or "-", sc, "" if good else "★"))
        print("     %-10s %-18s %s"
              % (part, str(list(vals)), "  ".join(cells)))
    print("     （★ = 认错或跌破门槛）")
    check(ok_all, "每类零件在它自己的实测 holes 取值下都认对且 > min_score")

    # ★ 轮盘 vs 底座：真机上它俩面积几乎一样（2993 vs 2972），
    #   全靠 holes/solidity/circularity 分开 —— 所以这两条必须单独钉住。
    #   ★ 只对**真机出现过的取值**断言（轮盘恒 0）；不能顺手扫全 0~4 ——
    #     轮盘 holes=2 已经出了它自己的容忍带 {0,1}，判定"认不出"是**正确行为**
    #     （那种观测在真机上不存在），拿它当失败是自欺。
    fw = dict(REAL_FEATS["P3_wheel"]); fw["holes"] = 0.0
    fb = dict(REAL_FEATS["P1_base"]); fb["holes"] = 1.0
    pw, sw, _ = clf.classify(fw)
    pb, sb, _ = clf.classify(fb)
    check(pw == "P3_wheel" and pb == "P1_base",
          "轮盘(holes=0)/底座(holes=1) 必须各自认对（%s %.2f / %s %.2f）"
          % (pw, sw, pb, sb))
    # ★★ 真正的安全边界：**两者都在自己带内时**，底座必须明显高于轮盘。
    #   不然"轮盘被遮一半 -> 面积掉到底座附近"时就会来回跳（§2.20 同类错）。
    _, sw0, _ = clf.classify(dict(REAL_FEATS["P3_wheel"], holes=0.0))
    _, sb0, _ = clf.classify(dict(REAL_FEATS["P1_base"], holes=0.0))
    print("     两者都在带内（holes=0）：轮盘 %.2f  底座 %.2f  差 %.2f"
          % (sw0, sb0, sb0 - sw0))
    check(sb0 > sw0, "同 holes 取值下底座必须高于轮盘（实测 %.2f vs %.2f）" % (sb0, sw0))

    # 轮盘自己的容忍带是 {0,1}：holes=2 应判"认不出"（这是**正确**的窄带行为，
    # 证明容忍带没有被无脑放大到"什么都接受"）
    pid_w2, sc_w2, _ = clf.classify(dict(REAL_FEATS["P3_wheel"], holes=2.0))
    check(pid_w2 is None, "轮盘 holes=2 已出带 {0,1}，应判认不出（实测 %s %.2f）"
          % (pid_w2, sc_w2))

    # ★★ 写清**代价**（这一条同样是断言，防止有人把 ±2 改成"更大更保险"）：
    #   底座的 ±2 让它与顶盖在 holes 上重叠（底座带 {0..4}、顶盖带 {1,3}）。
    #   实测：顶盖 fixtures 在 holes=0（真机未出现，比其真值小 2）时**仍能认对**
    #   （靠 area_mm2/length_mm 把它拉住，分 0.89）；但要盯住的是它
    #   不能掉到"认不出"，也不能被底座抢走。
    for h in (0, 1, 4):
        fc = dict(REAL_FEATS["P4_cap"]); fc["holes"] = float(h)
        pid_c, sc_c, _ = clf.classify(fc)
        print("     代价探针：顶盖 holes=%d（真机未出现）-> %s(%.2f)" % (h, pid_c, sc_c))
        check(sc_c > MIN_SCORE,
              "顶盖在这种退化取值下也不该掉到「认不出」（holes=%d 实测 %.3f）"
              % (h, sc_c))


def scene_config_wiring() -> None:
    print("== 场景 5：配置真的接上了（不是只写在 yaml 里） ==")
    from assemvision.config import load_config
    from assemvision.core.checks import Qualifier
    from assemvision.core.roi import TrayMap

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wp = load_config(os.path.join(root, "config", "workpiece.yaml"))
    tol = wp.get("part_id.holes_tol")
    print("     workpiece.yaml 的 part_id.holes_tol = %s" % tol)
    check(tol is not None, "workpiece.yaml 必须显式写出 holes_tol（不要靠代码默认值兜）")
    check(float(tol) >= 1.0, "holes_tol 至少为 1 —— 实测同一零件跨 0~4，0 会退回硬区间")

    # 没有标定文件时走内置初值，也必须拿到容忍带
    tray = TrayMap(wp, homography=None, px_per_mm=2.5, origin_px=(30.0, 30.0))
    q = Qualifier(wp, tray, None)
    clf = q._classifier()
    check(getattr(clf, "holes_tol", None) == float(tol),
          "Qualifier._classifier() 必须把 holes_tol 传进分类器（实测 %s）"
          % getattr(clf, "holes_tol", None))
    got = {t.part: t.tol_for("holes") for t in clf.templates}
    print("     内置初值模板的 holes 容忍带: %s" % got)
    check(all(v >= 1.0 for v in got.values()),
          "内置初值模板也必须带容忍带（否则没跑 c6 的现场认不出）")
    check("holes" in DISCRETE_FEATURES, "holes 必须登记在 DISCRETE_FEATURES 里")


def main() -> int:
    scene_disc_tol()
    print()
    scene_backward_compat()
    print()
    scene_unrecognizable_gap()
    print()
    scene_score_matrix()
    print()
    scene_config_wiring()
    print()
    print("零件识别整数容忍带测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
