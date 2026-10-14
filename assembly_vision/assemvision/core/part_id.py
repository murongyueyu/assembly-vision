# -*- coding: utf-8 -*-
"""零件识别：用旋转不变的特征向量判定"这是什么零件"。

═══ 为什么需要它（而不是只看槽位覆盖率）═══

原判据是「槽位 ROI 内的黄铜覆盖率」。它的隐含前提是**料盘位置和朝向固定**：
  · 料盘转一下 -> 所有 ROI 全部错位 -> 明明有件却判缺件
  · 零件在槽里放歪 -> 覆盖率掉下来 -> 误判
  · 相机被碰一下 -> 同样全崩

而**零件本身的形状特征与朝向无关**：轮盘就是"有 8 根辐条的圆环"，
底座就是"中心带孔的圆盘"，立柱就是"细长杆"。用这些特征判定，
料盘怎么转都能认出来。

═══ 特征向量的选取原则 ═══

只能用**旋转不变**的量（转多少度都不变的）：
  ✓ 圆度 circularity        面积/周长²   —— 旋转不变
  ✓ 凸包实度 solidity       面积/凸包面积 —— 旋转不变
  ✓ 伸长率 elongation       长边/短边     —— 旋转不变（只用比例，不用角度）
  ✓ 内孔数 holes            拓扑量        —— 旋转不变
  ✓ 端宽比 end_width_ratio  两端宽度/中部 —— 沿主轴算，旋转不变
  ✓ 归一化面积 area/roi     按 ROI 面积归一 -> 与像素尺度无关
  ✗ rect_angle              绝对角度      —— **不能用**，料盘一转就变
  ✗ 质心坐标                绝对位置      —— 位置另用 ROI 判

═══ 判分方式 ═══

每个特征给一个 0~1 的得分（落在期望区间内=1，越远越低），加权求和。
让"某个特征不可靠"时可以把它的权重调低，而不是非此即彼。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ── 特征名与默认权重 ──────────────────────────────────────────
# 权重反映"这个特征对区分四类零件的贡献"：
#   · holes / solidity / elongation 区分度最高（轮盘有辐条缺口、底座有孔、立柱细长）
#   · circularity 对圆件有用，但轮盘因为有辐条缺口反而不高
#   · area_frac 依赖 ROI 准确，权重压低一点
FEATURE_WEIGHTS: Dict[str, float] = {
    "holes": 2.0,
    # ★★ length_mm（最长边物理长度）：**对细长/弯曲/锥形件唯一可信的特征**。
    #   它由轮廓上相隔最远的两个点决定 —— 反光只影响中间段的宽度，
    #   **两端点永远在**。合成实验：只改"被照亮宽度"时它只变 4%。
    #   而 area 变 51%、circularity 变 39%、solidity 变 21%（见 features_from_blob 注释）。
    "length_mm": 2.5,
    # ★★ short_side_mm（短边物理长度）：圆件上很稳，区分"同形状不同大小"。
    "short_side_mm": 1.5,
    # solidity（凸包实度）：**圆件上稳（2.2%）、细长弯曲件上不稳（21%）**。
    #   实测教训：一开始误以为它全局最稳并给了 3.0 权重 ——
    #   那个 2.2% 是拿圆件平均出来的，对立柱恰恰是错的。
    #   所以默认权重降到 2.0，并在 part_templates 里给立柱单独设 0（见 per-part weights）。
    "solidity": 2.0,
    "elongation": 1.5,
    "circularity": 1.0,
    # ★ 物理面积（mm²）：含宽度因子，对细长/弯曲件最不稳（51%）。
    #   只用来区分**量级差很多**的件（立柱 735 vs 顶盖 555 这种就分不开）。
    "area_mm2": 0.8,
    # ── 色域特征（见 color_features）──────────────────────────
    # ★ 颜色**天然旋转不变**（转多少度，像素的 H/S/V 都不变），
    #   而且能补形状特征区分不了的场景：形状相近但材质/氧化/加工痕迹不同。
    #   权重给得比形状低一些：工位光照会漂移，颜色比形状更不稳。
    "c_hue_mean": 1.4,      # 平均色调（圆统计）—— 最能代表"这是什么颜色"
    "c_hue_std": 0.8,       # 色调离散度 —— 氧化不均/有镀层时会变大
    "c_sat_mean": 1.0,      # 平均饱和度 —— 镜面高光多的件会偏低
    "c_val_mean": 0.8,      # 平均亮度 —— 受光照影响最大，权重最低
    "c_sat_hi_frac": 0.8,   # 高饱和像素占比 —— 区分"哑光/抛光"表面
    "end_width_ratio": 1.0,
    "aspect": 0.6,
    # ★ rect_angle 故意不列入：绝对角度随料盘旋转而变，违反旋转不变性
    # ★ area_frac（相对槽面积）也保留但权重低，只作为辅助
    "area_frac": 0.4,
}

# ── 离散量特征：必须用「整数容忍带」而不是连续区间 ────────────────
# ★★ 为什么单列一类（真机实测踩过，见 docs/踩坑实录.md §2.30）：
#   `holes` 是**整数计数**，不是连续量。而它在真机上对**同一个零件**
#   会数出 0/1/2/3/4 —— 根因是金属镜面过曝把掩膜打穿、中央又露出一小块
#   黄铜"岛"（实测底座就是这样）。
#   而 c6 标定把区间收成宽度≈0：`[-0.00015, 0.00015]` / `[1.985, 2.015]`。
#   后果：观测值 1 时**四个零件全部 < min_score（认不出）**；
#   底座实测 0/1/2 时模板要 2 -> 单这一项就丢
#   `holes 权重 2.0 / wsum 18.1 = 11%` 的分，底座从 0.77 掉到 0.65，
#   而**轮盘拿 0.70** —— 两者贴住，`tracking.id_margin=0.12` 就不安全了。
#   => 离散量必须容忍 ±1（底座因为过曝更严重，单独给 ±2），
#      这样"数错一个"只是轻微扣分，而不是整条否掉。
DISCRETE_FEATURES = frozenset({"holes"})

# 容忍带的上限，纯属防御（防止配置里写个 999 把判据变成"永远满分"）
MAX_DISCRETE_TOL = 16.0


@dataclass
class PartTemplate:
    """一种零件的特征模板：每个特征给 [min, max] 期望区间。

    用区间而不是单点，因为：
      · 零件在槽里可以有不同的摆放角度/位置，特征的取值会有分布
      · 分割边缘有抖动

    `discrete_tol`：**离散量特征**（见 `DISCRETE_FEATURES`，目前只有 `holes`）
      的整数容忍带。取 0（默认）表示"关掉容忍带、按连续区间判"，
      这样任何没配 tolerance 的场景行为与改动前**完全一致**。
    """
    part: str
    name: str = ""
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    # 允许"缺失特征"（比如没标定过就留空），缺失的特征不参与判分
    min_score: float = 0.55        # 低于此分判为"认不出"
    # ★ 逐模板的离散容忍带覆盖（目前只有 holes 会用到）
    discrete_tol: Dict[str, float] = field(default_factory=dict)

    def tol_for(self, key: str) -> float:
        """该特征的有效容忍带。非离散特征恒为 0（= 老行为）。"""
        if key not in DISCRETE_FEATURES:
            return 0.0
        return float(self.discrete_tol.get(key, 0.0) or 0.0)

    def score(self, feats: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """给一个特征字典打分，返回 (总分 0~1, 明细)。"""
        total = 0.0
        wsum = 0.0
        detail: Dict[str, float] = {}
        for key, (lo, hi) in self.ranges.items():
            v = feats.get(key)
            if v is None or not np.isfinite(v):
                continue
            w = float(self.weights.get(key, FEATURE_WEIGHTS.get(key, 1.0)))
            tol = self.tol_for(key)
            if tol > 0.0:
                s = _band_score_discrete(float(v), float(lo), float(hi), tol)
            else:
                s = _band_score(float(v), float(lo), float(hi))
            detail[key] = s
            total += w * s
            wsum += w
        if wsum <= 0:
            return 0.0, detail
        return total / wsum, detail


def _band_score(v: float, lo: float, hi: float) -> float:
    """值落在 [lo,hi] 内得 1；落在外面按"相对宽度"衰减，最低 0。

    衰减宽度取区间宽度本身（lo==hi 时取 |v| 的 10% 兜底），
    这样"差一点点"比"差很多"得分高，便于看谁最像。
    """
    if lo > hi:
        lo, hi = hi, lo
    if lo <= v <= hi:
        return 1.0
    width = max(hi - lo, 0.1 * max(abs(lo), abs(hi), 1e-6))
    d = (lo - v) if v < lo else (v - hi)
    return float(max(0.0, 1.0 - d / width))


def _discrete_band(lo: float, hi: float, tol: float) -> Tuple[float, float]:
    """把标定区间换算成"整数容忍带"：**先取整成一个整数，再各放 tol**。

    ★★ 语义定成：**"这个零件的孔数是 n，允许数错 tol 个"**，其中
      `n = round(lo) == round(hi)`；若区间跨多个整数，就取 `[round(lo), round(hi)]`
      再各放 tol（保守，宁可多放一格）。

    ★★ 为什么"跨两个整数"必须收掉：c6 的区间是 `[中值 ± pad]` 算出来的，
      顶盖/底座的 `[1.985, 2.015]` **跨 {2, 3}**，但物理上各只有**一个**真值
      （顶盖 2、底座 2）—— 那是 pad 太小的产物，不是"有两个可能值"。
      如果不收，`[1.985, 2.015] + tol=1` 会变成 `[1, 4]`（接受 4 个数），
      对零孔零件更糟：`[-0.00015, 0.00015] + tol=1` 会变成 `[-2, 2]`
      （接受 3 个数，多放一格）。

    ★★ 我在这一处**连错三次**，最后靠这张表 + `tests/test_part_id_tol.py`
      的逐值断言才定下来（所以它必须留在注释里）：

        c6 区间                    n    tol  整数带    满分接受
        [-0.00015, 0.00015]（轮盘） 0    1    [-1, 1]    {0, 1}
        [-0.00015, 0.00015]（立柱） 0    1    [-1, 1]    {0, 1}
        [1.985, 2.015]（顶盖）      2    1    [1, 3]     {1, 2, 3}
        [1.985, 2.015]（底座）      2    2    [0, 4]     {0, 1, 2, 3, 4}

      对照真机实测分布（9 帧）：轮盘 {0}、立柱 {0}、顶盖 {2,3}、底座 {0,1,2,3,4}
      —— 四类零件**各自的实测值全部被覆盖**，一个都不漏。
    """
    t = max(0.0, min(float(tol), MAX_DISCRETE_TOL))
    n_lo = int(round(float(lo)))
    n_hi = int(round(float(hi)))
    if n_hi < n_lo:
        n_lo, n_hi = n_hi, n_lo
    # ★ 计数类离散量**没有负数** —— 下界夹到 0，否则零孔零件 ±1 会得到
    #   [-1, 1]（含 -1），语义上是个不存在的取值。
    return (max(0.0, float(n_lo) - t), float(n_hi) + t)


def _band_score_discrete(v: float, lo: float, hi: float, tol: float) -> float:
    """**离散量**（整数计数）的区间判分：把 [lo,hi] 扩成"整数容忍带"。

    规则：`[floor(lo), ceil(hi)]` 再各放 `tol`，带内得 1、带外按距离衰减。
      真值 0 的 `[-0.00015, 0.00015]` + tol=1 -> `[-1, 1]` -> 接受 {0, 1}
      真值跨 {2,3} 的 `[1.985, 2.015]` + tol=1 -> `[1, 4]` -> 接受 {1,2,3,4}
      真值跨 {2,3} 的 `[1.985, 2.015]` + tol=2 -> `[0, 5]` -> 接受 {0..5}

    参数
      v    实测值（浮点，但物理上是整数计数）
      tol  允许数错的个数。★ **tol<=0 时直接退化成 `_band_score`**，
           不做任何取整 —— 否则"关掉容忍带"也会悄悄改变判据行为
           （实测后果：`_band_score_discrete(1.0, 1.985, 2.015, 0.0)` 会从
           老的 0.0 变成 1.0，即默认路径被改了）。
    """
    if float(tol) <= 0.0:
        return _band_score(float(v), float(lo), float(hi))
    rlo, rhi = _discrete_band(lo, hi, tol)
    return _band_score(float(v), rlo, rhi)


class PartClassifier:
    """把一张图里的黄铜连通域逐个判定成零件种类。

    用法：
        clf = PartClassifier(templates)
        for blob in seg.brass_blobs:
            part, score, detail = clf.classify(blob.features)

    `holes_tol` 是**策略**（不是"这个零件有几个孔"的知识）：
      "对离散量特征，允许实测值与标定值差几个"，默认 1。
      ★ 知识（每个零件几个孔）在 `PartTemplate.ranges` 里；
        策略（数错了算多严重）在这里 —— 两者分开，配置才好调。
      ★ 逐模板可以用 `PartTemplate.discrete_tol` 覆盖本值
        （实测底座过曝最严重，观测值跨 0~4，单独给 2）。
    """

    def __init__(self, templates: Sequence[PartTemplate], min_score: float = 0.55,
                 holes_tol: float = 1.0):
        self.templates: List[PartTemplate] = list(templates)
        self.min_score = float(min_score)
        try:
            self.holes_tol = max(0.0, min(float(holes_tol), MAX_DISCRETE_TOL))
        except (TypeError, ValueError):
            self.holes_tol = 0.0
        # ★ 把策略**注入**模板：模板自己没写 discrete_tol 的离散特征就用它。
        #   放在这里而不是 score() 里传参，是为了不动 PartTemplate.score 的签名
        #   （已有测试与工具都直接调 score(feats)）。
        for t in self.templates:
            t.discrete_tol = _merge_tol(t.discrete_tol, self.holes_tol)

    def classify(self, feats: Dict[str, float]):
        """返回 (part, score, detail)。

        part 为 None 表示"认不出"（所有模板都没到 min_score）。
        detail 是每个模板的 (总分, 各特征得分)，用于调试"为什么认错"。
        """
        best_part: Optional[str] = None
        best_score = -1.0
        best_detail: Dict[str, float] = {}
        all_detail: Dict[str, Tuple[float, Dict[str, float]]] = {}
        for t in self.templates:
            s, det = t.score(feats)
            all_detail[t.part] = (s, det)
            if s > best_score:
                best_score, best_part, best_detail = s, t.part, det
        if best_score < self.min_score:
            return None, float(best_score), best_detail
        return best_part, float(best_score), best_detail

    def rank(self, feats: Dict[str, float]):
        """所有模板的得分排序（从高到低），用于诊断。"""
        out = []
        for t in self.templates:
            s, det = t.score(feats)
            out.append((t.part, s, det))
        out.sort(key=lambda x: -x[1])
        return out


def _merge_tol(existing: Optional[Dict[str, float]], holes_tol: float) -> Dict[str, float]:
    """把"分类器级"的容忍带默认值并进"模板级"的设置（模板级优先）。"""
    out: Dict[str, float] = {}
    for k, v in (existing or {}).items():
        try:
            out[str(k)] = max(0.0, min(float(v), MAX_DISCRETE_TOL))
        except (TypeError, ValueError):
            continue
    for k in DISCRETE_FEATURES:
        if k not in out:
            out[k] = max(0.0, min(float(holes_tol), MAX_DISCRETE_TOL))
    return out


# ── 内置默认模板 ──────────────────────────────────────────────
# ★★ 这个函数的定位（先读清，否则会拿它当真相）：
#   它是**没跑 c6 标定时的兜底**，判据区间是**由零件几何推算的初值**，
#   ★ **真机上的权威模板在 `config/calibration.yaml` 的 `part_templates`**
#     （c6 用 25 帧实测采集），运行时 `Qualifier._classifier()` 优先读它。
#   实测踩过：`holes` 这一项这里和真实画面**相反**（见下），
#   而"看起来像实测值"很容易让人信错。
#
# 推算依据（零件名义尺寸，见 workpiece.yaml 的 parts 与 radii_mm）：
#   P1_base  底座圆盘   Ø65 中心有孔 -> 圆度高、实度高、有孔、不伸长
#   P2_column 立柱      细长杆 ~85x15  -> 伸长率很高、实度高、无孔
#   P3_wheel 轮盘组件   Ø75 有 8 辐条 -> 实度低（辐条间有缺口）、不伸长
#   P4_cap   顶盖铭牌   Ø25           -> 小圆
#
# ★ `holes` 的取值依据（真机 9 帧 + 合成图，**两套都实测过**）：
#     零件       真机实测      合成图实测    本函数取值
#     轮盘 P3    0             8~9           (4, 16)    <- ★ 见下，刻意服务合成域
#     立柱 P2    0             0             (0, 1)
#     顶盖 P4    2~3           0             (1, 3)
#     底座 P1    0/1/2/3/4     1             (0, 3)     <- 过曝导致跨 5 个值
#   ★★ 轮盘为什么**保持** `(4.0, 16.0)` 不动（曾经想改成 (0,1)，实测证明是错的）：
#     真机轮盘确实 `holes=0`，而 `(4.0,16.0)` 认不出它 —— 看着像 bug。
#     但**本函数是合成域的兜底**（`calibration.synthetic.yaml` 没有
#     `part_templates`，离线端到端与 test_part_id 全靠它），而合成轮盘
#     `holes=8~9`。于是：
#       · 改成 (0,1)  -> 合成底座与合成轮盘在 holes 上**双双满分、并列 1.000**，
#                        `classify` 按模板顺序选 -> **底座被判成轮盘**
#                        （实测：test_part_id 与 test_segment 立刻挂）
#       · 保持 (4,16) -> 两域各自可用
#     ★ 为什么会并列 1.000：合成底座（圆盘）的 elong=1.000 / solidity=0.994 /
#       circ=0.897 / area=3244 全部落在轮盘的区间里，`holes` 是**唯一**
#       能分开它俩的判据 —— 一放开就并列，只能靠列表顺序决胜。
#     => 结论：**两个域的 holes 差 9 个（结构差异，不是噪声），不可能用同一个
#        区间**。真机轮盘的 0 由运行时模板 `calibration.yaml`（c6 实测）表达；
#        本函数保持合成域可用，并在注释里写清真机实测值。
#     ★ 真机轮盘"靠什么与底座分开"：solidity + circularity + area_mm2 的组合
#       + **整数容忍带**（见 DISCRETE_FEATURES）—— 而不是单靠 holes。
#     ★ 想理解这个坑，看 docs/踩坑实录.md §2.30。
#
# ★ area_mm2 = 零件的**物理面积**（mm²）= 像素面积 × (mm_per_px)²。
#   这是区分"同为圆形"的底座与顶盖的**唯一**手段，而且：
#     · 与像素尺度无关（实测 2.5 与 4.0 px/mm 下算出 4379 / 4377，差 0.05%）
#     · 与料盘朝向无关
#   ★ 不要用"面积/槽面积"代替：底座填满它的槽 Ø65、顶盖也填满它的小槽 Ø25，
#     两者比值都是 0.6 左右，互相误判（实测踩过）。
#
#   合成图上实测的物理面积（名义值见括号）：
#     轮盘 4379 mm²（名义 4348）    底座 3244（名义 2762）
#     立柱 2386（名义 1275，胶囊槽留了富余，所以偏大）
#     顶盖  462（名义  287）
#   ★ 真机上实测（分割会侵蚀边缘，所以偏小）：轮盘 2993 / 底座 2970 /
#     立柱 890 / 顶盖 497 mm² —— 所以 base 的区间下方要给得宽。
def default_templates() -> List[PartTemplate]:
    return [
        PartTemplate(
            part="P3_wheel", name="轮盘组件",
            ranges={
                # ★ 实度：辐条之间的空隙让凸包实度低于实心圆。
                #   注意实测（合成图、4 px/mm）轮盘 solidity=0.995 —— 因为
                #   外轮廓本身就是个完整圆，孔在内部，凸包与轮廓面积几乎相等。
                #   所以 **solidity 区分不了轮盘与底座**，别指望它。
                "solidity": (0.45, 1.0),
                # ★★ 轮盘的**决定性**判据本来是"辐条之间形成 8+ 个内孔"——
                #   它在**合成图**上成立（实测 8~9），在**真机上不成立**（实测 0）。
                #   保持 (4,16) 是为了让本函数在合成域可用，理由见上面 ★★ 长注释。
                #   ★ 真机轮盘的 holes 真值由 `calibration.yaml` 的
                #     `part_templates.P3_wheel.ranges.holes` 表达（c6 实测）。
                "holes": (4.0, 16.0),
                # ★ circularity 区间要给得宽：实测轮盘 0.90（合成图，接近满圆），
                #   真机只有 0.40（分割缺边），早期写 (0.10, 0.45) 太窄，
                #   在 6 px/mm 下把轮盘判成了底座。
                "circularity": (0.10, 0.98),
                "elongation": (1.0, 1.45),
                "area_mm2": (2400.0, 7000.0),
            },
        ),
        PartTemplate(
            part="P1_base", name="底座圆盘",
            ranges={
                "solidity": (0.88, 1.0),
                # ★ 中心一个螺纹孔 -> 实测 1；但镜面过曝会把它数成 0~4
                #   （见 DISCRETE_FEATURES 的说明），所以区间按观测跨度给。
                "holes": (0.0, 3.0),
                "circularity": (0.55, 1.05),
                "elongation": (1.0, 1.35),
                "area_mm2": (1900.0, 3900.0),
            },
        ),
        PartTemplate(
            part="P2_column", name="立柱",
            ranges={
                "elongation": (1.8, 8.0),     # ★ 最强判据：细长
                "solidity": (0.80, 1.0),
                "holes": (0.0, 1.0),
                "circularity": (0.05, 0.60),
                # 真实立柱约 85x15 = 1275 mm²；合成图按 85x16 画，实测约 1400 mm²
                # 真机实测只有 890 mm²（分割只切得到高光带）
                "area_mm2": (400.0, 2800.0),
            },
        ),
        PartTemplate(
            part="P4_cap", name="顶盖铭牌",
            ranges={
                "circularity": (0.60, 1.05),
                "solidity": (0.88, 1.0),
                # ★ `方案总纲 §3.3` 写的是"覆盖中心后内孔数 = 0"，
                #   那是**装配态**（顶盖盖住轮毂孔）的说法；料盘上的顶盖
                #   单独躺着，实测 2~3 个孔（刻字/校徽被分割成孤立块）。
                "holes": (1.0, 3.0),
                "elongation": (1.0, 1.30),
                # ★ 与底座同为圆形，靠物理面积区分（顶盖小一个量级）
                "area_mm2": (150.0, 1000.0),
            },
        ),
    ]


def color_features(bgr: np.ndarray, mask: np.ndarray,
                   min_pixels: int = 40) -> Dict[str, float]:
    """在给定掩膜内统计**色域特征**（全部旋转不变）。

    ★ 为什么颜色值得单列一类特征
      · **天然旋转不变**：料盘转多少度，同一块像素的 H/S/V 都不变。
      · 能补形状特征区分不了的场景：形状相近但材质/氧化程度/表面处理不同。
      · 代价小：分割时本来就算过 HSV，这里只是多统计几次。

    ★ 关键：色调（H，OpenCV 里 0~179）是**环形量** —— 0 和 179 其实相邻。
      所以**不能直接取平均**（5° 和 175° 的平均是 90°，完全错），
      必须用圆统计：把 H 映射到单位圆上求平均向量。

    返回的键（都会被 PartTemplate 当作普通特征处理）：
      c_hue_mean      圆平均色调（度）
      c_hue_std       色调圆离散度（度，0~90；越大说明表面颜色越杂）
      c_hue_hi_frac   高饱和像素中"暖色调"（H<45）的占比
      c_sat_mean      平均饱和度
      c_val_mean      平均亮度
      c_sat_hi_frac   高饱和（S>90）像素占比 —— 区分抛光/哑光表面
    """
    out: Dict[str, float] = {}
    if bgr is None or mask is None:
        return out
    m = mask > 0
    n = int(np.count_nonzero(m))
    if n < min_pixels:
        return out
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0][m].astype(np.float64) * 2.0      # OpenCV 的 H 是 0~179，换算成 0~358 度
    S = hsv[..., 1][m].astype(np.float64)
    V = hsv[..., 2][m].astype(np.float64)

    # 圆统计：只用饱和度够高的像素（低饱和像素的色调是噪声，没有意义）
    sel = S >= 40.0
    if int(np.count_nonzero(sel)) >= max(8, min_pixels // 4):
        hh = np.deg2rad(H[sel])
        cx = float(np.cos(hh).mean())
        cy = float(np.sin(hh).mean())
        r = float(np.hypot(cx, cy))
        mean_deg = float(np.rad2deg(np.arctan2(cy, cx)) % 360.0)
        # 圆离散度：R=1 表示所有像素色调一致；R 越小越散
        std_deg = float(np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(max(r, 1e-6))))))
        out["c_hue_mean"] = mean_deg / 2.0           # 换回 0~179 尺度，和配置一致
        out["c_hue_std"] = min(90.0, std_deg / 2.0)
        out["c_hue_hi_frac"] = float(np.count_nonzero(H[sel] < 45.0) / H[sel].size)
    out["c_sat_mean"] = float(S.mean())
    out["c_val_mean"] = float(V.mean())
    out["c_sat_hi_frac"] = float(np.count_nonzero(S > 90.0) / S.size)
    return out


def features_from_blob(blob, roi_area_px: Optional[float] = None,
                       mm_per_px: Optional[float] = None,
                       bgr: Optional[np.ndarray] = None,
                       mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """从 segment 的 blob 对象取出特征字典（形状 + 物理尺度 + 色域）。

    ★★ area_mm2（物理面积）是最重要的补充特征，必须传 mm_per_px 才能算。
       它 = 像素面积 × mm_per_px²，用来区分"同为圆形"的底座与顶盖。

    ★ 归一化分母的选择（踩过两次坑，记下来）：
      ① 用 bbox 面积：圆的"面积/bbox面积"恒为 π/4，**与大小无关**
         -> 底座和顶盖算出来一样，互相误判。
      ② 用【料盘面积】：对坐标映射的尺度很敏感（同一张合成图在 4.0 和 2.5 px/mm
         下算出 0.328 与 0.127），直接导致底座被判成顶盖。
      ③ 用【该槽面积】：与尺度无关、与朝向无关，但仍**区分不了同为圆形的底座与顶盖**
         （各自都填满自己的槽，比值都是 0.6 左右）。
      => 所以最终用 **area_mm2（物理面积）** 作为主判据，area_frac 只作辅助。

    参数
      blob         segment.py 输出的 Blob（有 .features / .area_px）
      roi_area_px  该槽的物理面积换算出的像素面积（辅助特征用，可省）
      mm_per_px    每像素多少毫米（**算 area_mm2 必须给**，来自 TrayMap）
      bgr          原图（**算色域特征必须给**）
      mask         该 blob 的掩膜（**算色域特征必须给**，一般传 seg.brass_mask）
    """
    f: Dict[str, float] = dict(getattr(blob, "features", {}) or {})
    area = float(getattr(blob, "area_px", f.get("area_px", 0.0)) or 0.0)
    if mm_per_px and mm_per_px > 0:
        f["area_mm2"] = area * float(mm_per_px) ** 2
        # ★★ length_mm：**最长边（两端点距离）的物理长度**。
        #
        # 为什么专门加这个（实测教训）：立柱是**弯曲 + 锥形 + 两端带台阶**的
        # 机加工件（手机照片确认）。它的中段是锥形回转体 + 抛光纵向亮纹，
        # 分割**只切得到顶面那条高光带**，所以：
        #
        #     长边变化  -1.7%   ✓ 稳（由两端最远点决定，端点永远在）
        #     宽边变化 -30.3%   ✗ 漂（取决于哪条亮纹被切中）
        #     面积变化 ±30~50%  ✗ 漂（含宽度因子）
        #
        # 合成实验量化（改变"被照亮的宽度比例"）：
        #     feret/rect_long 3.9~4.0% ✓ | elong 18% | solidity 21%
        #     rect_short 22% | circularity 39% | area 51.5% ✗
        #
        # **结论：对细长/弯曲/锥形件，只有"长度"类特征可信。**
        # `_band_score` 对长方形也是拿最长边当长，所以直接用 max(rect_w, rect_h)。
        rw_mm = float(f.get("rect_w", 0.0)); rh_mm = float(f.get("rect_h", 0.0))
        f["length_mm"] = max(rw_mm, rh_mm) * float(mm_per_px)
        # short_side_mm：短边物理长度。含宽度因子，对立柱不稳，但
        # **圆件上很稳**，用来区分顶盖(Ø25) 与底座(Ø65) 一类"同形状不同大小"。
        f["short_side_mm"] = min(rw_mm, rh_mm) * float(mm_per_px)
    if roi_area_px and roi_area_px > 0:
        f["area_frac"] = area / float(roi_area_px)
    else:
        rw = float(f.get("rect_w", 0.0)); rh = float(f.get("rect_h", 0.0))
        bb = rw * rh
        f["area_frac"] = (area / bb) if bb > 1e-6 else 0.0
    # 色域：只统计"这一个 blob"的像素，用它的外接框裁一下再用轮廓掩膜取交集
    if bgr is not None and mask is not None:
        x, y, w, h = getattr(blob, "bbox", (0, 0, bgr.shape[1], bgr.shape[0]))
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(bgr.shape[1], int(x + w)), min(bgr.shape[0], int(y + h))
        if x1 > x0 and y1 > y0:
            sub_bgr = bgr[y0:y1, x0:x1]
            sub_mask = np.zeros(sub_bgr.shape[:2], np.uint8)
            c = np.asarray(getattr(blob, "contour", []), dtype=np.int32)
            if c.size:
                cv2.drawContours(sub_mask, [c.reshape(-1, 1, 2) - [x0, y0]], -1, 255, -1)
            # 与全局前景掩膜取交集：避免把轮廓内部的孔（辐条空隙）算进颜色统计
            gm = mask[y0:y1, x0:x1]
            sub_mask = cv2.bitwise_and(sub_mask, gm)
            f.update(color_features(sub_bgr, sub_mask))
    return f


def slot_area_px(tray_map, slot_id: str, scale: float = 1.0) -> float:
    """某个槽的物理面积（像素²），用作该槽 area_frac 的归一化分母。

    ★ 用 shoelace 公式对**槽的多边形**求面积，能正确处理透视与斜置胶囊
      （立柱槽是 91x28 @-44° 的胶囊，外接框会是它的两倍多，不能用）。
    """
    poly = tray_map.slot_polygon_px(slot_id, scale)
    x = poly[:, 0]; y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def max_coverage_for(slot_id: str, templates: Sequence[PartTemplate],
                     tray_map, scale: float = 1.0) -> float:
    """某个槽里"零件完全在位"时的**理论最大覆盖率**。

    ★ 为什么需要它：覆盖率 = 零件面积 / 槽面积，而这个比值**每种零件天生不同**。
      实测（合成图，几何由名义尺寸决定）：
        轮盘 4378mm² / 槽 Ø75     -> 最大约 0.99（圆形填满圆槽）
        底座 3244mm² / 槽 Ø65     -> 最大约 0.98
        顶盖  462mm² / 槽 Ø25     -> 最大约 0.94
        立柱 1311mm² / 槽 91x28   -> 最大只有 **0.51**（细杆 vs 宽胶囊槽）
      用一个统一的 cov_present=0.45 去卡，立柱永远只能得 0.40 —— **必然误判**。

    做法：取该槽模板里 area_mm2 区间的**下界**（"零件完整但分割最不利"时的面积），
          除以槽的物理面积，再夹到 (0, 1]。

    ★★ 为什么必须用**下界**而不是上界（实测踩过，是我自己改坏的）：
      曾用上界 —— 逻辑是"零件完整在位时的最大面积"。但那算出来的阈值
      **比实际能测到的覆盖率还高**，必然误判：

      ```
      立柱：模板区间 [751.8, 1396.2]，槽 3280mm²
        用上界 1396 -> max_cov 0.426 -> 阈值 thr_present 0.192
        用下界  752 -> max_cov 0.229 -> 阈值 thr_present 0.103
        而**实测覆盖率只有 0.131**（分割只切得到高光带，面积 667~903mm²）
      -> 用上界时 0.131 < 0.192 **差 32%**，立柱 20 帧全被拒、c4 采不到样本
      ```

      阈值的作用是"判断零件在不在"，必须按**最坏情况**定：
      模板区间的下界本身就是"已标定过的、仍算合格的最小面积"。
      用上界等于要求"每次都测到最大面积"，与实测散布直接矛盾。

      ★ 区间下界仍比最坏实测低不少，所以有足够余量：
        立柱下界 752 vs 实测最小 667 —— 但仍过了（0.131 > 0.103）。
    """
    tpl = next((t for t in templates if t.part
                and slot_id and _part_of_slot(t, slot_id)), None)
    if tpl is None:
        return 1.0
    rng = tpl.ranges.get("area_mm2")
    if not rng:
        return 1.0
    mmpp = tray_map.mm_per_px(scale)
    slot_mm2 = slot_area_px(tray_map, slot_id, scale) * mmpp * mmpp
    if slot_mm2 <= 1e-9:
        return 1.0
    return float(min(1.0, max(0.05, float(rng[0]) / slot_mm2)))


def _part_of_slot(tpl: PartTemplate, slot_id: str) -> bool:
    """模板是否属于该槽（靠 part 名与槽 id 的对应关系判断）。

    槽 id 形如 slot_wheel / slot_column / slot_cap / slot_base，
    零件名形如 P3_wheel / P2_column / P4_cap / P1_base —— 取尾部词比对。
    """
    sid = slot_id.lower().replace("slot_", "")
    return tpl.part.lower().endswith(sid) or sid in tpl.part.lower()


def tray_area_px(tray_map, scale: float = 1.0) -> float:
    """料盘在画面里的面积（像素²）。

    ★ 保留是为了别的用途（比如判断整幅画面里料盘的占比）。
      **不要再用它给 area_frac 归一化** —— 见 features_from_blob 的说明。
    """
    quad = tray_map.mm_to_px(
        np.array([[0.0, 0.0], [160.0, 0.0], [160.0, 160.0], [0.0, 160.0]]),
        scale)
    x = quad[:, 0]; y = quad[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)
