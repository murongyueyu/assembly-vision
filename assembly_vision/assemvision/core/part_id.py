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


@dataclass
class PartTemplate:
    """一种零件的特征模板：每个特征给 [min, max] 期望区间。

    用区间而不是单点，因为：
      · 零件在槽里可以有不同的摆放角度/位置，特征的取值会有分布
      · 分割边缘有抖动
    """
    part: str
    name: str = ""
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    # 允许"缺失特征"（比如没标定过就留空），缺失的特征不参与判分
    min_score: float = 0.55        # 低于此分判为"认不出"

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


class PartClassifier:
    """把一张图里的黄铜连通域逐个判定成零件种类。

    用法：
        clf = PartClassifier(templates)
        for blob in seg.brass_blobs:
            part, score, detail = clf.classify(blob.features)
    """

    def __init__(self, templates: Sequence[PartTemplate], min_score: float = 0.55):
        self.templates: List[PartTemplate] = list(templates)
        self.min_score = float(min_score)

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


# ── 内置默认模板 ──────────────────────────────────────────────
# ★ 这些是**由零件几何推算的初值**，必须用 calib/c6_part_features.py
#   在真实画面上标定后覆盖。不要直接拿来下结论。
#
# 推算依据（零件名义尺寸，见 workpiece.yaml 的 parts 与 radii_mm）：
#   P1_base  底座圆盘   Ø59.3 中心有孔 -> 圆度高、实度高、有孔、不伸长
#   P2_column 立柱      细长杆 ~85x15  -> 伸长率很高、实度高、无孔
#   P3_wheel 轮盘组件   Ø74.4 有辐条   -> 实度低（辐条间有缺口）、多孔、不伸长
#   P4_cap   顶盖铭牌   Ø19.1         -> 小圆、无孔
#
# ★ area_mm2 = 零件的**物理面积**（mm²）= 像素面积 × (mm_per_px)²。
#   这是区分"同为圆形"的底座与顶盖的**唯一**手段，而且：
#     · 与像素尺度无关（实测 2.5 与 4.0 px/mm 下算出 4379 / 4377，差 0.05%）
#     · 与料盘朝向无关
#   ★ 不要用"面积/槽面积"代替：底座填满它的槽 Ø65、顶盖也填满它的小槽 Ø25，
#     两者比值都是 0.6 左右，互相误判（实测踩过）。
#
#   合成图上实测的物理面积（名义值见括号）：
#     轮盘 4378 mm²（名义 4348）    底座 3244（名义 2762）
#     立柱 2386（名义 1275，胶囊槽留了富余，所以偏大）
#     顶盖  462（名义  287）
#   ★ 区间下方给得比上方宽：真实画面的分割会侵蚀边缘，面积偏小的可能更大。
#   ★ 这些是**初值**，必须用 calib/c6_part_features.py 在真实画面上标定后覆盖。
def default_templates() -> List[PartTemplate]:
    return [
        PartTemplate(
            part="P3_wheel", name="轮盘组件",
            ranges={
                # ★ 实度：辐条之间的空隙让凸包实度低于实心圆。
                #   注意实测（合成图、4 px/mm）轮盘 solidity=0.996 —— 因为
                #   外轮廓本身就是个完整圆，孔在内部，凸包与轮廓面积几乎相等。
                #   所以 **solidity 区分不了轮盘与底座**，别指望它。
                "solidity": (0.45, 1.0),
                # ★ 轮盘的**决定性**判据：辐条之间形成 8+ 个内孔（实测 9）。
                #   底座只有中心 1 个孔，顶盖 0 个。这条最可靠。
                "holes": (4.0, 16.0),
                # ★ circularity 区间要给得宽：实测轮盘 0.90（接近满圆），
                #   早期写 (0.10, 0.45) 太窄，在 6 px/mm 下把轮盘判成了底座。
                "circularity": (0.10, 0.98),
                "elongation": (1.0, 1.45),
                "area_mm2": (2400.0, 7000.0),
            },
        ),
        PartTemplate(
            part="P1_base", name="底座圆盘",
            ranges={
                "solidity": (0.88, 1.0),
                "holes": (0.0, 1.5),          # 中心一个孔
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
                "holes": (0.0, 1.5),
                "circularity": (0.05, 0.60),
                # 真实立柱约 85x15 = 1275 mm²；合成图按 85x16 画，实测约 1400 mm²
                "area_mm2": (400.0, 2800.0),
            },
        ),
        PartTemplate(
            part="P4_cap", name="顶盖铭牌",
            ranges={
                "circularity": (0.60, 1.05),
                "solidity": (0.88, 1.0),
                "holes": (0.0, 1.5),
                "elongation": (1.0, 1.30),
                # ★ 与底座同为圆形、同样无内孔，靠物理面积区分（顶盖小一个量级）
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

    做法：取该槽模板里 area_mm2 区间的**上界**（"零件完整"时的面积），
          除以槽的物理面积，再夹到 (0, 1]。
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
    return float(min(1.0, max(0.05, float(rng[1]) / slot_mm2)))


def _part_of_slot(tpl: PartTemplate, slot_id: str) -> bool:
    """模板是否属于该槽（靠 part 名与槽 id 的对应关系判断）。

    槽 id 形如 slot_wheel / slot_column / slot_cap / slot_base，
    零件名形如 P3_wheel / P2_column / P4_cap / P1_base —— 取尾部词比对。
    """
    sid = slot_id.lower().replace("slot_", "")
    return tpl.part.lower().endswith(sid) or sid in tpl.part.lower()
    """某个槽的物理面积（像素²），用作该槽 area_frac 的归一化分母。

    ★ 用 shoelace 公式对**槽的多边形**求面积，能正确处理透视与斜置胶囊
      （立柱槽是 91x28 @-44° 的胶囊，外接框会是它的两倍多，不能用）。
    """
    poly = tray_map.slot_polygon_px(slot_id, scale)
    x = poly[:, 0]; y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


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
