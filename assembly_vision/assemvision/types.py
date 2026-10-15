# -*- coding: utf-8 -*-
"""全项目共用的数据类。

设计原则：线程之间只传这些小对象和 numpy 掩膜，不传大图。
所有坐标一律写清单位：px = 像素，mm = 料盘毫米。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 判定级别
OK = "ok"
WARN = "warn"
ERROR = "error"
IDLE = "idle"

# 槽位判定
SLOT_OK = "OK"
SLOT_NG = "NG"
SLOT_SUSPECT = "SUSPECT"


@dataclass
class Frame:
    """一次采集的产物。实时通道只需要 left/right 小图；抽检时才带 left_full。"""
    index: int
    ts: float
    left: Optional[np.ndarray] = None        # BGR，实时分辨率
    right: Optional[np.ndarray] = None
    left_full: Optional[np.ndarray] = None   # BGR，全分辨率（可能为 None）
    right_full: Optional[np.ndarray] = None  # 全分辨率右眼（低频深度用）
    scale: float = 1.0                       # 实时图相对 left_full 的缩放比

    @property
    def size(self) -> Tuple[int, int]:
        if self.left is None:
            return (0, 0)
        h, w = self.left.shape[:2]
        return (w, h)


@dataclass
class Blob:
    """一个分割出来的前景连通域。"""
    contour: np.ndarray                      # (N,1,2) int32
    bbox: Tuple[int, int, int, int]          # x, y, w, h（像素）
    centroid: Tuple[float, float]            # 像素
    area_px: float
    features: Dict[str, float] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        return self.centroid


@dataclass
class SegResult:
    """分割结果。掩膜分辨率与传入的图像一致。"""
    ts: float = 0.0
    scale: float = 1.0                                  # 相对全分辨率的缩放
    brass_mask: Optional[np.ndarray] = None
    # ★ "金属"掩膜：**不靠色相**，判据是"亮 + 不是绿垫"。
    #   为什么需要它（c5 实测踩过）：装配体上的**球头只有 Ø5mm**，
    #   在 0.86mm/px 下只有 6 像素宽，边缘与绿垫抗锯齿混色后
    #   色相从黄铜的 H=30 跳到 **H=59**（被绿垫染成淡绿），
    #   于是被 brass 的 `h_max=45` 整条挡掉 —— 辐条一个都数不出来。
    #   实测同一帧：brass 掩膜在 28mm 处只有 8 峰但角度抖 33°（凑出来的），
    #   而 metal 掩膜在 30~35mm 带 **8 峰、角度抖仅 5.9°**（真的）。
    #   见 docs/踩坑实录.md §2.13。
    metal_mask: Optional[np.ndarray] = None
    glove_mask: Optional[np.ndarray] = None
    brass_blobs: List[Blob] = field(default_factory=list)
    glove_bbox: Optional[Tuple[int, int, int, int]] = None
    glove_area_px: float = 0.0
    hand_on: Dict[str, bool] = field(default_factory=dict)   # slot_id -> 手套是否压在该槽
    detail: Dict[str, float] = field(default_factory=dict)   # 耗时等调试信息

    def glove_missing(self) -> bool:
        return self.glove_bbox is None


@dataclass
class SlotResult:
    """一个槽位的判定结果。"""
    slot_id: str
    part: str
    present: bool = False
    coverage: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)
    verdict: str = SLOT_SUSPECT
    reasons: List[str] = field(default_factory=list)
    roi_px: Optional[np.ndarray] = None      # ROI 多边形（用于绘制）


@dataclass
class SpokeResult:
    """辐条计数结果（本项目唯一的重点指标）。"""
    count: int = 0
    required: int = 8
    curve: Optional[np.ndarray] = None       # 角度 -> 覆盖率曲线
    peaks_deg: List[float] = field(default_factory=list)
    missing_deg: List[float] = field(default_factory=list)
    ok: bool = False
    # ★ 峰角度的均布抖动（度）。`inf` = 峰数不等于 required，算不出抖动。
    #   实测判据：真辐条抖动 **5.9°**，凑出来的 8 峰抖动 **33°**。
    #   `count_spokes` 用它来**挑最好的半径带**（见 polar.py 的 docstring）——
    #   因为球头只有 Ø5mm，落进哪个子带是碰运气，但真辐条的角度是均布的。
    jitter_deg: float = float("inf")


@dataclass
class Event:
    """流程事件（取件 / 装配 / 核验）。"""
    ts: float
    kind: str                                # pick | assemble | inspect
    part: Optional[str] = None
    slot: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Verdict:
    """给界面看的结论。"""
    level: str = IDLE                        # ok | warn | error | idle
    title: str = ""
    detail: List[str] = field(default_factory=list)
    step: int = 0
    total: int = 0


@dataclass
class AssemblyResult:
    """装配体检查结果（要求④）。"""
    complete: bool = False
    verdict: str = SLOT_SUSPECT
    missing: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    ring_coverage: float = 0.0
    hub_coverage: float = 0.0
    cap_coverage: float = 0.0
    cap_holes: int = -1
    wheel_height_mm: Optional[float] = None
    height_baseline_mm: Optional[float] = None
    spoke: Optional[SpokeResult] = None
    center_px: Optional[Tuple[float, float]] = None
    outer_radius_px: float = 0.0


@dataclass
class PoseState:
    """位姿自检结果。"""
    ok: bool = True
    max_shift_px: float = 0.0
    warn: bool = False
    fail: bool = False
    markers_found: int = 0
    scale_change: float = 0.0                # 对角线长度变化比例（高度变化）
    edge_ratio: float = 1.0                  # 对边长度比（倾斜变化）
    message: str = ""
