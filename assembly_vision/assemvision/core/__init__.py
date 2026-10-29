# -*- coding: utf-8 -*-
"""核心算法层。只吃 numpy 数组与数据类，不碰相机、不碰界面、不依赖标定文件存在。"""
from __future__ import annotations

from .segment import Segmenter
from .features import contour_features, fit_circle, min_area_rect, count_holes
from .polar import count_spokes, polar_profile, count_peaks
from .roi import TrayMap, coverage_ratio, polygon_mask, circle_polygon, capsule_polygon
from .checks import Qualifier
from .fsm import PreconditionNet

__all__ = [
    "Segmenter", "contour_features", "fit_circle", "min_area_rect", "count_holes",
    "count_spokes", "polar_profile", "count_peaks",
    "TrayMap", "coverage_ratio", "polygon_mask", "circle_polygon", "capsule_polygon",
    "Qualifier", "PreconditionNet",
]
