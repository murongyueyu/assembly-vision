# -*- coding: utf-8 -*-
"""碎孔身份特征回归：冻结兼容、结构孔保留、转角及尺寸拒识。"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assemvision.core.features import contour_features, count_holes
from part_dataset import (Config, ROOT, load_config, segment_variant, extract_features,
                          classifier_from_templates, predict)


def outer(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


def main():
    mask = np.zeros((240, 240), np.uint8)
    cv2.circle(mask, (120, 120), 90, 255, -1)
    # 九个可分辨结构孔，另加二像素碎孔；没有修改原金属/空洞像素。
    cv2.circle(mask, (120, 120), 8, 0, -1)
    for angle in range(0, 360, 45):
        point = tuple(np.rint(np.array([120, 120])+50*np.array([
            np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])).astype(int))
        cv2.circle(mask, point, 6, 0, -1)
    mask[100, 100:102] = 0
    frozen = mask.copy()
    contour = outer(mask)
    assert count_holes(contour, mask) == 10
    old = contour_features(contour, mask)
    new = contour_features(contour, mask, hole_min_area_px=10)
    assert old['holes'] == 10 and new['holes'] == 9
    assert {k:v for k,v in old.items() if k != 'holes'} == {k:v for k,v in new.items() if k != 'holes'}
    assert np.array_equal(mask, frozen)
    boundary = np.zeros((50, 50), np.uint8)
    boundary[5:45, 5:45] = 255
    boundary[20:22, 20:23] = 0  # 六个黑像素，对应孔轮廓面积恰为10px²。
    assert count_holes(outer(boundary), boundary, min_area_px=10) == 1
    assert count_holes(outer(boundary), boundary, min_area_px=10.01) == 0
    for angle in (0, 30, 60, 90, 135, 180, 270):
        turned = cv2.warpAffine(mask, cv2.getRotationMatrix2D((120, 120), angle, 1),
                               (240, 240), flags=cv2.INTER_NEAREST)
        assert contour_features(outer(turned), turned, hole_min_area_px=10)['holes'] == 9, angle
    # 减少/增加一个结构孔仍反映在特征中，不无差别抹平拓扑。
    missing = mask.copy()
    cv2.circle(missing, (120, 120), 8, 255, -1)
    extra = mask.copy()
    cv2.circle(extra, (145, 120), 5, 0, -1)
    assert contour_features(outer(missing), missing, hole_min_area_px=10)['holes'] == 8
    assert contour_features(outer(extra), extra, hole_min_area_px=10)['holes'] == 10
    for bad in (0, -1, float('nan'), float('inf'), True, '10', None, [10]):
        try:
            contour_features(contour, mask, hole_min_area_px=bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('非法孔面积门槛未拒绝：%r' % (bad,))
    # 真分割链：候选只改特征，默认输出、掩膜、外轮廓和其他特征逐项相同。
    wp = load_config(str(ROOT/'config/workpiece.yaml'))
    policy = load_config(str(ROOT/'config/part_dataset.yaml'))
    image = cv2.cvtColor(np.full((240, 240, 3), (20, 150, 180), np.uint8), cv2.COLOR_HSV2BGR)
    image[mask == 0] = 0
    old_mask, old_blobs, _ = segment_variant(image, wp, policy, 'brass')
    candidate = Config(copy.deepcopy(policy.data))
    candidate.set('experiment.hole_min_area_px', 10)
    new_mask, new_blobs, _ = segment_variant(image, wp, candidate, 'brass')
    assert np.array_equal(old_mask, new_mask) and len(old_blobs) == len(new_blobs) == 1
    old_features = extract_features(old_blobs[0], image, old_mask, .7)
    new_features = extract_features(new_blobs[0], image, new_mask, .7)
    assert {k:v for k,v in old_features.items() if k != 'holes'} == {k:v for k,v in new_features.items() if k != 'holes'}
    # 未知尺寸与其他类不会因为孔项变满分而绕过尺寸必检。
    gate_policy = Config({'experiment': {'min_score': .55, 'min_margin': .1,
        'holes_tol': 1, 'required_features': ['length_mm', 'short_side_mm']}})
    templates = {'P3_wheel': {'ranges': {'length_mm': [55, 80], 'short_side_mm': [55, 80], 'holes': [8, 10]}}}
    classifier = classifier_from_templates(templates, gate_policy)
    for values in ((70,70), (90,70), (70,20)):
        pid, _, _ = predict(classifier, dict(length_mm=values[0], short_side_mm=values[1], holes=9), gate_policy)
        assert (pid == 'P3_wheel') == (values == (70,70))
    print('轮盘碎孔：默认兼容、九结构孔、七转角、拓扑变化、掩膜不变与尺寸必检通过')


if __name__ == '__main__':
    main()
