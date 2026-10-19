# -*- coding: utf-8 -*-
"""只读 dev 对比孔面积阈值，再另存 A 候选；不覆盖冻结模型与正式配置。"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from part_dataset import (ROOT, DEFAULT_DATA, Config, load_samples, load_config, read_json,
                          write_json, file_hash, segment_variant, extract_features,
                          classifier_from_templates, predict, score_frame, PARTS, sample_scale)
from assemvision.core.features import count_holes
from replay_wheel_holes import hole_areas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-area', type=float, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.min_area) or args.min_area <= 0:
        raise ValueError('孔面积门槛必须为有限正数')
    args.output.mkdir(parents=True, exist_ok=False)
    original = DEFAULT_DATA/'offline_20261017_visible_follow_v5.json'
    artifact = read_json(original)
    wp = load_config(str(ROOT/'config/workpiece.yaml'))
    policy = Config(artifact['policy'])
    classifier = classifier_from_templates(artifact['templates'], policy)
    thresholds = sorted(set([3, 6, 10, 20, 30, args.min_area]))
    totals = {k: {p: dict(tp=0, fp=0, fn=0) for p in PARTS} for k in thresholds}
    rows = []
    for sample in load_samples(DEFAULT_DATA, 'dev'):
        image = sample['image']
        mask, blobs, _ = segment_variant(image, wp, policy, artifact['variant'])
        truth = [o for o in sample['annotation']['objects'] if o['part'] in PARTS]
        row = dict(image=str(sample['path']), sha256=file_hash(sample['path']), truth=truth, sweep={})
        for k in thresholds:
            predictions = []
            for blob in blobs:
                features = extract_features(blob, image, mask, sample_scale(sample))
                features['holes'] = float(count_holes(blob.contour, mask, min_area_px=k))
                pid, score, margin = predict(classifier, features, policy)
                predictions.append(dict(part=pid, score=score, margin=margin,
                    bbox=list(blob.bbox), features=features, hole_areas=hole_areas(blob.contour, mask)))
            counts, _, _ = score_frame(truth, predictions, policy.get('evaluation.bbox_iou'))
            for part in PARTS:
                for key in counts[part]:
                    totals[k][part][key] += counts[part][key]
            row['sweep'][k] = predictions
        rows.append(row)
    write_json(args.output/'dev_sweep.json', rows)
    write_json(args.output/'dev_summary.json', dict(images=len(rows), totals=totals,
        scope='原开发集参数敏感性回归；不读取 test，不是新增独立验收'))
    candidate = copy.deepcopy(artifact)
    candidate['policy']['experiment']['hole_min_area_px'] = args.min_area
    policy_path = args.output/'policy.json'
    write_json(policy_path, candidate['policy'])
    candidate['policy_sha256'] = file_hash(policy_path)
    # 当前配置仅作为绑定来源；正式文件不改。模型旧分割覆盖保持不变。
    candidate['workpiece_sha256'] = file_hash(ROOT/'config/workpiece.yaml')
    candidate['parent_model_sha256'] = file_hash(original)
    candidate['candidate_note'] = ('仅启用孔数面积过滤；保留全部旧模板区间/权重、分割、尺寸必检和七帧0.85门槛；'
                                 '旧孔数区间作为保守兼容带，以dev及历史图回归检查，不重新拟合或使用失败图扩带')
    write_json(args.output/'model.json', candidate)
    print(dict(images=len(rows), totals=totals, model_sha256=file_hash(args.output/'model.json')), flush=True)


if __name__ == '__main__':
    main()
