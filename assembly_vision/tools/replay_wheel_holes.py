# -*- coding: utf-8 -*-
"""A 碎孔开发回放：保留冻结预测并比较孔数敏感性，不读取槽位作为身份。

例：python tools/replay_wheel_holes.py --output NEW_DIR
输出拒绝覆盖；旧失败 / 成功序列均为开发回归，不是独立验收。
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import cv2
import numpy as np

from part_dataset import (ROOT, Config, load_config, read_json, write_json, file_hash,
                          segment_variant, extract_features, classifier_from_templates, predict)
from assemvision.core.features import count_holes
from assemvision.core.part_tracker import IdentityConfirmation
from assemvision.imgio import imread_safe


def hole_areas(contour, mask):
    x, y, w, h = cv2.boundingRect(contour)
    found = cv2.findContours(mask[y:y+h, x:x+w], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return sorted(float(cv2.contourArea(c)) for c, hi in zip(found[-2], found[-1][0])
                  if hi[3] != -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT.parent/'ai_测试工具/阶段1数据/offline_20261017_visible_follow_v5.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    artifact = read_json(args.model)
    policy = Config(artifact['policy'])
    wp = load_config(str(ROOT/'config/workpiece.yaml'))
    classifier = classifier_from_templates(artifact['templates'], policy)
    cases = [('old_success', 'D模块联检/20261018_01/live_correct_01', 'frame_*.png'),
             ('square_failure', 'E模块流程/live_reorientation_01', 'raw_*.png'),
             ('diamond_failure', 'E模块流程/live_reorientation_02', 'raw_*.png')]
    summary = {}
    for name, directory, pattern in cases:
        folder = ROOT.parent/'ai_测试工具'/directory
        saved = read_json(folder/'report.json')
        intr = saved['metadata']['intrinsics']
        mmpp = policy.get('capture.distance_mm')/intr['fx']
        thresholds = [3, 6, 10, 20, 30]
        confirmations = {k: IdentityConfirmation(policy.get('experiment.confirmation')) for k in thresholds}
        actual_confirmation = IdentityConfirmation(policy.get('experiment.confirmation'))
        errors = []
        rows = []
        for index, path in enumerate(sorted(folder.glob(pattern))):
            image = imread_safe(str(path))
            mask, blobs, _ = segment_variant(image, wp, policy, artifact['variant'])
            by_threshold = {k: [] for k in thresholds}
            observations = []
            actual_predictions = []
            for blob in blobs:
                feats = extract_features(blob, image, mask, mmpp)
                pid, score, margin = predict(classifier, feats, policy)
                actual_predictions.append(dict(part=pid, score=score, margin=margin,
                                               bbox=list(blob.bbox), area_px=float(blob.area_px)))
                ranks = classifier.rank(feats)
                is_wheel = ranks and ranks[0][0] == 'P3_wheel'
                item = dict(bbox=list(blob.bbox), features=feats, sweep={})
                if is_wheel:
                    item['hole_areas'] = hole_areas(blob.contour, mask)
                for threshold in thresholds:
                    trial = copy.copy(feats)
                    trial['holes'] = float(count_holes(blob.contour, mask, min_area_px=threshold))
                    pid, score, margin = predict(classifier, trial, policy)
                    prediction = dict(part=pid, score=score, margin=margin, bbox=list(blob.bbox), area_px=float(blob.area_px))
                    by_threshold[threshold].append(prediction)
                    if threshold == 3:
                        old = next((p for p in saved['samples'][index]['predictions']
                                    if p['bbox'] == list(blob.bbox)), None)
                        if old is None:
                            raise ValueError('冻结候选轮廓不一致：'+str(path))
                        errors.append(abs(score-old['score']))
                    if is_wheel:
                        item['sweep'][threshold] = dict(prediction, holes=trial['holes'],
                            subscores=classifier.templates[[t.part for t in classifier.templates].index('P3_wheel')].score(trial)[1])
                if is_wheel:
                    observations.append(item)
            actual_passed = actual_confirmation.update(actual_predictions, saved['samples'][index]['captured_at'])
            for item in observations:
                pi = next(i for i,p in enumerate(actual_predictions) if p['bbox'] == item['bbox'])
                item['actual'] = dict(actual_predictions[pi], confirmed=actual_passed[pi],
                                      holes=item['features']['holes'])
            for threshold in thresholds:
                passed = confirmations[threshold].update(by_threshold[threshold], saved['samples'][index]['captured_at'])
                for item in observations:
                    prediction = item['sweep'][threshold]
                    pi = next(i for i, p in enumerate(by_threshold[threshold]) if p['bbox'] == prediction['bbox'])
                    prediction['confirmed'] = passed[pi]
            rows.append(dict(file=path.name, sha256=file_hash(path), wheels=observations))
        summary[name] = dict(frames=len(rows), frozen_score_max_abs_error=max(errors, default=0),
            actual=dict(score_median=float(np.median([w['actual']['score'] for r in rows for w in r['wheels']])),
                        confirmed=sum(w['actual']['confirmed'] for r in rows for w in r['wheels']),
                        over_confirmation=sum(w['actual']['score'] >= .85 for r in rows for w in r['wheels'])),
            sweep={k: dict(
            wheel_observations=sum(len(r['wheels']) for r in rows),
            score_median=float(np.median([w['sweep'][k]['score'] for r in rows for w in r['wheels']])),
            over_confirmation=sum(w['sweep'][k]['score'] >= .85 for r in rows for w in r['wheels']),
            confirmed=sum(w['sweep'][k]['confirmed'] for r in rows for w in r['wheels'])) for k in thresholds})
        write_json(args.output/(name+'.json'), rows)
        print(name, summary[name], flush=True)
    write_json(args.output/'summary.json', dict(model_sha256=file_hash(args.model),
        policy_sha256=artifact['policy_sha256'], workpiece_sha256=file_hash(ROOT/'config/workpiece.yaml'),
        scope='开发回归与单因素敏感性比较；未改掩膜/分割/模板/确认门槛；非独立验收', cases=summary))


if __name__ == '__main__':
    main()
