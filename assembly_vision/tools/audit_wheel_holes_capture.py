# -*- coding: utf-8 -*-
"""同批新原图做冻结/候选 A 对照；不重用槽位，不修改任何正式模型。"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from part_dataset import (ROOT, Config, load_config, read_json, write_json, file_hash,
    segment_variant, extract_features, classifier_from_templates, predict, PARTS)
from assemvision.core.part_tracker import IdentityConfirmation
from assemvision.imgio import imread_safe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    target = args.capture/'comparison.json'
    if target.exists():
        raise FileExistsError(target)
    capture = read_json(args.capture/'capture.json')
    baseline_path = ROOT.parent/'ai_测试工具/阶段1数据/offline_20261017_visible_follow_v5.json'
    models = dict(frozen=read_json(baseline_path), candidate=read_json(args.model))
    wp = load_config(str(ROOT/'config/workpiece.yaml'))
    frames, summary = [], {}
    for label, artifact in models.items():
        policy = Config(artifact['policy'])
        classifier = classifier_from_templates(artifact['templates'], policy)
        confirmation = IdentityConfirmation(policy.get('experiment.confirmation'))
        rows, admitted, raw, ranked, rejected_dims = [], Counter(), Counter(), Counter(), Counter()
        for record in capture['samples']:
            path = args.capture/record['raw']
            if file_hash(path) != record['sha256']:
                raise ValueError('现场原图hash变化')
            image = imread_safe(str(path))
            mask, blobs, _ = segment_variant(image, wp, policy, artifact['variant'])
            mmpp = policy.get('capture.distance_mm')/capture['metadata']['intrinsics']['fx']
            predictions = []
            for blob in blobs:
                features = extract_features(blob, image, mask, mmpp)
                pid, score, margin = predict(classifier, features, policy)
                top = classifier.rank(features)[0][0]
                ranked[top] += 1
                if pid is None:
                    rejected_dims[top] += 1
                predictions.append(dict(part=pid, score=score, margin=margin, bbox=list(blob.bbox),
                    area_px=float(blob.area_px), features=features, top_before_gates=top))
                if pid:
                    raw[pid] += 1
            confirmed = confirmation.update(predictions, record['captured_at'])
            for p, ok in zip(predictions, confirmed):
                p['confirmed'] = ok
                if ok:
                    admitted[p['part']] += 1
            rows.append(dict(raw=record['raw'], predictions=predictions,
                             mask_sha256=__import__('hashlib').sha256(mask.tobytes()).hexdigest()))
        frames.append(rows)
        summary[label] = dict(frames=len(rows), raw_labels=dict(raw), confirmed=dict(admitted),
            top_before_gates=dict(ranked), rejected_after_ranking=dict(rejected_dims),
            scores={p: dict(median=float(np.median(vals)), minimum=min(vals), maximum=max(vals))
                    for p in PARTS if (vals:=[d['score'] for row in rows for d in row['predictions'] if d['top_before_gates']==p])})
    changed_other, changed_masks = 0, 0
    for old, new in zip(*frames):
        changed_masks += old['mask_sha256'] != new['mask_sha256']
        assert len(old['predictions']) == len(new['predictions'])
        for before, after in zip(old['predictions'], new['predictions']):
            if before['top_before_gates'] != 'P3_wheel':
                changed_other += before['part'] != after['part']
            assert before['bbox'] == after['bbox']
            assert {k:v for k,v in before['features'].items() if k!='holes'} == {k:v for k,v in after['features'].items() if k!='holes'}
    result = dict(scene=capture['scene'], source_model_sha256=capture['model_sha256'],
        masks_changed=changed_masks, nonwheel_labels_changed=changed_other, summary=summary,
        frozen=frames[0], candidate=frames[1])
    write_json(target, result)
    print(dict(scene=capture['scene'], masks_changed=changed_masks,
               nonwheel_labels_changed=changed_other, summary=summary), flush=True)


if __name__ == '__main__':
    main()
