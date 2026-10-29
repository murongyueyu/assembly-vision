# -*- coding: utf-8 -*-
"""经授权的A v5复用适配层：只组合冻结函数，不修改识别/追踪算法。"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import Config
from .part_tracker import IdentityConfirmation, PartTracker
from .checks import IdentityObject
from tools.part_dataset import (segment_variant, extract_features, predict,
                                classifier_from_templates, visible_hand_candidates, sample_scale)


class FrozenPartIdentity:
    def __init__(self, wp, camera, config_path, settings):
        cfg = settings['identity']
        base = Path(config_path).resolve().parent
        self.model_path = (base / cfg['model']).resolve()
        policy_path = (base / cfg['policy']).resolve()
        for path, expected in ((self.model_path, cfg['model_sha256']),
                               (policy_path, cfg['policy_sha256']),
                               (Path(wp.path), cfg['workpiece_sha256']),
                               (Path(camera.path), cfg['camera_sha256'])):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('D冻结来源hash不符：' + str(path))
        self.artifact = json.loads(self.model_path.read_text(encoding='utf-8'))
        policy = json.loads(policy_path.read_text(encoding='utf-8'))
        if self.artifact['policy'] != policy or self.artifact['policy_sha256'] != cfg['policy_sha256']:
            raise ValueError('v5模型内嵌policy与冻结policy不一致')
        self.wp, self.policy = wp, Config(policy)
        self.cfg = copy.deepcopy(wp.get('tracking', {}) or {})
        self.cfg.update(self.policy.get('experiment.tracking_overrides', {}))
        if self.cfg.get('backend') != 'detection':
            raise ValueError('v5适配层只支持冻结逐帧检测后端')
        self.clock = 0.0
        self.reset()

    def reset(self):
        self.tracker = PartTracker(self.cfg, clock=lambda: self.clock)
        self.classifier = classifier_from_templates(self.artifact['templates'], self.policy)
        self.confirmation = IdentityConfirmation(self.policy.get('experiment.confirmation', {}))
        self.last_index, self.last_capture = -1, None

    def observe(self, image, stamp, intrinsics):
        if stamp.index <= self.last_index:
            raise ValueError('A身份输入帧号必须严格前进')
        if self.last_capture is not None and stamp.captured_at <= self.last_capture:
            raise ValueError('A身份采集时刻必须前进')
        mmpp = sample_scale({'meta': {'intrinsics': intrinsics,
                                     'distance_mm': self.policy.get('capture.distance_mm')}})
        if mmpp is None or not np.isfinite(mmpp) or mmpp <= 0:
            raise ValueError('缺少当前SDK内参/冻结工作距离')
        self.clock = stamp.captured_at
        mask, blobs, result = segment_variant(image, self.wp, self.policy, self.artifact['variant'])
        labels, predictions = [], []
        for blob in blobs:
            blob.features = extract_features(blob, image, mask, mmpp)
            pid, score, margin = predict(self.classifier, blob.features, self.policy)
            labels.append(pid)
            predictions.append(dict(part=pid, score=score, margin=margin, bbox=list(blob.bbox),
                                    area_px=float(blob.area_px), features=blob.features))
        confirmed = self.confirmation.update(predictions, self.clock)
        continued = self.tracker.known_continuations(predictions)
        admitted = [passed or i in continued for i, passed in enumerate(confirmed)]
        new_candidates = {id(b) for b, pid, passed in zip(blobs, labels, confirmed) if pid and passed}
        observed = [b for b, passed in zip(blobs, admitted) if passed]
        identified = {pid for pid, passed in zip(labels, admitted) if pid and passed}
        unidentified = any((not pid or not passed)
                           and self.tracker.min_area_px <= b.area_px <= self.tracker.max_area_px
                           for pid, b, passed in zip(labels, blobs, admitted))
        partial = visible_hand_candidates(mask, result.glove_mask if result.glove_bbox is not None else None,
                                          self.policy)
        if partial:
            normal_boxes = {tuple(b.bbox) for b in observed}
            unidentified = unidentified or any(tuple(b.bbox) not in normal_boxes for b in partial)
        tracks = self.tracker.update(image, observed, detect=True,
            classify=lambda features: predict(self.classifier, features, self.policy)[:2],
            seen_ids=identified, unidentified=unidentified, occluder=result.glove_mask,
            allow_new=lambda b: id(b) in new_candidates, partial_blobs=partial)
        # ★ 只有本帧可见且有唯一相同轮廓的轨迹才提供身份；未知原始块仍保留用于冲突检查。
        visible = [t for t in tracks if t.visible and t.lost_frames == 0
                   and t.detail.get('visible_observed') == 1.0 and t.contour is not None]
        objects, used = [], set()
        for blob in blobs:
            matches = [(i, t) for i, t in enumerate(visible)
                       if np.array_equal(t.contour, blob.contour)]
            if len(matches) == 1:
                i, track = matches[0]
                used.add(i)
                objects.append(IdentityObject(track.part_id, blob.contour.copy(), True, True,
                                               'partial' if track.detail.get('partial_observed') else 'detected'))
            else:
                objects.append(IdentityObject('', blob.contour.copy()))
        for i, track in enumerate(visible):
            if i not in used:
                objects.append(IdentityObject(track.part_id, track.contour.copy(), True, True, 'partial'))
        self.last_index, self.last_capture = stamp.index, stamp.captured_at
        return objects, dict(predictions=predictions, mask=mask, tracks=copy.deepcopy(tracks), objects=objects)
