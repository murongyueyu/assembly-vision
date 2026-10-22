# -*- coding: utf-8 -*-
"""ABCDE接线辅助：绑定冻结来源，B只接受一次人工确认的同帧测区。"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

from .config import Config, load_config
from .core.depth import NativeHeightEstimator, compare_native_height, rectangle_mask


def assess_height(height_mm, baseline_mm, tolerance_mm, upper_limit_mm=None):
    """原下限继续生效；额外上限按未舍入数值比较，等于上限可通过。"""
    conforms = compare_native_height(height_mm, baseline_mm, tolerance_mm)
    if conforms is None:
        return None
    return bool(conforms and (upper_limit_mm is None or height_mm <= upper_limit_mm))


class Integration:
    def __init__(self, path, camera):
        self.path = Path(path).resolve()
        self.settings = load_config(str(self.path)).data
        if self.settings['association'] != 'manual_same_frame_cap_region':
            raise ValueError('集成候选仅支持人工同帧顶盖表面关联')
        bound = {}
        for name in ('sequence', 'native_reference'):
            target = (self.path.parent / self.settings[name]).resolve()
            if hashlib.sha256(target.read_bytes()).hexdigest() != self.settings[name + '_sha256']:
                raise ValueError('集成冻结来源hash不符：' + str(target))
            bound[name] = target
        self.sequence_path = str(bound['sequence'])
        native = copy.deepcopy(load_config(str(bound['native_reference'])).get('native_depth'))
        if native['baseline_accepted'] or native['measurement_surface'] != 'cap_region':
            raise ValueError('集成候选必须保持顶盖表面及未接受基线')
        assessment = self.settings.get('height_assessment', {})
        self.height_accepted = assessment.get('enabled', False) is True
        self.height_tolerance = None
        self.height_upper_limit = None
        self.height_min = self.height_max = None
        if self.height_accepted:
            self.height_tolerance = float(assessment['tolerance_mm'])
            maximum = float(assessment['max_tolerance_mm'])
            if (assessment['scope'] != self.settings['association'] or not assessment.get('authorization')
                    or not math.isfinite(maximum) or not math.isfinite(self.height_tolerance)
                    or not 0 < self.height_tolerance <= maximum):
                raise ValueError('受控高度判定范围或授权无效')
            baseline = float(native['baseline_height_mm'])
            self.height_min = baseline - self.height_tolerance
            self.height_max = baseline + self.height_tolerance
            if 'upper_limit_mm' in assessment:
                self.height_upper_limit = float(assessment['upper_limit_mm'])
                if not math.isfinite(self.height_upper_limit) or self.height_upper_limit < self.height_min:
                    raise ValueError('高度上限必须有限且不低于原下限')
                self.height_max = min(self.height_max, self.height_upper_limit)
        # ★ 仅复制B参数，不采用旧运行副本的相机控件，也绝不复用旧像素测区。
        native['reference_rect'] = native['target_rect'] = None
        if self.height_accepted:
            native['height_tolerance_mm'] = self.height_tolerance
        self.camera = Config(copy.deepcopy(camera.data))
        self.camera.set('depth.enabled', True)
        self.camera.set('depth.backend', 'realsense')
        self.camera.set('native_depth', native)
        self.estimator = NativeHeightEstimator(self.camera)
        self.result = None
        self.measurement = None
        self.reason = '待测（四步后按i选区）'

    @property
    def height_rule(self):
        if not self.height_accepted:
            return '基线未接受'
        return '%.3f～%.3f mm' % (self.height_min, self.height_max)

    def reset(self):
        self.result = self.measurement = None
        self.reason = '待测（四步后按i选区）'

    def measure(self, frame, stamp, reference, target, generation):
        """操作者在冻结原图选区；结论只属于该快照，不推断后续帧的表面。"""
        self.reset()
        shape = frame.left.shape[:2]
        if (frame.index != stamp.index or shape != stamp.shape or frame.scale != 1.0
                or frame.depth_points_mm is None or frame.depth_points_mm.shape != (*shape, 3)):
            raise ValueError('测量需要同帧同尺寸原生XYZ及原图')
        ref, roi = rectangle_mask(shape, reference), rectangle_mask(shape, target)
        if (ref & roi).any():
            raise ValueError('参考与顶盖测区重叠')
        cfg = Config(copy.deepcopy(self.camera.data))
        cfg.set('native_depth.reference_rect', list(reference))
        cfg.set('native_depth.target_rect', list(target))
        result = NativeHeightEstimator(cfg).measure_frame(frame)
        conforms = (assess_height(result['height_mm'], cfg.get('native_depth.baseline_height_mm'),
                                 self.height_tolerance, self.height_upper_limit)
                    if result['ok'] and self.height_accepted else None)
        self.result = dict(result, frame_index=stamp.index, captured_at=stamp.captured_at,
                           generation=generation, association='manual_same_frame_cap_region',
                           reference_rect=list(reference), target_rect=list(target),
                           baseline_accepted=self.height_accepted, baseline_source_accepted=False,
                           baseline_height_mm=cfg.get('native_depth.baseline_height_mm'),
                           height_tolerance_mm=self.height_tolerance, height_conforms=conforms,
                           height_min_mm=self.height_min, height_max_mm=self.height_max,
                           upper_limit_mm=self.height_upper_limit,
                           assessment_scope='仅人工同帧顶盖高度；不代表整机全部质量',
                           scope='仅该帧人工确认表面；不证明后续帧仍在同一位置')
        self.measurement = (frame, copy.deepcopy(self.result))
        return self.result

    def decorate(self, state, started):
        # ★ 数值有效和高度合格分列；四步永远不产生「装配全部合格」。
        if state.level == 'ok':
            state.level = 'warn'
        state.title = '取件%d/4 · 质量待判' % state.step
        result = self.result
        lines = ['开工：' + ('已通过D并开工' if started else '等待D稳定放行'),
                 '取件顺序：%d/4' % state.step]
        if result:
            lines.append('快照测量：' + ('有效 %.4f mm' % result['height_mm'] if result['ok'] else '无效'))
            lines.append('测量帧：%s（人工快照）' % result['frame_index'])
            if not result['ok']:
                lines.append(result['reason'])
        else:
            lines.append('测量：' + self.reason)
        if not self.height_accepted:
            lines.extend(['高度符合性：无法判断', '原因：基线未正式接受'])
        elif result and result['height_conforms'] is not None:
            passed = result['height_conforms']
            lines.append('高度符合性：' + ('合格' if passed else '不合格'))
            state.title = '取件%d/4 · 高度%s' % (state.step, '合格' if passed else '不合格')
            if not passed:
                state.level = 'error'
                state.banner = '高度不合格：超出受控范围'
            elif state.step == 4 and state.level != 'error':
                state.level = 'ok'
        else:
            lines.append('高度符合性：待测')
        if self.height_accepted:
            lines.extend(['上限：%.3f mm（含）' % self.height_max,
                          '下限：%.3f mm（含）' % self.height_min])
        lines.extend(['紧固性：队友负责', '安装等其他质量：未验证'])
        state.detail = lines + state.detail
        state.stats['B基线'] = '仅受控高度接受' if self.height_accepted else '未接受'
        state.stats['测量模式'] = '单帧人工关联'
