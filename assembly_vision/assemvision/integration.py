# -*- coding: utf-8 -*-
"""ABCDE接线辅助：绑定冻结来源，B只接受一次人工确认的同帧测区。

★ 2026-10-28 扩展：高度判定从「单一顶盖基线」改为**分阶段**（stage）。
  每个 stage 绑定一个取件进度 `at_step`、一条基线/容差和一份授权：
    - 2/4：底座+立柱装好 → 测立柱顶端，64±3 mm（61.000～67.000）
    - 4/4：整机装好     → 测顶盖中心圆盘，65±3 mm（62.000～68.000）
  没有 `height_stages` 的旧配置按单阶段（at_step=4、顶盖）处理，
  数值路径与旧版逐字节一致（`height_assessment` 仍然是合法写法）。
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

from .config import Config, load_config
from .core.depth import NativeHeightEstimator, compare_native_height, rectangle_mask
from .core.length import measure_length as measure_length_image

DEFAULT_STEP = 4


def assess_height(height_mm, baseline_mm, tolerance_mm, upper_limit_mm=None):
    """原下限继续生效；额外上限按未舍入数值比较，等于上限可通过。"""
    conforms = compare_native_height(height_mm, baseline_mm, tolerance_mm)
    if conforms is None:
        return None
    return bool(conforms and (upper_limit_mm is None or height_mm <= upper_limit_mm))


def build_height_stages(settings, native_baseline, association):
    """把配置解析成 stage 列表。任何一处不合规都抛 ValueError，绝不静默降级。

    一个 stage 的字段：
        name/label   机器名与界面名（如 column/立柱）
        at_step      允许测量的取件进度（1..4，且各 stage 不得重复）
        target       目标区提示词（COLUMN / CAP），只影响选区窗口标题
        enabled      是否启用判定；False 时只测量不判合格
        baseline_mm/tolerance_mm/max_tolerance_mm/upper_limit_mm/authorization
    """
    raw_stages = settings.get('height_stages')
    if raw_stages is None:
        # ★ 旧配置兼容路径：height_assessment 单条即一个 stage。
        assessment = dict(settings.get('height_assessment', {}) or {})
        assessment.setdefault('name', 'cap')
        assessment.setdefault('label', '顶盖')
        assessment.setdefault('at_step', DEFAULT_STEP)
        assessment.setdefault('target', 'CAP')
        raw_stages = [assessment]
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError('高度阶段配置必须是非空列表')

    stages, names, steps = [], set(), set()
    for index, raw in enumerate(raw_stages):
        if not isinstance(raw, dict):
            raise ValueError('高度阶段 #%d 不是映射' % index)
        name = str(raw.get('name', '')).strip()
        label = str(raw.get('label', name)).strip() or name
        if not name or name in names:
            raise ValueError('高度阶段名称缺失或重复：%r' % (name,))
        names.add(name)
        at_step = raw.get('at_step', DEFAULT_STEP)
        if type(at_step) is not int or not 1 <= at_step <= 4 or at_step in steps:
            raise ValueError('高度阶段取件进度必须为1..4且不重复：%r' % (at_step,))
        steps.add(at_step)
        stage = dict(name=name, label=label, at_step=at_step,
                     target=str(raw.get('target', 'CAP')).strip().upper() or 'CAP',
                     mode=str(raw.get('mode', 'height')).strip().lower() or 'height',
                     auto_zone=False,
                     accepted=raw.get('enabled', False) is True,
                     scope=raw.get('scope'), authorization=raw.get('authorization'),
                     baseline_mm=None, tolerance_mm=None, upper_limit_mm=None,
                     min_mm=None, max_mm=None, min_points=None, min_valid_ratio=None)
        # ★ 2026-10-29：阶段有两种量法
        #     height —— B 的深度法向高度（4/4 顶盖仍在用）
        #     length —— 2D 轮廓长边（立柱横放量长度，不依赖深度）
        if stage['mode'] not in ('height', 'length'):
            raise ValueError('高度阶段 %s 的 mode 只能是 height / length' % name)
        # ★ 自动长度框：按 I 后不拉人工框，直接用固定检测区 + 2 秒稳定门。
        auto_zone = raw.get('auto_zone', False)
        if type(auto_zone) is not bool:
            raise ValueError('高度阶段 %s 的 auto_zone 必须是布尔值' % name)
        if auto_zone and stage['mode'] != 'length':
            raise ValueError('只有 length 阶段可以使用 auto_zone')
        stage['auto_zone'] = auto_zone
        # ★ 可选的深度门禁覆盖：立柱顶端只有十几个像素、且是高反光小面，
        #   B 的默认门禁（≥50 点、有效率 ≥0.6）在 10×10 的框里必然过不去。
        #   只允许**单个阶段**收紧/放宽，不动 camera.yaml 与全局判据。
        if raw.get('min_points') is not None:
            points = raw['min_points']
            if type(points) is not int or not 3 <= points <= 5000:
                raise ValueError('高度阶段 %s 的 min_points 必须是 3..5000 的整数' % name)
            stage['min_points'] = points
        if raw.get('min_valid_ratio') is not None:
            ratio = float(raw['min_valid_ratio'])
            if not math.isfinite(ratio) or not 0 < ratio <= 1:
                raise ValueError('高度阶段 %s 的 min_valid_ratio 必须在 (0,1]' % name)
            stage['min_valid_ratio'] = ratio
        # ★ 2026-10-29：长度阶段的现场标定比例（可选）。
        #   TrayMap 推出来的 0.731 mm/px 与卡尺实测差 ~12%（标定 homography 过期），
        #   长度判定必须用**现场标定值**；不写则沿用 TrayMap（并记录来源，便于排查）。
        if raw.get('mm_per_px') is not None:
            mmpp = float(raw['mm_per_px'])
            if not math.isfinite(mmpp) or not 0.05 <= mmpp <= 5.0:
                raise ValueError('长度阶段 %s 的 mm_per_px 必须在 0.05..5.0' % name)
            stage['mm_per_px'] = mmpp
        if stage['accepted']:
            if stage['scope'] != association or not stage['authorization']:
                raise ValueError('高度阶段 %s 的判定范围或授权无效' % name)
            tolerance = float(raw['tolerance_mm'])
            maximum = float(raw['max_tolerance_mm'])
            if (not math.isfinite(tolerance) or not math.isfinite(maximum)
                    or not 0 < tolerance <= maximum):
                raise ValueError('高度阶段 %s 的容差无效' % name)
            baseline = float(raw.get('baseline_mm', native_baseline))
            if not math.isfinite(baseline) or baseline <= 0:
                raise ValueError('高度阶段 %s 的基线无效' % name)
            stage.update(tolerance_mm=tolerance, baseline_mm=baseline,
                         min_mm=baseline - tolerance, max_mm=baseline + tolerance)
            upper = raw.get('upper_limit_mm')
            if upper is not None:
                upper = float(upper)
                if not math.isfinite(upper) or upper < stage['min_mm']:
                    raise ValueError('高度阶段 %s 的上限必须有限且不低于下限' % name)
                stage['upper_limit_mm'] = upper
                stage['max_mm'] = min(stage['max_mm'], upper)
        stages.append(stage)
    return stages


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

        self.stages = build_height_stages(self.settings, native['baseline_height_mm'],
                                          self.settings['association'])
        # ★ 主阶段（4/4 顶盖）提供旧属性，保证老调用方与老测试的读数不变。
        primary = next((s for s in self.stages if s['at_step'] == DEFAULT_STEP), self.stages[0])
        self.primary = primary
        self.height_accepted = primary['accepted']
        self.height_tolerance = primary['tolerance_mm']
        self.height_upper_limit = primary['upper_limit_mm']
        self.height_min = primary['min_mm']
        self.height_max = primary['max_mm']

        # ★ 仅复制B参数，不采用旧运行副本的相机控件，也绝不复用旧像素测区。
        native['reference_rect'] = native['target_rect'] = None
        if self.height_accepted:
            native['baseline_height_mm'] = primary['baseline_mm']
            native['height_tolerance_mm'] = self.height_tolerance
        self.camera = Config(copy.deepcopy(camera.data))
        self.camera.set('depth.enabled', True)
        self.camera.set('depth.backend', 'realsense')
        self.camera.set('native_depth', native)
        self.estimator = NativeHeightEstimator(self.camera)
        self.workflow = None
        ui_settings = self.settings.get('operator_ui')
        if ui_settings is not None:
            from .ui.workflow import SixStepWorkflow
            if (ui_settings.get('mode') != 'six_step'
                    or [(s['name'], s['at_step'], s['mode']) for s in self.stages]
                    != [('column', 2, 'length'), ('cap', 4, 'height')]):
                raise ValueError('六步界面需要column@2长度和cap@4高度阶段')
            self.workflow = SixStepWorkflow(ui_settings)
        self.generation = 0
        self.results = {}
        self.result = None
        self.measurement = None
        self.reason = self.waiting_reason()

    # ── 阶段查询 ──────────────────────────────────────────────
    def stage_for_step(self, step):
        """取件进度 -> 该进度允许的 stage；没有检测项时返回 None。"""
        for stage in self.stages:
            if stage['at_step'] == step:
                return stage
        return None

    def allowed_steps_text(self):
        return '、'.join('%d/4' % s['at_step'] for s in sorted(self.stages, key=lambda s: s['at_step']))

    def waiting_reason(self):
        if any(stage.get('auto_zone') for stage in self.stages):
            return '待测（2/4 按I自动框；4/4 按I选区）'
        return '待测（%s 后按i选区）' % self.allowed_steps_text()

    @property
    def height_rule(self):
        if not self.height_accepted:
            return '基线未接受'
        return '%.3f～%.3f mm' % (self.height_min, self.height_max)

    @staticmethod
    def rule_text(stage):
        if not stage['accepted']:
            return '基线未接受'
        return '%.3f～%.3f mm' % (stage['min_mm'], stage['max_mm'])

    def reset(self):
        if self.workflow is not None:
            self.workflow.reset()
        self.results = {}
        self.result = self.measurement = None
        self.reason = self.waiting_reason()

    # ── 测量 ──────────────────────────────────────────────────
    def measure(self, frame, stamp, reference, target, generation, stage=None):
        """操作者在冻结原图选区；结论只属于该快照，不推断后续帧的表面。

        `stage` 缺省为 4/4 顶盖阶段 —— 与旧版单阶段调用的行为完全一致。
        """
        if stage is None:
            stage = self.stage_for_step(DEFAULT_STEP) or self.primary
        step = int(stage['at_step'])
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
        if stage['accepted']:
            cfg.set('native_depth.baseline_height_mm', stage['baseline_mm'])
            cfg.set('native_depth.height_tolerance_mm', stage['tolerance_mm'])
        # ★ 阶段级门禁覆盖（见 build_height_stages 的说明）；不设则完全沿用 camera.yaml。
        if stage.get('min_points') is not None:
            cfg.set('native_depth.min_points', int(stage['min_points']))
        if stage.get('min_valid_ratio') is not None:
            cfg.set('native_depth.min_valid_ratio', float(stage['min_valid_ratio']))
        result = NativeHeightEstimator(cfg).measure_frame(frame)
        conforms = (assess_height(result['height_mm'], stage['baseline_mm'],
                                  stage['tolerance_mm'], stage['upper_limit_mm'])
                    if result['ok'] and stage['accepted'] else None)
        result = dict(result, frame_index=stamp.index, captured_at=stamp.captured_at,
                      generation=generation, association='manual_same_frame_cap_region',
                      stage=stage['name'], stage_label=stage['label'], step=step,
                      reference_rect=list(reference), target_rect=list(target),
                      baseline_accepted=stage['accepted'], baseline_source_accepted=False,
                      baseline_height_mm=stage['baseline_mm'],
                      height_tolerance_mm=stage['tolerance_mm'], height_conforms=conforms,
                      height_min_mm=stage['min_mm'], height_max_mm=stage['max_mm'],
                      upper_limit_mm=stage['upper_limit_mm'],
                      assessment_scope='仅人工同帧%s高度；不代表整机全部质量' % stage['label'],
                      scope='仅该帧人工确认表面；不证明后续帧仍在同一位置')
        self.results[stage['name']] = result
        self.result = result                      # 旧版别名：最近一次测量
        self.measurement = (frame, copy.deepcopy(result))
        self.reason = self.waiting_reason()
        return result

    def measure_length(self, frame, stamp, target, generation, stage=None,
                       mask=None, mm_per_px=None):
        """★ 2026-10-29：横放件的**长度**（2D 轮廓），完全不依赖深度。

        立柱是 Ø15 的高反光细杆，2026-10-28 实测 D435 在它身上没有有效深度点，
        B 的高度法测不了；横放后长度只在图像平面内，用标定 mm/px 量掩膜长边。
        """
        if stage is None:
            stage = self.primary
        step = int(stage['at_step'])
        shape = frame.left.shape[:2]
        if frame.index != stamp.index or shape != stamp.shape or frame.scale != 1.0:
            raise ValueError('长度测量需要同帧同尺寸原图及 scale=1')
        if mask is None or mask.shape[:2] != shape:
            raise ValueError('长度测量需要同帧同尺寸金属掩膜')
        # ★ 阶段里写了 mm_per_px 就用现场标定值，否则退回 TrayMap（并标注来源）。
        if stage.get('mm_per_px') is not None:
            scale, scale_src = float(stage['mm_per_px']), 'stage_calibrated'
        else:
            scale, scale_src = float(mm_per_px), 'tray_map'
        measured = measure_length_image(mask, target, scale)
        conforms = (assess_height(measured['length_mm'], stage['baseline_mm'],
                                  stage['tolerance_mm'], stage['upper_limit_mm'])
                    if measured['ok'] and stage['accepted'] else None)
        result = dict(measured, frame_index=stamp.index, captured_at=stamp.captured_at,
                      generation=generation, association='manual_same_frame_cap_region',
                      stage=stage['name'], stage_label=stage['label'], step=step,
                      mode='length', reference_rect=None, target_rect=list(target),
                      scale_mm_per_px=scale, scale_source=scale_src,
                      baseline_accepted=stage['accepted'], baseline_source_accepted=False,
                      baseline_height_mm=stage['baseline_mm'],
                      height_tolerance_mm=stage['tolerance_mm'], height_conforms=conforms,
                      height_min_mm=stage['min_mm'], height_max_mm=stage['max_mm'],
                      upper_limit_mm=stage['upper_limit_mm'],
                      assessment_scope='仅人工同帧%s长度；不代表整机全部质量' % stage['label'],
                      scope='仅该帧人工确认轮廓；不证明装配到位')
        self.results[stage['name']] = result
        self.result = result
        self.measurement = (frame, copy.deepcopy(result))
        self.reason = self.waiting_reason()
        return result

    def accept_length_result(self, frame, stamp, measured, generation, stage, zone):
        """接受自动检测框已经算好的长度结果（2 秒稳定后触发）。

        与 `measure_length` 的区别：不再用人工目标框和单一 mm/px 重算，
        而是直接接收 `zone_geometry.calibrated_measure` 的结果（含 length_scale）。
        """
        step = int(stage['at_step'])
        if frame.index != stamp.index or frame.scale != 1.0:
            raise ValueError('自动长度结果需要同帧同尺寸原图')
        conforms = (assess_height(measured['length_mm'], stage['baseline_mm'],
                                  stage['tolerance_mm'], stage['upper_limit_mm'])
                    if measured.get('ok') and measured.get('length_mm') is not None
                    and stage['accepted'] else None)
        result = dict(measured, frame_index=stamp.index, captured_at=stamp.captured_at,
                      generation=generation, association='auto_zone_2s',
                      stage=stage['name'], stage_label=stage['label'], step=step,
                      mode='length', reference_rect=None, target_rect=list(zone),
                      scale_mm_per_px=None, scale_source='auto_zone_calibrated',
                      length_scale=measured.get('length_scale'),
                      baseline_accepted=stage['accepted'], baseline_source_accepted=False,
                      baseline_height_mm=stage['baseline_mm'],
                      height_tolerance_mm=stage['tolerance_mm'], height_conforms=conforms,
                      height_min_mm=stage['min_mm'], height_max_mm=stage['max_mm'],
                      upper_limit_mm=stage['upper_limit_mm'],
                      assessment_scope='仅自动检测框2秒保持后的%s长度；不代表整机全部质量' % stage['label'],
                      scope='自动框内2秒稳定后测量；不证明三维姿态')
        self.results[stage['name']] = result
        self.result = result
        self.measurement = (frame, copy.deepcopy(result))
        self.reason = self.waiting_reason()
        return result

    # ── 界面 ──────────────────────────────────────────────────
    def decorate(self, state, started):
        # ★ 数值有效和高度合格分列；四步永远不产生「装配全部合格」。
        step = int(getattr(state, 'step', 0) or 0)
        stage = self.stage_for_step(step)
        result = None
        if stage is not None:
            candidate = self.results.get(stage['name'])
            # ★ 结果只在"当初测量的那个进度"显示，回退/继续后不会串台。
            if candidate is not None and candidate.get('step') == step:
                result = candidate
        label = stage['label'] if stage is not None else ''
        # ★ 2026-10-29：立柱阶段量的是"长度"（横放），顶盖仍是"高度"。
        unit = '长度' if (stage is not None and stage.get('mode') == 'length') else '高度'

        if state.level == 'ok':
            state.level = 'warn'
        state.title = '取件%d/4 · 质量待判' % step
        lines = ['开工：' + ('已通过D并开工' if started else '等待D稳定放行'),
                 '取件顺序：%d/4' % step]
        if stage is None:
            lines.append('本进度无检测项（%s 才有）' % self.allowed_steps_text())
        elif result is None:
            lines.append('测量：' + self.reason)
        else:
            value = result.get('length_mm', result.get('height_mm'))
            lines.append('快照测量：' + ('有效 %.4f mm' % value if result['ok'] else '无效'))
            lines.append('测量帧：%s（人工快照）' % result['frame_index'])
            if not result['ok']:
                lines.append(result['reason'])
                # ★ 交接文档待办#2：把深度质量摆到面板上，离群时能直接看出
                #   是"点太少"还是"参考区被污染"，而不是只看到一句"无效"。
                if result.get('target_points') is not None:
                    lines.append('深度质量：目标 %d 点 / 有效率 %.2f / 参考 %s 点' % (
                        int(result['target_points']), float(result.get('target_valid_ratio') or 0.0),
                        int(result.get('reference_points', 0))))
            if result.get('width_mm') is not None:
                lines.append('轮廓：长 %.4f mm / 宽 %.2f mm / 角度 %.1f°' % (
                    float(value), float(result['width_mm']), float(result.get('angle_deg') or 0.0)))
        if stage is None:
            pass
        elif not stage['accepted']:
            lines.extend(['%s%s符合性：无法判断' % (label, unit), '原因：基线未正式接受'])
        elif result is not None and result['height_conforms'] is not None:
            passed = result['height_conforms']
            lines.append('%s%s符合性：' % (label, unit) + ('合格' if passed else '不合格'))
            state.title = '取件%d/4 · %s%s%s' % (step, label, unit, '合格' if passed else '不合格')
            if not passed:
                state.level = 'error'
                state.banner = '%s%s不合格：超出受控范围' % (label, unit)
            elif step == DEFAULT_STEP and state.level != 'error':
                state.level = 'ok'
        else:
            lines.append('%s%s符合性：待测' % (label, unit))
        if stage is not None and stage['accepted']:
            lines.extend(['上限：%.3f mm（含）' % stage['max_mm'],
                          '下限：%.3f mm（含）' % stage['min_mm']])
        lines.extend(['紧固性：队友负责', '安装等其他质量：未验证'])
        state.detail = lines + state.detail
        state.stats['B基线'] = ('仅%s受控%s' % (label, unit)) if (stage is not None and stage['accepted']) else '未接受'
        state.stats['测量模式'] = ('2D轮廓长度' if unit == '长度' else '单帧人工关联')
        if result is not None and result.get('target_points') is not None:
            state.stats['深度质量'] = '%d点/有效率%.2f' % (
                int(result['target_points']), float(result.get('target_valid_ratio') or 0.0))
        if result is not None and result.get('scale_mm_per_px') is not None:
            state.stats['长度比例'] = '%.4f mm/px（%s）' % (
                float(result['scale_mm_per_px']),
                '现场标定' if result.get('scale_source') == 'stage_calibrated' else 'TrayMap')

        if self.workflow is not None:
            view = self.workflow.update(step, started, self.results, self.generation, state.level)
            view['rules'] = tuple((s['label'], self.rule_text(s)) for s in self.stages)
            view['reason'] = self.reason
            state.workflow = view
