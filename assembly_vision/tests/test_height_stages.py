# -*- coding: utf-8 -*-
"""分阶段高度判定的纯逻辑回归（不碰相机）：2/4 测立柱、4/4 测顶盖。

覆盖点
  1) 旧配置（只有 height_assessment）解析成单阶段，数值与旧版逐字节一致；
  2) 新配置（height_stages）解析出 2/4 与 4/4 两条规则，且互不串台；
  3) 违规配置必须**抛错**而不是静默降级（容差超上限/缺授权/上下限倒置/进度重复）；
  4) 界面按"当前取件进度"取结果，回退或继续后旧结果不冒充当前结论。
"""
import copy
from pathlib import Path

from assemvision.config import load_config
from assemvision.integration import Integration, build_height_stages

ROOT = Path(__file__).resolve().parents[1]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print('  ok  ' + msg)


def expect_error(fn, msg):
    try:
        fn()
    except ValueError:
        print('  ok  ' + msg)
        return
    raise AssertionError('应报错但没报：' + msg)


LEGACY = {
    'association': 'manual_same_frame_cap_region',
    'height_assessment': {
        'enabled': True, 'scope': 'manual_same_frame_cap_region',
        'baseline_mm': 65.0, 'tolerance_mm': 3.0, 'max_tolerance_mm': 3.0,
        'upper_limit_mm': 68.0, 'authorization': 'legacy',
    },
}

TWO_STAGE = {
    'association': 'manual_same_frame_cap_region',
    'height_stages': [
        {'name': 'column', 'label': '立柱', 'at_step': 2, 'target': 'COLUMN',
         'mode': 'length', 'mm_per_px': 0.638,
         'enabled': True, 'scope': 'manual_same_frame_cap_region',
         'baseline_mm': 64.0, 'tolerance_mm': 3.0, 'max_tolerance_mm': 3.0,
         'authorization': 'user-2026-10-29'},
        {'name': 'cap', 'label': '顶盖', 'at_step': 4, 'target': 'CAP',
         'mode': 'height',
         'enabled': True, 'scope': 'manual_same_frame_cap_region',
         'baseline_mm': 65.0, 'tolerance_mm': 3.0, 'max_tolerance_mm': 3.0,
         'upper_limit_mm': 68.0, 'authorization': 'user-2026-10-25'},
    ],
}


def main():
    print('== 旧配置兼容（单阶段 = 4/4 顶盖）')
    stages = build_height_stages(LEGACY, 68.025, LEGACY['association'])
    check(len(stages) == 1 and stages[0]['at_step'] == 4 and stages[0]['name'] == 'cap',
          '只有 height_assessment 时解析为单阶段 cap@4')
    check(stages[0]['min_mm'] == 62.0 and stages[0]['max_mm'] == 68.0
          and stages[0]['upper_limit_mm'] == 68.0,
          '旧窗口仍是 62.000～68.000（含 68 上限）')

    print('== 新配置：2/4 立柱 + 4/4 顶盖')
    stages = build_height_stages(TWO_STAGE, 68.025, TWO_STAGE['association'])
    check([s['at_step'] for s in stages] == [2, 4], '两条阶段分别绑 2/4 与 4/4')
    column = stages[0]
    check(column['label'] == '立柱' and column['target'] == 'COLUMN', '立柱阶段名与目标提示词正确')
    check(column['mode'] == 'length' and stages[1]['mode'] == 'height',
          '立柱走长度法、顶盖仍走高度法')
    check(column['min_mm'] == 61.0 and column['max_mm'] == 67.0
          and column['upper_limit_mm'] is None,
          '立柱窗口 61.000～67.000 且无额外上限')
    cap = stages[1]
    check(cap['min_mm'] == 62.0 and cap['max_mm'] == 68.0, '顶盖窗口 62.000～68.000 不受影响')
    check(column.get('min_points') is None and column.get('min_valid_ratio') is None,
          '未写门禁覆盖时保持 camera.yaml 的默认门禁')

    print('== 阶段取值与"无检测项"的进度')
    by_step = {s['at_step']: s for s in stages}
    check(by_step.get(2)['name'] == 'column' and by_step.get(4)['name'] == 'cap',
          '2/4 取立柱、4/4 取顶盖')
    check(by_step.get(1) is None and by_step.get(3) is None, '1/4、3/4 没有高度检测项')

    print('== 阶段级门禁覆盖（显式写才生效）')
    gated = copy.deepcopy(TWO_STAGE)
    gated['height_stages'][0]['min_points'] = 20
    gated['height_stages'][0]['min_valid_ratio'] = 0.35
    st_gated = build_height_stages(gated, 68.025, gated['association'])
    check(st_gated[0]['min_points'] == 20 and abs(st_gated[0]['min_valid_ratio'] - 0.35) < 1e-9,
          '显式写入 min_points / min_valid_ratio 时按值生效')
    check(st_gated[1]['min_points'] is None, '未写的阶段不受影响')

    print('== 违规配置必须报错')
    bad = copy.deepcopy(TWO_STAGE)
    bad['height_stages'][0]['tolerance_mm'] = 15.0
    expect_error(lambda: build_height_stages(bad, 68.025, bad['association']),
                 '容差超过 max_tolerance 被拒绝')
    bad = copy.deepcopy(TWO_STAGE)
    bad['height_stages'][0]['at_step'] = 4
    expect_error(lambda: build_height_stages(bad, 68.025, bad['association']),
                 '两个阶段抢同一个取件进度被拒绝')
    bad = copy.deepcopy(TWO_STAGE)
    bad['height_stages'][0].pop('authorization')
    expect_error(lambda: build_height_stages(bad, 68.025, bad['association']),
                 '缺授权被拒绝')
    bad = copy.deepcopy(TWO_STAGE)
    bad['height_stages'][1]['upper_limit_mm'] = 60.0
    expect_error(lambda: build_height_stages(bad, 68.025, bad['association']),
                 '上限低于下限被拒绝')
    bad = copy.deepcopy(TWO_STAGE)
    bad['height_stages'][0]['scope'] = 'other_scope'
    expect_error(lambda: build_height_stages(bad, 68.025, bad['association']),
                 '判定范围与 association 不一致被拒绝')
    bad = copy.deepcopy(TWO_STAGE)
    bad['height_stages'][0]['min_points'] = 2
    expect_error(lambda: build_height_stages(bad, 68.025, bad['association']),
                 'min_points 过小被拒绝')
    bad = copy.deepcopy(TWO_STAGE)
    bad['height_stages'][0]['min_valid_ratio'] = 0
    expect_error(lambda: build_height_stages(bad, 68.025, bad['association']),
                 'min_valid_ratio 必须大于 0')

    print('== 现场候选文件（需要冻结来源在位）')
    cam = load_config(str(ROOT / 'config/camera.yaml'))
    integ = Integration(str(ROOT / 'config/abcde_integration_pi_column_candidate.yaml'), cam)
    check([s['at_step'] for s in integ.stages] == [2, 4], '现场候选解析出 2/4 与 4/4')
    check(integ.allowed_steps_text() == '2/4、4/4', '提示文案列出 2/4、4/4')
    check(integ.stage_for_step(3) is None, '3/4 不可测')
    # 旧属性仍取 4/4 顶盖，供老调用方/老测试读取。
    # ★ 2026-10-29 用户改判：顶盖基线 65 → 68.5，且要求 ±1 → 67.500～69.500。
    check(integ.height_min == 67.5 and integ.height_max == 69.5,
          'height_rule 指向 4/4 顶盖 67.500～69.500')
    col = integ.stage_for_step(2)
    # ★ 2026-10-28 收尾：放宽门禁会让"框到台面"也安静出数，现场候选回到默认门禁。
    check(col['min_points'] is None and col['min_valid_ratio'] is None,
          '立柱阶段使用 camera.yaml 的默认门禁（≥50 点 / ≥0.6）')
    check(integ.stage_for_step(4)['min_points'] is None,
          '顶盖阶段同样不受门禁覆盖影响')
    check(abs(integ.stage_for_step(2)['mm_per_px'] - 0.638) < 1e-9,
          '立柱阶段使用现场标定比例 0.638 mm/px')
    check(integ.stage_for_step(4).get('mm_per_px') is None,
          '顶盖阶段不写比例覆盖（仍用深度法）')

    print('== 界面按进度取结果，不串台')
    from assemvision.ui.base import UIState
    integ.results['column'] = dict(ok=True, length_mm=64.2, frame_index=11, step=2,
                                   stage='column', stage_label='立柱', height_conforms=True)
    integ.results['cap'] = dict(ok=True, height_mm=66.5, frame_index=99, step=4,
                                stage='cap', stage_label='顶盖', height_conforms=True)
    st = UIState()
    st.step = 2
    integ.decorate(st, True)
    check('立柱长度合格' in st.title, '2/4 显示"长度"结论而不是高度')
    check(any('64.2000' in line for line in st.detail), '2/4 显示立柱读数')
    check(not any('66.5000' in line for line in st.detail), '2/4 不显示顶盖读数')

    st = UIState()
    st.step = 3
    integ.decorate(st, True)
    check('无检测项' in ''.join(st.detail), '3/4 明确提示无检测项')

    st = UIState()
    st.step = 4
    integ.decorate(st, True)
    check('顶盖高度合格' in st.title, '4/4 显示顶盖结论')

    print('== 不合格时必须报错级别')
    integ.results['column'] = dict(ok=True, length_mm=68.0, frame_index=12, step=2,
                                   stage='column', stage_label='立柱', height_conforms=False)
    st = UIState()
    st.step = 2
    integ.decorate(st, True)
    check(st.level == 'error' and '立柱长度不合格' in st.banner, '立柱超长给红条并写明是立柱')

    print('== 2D 长度测量（纯函数）')
    import numpy as np
    from assemvision.core.length import measure_length

    mmpp = 0.86
    mask = np.zeros((480, 640), np.uint8)
    # 74 x 12 px 的横条 ≈ 63.6 x 10.3 mm
    mask[200:212, 100:174] = 255
    got = measure_length(mask, (90, 190, 100, 40), mmpp)
    check(got['ok'] and abs(got['length_mm'] - 74 * mmpp) < 0.9,
          '横条长度 %.2f mm ≈ 74px×%.2f' % (got['length_mm'], mmpp))
    check(abs(got['width_mm'] - 12 * mmpp) < 0.9, '短边同样按 mm/px 换算')

    empty = measure_length(np.zeros((480, 640), np.uint8), (90, 190, 100, 40), mmpp)
    check(not empty['ok'] and '太少' in empty['reason'], '空测区明确失败而不是出数')

    thin = np.zeros((480, 640), np.uint8)
    thin[200:202, 100:174] = 255          # 2px 高 ≈ 1.7mm，低于宽限
    narrow = measure_length(thin, (90, 190, 100, 40), mmpp)
    check(not narrow['ok'] and '过窄' in narrow['reason'], '过窄的轮廓被拒绝（防反光边/碎片）')

    check(not measure_length(mask, (90, 190, 100, 40), None)['ok'], '缺 mm/px 时明确失败')
    check(not measure_length(mask, (600, 460, 100, 40), mmpp)['ok'], '测区越界时明确失败')

    print('== 现场标定比例优先于 TrayMap')
    from types import SimpleNamespace
    frame = SimpleNamespace(left=np.zeros((480, 640, 3), np.uint8), index=7, scale=1.0, ts=0.0)
    stamp = SimpleNamespace(index=7, captured_at=0.0, shape=(480, 640))
    rod = np.zeros((480, 640), np.uint8)
    rod[220:235, 200:300] = 255                 # 100 x 15 px
    res = integ.measure_length(frame, stamp, (190, 210, 120, 45), 0,
                               stage=integ.stage_for_step(2), mask=rod, mm_per_px=0.731)
    check(res['ok'] and res['scale_source'] == 'stage_calibrated',
          '阶段写了 mm_per_px 就用它，并标注来源')
    check(abs(res['length_mm'] - 100 * 0.638) < 1.0,
          '100 px 按 0.638 得 %.2f mm（不是 0.731 的 %.2f）'
          % (res['length_mm'], 100 * 0.731))
    check(res['height_conforms'] is True, '64±3 下这根 63.8mm 判合格')


if __name__ == '__main__':
    main()
