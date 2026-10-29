"""静止2秒、轻微手抖不清零、明显移动/空框/断帧仍撤销、每次放置仅触发一次。"""
from assemvision.ui.auto_length_gate import PlacementGate


def gate(**over):
    config = dict(hold_seconds=2., max_frame_gap_seconds=.5, min_frames=5,
                  max_center_drift_px=15., max_length_drift_px=12.,
                  max_lost_seconds=.6, rearm_empty_seconds=.6)
    config.update(over)
    return PlacementGate(config)


def steady(g, start, end, metrics=(10., 10., 100.), step=.4):
    result = None
    t = start
    while t <= end + 1e-9:
        result = g.update(t, metrics)
        t += step
    return result


def main():
    # 1) 基本两秒触发、每次放置只触发一次、移出后可再测
    g = gate()
    assert g.update(0.)['color'] == 'yellow'
    for t in [0., .4, .8, 1.2, 1.6]:
        r = g.update(t, (10., 10., 100.))
        assert r['color'] == 'green' and not r['trigger']
    assert g.update(2., (10., 10., 100.))['trigger']
    assert not g.update(2.4, (10., 10., 100.))['trigger']
    g.update(3., empty=True)
    assert g.update(3.7, empty=True)['color'] == 'yellow'
    g.update(4., (10., 10., 100.))
    assert g.update(4.4)['color'] == 'yellow'      # 短暂丢失：黄框但不立即清零
    assert g.update(4.8, (10., 10., 100.))['elapsed'] == 0.
    g.update(5.2, (10., 10., 100.))
    assert g.update(5.6, (30., 10., 100.))['elapsed'] == 0.  # 20px 移动，明显移动后从零
    assert g.update(6.5, (30., 10., 100.))['elapsed'] == 0.  # 断帧不能累计
    assert g.update(6.8, (float('nan'), 10., 100.))['color'] == 'yellow'

    # 2) 手抖：中心 ±13px、长度 ±11px（超过旧 12/9，但在新 15/12 内），仍应在 2 秒触发
    g = gate()
    jitter = [(10., 10., 100.), (23., 9., 111.), (0., 11., 89.), (22., 8., 110.),
              (1., 12., 90.), (21., 11., 109.), (2., 12., 91.), (23., 12., 111.),
              (0., 8., 89.), (10., 10., 100.), (10., 10., 100.)]
    for i, m in enumerate(jitter):
        r = g.update(i*.2, m)
        if i < len(jitter)-1:
            assert not r['trigger'], ('轻微抖动提前触发', i)
    assert r['trigger'], '轻微抖动始终没有触发'

    # 3) 明显移动必须重置
    g = gate()
    steady(g, 0., .8)
    r = g.update(1., (30., 10., 100.))
    assert r['elapsed'] == 0. and not r['trigger']
    assert steady(g, 1.4, 3.0, (30., 10., 100.))['trigger']

    # 4) 短暂丢失：冻结计时，不把丢失时间算进 2 秒
    g = gate()
    steady(g, 0., .8)
    r = g.update(1.)
    assert r['color'] == 'yellow' and abs(r['elapsed']-.8) < 1e-9
    r = g.update(1.2)
    assert abs(r['elapsed']-.8) < 1e-9
    r = g.update(1.4, (10., 10., 100.))
    assert abs(r['elapsed']-.8) < 1e-9, '恢复首帧不能补算丢失时间'
    assert steady(g, 1.6, 2.6, step=.2)['trigger']

    # 5) 丢失超过宽限必须撤销，不能累计成完成
    g = gate()
    steady(g, 0., .8)
    for t in [1., 1.2, 1.4, 1.6]:
        assert not g.update(t)['trigger']
    r = g.update(1.8)
    assert r['elapsed'] == 0. and r['color'] == 'yellow'
    assert g.update(1.8, (10., 10., 100.))['elapsed'] == 0.

    # 6) 空框立即撤销
    g = gate()
    steady(g, 0., .8)
    assert g.update(1., empty=True)['elapsed'] == 0.

    # 7) hard=True（真正取帧失败）立即撤销，且不吃短暂丢失宽限
    g = gate()
    steady(g, 0., .8)
    assert g.update(1., hard=True)['elapsed'] == 0.
    assert g.update(1.2, (10., 10., 100.))['elapsed'] == 0.

    # 8) 帧间隔超过上限不能续计时
    g = gate()
    steady(g, 0., .4)
    assert g.update(1., (10., 10., 100.))['elapsed'] == 0.

    # 9) 触发后空框保持 rearm 秒才允许重新测量
    g = gate()
    assert steady(g, 0., 2.)['trigger']
    g.update(2.2, empty=True)
    assert g.update(2.6, empty=True)['measured']
    assert g.update(2.8, empty=True)['color'] == 'yellow'
    assert g.update(3., (10., 10., 100.))['elapsed'] == 0.

    print('test_auto_length_gate 全部通过')


if __name__ == '__main__':
    main()
