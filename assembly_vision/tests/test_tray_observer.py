# -*- coding: utf-8 -*-
"""C槽形朝向与占用观测：16种占用×四方向，并测无证据拒绝。"""
import copy
import os
import sys
from multiprocessing import shared_memory

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.config import Config, load_config
from assemvision.core.roi import capsule_polygon, circle_polygon
from assemvision.core.tray_observer import TrayObserver, geometry_binding

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def same_observation(left, right):
    """所有离散决策必须一致；OpenCV并发路径的浮点舍入误差最多1e-6。"""
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same_observation(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same_observation(a, b)
    elif isinstance(left, (float, np.floating)):
        np.testing.assert_allclose(left, right, atol=1e-6, rtol=0, equal_nan=True)
    else:
        assert left == right, (left, right)


def main():
    wp = load_config(os.path.join(ROOT, 'config/workpiece.yaml'))
    data = copy.deepcopy(load_config(os.path.join(ROOT, 'config/tray_observation.yaml')).data)
    # ★ 以独立OpenCV图像旋转生成真值，不用被测pose产生期望布局。
    wp = Config(copy.deepcopy(wp.data))
    wp.set('tray_detect.nominal_angle_deg', 0.)
    cal = Config({'tray_map': {'homography': [[2.,0.,160.],[0.,2.,80.],[0.,0.,1.]], 'scale': 1.}})
    data['geometry_binding'] = geometry_binding(wp, cal)
    reference = np.full((160,160), 72, 'uint8')
    for sid, geom in data['rim_geometry'].items():
        if 'rim_diameter_mm' in geom:
            poly = circle_polygon(geom['center_mm'],geom['rim_diameter_mm'])
        else:
            poly = capsule_polygon(geom['center_mm'],geom['rim_length_mm'],geom['rim_width_mm'],geom['angle_deg'])
        poly = np.rint(poly).astype('int32')
        cv2.fillPoly(reference,[poly],55)
        cv2.polylines(reference,[poly],True,25,2)
    observer = TrayObserver(wp, cal, data, reference)
    revised_data = copy.deepcopy(data)
    revised_data['outline']['s_max'] = 120
    revised_data['matching']['exclude_brass'] = True
    revised_data['occupancy']['evidence'] = 'metal_or_brass'
    revised = TrayObserver(wp, cal, revised_data, reference)
    parallel = TrayObserver(wp, cal, revised_data, reference, workers=3)
    fast = TrayObserver(wp, cal, revised_data, reference, workers=4, correlation_backend='box_sum_cached')
    matching_only_data = copy.deepcopy(revised_data)
    matching_only_data['occupancy']['evidence'] = 'metal'
    matching_only = TrayObserver(wp, cal, matching_only_data, reference)
    checked = 0
    for bits in range(16):
        canonical = cv2.cvtColor(reference,cv2.COLOR_GRAY2BGR)
        mask = np.zeros((160,160),'uint8')
        for i,sid in enumerate(data['slot_order']):
            if bits & (1<<i):
                mask[observer.slot_masks[sid]] = 255
                canonical[observer.slot_masks[sid]] = (30,160,225)
        frame = np.full((480,640,3),(25,180,25),'uint8')
        patch = cv2.resize(canonical,(320,320),interpolation=cv2.INTER_NEAREST)
        frame[80:400,160:480] = patch
        metal = np.zeros((480,640),'uint8')
        metal[80:400,160:480] = cv2.resize(mask,(320,320),interpolation=cv2.INTER_NEAREST)
        for turn in range(0,360,30):
            affine = cv2.getRotationMatrix2D((320.,240.),turn,1.)
            image = cv2.warpAffine(frame,affine,(640,480),borderValue=(25,180,25))
            current_metal = cv2.warpAffine(metal,affine,(640,480),flags=cv2.INTER_NEAREST)
            result = observer.observe(image,current_metal)
            assert result.orientation_ok, (bits,turn,result.to_dict())
            for i,sid in enumerate(data['slot_order']):
                expected = 'occupied' if bits & (1<<i) else 'empty'
                assert result.slots[sid]['state']==expected,(bits,turn,sid,result.to_dict())
            assert result.state == ('empty' if bits==0 else 'full' if bits==15 else 'partial')
            # 检查物理方向，而不是只比较占用数目；满盘和空盘也不得错90°。
            expected_rotation = cv2.getRotationMatrix2D((0.,0.),turn,1.)[:,:2]
            np.testing.assert_allclose(result.pose.R,expected_rotation,atol=.03)
            checked += 1
            # ★ 新匹配遮挡入口必须保留全组合朝向/占用，不只测满盘数量。
            current = revised.observe(image, current_metal,
                                      brass_mask=current_metal)
            concurrent = parallel.observe(image, current_metal, brass_mask=current_metal)
            same_observation(concurrent.to_dict(), current.to_dict())
            same_observation(fast.observe(image,current_metal,brass_mask=current_metal).to_dict(),
                             current.to_dict())
            assert current.orientation_ok, (bits, turn, current.to_dict())
            assert current.state == result.state
            assert {s:r['state'] for s,r in current.slots.items()} == {
                s:r['state'] for s,r in result.slots.items()}
            np.testing.assert_allclose(current.pose.R, expected_rotation, atol=.03)
    blank = np.full((480,640,3),(25,180,25),'uint8')
    no_mask = np.zeros((480,640),'uint8')
    # 附加黄铜掩膜不可以偷偷成为「有件」证据；占用仍只读原金属掩膜。
    isolated = matching_only.observe(frame, no_mask, brass_mask=metal)
    assert isolated.orientation_ok and isolated.state == 'empty', isolated.to_dict()
    combined = revised.observe(frame, no_mask, brass_mask=metal)
    assert combined.orientation_ok and combined.state == 'full', combined.to_dict()
    assert all(r['metal_fraction'] == 0 and r['brass_fraction'] > 0
               and r['occupancy_fraction'] == r['brass_fraction'] for r in combined.slots.values())
    assert revised.observe(frame, metal).state == 'unknown'
    assert '黄铜观测掩膜' in revised.observe(frame, metal).reason
    try:
        revised.observe(frame, metal, brass_mask=np.zeros((10,10),'uint8'))
    except ValueError:
        pass
    else:
        raise AssertionError('匹配遮挡掩膜也必须与当前原图同尺寸')
    assert not observer.observe(blank,no_mask).found
    blank[80:400,160:480] = 72
    bad = observer.observe(blank,no_mask)
    assert not bad.orientation_ok and bad.state=='unknown' and not bad.slots
    # ★ 已经识别过满盘，下一帧没有可观测方向时不能沿用旧结果。
    all_blocked = np.full_like(no_mask,255)
    bad = observer.observe(frame,all_blocked)
    assert not bad.orientation_ok and bad.pose is None and bad.state=='unknown'
    assert observer.observe(frame,None).state=='unknown'
    for image, mask, hand in ((frame, all_blocked, None), (blank, no_mask, None),
                              (frame, metal, all_blocked)):
        same_observation(parallel.observe(image, mask, hand, brass_mask=metal).to_dict(), revised.observe(
            image, mask, hand, brass_mask=metal).to_dict())
        same_observation(fast.observe(image,mask,hand,brass_mask=metal).to_dict(),revised.observe(
            image,mask,hand,brass_mask=metal).to_dict())
    before=fast._correlation_cache.hits
    for _ in range(2):
        same_observation(fast.observe(frame,metal,brass_mask=metal).to_dict(),
                         revised.observe(frame,metal,brass_mask=metal).to_dict())
    assert fast._correlation_cache.hits>before
    parallel.close()
    fast.close()
    processes = TrayObserver(wp,cal,revised_data,reference,workers=4,
                            correlation_backend='box_sum_cached',execution_backend='process')
    name = None
    try:
        for image,mask,hand in ((frame,metal,None),(blank,no_mask,None),(frame,metal,all_blocked)):
            same_observation(processes.observe(image,mask,hand,brass_mask=metal).to_dict(),
                             revised.observe(image,mask,hand,brass_mask=metal).to_dict())
        name = processes._shared_memory.name
    finally:
        processes.close()
    assert processes._shared_memory is None and processes._executor is None
    try:
        leaked = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        pass
    else:
        leaked.close()
        raise AssertionError('C方向共享内存未释放')
    assert parallel._executor is None
    try:
        parallel.observe(frame, metal, brass_mask=metal)
    except RuntimeError:
        pass
    else:
        raise AssertionError('关闭后不允许继续提交方向任务')
    for workers in (0, 5, True, 1.5):
        try:
            TrayObserver(wp, cal, data, reference, workers=workers)
        except ValueError:
            pass
        else:
            raise AssertionError('无效C并行数未拒绝')
    glove = np.zeros_like(no_mask)
    glove[80:400,160:480] = cv2.resize((observer.slot_masks['slot_cap']*255).astype('uint8'),(320,320),interpolation=cv2.INTER_NEAREST)
    hand_result = observer.observe(frame,metal,glove)
    assert hand_result.orientation_ok and hand_result.slots['slot_cap']['state']=='unknown'
    assert hand_result.state=='unknown'
    changed = Config(copy.deepcopy(cal.data))
    changed.data['tray_map']['homography'][0][2] += 1
    try:
        TrayObserver(wp,changed,data,reference)
    except ValueError:
        pass
    else:
        raise AssertionError('旧C模板必须拒绝不匹配的新标定')
    print('C观测：%d种占用/朝向组合及遮挡、无盘、歧义、旧标定拒绝全部通过'%checked)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
