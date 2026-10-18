# -*- coding: utf-8 -*-
"""C槽形朝向与占用观测：16种占用×四方向，并测无证据拒绝。"""
import copy
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.config import Config, load_config
from assemvision.core.roi import capsule_polygon, circle_polygon
from assemvision.core.tray_observer import TrayObserver, geometry_binding

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    blank = np.full((480,640,3),(25,180,25),'uint8')
    no_mask = np.zeros((480,640),'uint8')
    assert not observer.observe(blank,no_mask).found
    blank[80:400,160:480] = 72
    bad = observer.observe(blank,no_mask)
    assert not bad.orientation_ok and bad.state=='unknown' and not bad.slots
    # ★ 已经识别过满盘，下一帧没有可观测方向时不能沿用旧结果。
    all_blocked = np.full_like(no_mask,255)
    bad = observer.observe(frame,all_blocked)
    assert not bad.orientation_ok and bad.pose is None and bad.state=='unknown'
    assert observer.observe(frame,None).state=='unknown'
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
