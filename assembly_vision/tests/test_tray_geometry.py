# -*- coding: utf-8 -*-
"""C模块：整盘绕自身中心转动时，中心、槽形与方向必须一致。"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.config import Config
from assemvision.core.roi import TrayMap, capsule_polygon
from assemvision.core.tray_detect import TrayDetect, pose_from_tray_detect
from assemvision.core.tray_pose import apply_pose_to_slot


def main():
    # ★ 用OpenCV绕盘中心旋转作为独立几何真值，不用被测函数生成期望值。
    slot = {'id': 'column', 'type': 'capsule', 'center': [120., 40.],
            'length': 85., 'width': 16., 'angle_deg': 35.}
    wp = Config({'tray': {'size': [160., 160.]}, 'slots': [slot]})
    tray = TrayMap(wp, px_per_mm=2., origin_px=(100., 50.))
    nominal_center = np.array([260., 210.])
    original = tray.slot_polygon_px('column')
    for angle in (-30., -10., 0., 17., 40.):
        for translation in ((0., 0.), (60., -24.)):
            # 图像中正角为顺时针；TrayPose正角为逆时针。
            det = TrayDetect(ok=True,
                             center_px=tuple(nominal_center + translation),
                             size_px=(320., 320.), angle_deg=angle)
            pose = pose_from_tray_detect(det, nominal_center, 320., .5,
                                         nominal_angle_deg=0.)
            actual_center = tray.mm_to_px(pose.apply([[80., 80.]]))[0]
            np.testing.assert_allclose(actual_center, det.center_px, atol=1e-8,
                                       err_msg='整盘转动不得把盘中心绕原点甩开')
            affine = cv2.getRotationMatrix2D(tuple(nominal_center), -angle, 1.)
            affine[:, 2] += translation
            expected = cv2.transform(original[None, :, :], affine)[0]
            center = pose.apply([slot['center']])[0]
            actual = tray.slot_polygon_px('column', center_override=center,
                                           angle_offset_deg=pose.theta_deg)
            np.testing.assert_allclose(actual, expected, atol=1e-6,
                                       err_msg='槽位中心与长轴必须跟随同一个刚体变换')
            corrected = apply_pose_to_slot(slot, pose)
            poly = capsule_polygon(corrected['center'], corrected['length'],
                                   corrected['width'], corrected['angle_deg'])
            np.testing.assert_allclose(tray.mm_to_px(poly), expected, atol=1e-6)
    assert slot['center'] == [120., 40.] and slot['angle_deg'] == 35.
    # ★ 非仿射单应：跨视野移动后，像素方向也会变，不能只乘一个GSD。
    projective = TrayMap(wp, homography=np.array(
        [[1.4, .08, 220.], [-.05, 1.45, 130.], [-.00015, .0002, 1.]]))
    nominal = projective.mm_to_px([[80., 80.], [81., 80.]])
    ref_axis = nominal[1] - nominal[0]
    ref_angle = float(np.degrees(np.arctan2(ref_axis[1], ref_axis[0])))
    for turn in (-15., 0., 20.):
        affine = cv2.getRotationMatrix2D((80., 80.), turn, 1.)
        affine[:, 2] += (115., -25.)
        moved_mm = cv2.transform(np.array([[[80., 80.], [81., 80.]]]), affine)[0]
        observed = projective.mm_to_px(moved_mm)
        axis = observed[1] - observed[0]
        det = TrayDetect(ok=True, center_px=tuple(observed[0]),
                         angle_deg=float(np.degrees(np.arctan2(axis[1], axis[0]))))
        pose = pose_from_tray_detect(det, nominal[0], 224., .73,
                                     nominal_angle_deg=ref_angle,
                                     pixel_to_mm=projective.px_to_mm)
        expected = cv2.transform(np.array([[[40., 40.], [120., 40.], [40., 120.], [120., 120.]]]), affine)[0]
        np.testing.assert_allclose(pose.apply([[40., 40.], [120., 40.], [40., 120.], [120., 120.]]),
                                   expected, atol=1e-7,
                                   err_msg='跨视野平移转动必须在标定平面内解几何')
    print('C模块中心旋转、平移与长槽方向测试全部通过')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
