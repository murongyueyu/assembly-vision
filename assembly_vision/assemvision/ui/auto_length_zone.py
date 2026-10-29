"""自动长度框的候选几何检查；独立工具与正式六步共用。

只检查框内可见金属轮廓是否可测，不推断三维姿态，也不因蓝手套靠近端点就否决。
"""
import cv2
import numpy as np


def candidate(seg, zone, settings, mask=None):
    x, y, w, h = zone
    if mask is None:
        mask = seg.brass_mask if seg.brass_mask is not None else seg.metal_mask
    if mask is None:
        return None, '没有有效金属掩膜', False
    sub = mask[y:y+h, x:x+w]
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= settings['min_contour_area_px']]
    if not contours:
        return None, '请将底座和立柱成品平放入框', True
    if (not settings.get('allow_glove_support', False) and seg.glove_mask is not None
            and np.count_nonzero(seg.glove_mask[y:y+h, x:x+w]) > settings['max_glove_pixels']):
        return None, '请将手移出检测框', False
    if len(contours) != 1:
        return None, '请只放一个组件，等待完整轮廓', False
    c = contours[0]
    bx, by, bw, bh = cv2.boundingRect(c)
    margin = settings['edge_margin_px']
    if min(bx, by, w-bx-bw, h-by-bh) < margin:
        return None, '请将组件完整移入框内，避开边缘', False
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(c)
    ratio = max(rw, rh) / max(1., min(rw, rh))
    if not settings['min_aspect'] <= ratio <= settings['max_aspect']:
        return None, '请按示意平放底座和立柱组件', False
    return (cx+x, cy+y, max(rw, rh)+1.), '轮廓完整，保持静止', False
