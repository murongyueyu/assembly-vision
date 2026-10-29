"""独立长度框的初步几何校准；不修改原B算法或生产判据。"""
import cv2
import numpy as np
from ..core.length import measure_length


def endpoints_visible(image, mask, zone, settings):
    """允许手套贴着底座扶持；仅验证可见轮廓几何，不凭手套距离推断遮挡。"""
    x,y,w,h=zone
    local=mask[y:y+h,x:x+w]
    yy,xx=np.nonzero(local)
    if len(xx)<int(settings['min_seed_pixels']):return False
    points=np.column_stack([xx,yy]).astype(np.float64)
    distance=cv2.distanceTransform(local,cv2.DIST_L2,5)
    _,_,_,root=cv2.minMaxLoc(distance)
    root=np.array(root,float)
    offsets=points-root
    tip=offsets[np.argmax(np.sum(offsets**2,axis=1))]
    length=np.linalg.norm(tip)
    if length<1:return False
    direction=tip/length
    u=offsets@direction
    lo,hi=settings['endpoint_fit_fraction']
    shaft=points[(u>float(lo)*length)&(u<float(hi)*length)]
    if len(shaft)<int(settings['min_seed_pixels']):return False
    ux,uy,cx,cy=cv2.fitLine(shaft.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel()
    direction=np.array([ux,uy]);center=np.array([cx,cy])
    projected=(points-center)@direction
    # ★ 金属与蓝手套相邻，既可能是正常扶持，也可能是真遮挡。
    #   不能只凭距离判遮挡；当前仅确认可见轮廓能形成杆身测量证据。
    #   精度与完整性尚未验收，不将此结果用于正式质量放行。
    return True


def measurement_mask(image, seg, settings):
    """局部绿幕/蓝手套排除，并要求与既有金属种子相连，补足反光外边缘。"""
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    green=cv2.inRange(hsv,tuple(settings['green_hsv_lower']),tuple(settings['green_hsv_upper']))
    blue=cv2.inRange(hsv,tuple(settings['glove_hsv_lower']),tuple(settings['glove_hsv_upper']))
    foreground=cv2.bitwise_not(cv2.bitwise_or(green,blue))
    x,y,w,h=settings['zone']
    local=foreground[y:y+h,x:x+w].copy()
    cs,_=cv2.findContours(local,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    seed=seg.brass_mask if seg.brass_mask is not None else seg.metal_mask
    result=np.zeros_like(foreground)
    if seed is None:return result
    for c in cs:
        if cv2.contourArea(c)<settings['min_contour_area_px']:continue
        item=np.zeros_like(local);cv2.drawContours(item,[c],-1,255,-1)
        if np.count_nonzero((item>0)&(seed[y:y+h,x:x+w]>0))>=settings['min_seed_pixels']:
            cv2.drawContours(result,[c+np.array([[[x,y]]],np.int32)],-1,255,-1)
    return result


def calibrated_measure(mask, zone, calibration):
    """相机射线投到已校准的轴线平面，正射到毫米网格，再调用原长度函数。

    ★ 假设圆盘边缘接触绿幕、轴线平行绿幕；若手垫高或轴线倾斜，数值仍可能偏。
    """
    intr=calibration['intrinsics']
    if list(mask.shape[::-1])!=calibration['image_size']:
        raise ValueError('标定分辨率与当前图像不符')
    if any(abs(v)>1e-12 for v in calibration['distortion_coefficients']):
        raise ValueError('此候选尚未支持非零畸变，不能跳过校正')
    K=np.array([[intr['fx'],0,intr['ppx']],[0,intr['fy'],intr['ppy']],[0,0,1.]],float)
    n=np.array(calibration['floor_plane_normal'],float);n/=np.linalg.norm(n)
    offset=float(calibration['floor_plane_offset_calibrated_mm'])-float(calibration['axis_height_mm'])
    if not np.isfinite(offset) or offset<=0:raise ValueError('无效测量平面')
    b1=np.cross(n,[1.,0.,0.]);b1/=np.linalg.norm(b1);b2=np.cross(n,b1)
    inv=np.linalg.inv(K)
    H=np.stack([offset*(b1@inv),offset*(b2@inv),n@inv])
    x,y,w,h=zone
    corners=np.array([[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]],np.float64)
    projected=cv2.perspectiveTransform(corners,H)[0]
    low=projected.min(0)-2.;high=projected.max(0)+2.
    ppm=float(calibration['rectified_pixels_per_mm'])
    if not np.isfinite(ppm) or ppm<=0:raise ValueError('正射分辨率无效')
    size=np.ceil((high-low)*ppm).astype(int)
    if (size<1).any() or (size>4096).any():raise ValueError('正射检测区异常')
    canvas=np.array([[ppm,0,-low[0]*ppm],[0,ppm,-low[1]*ppm],[0,0,1.]])
    local_to_image=np.array([[1.,0,x],[0,1.,y],[0,0,1.]])
    rectified=cv2.warpPerspective(mask[y:y+h,x:x+w],canvas@H@local_to_image,
                                 tuple(int(v) for v in size),flags=cv2.INTER_NEAREST)
    result=measure_length(rectified,(0,0,int(size[0]),int(size[1])),1./ppm)
    # ★ 单件比例校正：160mm 料盘锚定的是几何尺度；这件工件用自己的已知长度
    #   再做一次单件平均校正。它是“单件比例拟合”，不是独立精度验收。
    scale=float(calibration.get('length_scale',1.0))
    if not np.isfinite(scale) or scale<=0:raise ValueError('长度比例校正无效')
    if result.get('length_mm') is not None:
        result['length_unscaled_mm']=result['length_mm']
        result['length_mm']=result['length_mm']*scale
    if result.get('width_mm') is not None:
        result['width_unscaled_mm']=result['width_mm']
        result['width_mm']=result['width_mm']*scale
    result.update(scale_source='tray160_floor_plane_axis_height',
                  calibration_scope=calibration['calibration_scope'],
                  axis_height_mm=calibration['axis_height_mm'],
                  length_scale=scale,
                  length_scale_status=calibration.get('length_scale_status',''))
    return result,rectified
