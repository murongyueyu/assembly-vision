"""毫米平面变换不使用工件真值拟合；蓝手套辅助不阻止完整金属候选。"""
import copy
from types import SimpleNamespace
import cv2
import numpy as np
from assemvision.ui.zone_geometry import calibrated_measure,measurement_mask,endpoints_visible
from assemvision.ui.auto_length_zone import candidate


def main():
    cal=dict(image_size=[320,240],intrinsics=dict(fx=100.,fy=100.,ppx=160.,ppy=120.),
             floor_plane_normal=[0.,0.,1.],floor_plane_offset_calibrated_mm=100.,
             axis_height_mm=0.,rectified_pixels_per_mm=4.,distortion_coefficients=[0.]*5,
             calibration_scope='synthetic',validation_truth_length_mm=64.3,
             axis_band_half_width_mm=1.5,shaft_fit_fraction=[.35,.85],min_axis_pixels=5)
    values=[]
    for x,y in [(30,40),(190,150)]:
        mask=np.zeros((240,320),np.uint8);mask[y:y+15,x:x+64]=255
        a,_=calibrated_measure(mask,(x-10,y-10,90,40),cal)
        assert a['ok'] and abs(a['length_mm']-64.)<.6
        other=copy.deepcopy(cal);other['validation_truth_length_mm']=999.
        b,_=calibrated_measure(mask,(x-10,y-10,90,40),other)
        assert a['length_mm']==b['length_mm']
        values.append(a['length_mm'])
    assert abs(values[0]-values[1])<.6
    assert a['length_scale']==1.
    scaled=copy.deepcopy(cal);scaled['length_scale']=.5
    c,_=calibrated_measure(mask,(180,140,90,40),scaled)
    assert abs(c['length_mm']-a['length_mm']*.5)<1e-6
    assert abs(c['width_mm']-a['width_mm']*.5)<1e-6
    assert abs(c['length_unscaled_mm']-a['length_mm'])<1e-6
    bad=copy.deepcopy(cal);bad['distortion_coefficients'][0]=.1
    try:calibrated_measure(mask,(180,140,90,40),bad)
    except ValueError:pass
    else:raise AssertionError('不能静默忽略非零畸变')
    settings=dict(zone=[10,10,180,100],green_hsv_lower=[35,40,0],green_hsv_upper=[95,255,255],
                  glove_hsv_lower=[91,60,0],glove_hsv_upper=[140,255,255],min_seed_pixels=20,
                  min_contour_area_px=60.,edge_margin_px=8,min_aspect=1.15,max_aspect=5.,
                  max_glove_pixels=10,allow_glove_support=True)
    image=np.full((130,220,3),(40,140,40),np.uint8)
    seed=np.zeros((130,220),np.uint8)
    image[45:60,40:150]=(90,160,190);image[30:90,145:158]=(100,180,200)
    seed[45:60,40:150]=255;seed[30:90,145:158]=255
    image[80:100,60:100]=(220,80,30)
    glove=np.zeros_like(seed);glove[80:100,60:100]=255
    seg=SimpleNamespace(brass_mask=seed,metal_mask=seed,glove_mask=glove)
    mask=measurement_mask(image,seg,settings)
    assert not mask[90,70] and mask[50,100]
    metrics,_,_=candidate(seg,settings['zone'],settings,mask)
    assert metrics is not None
    settings.update(endpoint_fit_fraction=[.35,.85],endpoint_axis_band_px=4,
                    endpoint_margin_px=6,max_endpoint_glove_pixels=3)
    assert endpoints_visible(image,mask,settings['zone'],settings)
    occluded=image.copy();occluded[47:57,155:170]=(220,80,30)
    shortened=measurement_mask(occluded,seg,settings)
    assert endpoints_visible(occluded,shortened,settings['zone'],settings)  # 邻近手套不能当作遮挡
    assert not endpoints_visible(image,np.zeros_like(mask),settings['zone'],settings)
    edge=np.zeros_like(mask);edge[50:66,13:81]=255
    settings['edge_margin_px']=3
    assert candidate(seg,settings['zone'],settings,edge)[0] is not None
    edge=np.zeros_like(mask);edge[50:66,12:80]=255
    assert candidate(seg,settings['zone'],settings,edge)[0] is None
    settings['allow_glove_support']=False
    assert candidate(seg,settings['zone'],settings,mask)[0] is None
    print('test_zone_geometry 全部通过')


if __name__=='__main__':main()
