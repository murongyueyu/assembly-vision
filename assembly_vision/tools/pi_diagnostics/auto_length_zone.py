"""独立自动检测框联调：黄框等待，完整候选稳定2秒后自动参考测量。

★ 不证明三维平放，不判尺寸合格，不推进六步。独立标定完成后再接生产流程。
"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from assemvision.config import load_config
from assemvision.core.segment import Segmenter
from assemvision.core.length import measure_length
from assemvision.sources.factory import make_source
from assemvision.imgio import imwrite_safe
from assemvision.ui.auto_length_gate import PlacementGate
from assemvision.ui.auto_length_zone import candidate
from assemvision.ui.zone_geometry import measurement_mask, calibrated_measure, endpoints_visible
from assemvision.ui.opencv_renderer import _Text
from assemvision.ui.base import find_cjk_font


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/auto_length_zone_preview.yaml')
    parser.add_argument('--windowed',action='store_true')
    args=parser.parse_args()
    settings=load_config(str(ROOT/args.config)).data
    zone=tuple(int(v) for v in settings['zone'])
    calibration=json.loads((ROOT/'config'/settings['calibration_file']).read_text(encoding='utf-8'))
    gate=PlacementGate(settings)
    camera=load_config(str(ROOT/'config/camera.yaml'))
    camera.set('depth.enabled',False)
    source=make_source(camera)
    segmenter=Segmenter(load_config(str(ROOT/'config/workpiece.yaml')))
    import app
    W,H=app._detect_screen()
    text=_Text(find_cjk_font(),1.)
    cv2.setNumThreads(1)
    output=ROOT/'data/probe'/time.strftime('f_auto_length_zone_%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    (output/'settings.json').write_text(json.dumps(settings,indent=2,ensure_ascii=False),encoding='utf-8')
    window='Automatic Length Zone - PREVIEW'
    measured=None;number=0;previous_measured=False;last_image=None
    try:
        source.open()
        cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window,cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_NORMAL if args.windowed else cv2.WINDOW_FULLSCREEN)
        if args.windowed:cv2.resizeWindow(window,min(W,1280),min(H-60,720))
        while True:
            tick=time.monotonic()
            frame=source.read(want_full=False)
            if frame is None:
                gate.update(time.monotonic(), hard=True)
                print('取帧失败；稳定计时撤销',flush=True)
                if cv2.waitKey(1)&255 in (ord('q'),27):break
                continue
            image=frame.left
            fh,fw=image.shape[:2]
            x,y,w,h=zone
            if x<0 or y<0 or x+w>fw or y+h>fh:
                raise ValueError('配置检测框超出D435原始画面')
            seg=segmenter.segment(image,ts=frame.ts,scale=1.,want_features=False,
                                  work_roi=None,mm_per_px=settings['reference_mm_per_px'])
            local_mask=measurement_mask(image,seg,settings)
            metrics,reason,empty=candidate(seg,zone,settings,mask=local_mask)
            if metrics is not None and not endpoints_visible(image,local_mask,zone,settings):
                metrics=None
                reason='手可扶，请露出底座外侧面和螺纹尖端'
            status=gate.update(time.monotonic(),metrics,empty)
            if previous_measured and not status['measured']:measured=None
            if status['trigger']:
                measured,rectified=calibrated_measure(local_mask,zone,calibration)
                measured['old_fixed_scale_length_mm']=measure_length(local_mask,zone,settings['reference_mm_per_px']).get('length_mm')
                measured.update(frame_index=frame.index,zone=zone,scope='检测框联调参考值；未验证毫米精度；不作质量放行',
                                source_scale_mm_per_px=settings['reference_mm_per_px'])
                number+=1
                name='placement_%02d'%number
                imwrite_safe(str(output/(name+'.raw.png')),image)
                imwrite_safe(str(output/(name+'.mask.png')),local_mask)
                imwrite_safe(str(output/(name+'.rectified.png')),rectified)
                (output/(name+'.json')).write_text(json.dumps(measured,ensure_ascii=False,indent=2),encoding='utf-8')
                print('AUTO_MEASURE',json.dumps(measured,ensure_ascii=False),flush=True)
            previous_measured=status['measured']
            canvas=np.full((H,W,3),(23,19,14),np.uint8)
            scale=min(W*.70/fw,(H-100)/fh)
            nw,nh=round(fw*scale),round(fh*scale)
            ox,oy=24,round((H-nh)/2)
            canvas[oy:oy+nh,ox:ox+nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
            green=(154,218,65);amber=(81,196,248);white=(244,240,232);grey=(173,160,141)
            color=green if status['color']=='green' else amber
            a=(round(x*scale+ox),round(y*scale+oy));b=(round((x+w)*scale+ox),round((y+h)*scale+oy))
            cv2.rectangle(canvas,a,b,color,3,cv2.LINE_AA)
            mid=(a[1]+b[1])//2
            for px in range(a[0]+12,b[0]-12,20):
                cv2.line(canvas,(px,mid),(min(px+10,b[0]-12),mid),color,1)
            fs=max(18,round(H/600*20));items=[]
            def line(value,xx,yy,size=20,col=white):
                items.append((value,(int(xx),int(yy)),col,max(18,round(size*H/600)),'la'))
            line('自动长度检测 · 校准联调',24,15,24)
            line('检测区：平放完整组件',a[0],max(60,a[1]-fs-12),20,color)
            rx=round(W*.73)
            line('黄框等待 / 绿框计时',rx,H*.16,23,color)
            for i,row in enumerate((reason[:13],reason[13:])):
                if row:line(row,rx,H*(.24+i*.045),20)
            if status['measured'] and measured:
                line('自动测量已完成',rx,H*.39,24,green)
                value=measured.get('length_mm')
                line(('%.2f mm'%value) if value is not None else '测量无效',rx,H*.48,34,amber)
                line('按当前姿态校64',rx,H*.60,19,amber)
                line('未验收，仅参考',rx,H*.67,19,amber)
                line('移出组件后可再测',rx,H*.72,19,grey)
            else:
                elapsed=min(settings['hold_seconds'],status['elapsed'])
                line('稳定 %.1f / %.1f 秒'%(elapsed,settings['hold_seconds']),rx,H*.43,25,color)
                line('连续绿框2秒自动测量',rx,H*.54,20,grey)
                line('允许轻微手抖',rx,H*.60,19,grey)
            line('手可辅助扶平；勿垫高底座或遮住端点',24,H-48,20,amber)
            line('R 重测   Q 退出',rx,H-48,20,grey)
            text.draw(canvas,items)
            if status['trigger']:
                imwrite_safe(str(output/('placement_%02d.ui.png'%number)),canvas)
            if last_image is None:
                imwrite_safe(str(output/'initial.ui.png'),canvas)
                print('ZONE_READY',output,zone,flush=True)
            last_image=canvas
            cv2.imshow(window,canvas)
            key=cv2.waitKey(1)&255
            if key in (ord('q'),27):break
            if key==ord('r'):gate.reset();measured=None;previous_measured=False
            delay=max(0.,1./settings['render_fps']-(time.monotonic()-tick))
            if delay:time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__=='__main__':
    main()
