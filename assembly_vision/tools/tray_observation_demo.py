# -*- coding: utf-8 -*-
"""C空/满/部分占用现场检查；s采5帧，q退出，不改A/B、工序或合格判据。

--plan gray16每次只增减一件，覆盖四槽全部16种组合；保存声明标签待原图复核。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from assemvision.config import load_config
from assemvision.core.segment import Segmenter
from assemvision.core.tray_observer import TrayObserver
from assemvision.imgio import imread_safe,imwrite_safe
from assemvision.sources.factory import make_source


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paint(frame,observation,observer):
    vis=frame.copy()
    if observation.orientation_ok:
        for slot in observer.tray.slots:
            center=observation.pose.apply([slot['center']])[0]
            polygon=observer.tray.slot_polygon_px(slot['id'],center_override=center,
                                                   angle_offset_deg=observation.pose.theta_deg)
            state=observation.slots[slot['id']]['state']
            color={'empty':(255,255,0),'occupied':(0,255,0),'unknown':(0,180,255)}[state]
            cv2.polylines(vis,[np.rint(polygon).astype('int32')],True,color,2)
            pixel=tuple(np.rint(observer.tray.mm_to_px(center[None,:])[0]).astype(int))
            cv2.putText(vis,slot['id'].replace('slot_','')+':'+state,pixel,
                        cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
    return vis


def instruction_panel(vis, lines, font):
    """中文提示置于原图上方，避免覆盖料盘，也不改保存的原图。"""
    panel=Image.new('RGB',(vis.shape[1],160),(20,20,20))
    draw=ImageDraw.Draw(panel)
    for i,line in enumerate(lines):
        draw.text((10,5+30*i),line,font=font,
                  fill=(255,225,70) if i in (1,3) else (245,245,245))
    return np.vstack([cv2.cvtColor(np.asarray(panel),cv2.COLOR_RGB2BGR),vis])


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--model',default=str(ROOT/'config/tray_observation.yaml'))
    parser.add_argument('--plan',choices=['free','gray16','rotation18'],default='free')
    parser.add_argument('--image',default='',help='离线单图冒烟，不打开相机')
    parser.add_argument('--max-frames',type=int,default=0)
    args=parser.parse_args()
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=False)
    wp=load_config(str(ROOT/'config/workpiece.yaml')); cal=load_config(str(ROOT/'config/calibration.yaml'))
    cam=load_config(str(ROOT/'config/camera.yaml'))
    observer=TrayObserver.from_file(wp,cal,args.model); segmenter=Segmenter(wp)
    font=ImageFont.truetype(str(Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/msyh.ttc'),18)
    cfg=observer.settings; order=cfg['slot_order']
    names={'slot_base':'底座','slot_column':'立柱','slot_wheel':'轮盘','slot_cap':'顶盖'}
    targets=[15^(i^(i>>1)) for i in range(16)] if args.plan=='gray16' else []
    # ★ 补测计划只声明摆盘目标，必须另看原图，不把提示当人工真值。
    directions=['右下','正下','左下','正上','右上','正右']
    if args.plan=='rotation18':
        targets=[15]*6+[0]*6+[1,2,4,8,5,10]
    protected=[ROOT/'config'/name for name in ['camera.yaml','workpiece.yaml','calibration.yaml']]
    protected += [Path(args.model),Path(args.model).parent/cfg['reference_image']]
    protected += [Path(args.model).parent/view['image'] for view in cfg.get('reference_views', [])]
    hashes={str(p):file_hash(p) for p in protected}
    report={'config_sha256':hashes,'plan':args.plan,'slot_order':order,'targets':targets,
            'labels_reviewed':False,'scenes':[],'frames':0,'error':None,
            'scope':'C geometry/occupancy only; frozen A masks; no assembly identity/steps/quality'}
    if args.plan=='rotation18': report['direction_targets']=directions*3
    source=None; scene=None; remaining=0; index=0; frame_index=0; last_arm=-1e9
    feedback='尚未保存；先点击图像窗口，让键盘焦点进入窗口'
    window='C TRAY OBSERVATION | s=sample  q=quit | no assembly verdict'
    image=imread_safe(args.image) if args.image else None
    if args.image and image is None:
        raise ValueError('读不到指定的离线检查图')
    def guide():
        if not targets: return
        target=targets[index]
        occupied=[names[s] for i,s in enumerate(order) if target&(1<<i)]
        if args.plan=='rotation18':
            print('第%d/18组：轮盘槽朝画面%s；有件：%s；其余槽空，手退出按S。'%(
                index+1,directions[index%6],'、'.join(occupied) or '全部为空'),flush=True)
            return
        action='四件都在各自槽内'
        if index:
            changed=target^targets[index-1]; i=changed.bit_length()-1
            action=('放回' if target&changed else '取走')+'【'+names[order[i]]+'】'
        print('第%d/16组：%s；目标有件：%s；手退出后轻按s。'%(index+1,action,'、'.join(occupied) or '空盘'),flush=True)
    try:
        if image is None:
            source=make_source(cam,None,None); source.open()
            if hasattr(source,'apply_locked_controls'): source.apply_locked_controls()
            for _ in range(30): source.read(want_full=True)
        cv2.namedWindow(window,cv2.WINDOW_NORMAL); cv2.resizeWindow(window,900,900)
        guide()
        while True:
            if image is None:
                frame=source.read(want_full=True)
                if frame is None: raise RuntimeError('当前帧读取失败')
                bgr=frame.left_full if frame.left_full is not None else frame.left
            else: bgr=image.copy()
            frame_index+=1
            start=time.perf_counter()
            segmentation=segmenter.segment(bgr,scale=1.,mm_per_px=observer.tray.mm_per_px())
            observation=observer.observe(bgr,segmentation.metal_mask,segmentation.glove_mask,
                                         brass_mask=segmentation.brass_mask)
            duration=(time.perf_counter()-start)*1000
            vis=paint(bgr,observation,observer)
            if targets:
                goal='  '.join(names[s]+('：有件' if targets[index]&(1<<i) else '：空') for i,s in enumerate(order))
                action='保持四件都在正确槽内'
                if index and args.plan!='rotation18':
                    changed=targets[index]^targets[index-1]; bit=changed.bit_length()-1
                    action=('放回' if targets[index]&changed else '取走')+'【'+names[order[bit]]+'】'
                prompt='第%d/16组：%s；手退出后按 S'%(index+1,action)
                if args.plan=='rotation18':
                    prompt='第%d/18组：轮盘槽朝%s；手退出后按 S'%(index+1,directions[index%6])
            else:
                goal='青框：空槽  绿框：有物体  黄框：不确定'
                prompt='自由检查：手退出后按 S 保存'
            state={'full':'满盘','empty':'空盘','partial':'部分占用','unknown':'不确定'}[observation.state]
            vis=instruction_panel(vis,[
                '当前识别：'+state+'；朝向'+('已确认' if observation.orientation_ok else '不确定'),
                prompt,goal,feedback,
                'S 保存（每组按一次）  Q 退出 | %.1f ms'%duration],font)
            if remaining:
                n=5-remaining
                raw=scene/('frame_%02d.png'%n)
                # ★ 保存原图成功后才登记该帧，不无脑打印已保存。
                if not imwrite_safe(str(raw),bgr): raise OSError('原图保存失败：'+str(raw))
                if not imwrite_safe(str(scene/('overlay_%02d.png'%n)),vis): raise OSError('叠加保存失败')
                row={'frame':frame_index,'raw_sha256':file_hash(raw),'pipeline_ms':duration,
                     'observation':observation.to_dict()}
                (scene/('observation_%02d.json'%n)).write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
                report['scenes'][-1]['samples'].append(row)
                remaining-=1
                if not remaining:
                    feedback='第%d组已保存（5帧）；请按上方提示摆放下一组'%len(report['scenes'])
                    print('保存成功：'+scene.name+'，5帧；预测='+observation.state,flush=True)
                    if targets:
                        index+=1
                        if index==len(targets): break
                        guide()
            cv2.imshow(window,vis)
            key=cv2.waitKey(1)&255
            if key in (ord('q'),ord('Q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1: break
            if key in (ord('s'),ord('S')) and not remaining and time.perf_counter()-last_arm>=1.:
                last_arm=time.perf_counter(); remaining=5
                feedback='已收到 S，正在保存5帧，请保持摆放不动'
                scene=out/('scene_%02d'%(len(report['scenes'])+1)); scene.mkdir()
                target=targets[index] if targets else None
                report['scenes'].append({'directory':scene.name,'user_declared_mask':target,
                                         'samples':[],'labels_reviewed':False})
                if args.plan=='rotation18':
                    report['scenes'][-1]['direction_target']=directions[index%6]
                (scene/'label.json').write_text(json.dumps({'slot_order':order,'user_declared_mask':target,
                    'source':'user pressed s after arranging target; raw QA required'},ensure_ascii=False,indent=2),encoding='utf-8')
            if args.max_frames and frame_index>=args.max_frames: break
    except Exception as error:
        report['error']=str(error); print('检查中止：'+str(error),flush=True)
    finally:
        if source is not None: source.close()
        cv2.destroyAllWindows()
        report['frames']=frame_index
        report['config_unchanged']=all(file_hash(p)==digest for p,digest in hashes.items())
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('报告已保存：'+str(out/'report.json'),flush=True)
    return int(report['error'] is not None)


if __name__=='__main__': raise SystemExit(main())
