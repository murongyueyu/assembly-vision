"""小屏操作视图：文字在最终画布绘制，视频保持比例，完成卡不阻塞。"""
from __future__ import annotations

import time
import cv2
import numpy as np

BG = (23, 19, 14)
CARD = (42, 34, 26)
INK = (244, 240, 232)
MUTED = (173, 160, 141)
GREEN = (154, 218, 65)
AMBER = (81, 196, 248)
RED = (103, 105, 246)


class OperatorView:
    def __init__(self, text, clock=time.monotonic):
        self.text, self.clock = text, clock
        self.seen = None
        self.popup_at = None
        self.frame = self.video = self.video_shape = None

    def render(self, state, width, height):
        view = state.workflow
        s = min(width / 1024., height / 600.)
        pad = max(12, round(20 * s))
        header = round(75 * s)
        footer = round(40 * s)
        grid_h = round(152 * s)
        bottom = height - footer - grid_h
        split = round(width * .56)
        canvas = np.full((height, width, 3), BG, np.uint8)
        items = []

        def text(value, x, y, size=20, color=INK, anchor='la'):
            items.append((str(value), (int(x), int(y)), color, max(18, round(size * s)), anchor))

        def lines(value, x, y, max_width, size=22, limit=2, color=INK):
            # ★ 按实际字体宽度换行；不把长错误原因缩成看不清的小字。
            font_size = max(18, round(size * s))
            font = self.text.font(font_size) if self.text.ok else None
            chunks, line = [], ''
            for ch in str(value):
                candidate = line + ch
                length = font.getlength(candidate) if font else len(candidate) * font_size
                if length > max_width and line:
                    chunks.append(line)
                    line = ch
                else:
                    line = candidate
            if line:
                chunks.append(line)
            if len(chunks) > limit:
                chunks = chunks[:limit]
                chunks[-1] = chunks[-1][:-1] + '…'
            for i, line in enumerate(chunks):
                text(line, x, y + i * round(font_size * 1.4), size, color)

        text('ASSEMBLY / 装配检测', pad, pad, 24)
        text('工序 %d / 6' % view['completed'], width - pad, pad, 24, GREEN, 'ra')
        cv2.line(canvas, (pad, header - 12), (width-pad, header-12), CARD, 1)
        # 视频只缩放一次，且相同相机帧复用缩放缓冲。
        vx, vy, vw, vh = pad, header, split-pad*2, bottom-header-pad
        if state.frame is not None:
            fh, fw = state.frame.shape[:2]
            scale = min(vw / fw, vh / fh)
            nw, nh = max(1, round(fw*scale)), max(1, round(fh*scale))
            ox, oy = vx+(vw-nw)//2, vy+(vh-nh)//2
            if self.frame is not state.frame or self.video_shape != (nw, nh):
                self.video = cv2.resize(state.frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                self.frame, self.video_shape = state.frame, (nw, nh)
            canvas[oy:oy+nh, ox:ox+nw] = self.video
            for poly, color, thick in state.polygons:
                points = np.asarray(poly).reshape(-1, 2)
                if len(points) >= 3:
                    points = np.rint(points*scale+(ox, oy)).astype(np.int32)
                    cv2.polylines(canvas, [points], True, state.color(color), max(1, thick), cv2.LINE_8)
            for (x,y,w,h), label, color in state.boxes:
                p = (round(x*scale+ox),round(y*scale+oy))
                cv2.rectangle(canvas,p,(round((x+w)*scale+ox),round((y+h)*scale+oy)),state.color(color),2)
                # 仅主标签；字符按最终像素画，不再二次缩小。
                text(label, p[0], max(vy, p[1]-round(24*s)), 18, state.color(color))
        else:
            cv2.rectangle(canvas,(vx,vy),(vx+vw,vy+vh),CARD,-1)
            text('等待相机画面',vx+vw//2,vy+vh//2,24,MUTED,'mm')

        rx, rw = split, width-split-pad
        text('当前操作', rx, header, 18, MUTED)
        lines(view['action'],rx,header+round(30*s),rw,30,2,GREEN if view['complete'] else INK)
        y = header+round(120*s)
        measurement = view.get('measurement', {})
        if measurement:
            value = measurement.get('length_mm',measurement.get('height_mm'))
            passed = measurement.get('ok') and measurement.get('height_conforms') is True
            text(('%.2f mm' % value) if value is not None else '测量无效', rx,y,32,GREEN if passed else RED)
            text('合格' if passed else '请复测',width-pad,y,22,GREEN if passed else RED,'ra')
        else:
            text('组装用时（含检测）',rx,y,18,MUTED)
            text(view['elapsed'],rx,y+round(28*s),32,AMBER)
        message = (measurement.get('reason', '') if measurement and not measurement.get('ok')
                   else view.get('reason', '') if not measurement and view['completed'] in (2,5)
                   else '')
        if message:
            lines(message,rx,header+round(205*s),rw,18,2,AMBER)
        for i,(name,rule) in enumerate(view.get('rules',())):
            text('%s  %s' % (name,rule),rx,bottom-round((62-i*27)*s),18,MUTED)

        cell_w = (width-pad*4)//3
        cell_h = (grid_h-pad)//2
        for i,(label,status) in enumerate(zip(view['labels'],view['statuses'])):
            x = pad+(i%3)*(cell_w+pad)
            y = bottom+(i//3)*(cell_h+pad//2)
            color = GREEN if status=='done' else RED if status=='failed' else AMBER if status=='current' else MUTED
            cv2.rectangle(canvas,(x,y),(x+cell_w,y+cell_h),CARD,-1)
            if status in ('current','failed'):
                cv2.rectangle(canvas,(x,y),(x+cell_w,y+cell_h),color,2)
            text('%02d' % (i+1),x+round(12*s),y+round(14*s),22,color)
            text(label,x+round(55*s),y+round(11*s),22,INK if status!='pending' else MUTED)
            text({'done':'已完成','failed':'未通过 · 按 I 复测','current':'当前步骤','pending':'待进行'}[status],
                 x+round(55*s),y+round(39*s),18,color)

        text('I 检测  /  R 复位  /  F 全屏  /  Q 退出',pad,height-footer+round(10*s),18,MUTED)
        text(view['elapsed'],width-pad,height-footer+round(8*s),22,AMBER,'ra')
        warning = state.banner or ('长度检测未完成，请复位后按六步操作' if view.get('early_pick') else '')
        if warning:
            cv2.rectangle(canvas,(vx,vy),(vx+vw,vy+round(65*s)),(56,45,99),-1)
            lines(warning,vx+round(10*s),vy+round(8*s),vw-round(20*s),20,2,INK)
        self.text.draw(canvas,items)

        # ★ 第一次实际渲染才开始计展示时长；缓存不能冻结到期判断。
        now = self.clock()
        ident = tuple(view['completion_id'])
        safe = view['complete'] and not warning and not view.get('fault')
        if safe and ident != self.seen:
            self.seen, self.popup_at = ident, now
        if not safe:
            self.popup_at = None
        if self.popup_at is not None and now-self.popup_at < view['completion_seconds']:
            self._completion(canvas,view,s,now-self.popup_at)
        return canvas

    def _completion(self,canvas,view,scale,age):
        h,w = canvas.shape[:2]
        # 轻量淡入，无遮罩阻塞、无粒子动画。
        strength = min(1.,age/.18)
        cv2.convertScaleAbs(canvas,alpha=1.-.52*strength,dst=canvas)
        cw,ch = min(w-40,round(620*scale)),round(340*scale)
        x,y = (w-cw)//2,(h-ch)//2
        r0 = round(22*scale)
        cv2.rectangle(canvas,(x+r0,y),(x+cw-r0,y+ch),CARD,-1)
        cv2.rectangle(canvas,(x,y+r0),(x+cw,y+ch-r0),CARD,-1)
        for px,py in ((x+r0,y+r0),(x+cw-r0,y+r0),(x+r0,y+ch-r0),(x+cw-r0,y+ch-r0)):
            cv2.circle(canvas,(px,py),r0,CARD,-1,cv2.LINE_AA)
        cv2.line(canvas,(x+r0,y),(x+cw-r0,y),GREEN,3,cv2.LINE_AA)
        cx = w//2
        cy = y+round(56*scale)
        r = round(25*scale)
        cv2.circle(canvas,(cx,cy),r,GREEN,-1,cv2.LINE_AA)
        check=np.array([(cx-r//2,cy),(cx-r//8,cy+r//3),(cx+r//2,cy-r//3)],np.int32)
        cv2.polylines(canvas,[check],False,BG,max(2,round(4*scale)),cv2.LINE_AA)
        data=[('BINGO!',(cx,y+round(110*scale)),INK,round(56*scale),'mm'),
              ('本轮检测完成',(cx,y+round(163*scale)),GREEN,round(28*scale),'mm'),
              ('组装总用时  '+view['elapsed'],(cx,y+round(219*scale)),INK,round(34*scale),'mm'),
              ('含检测用时 · 按 R 开始下一轮',(cx,y+round(279*scale)),MUTED,round(19*scale),'mm')]
        self.text.draw(canvas,data)
