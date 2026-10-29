# -*- coding: utf-8 -*-
"""一次性深度探针：不改任何产品代码，只为回答"这块区域里到底有没有 64mm 的面"。

做法：开 D435（对齐深度到彩色）→ 用裸垫参考区拟合桌面平面 → 逐像素算法向高度 →
在装配件周围打印一张 10px 一张的"高度地图"，并保存热力图。
"""
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

W, H = 640, 480
REF = (250, 340, 60, 60)          # x, y, w, h —— 与实测用的参考区一致
REGION = (330, 285, 470, 415)     # 打印区域 x0, y0, x1, y1
CELL = 10


def main():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    try:
        for _ in range(40):                      # 等自动曝光与深度流稳定
            pipe.wait_for_frames(5000)
        frames = align.process(pipe.wait_for_frames(5000))
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        z = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale * 1000.0
        img = np.asanyarray(color_frame.get_data()).copy()
        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float32)
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        valid = np.isfinite(z) & (z > 100) & (z < 2000)

        # 参考区平面（与 NativeHeightEstimator 同一套：Z=ax+by+c 迭代剔除离群）
        rx, ry, rw, rh = REF
        ref = np.zeros((H, W), bool)
        ref[ry:ry + rh, rx:rx + rw] = True
        sel = ref & valid
        pts = np.stack([x[sel], y[sel], z[sel]], axis=1).astype(np.float64)
        print('参考点数=%d  有效率=%.2f' % (len(pts), float(sel.sum()) / float(ref.sum())))
        if len(pts) < 50:
            print('参考区有效深度不足，无法拟合')
            return 2
        keep = np.ones(len(pts), bool)
        for _ in range(4):
            A = np.column_stack((pts[:, 0], pts[:, 1], np.ones(len(pts))))
            abc, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
            n = np.array([-abc[0], -abc[1], 1.0])
            n /= np.linalg.norm(n)
            off = float(abc[2] / np.linalg.norm([-abc[0], -abc[1], 1.0]))
            res = pts @ n - off
            mad = float(np.median(np.abs(res - np.median(res))))
            keep = np.abs(res - np.median(res)) <= max(2.0, 3.0 * mad)
            pts = pts[keep]
        A = np.column_stack((pts[:, 0], pts[:, 1], np.ones(len(pts))))
        abc, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
        n = np.array([-abc[0], -abc[1], 1.0])
        n /= np.linalg.norm(n)
        off = float(abc[2] / np.linalg.norm([-abc[0], -abc[1], 1.0]))
        rms = float(np.sqrt(np.mean((pts @ n - off) ** 2)))
        print('平面残差 rms=%.2f mm  内点=%d  法向=%s' % (rms, len(pts), np.round(n, 4)))

        hmap = off - (x * n[0] + y * n[1] + z * n[2])
        hmap[~valid] = np.nan

        x0, y0, x1, y1 = REGION
        print('\n== 高度地图（mm，%dpx 一格，NaN 记 -）==' % CELL)
        header = '      ' + ''.join('%5d' % xx for xx in range(x0, x1, CELL))
        print(header)
        tall = []
        for yy in range(y0, y1, CELL):
            row = []
            for xx in range(x0, x1, CELL):
                cell = hmap[yy:yy + CELL, xx:xx + CELL]
                v = cell[np.isfinite(cell)]
                if v.size < 10:
                    row.append('    -')
                else:
                    m = float(np.median(v))
                    row.append('%5.0f' % m)
                    if m > 20:
                        tall.append((xx, yy, m, float(np.max(v))))
            print('%5d ' % yy + ''.join(row))
        print('\n高于 20mm 的格子：%d 个' % len(tall))
        for xx, yy, m, mx in sorted(tall, key=lambda t: -t[2])[:8]:
            print('   (%d,%d) 中位 %.1f 最大 %.1f' % (xx, yy, m, mx))
        finite = hmap[np.isfinite(hmap)]
        print('全图高度：中位 %.1f  p99 %.1f  最大 %.1f  mm（有效像素 %d）'
              % (np.median(finite), np.percentile(finite, 99), finite.max(), finite.size))

        # 存图：彩色 + 热力图（-5..70mm）
        heat = np.clip((hmap + 5.0) / 75.0, 0.0, 1.0)
        heat = np.nan_to_num(heat, nan=0.0)
        heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite('/tmp/probe_color.png', img)
        cv2.imwrite('/tmp/probe_heat.png', heat)
        print('\n已保存 /tmp/probe_color.png 与 /tmp/probe_heat.png')
        return 0
    finally:
        pipe.stop()


if __name__ == '__main__':
    sys.exit(main())
