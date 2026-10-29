# -*- coding: utf-8 -*-
"""深度可用性探针：对指定区域逐格打印"中位高度 + 有效点数"。

用来区分两件事：
  A) 那里真的没有高面（高度接近 0，且有效点很多）
  B) 那里有东西但深度打不到（高度 NaN 或有效点极少）
"""
import argparse
import sys

import numpy as np
import pyrealsense2 as rs

W, H = 640, 480


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', nargs=4, type=int, default=[390, 140, 530, 270],
                    metavar=('X0', 'Y0', 'X1', 'Y1'))
    ap.add_argument('--cell', type=int, default=8)
    ap.add_argument('--ref', nargs=4, type=int, default=[250, 340, 60, 60])
    args = ap.parse_args()
    x0, y0, x1, y1 = args.region

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    try:
        for _ in range(40):
            pipe.wait_for_frames(5000)
        frames = align.process(pipe.wait_for_frames(5000))
        d = frames.get_depth_frame()
        c = frames.get_color_frame()
        z = np.asanyarray(d.get_data()).astype(np.float32) * scale * 1000.0
        intr = c.profile.as_video_stream_profile().get_intrinsics()
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float32)
        x = (uu - intr.ppx) * z / intr.fx
        y = (vv - intr.ppy) * z / intr.fy
        valid = np.isfinite(z) & (z > 100) & (z < 2000)
        rx, ry, rw, rh = args.ref
        ref = np.zeros((H, W), bool)
        ref[ry:ry + rh, rx:rx + rw] = True
        sel = ref & valid
        pts = np.stack([x[sel], y[sel], z[sel]], axis=1).astype(np.float64)
        for _ in range(4):
            A = np.column_stack((pts[:, 0], pts[:, 1], np.ones(len(pts))))
            abc, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
            nn = np.array([-abc[0], -abc[1], 1.0])
            ln = np.linalg.norm(nn)
            nn /= ln
            off = float(abc[2] / ln)
            res = pts @ nn - off
            mad = float(np.median(np.abs(res - np.median(res))))
            pts = pts[np.abs(res - np.median(res)) <= max(2.0, 3.0 * mad)]
        A = np.column_stack((pts[:, 0], pts[:, 1], np.ones(len(pts))))
        abc, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
        nn = np.array([-abc[0], -abc[1], 1.0])
        ln = np.linalg.norm(nn)
        nn /= ln
        off = float(abc[2] / ln)
        hmap = off - (x * nn[0] + y * nn[1] + z * nn[2])
        hmap[~valid] = np.nan

        cell = args.cell
        print('== 中位高度 mm（%dpx/格，- 表示无有效点）==' % cell)
        print('      ' + ''.join('%6d' % xx for xx in range(x0, x1, cell)))
        for yy in range(y0, y1, cell):
            row = []
            for xx in range(x0, x1, cell):
                v = hmap[yy:yy + cell, xx:xx + cell]
                v = v[np.isfinite(v)]
                row.append('     -' if v.size == 0 else '%6.0f' % np.median(v))
            print('%5d ' % yy + ''.join(row))
        print('\n== 有效像素数 / 该格面积 ==')
        print('      ' + ''.join('%6d' % xx for xx in range(x0, x1, cell)))
        for yy in range(y0, y1, cell):
            row = []
            for xx in range(x0, x1, cell):
                v = hmap[yy:yy + cell, xx:xx + cell]
                n = int(np.isfinite(v).sum())
                row.append('%6d' % n)
            print('%5d ' % yy + ''.join(row))
        area = cell * cell
        sub = hmap[y0:y1, x0:x1]
        print('\n区域有效像素占比 %.2f（每格满分 %d）' % (float(np.isfinite(sub).mean()), area))
        return 0
    finally:
        pipe.stop()


if __name__ == '__main__':
    sys.exit(main())
