# -*- coding: utf-8 -*-
"""一次性长度测量工具：不跑 ABCDE 流程，直接开相机量一件横放件的轮廓长度。

用途：验证"长度法"在真机上准不准、标定 mm/px 对不对。
      ABCDE 里的 2/4 立柱阶段用的是同一套 `core/length.py`，所以这里的读数
      就是流程里会得到的读数（差别只是测区由人框还是自动取最大连通域）。

用法：
    length_probe.py                      # 自动取全图最大的黄铜连通域
    length_probe.py --rect 100 200 120 40  # 指定测区 x y w h（640x480 坐标）
    length_probe.py --auto --min-len 20    # 只看长度 > 20mm 的连通域
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from assemvision.config import load_config                      # noqa: E402
from assemvision.core.length import measure_length             # noqa: E402
from assemvision.core.roi import TrayMap                       # noqa: E402
from assemvision.core.segment import Segmenter                 # noqa: E402

W, H = 640, 480


def tray_mm_per_px(calib, wp):
    H_m = np.asarray(calib.get('tray_map.homography'), dtype=np.float64)
    ref_scale = float(calib.get('tray_map.scale', 1.0))
    tray = TrayMap(wp, homography=H_m, ref_scale=ref_scale, px_per_mm=1.0, origin_px=(0.0, 0.0))
    nominal = float(calib.get('calibration.gsd_mm_per_px',
                              calib.data.get('gsd_mm_per_px', 0.0)) or 0.0)
    return float(tray.mm_per_px(1.0)), nominal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rect', nargs=4, type=int, default=None)
    ap.add_argument('--out', default='/tmp/length_probe.png')
    args = ap.parse_args()

    wp = load_config(str(ROOT / 'config/workpiece.yaml'))
    calib = load_config(str(ROOT / 'config/calibration.yaml'))
    mmpp, nominal = tray_mm_per_px(calib, wp)
    print('mm/px: TrayMap=%.5f  标定文件 gsd=%.5f' % (mmpp, nominal))

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    pipe.start(cfg)
    try:
        for _ in range(30):
            pipe.wait_for_frames(5000)
        frame = pipe.wait_for_frames(5000).get_color_frame()
        img = np.asanyarray(frame.get_data()).copy()
    finally:
        pipe.stop()

    seg = Segmenter(wp).segment(img, ts=0.0, scale=1.0, want_features=False,
                                work_roi=None, mm_per_px=mmpp)
    mask = seg.brass_mask if seg.brass_mask is not None else seg.metal_mask
    if mask is None:
        print('没有黄铜/金属掩膜')
        return 2

    rect = args.rect
    if rect is None:
        # 自动：取最大连通域，用它的轴对齐 bbox 当测区（长度仍由最小外接矩形算）
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            print('画面里没有黄铜连通域')
            return 2
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = (int(stats[idx, cv2.CC_STAT_LEFT]), int(stats[idx, cv2.CC_STAT_TOP]),
                      int(stats[idx, cv2.CC_STAT_WIDTH]), int(stats[idx, cv2.CC_STAT_HEIGHT]))
        pad = 6
        rect = [max(0, x - pad), max(0, y - pad),
                min(W - max(0, x - pad), w + 2 * pad), min(H - max(0, y - pad), h + 2 * pad)]
        print('自动测区（最大连通域 + %d px 外扩）: %s  面积=%d' % (pad, rect, stats[idx, cv2.CC_STAT_AREA]))

    got = measure_length(mask, rect, mmpp)
    print('结果: ok=%s  长度=%.3f mm  宽=%.2f mm  角度=%.1f°  面积=%.0f px²  理由=%s'
          % (got['ok'], got['length_mm'] or -1, got['width_mm'] or -1,
             got['angle_deg'] or 0, got['area_px'], got['reason']))

    vis = img.copy()
    x, y, w, h = rect
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 1)
    sub = (mask[y:y + h, x:x + w] > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        c = max(cnts, key=cv2.contourArea) + np.array([[x, y]])
        box = cv2.boxPoints(cv2.minAreaRect(c.astype(np.float32))).astype(int)
        cv2.drawContours(vis, [box], 0, (0, 0, 255), 2)
    cv2.putText(vis, '%.2f mm' % (got['length_mm'] or 0), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(args.out, vis)
    print('已保存 %s' % args.out)
    return 0 if got['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
