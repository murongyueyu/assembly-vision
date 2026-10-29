#!/usr/bin/env python3
"""G性能微基准：同一批既有原图、同一搜索参数，对照C串行与方向并行。"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from assemvision.config import load_config
from assemvision.core.segment import Segmenter
from assemvision.core.tray_observer import TrayObserver
from assemvision.imgio import imread_safe


def equivalent(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equivalent(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    if isinstance(left, (float, np.floating)):
        return bool(np.isclose(left, right, atol=1e-6, rtol=0, equal_nan=True))
    return left == right


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--images', type=int, default=3)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--process-only', action='store_true')
    ap.add_argument('--cached-only', action='store_true')
    ap.add_argument('--fft-only', action='store_true')
    args = ap.parse_args()
    if args.images < 1 or args.repeats < 1:
        ap.error('images/repeats必须为正整数')
    args.output.mkdir(parents=True, exist_ok=False)
    wp = load_config(str(ROOT/'config/workpiece.yaml'))
    cal = load_config(str(ROOT/'config/calibration.yaml'))
    settings = ROOT/'config/tray_observation_v2.yaml'
    segmenter = Segmenter(wp)
    seed = TrayObserver.from_file(wp, cal, settings)
    inputs, files = [], sorted(args.capture.glob('*.raw.png'))[:args.images]
    if not files:
        raise ValueError('没有既有raw原图')
    for path in files:
        frame = imread_safe(str(path))
        seg = segmenter.segment(frame, scale=1., mm_per_px=seed.tray.mm_per_px())
        inputs.append((frame, seg.metal_mask, seg.glove_mask, seg.brass_mask))
    seed.close()
    baseline, rows = [], []
    original_threads = cv2.getNumThreads()
    try:
        modes = [(w, n, 'opencv','thread') for w,n in ((1,original_threads),(1,1),(2,1),(3,1),(4,1))]
        modes += [(w, 1, 'box_sum','thread') for w in (1,2,3,4)]
        modes += [(w, 1, 'box_sum','process') for w in (2,3,4)]
        if args.process_only:
            modes = [modes[0]]+modes[-3:]
        if args.cached_only:
            modes = [modes[0]]+[(w,1,'box_sum_cached',e)
                                for e in ('thread','process') for w in (2,3,4)]
        if args.fft_only:
            modes = [modes[0],(4,1,'box_sum_cached','thread')]+[
                (w,1,'fft_cached','thread') for w in (1,2,3,4)]
        for workers, native, backend, execution in modes:
            cv2.setNumThreads(native)
            observer = TrayObserver.from_file(wp, cal, settings, workers=workers,
                                             correlation_backend=backend,execution_backend=execution)
            timings, agrees = [], True
            try:
                warmup_started = time.perf_counter()
                for _ in range(3):
                    observer.observe(*inputs[0])
                warmup_ms = (time.perf_counter()-warmup_started)*1000
                for _ in range(args.repeats):
                    for number, image in enumerate(inputs):
                        start = time.perf_counter()
                        result = observer.observe(*image).to_dict()
                        timings.append(time.perf_counter()-start)
                        if not rows and len(baseline) < len(inputs):
                            baseline.append(result)
                        agrees = agrees and equivalent(result, baseline[number])
            finally:
                observer.close()
            row = dict(workers=workers, opencv_threads=native, backend=backend,execution=execution,calls=len(timings),
                       warmup_ms=warmup_ms,
                       median_ms=float(np.median(timings)*1000),
                       p90_ms=float(np.percentile(timings,90)*1000), equivalent=agrees)
            rows.append(row)
            report = dict(scope='既有原图C微基准；非真实新试验/实时帧率；同场可能运行旧UI',
                          raw=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
                          results=rows)
            (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(row), flush=True)
            if not agrees:
                raise AssertionError('串行/并行输出不一致，停止性能候选部署')
    finally:
        cv2.setNumThreads(original_threads)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
