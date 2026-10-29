#!/usr/bin/env python3
"""Measure ABCDE integration pipeline rates (read-only instrumentation).

video side  : frames delivered by the camera source (source.read)
detect side : frames fully processed through A/C/D/E (Pipeline._c_process_once)

Wrapping only counts calls; no algorithm, threshold, config or output changes.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = '/home/pi/assembly_vision'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2  # noqa: E402

import app  # noqa: E402
from assemvision.config import load_config  # noqa: E402
from assemvision.pipeline import Pipeline  # noqa: E402


def temp_c():
    try:
        out = subprocess.run(['vcgencmd', 'measure_temp'], capture_output=True, text=True, timeout=3)
        return out.stdout.strip()
    except Exception:
        return ''


def pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/abcde_integration_pi_functional_candidate.yaml')
    ap.add_argument('--seconds', type=float, default=40.0)
    ap.add_argument('--warmup', type=float, default=15.0)
    ap.add_argument('--label', default='run')
    args = ap.parse_args()

    cam = load_config('config/camera.yaml')
    wp = load_config('config/workpiece.yaml')
    cal = load_config('config/calibration.yaml')
    pipe = Pipeline(cam, wp, cal, integration_path=args.config)

    runtime = getattr(pipe, 'deployment_runtime', None) or {}
    if runtime:
        cv2.setNumThreads(int(runtime.get('opencv_threads', 0)))
    thr = app._gc_tune(runtime.get('gc_threshold'))

    acq_times, det_times = [], []
    original_read = pipe.source.read
    original_process = pipe._c_process_once

    def counted_read(*a, **k):
        frame = original_read(*a, **k)
        if frame is not None:
            acq_times.append(time.monotonic())
        return frame

    def counted_process():
        ok = original_process()
        if ok:
            det_times.append(time.monotonic())
        return ok

    pipe.source.read = counted_read
    pipe._c_process_once = counted_process

    print('[measure] temp_before=%s' % temp_c(), flush=True)
    pipe.start()
    frozen = app._gc_freeze()
    print('[measure] label=%s config=%s source=%s gc=%s frozen=%d runtime=%s' %
          (args.label, args.config, pipe.source.name, thr, frozen, runtime), flush=True)

    t0 = time.monotonic()
    started = None
    next_log = t0 + 3.0
    end = t0
    while True:
        now = time.monotonic()
        if started is None and now - t0 >= args.warmup:
            started = now
            print('[measure] window starts (warmup %.1fs, acq=%d det=%d)' %
                  (args.warmup, len(acq_times), len(det_times)), flush=True)
        if now >= next_log:
            next_log += 5.0
            print('[measure] t=%5.1f acq=%d det=%d' % (now - t0, len(acq_times), len(det_times)), flush=True)
        if now - t0 > args.warmup + args.seconds:
            end = now
            break
        if pipe.sh.quit:
            end = now
            print('[measure] pipeline quit early: %s' % pipe.c_error, flush=True)
            break
        time.sleep(0.05)

    error = pipe.c_error
    pipe.stop()
    print('[measure] temp_after=%s' % temp_c(), flush=True)

    acq = [t for t in acq_times if t >= started]
    det = [t for t in det_times if t >= started]
    window = max(1e-6, end - started)
    video_fps = len(acq) / window
    det_fps = len(det) / window
    gaps = [det[i + 1] - det[i] for i in range(len(det) - 1)]
    agaps = [acq[i + 1] - acq[i] for i in range(len(acq) - 1)]
    summary = {
        'label': args.label,
        'config': args.config,
        'runtime': runtime,
        'window_s': round(window, 2),
        'video_frames': len(acq),
        'det_frames': len(det),
        'video_fps': round(video_fps, 2),
        'detection_fps': round(det_fps, 2),
        'frame_time_ms': round(1000.0 / det_fps, 1) if det_fps else None,
        'video_gap_ms_p50': round(pct(agaps, 0.5) * 1000, 1) if agaps else None,
        'det_gap_ms_p50': round(pct(gaps, 0.5) * 1000, 1) if gaps else None,
        'det_gap_ms_p95': round(pct(gaps, 0.95) * 1000, 1) if gaps else None,
        'det_gap_ms_max': round(max(gaps) * 1000, 1) if gaps else None,
        'processing_dropped': len(acq) - len(det),
        'c_error': error,
    }
    print('[measure] SUMMARY ' + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
