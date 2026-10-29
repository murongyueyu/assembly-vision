#!/usr/bin/env python3
"""Split ABCDE frame cost by thread: D-prep / D-check / C-angle workers.
Read-only instrumentation (timing wrappers only, no algorithm change)."""
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, '/home/pi/assembly_vision')
os.chdir('/home/pi/assembly_vision')

import cv2  # noqa: E402
import app  # noqa: E402
from assemvision.config import load_config  # noqa: E402
from assemvision.core.part_identity import FrozenPartIdentity  # noqa: E402
from assemvision.core.segment import Segmenter  # noqa: E402
from assemvision.core.tray_observer import TrayObserver  # noqa: E402
from assemvision.pipeline import Pipeline  # noqa: E402

BUCKETS = {}
CONFIG = 'config/abcde_integration_pi_performance_candidate.yaml'


def cpu_snapshot():
    line = Path('/proc/stat').read_text().splitlines()[0].split()[1:]
    vals = list(map(int, line))
    idle = vals[3] + vals[4]
    return sum(vals[:8]), idle


def timed(name, fn):
    def wrapper(*a, **k):
        key = '%s@%s' % (name, threading.current_thread().name)
        start = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            BUCKETS.setdefault(key, []).append((time.perf_counter() - start) * 1000.0)
    return wrapper


Segmenter.segment = timed('segment', Segmenter.segment)
TrayObserver.observe = timed('C_observe', TrayObserver.observe)
FrozenPartIdentity.observe = timed('A_identity', FrozenPartIdentity.observe)


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    pipe = Pipeline(load_config('config/camera.yaml'), load_config('config/workpiece.yaml'),
                    load_config('config/calibration.yaml'), integration_path=CONFIG)
    runtime = pipe.deployment_runtime or {}
    cv2.setNumThreads(int(runtime.get('opencv_threads', 0)))
    app._gc_tune(None)
    counts = {'det': 0, 'acq': 0}
    proc0, read0 = pipe._d_process_once, pipe.source.read

    def process():
        ok = proc0()
        if ok:
            counts['det'] += 1
        return ok

    def read(*a, **k):
        frame = read0(*a, **k)
        if frame is not None:
            counts['acq'] += 1
        return frame

    pipe._d_process_once = process
    pipe.source.read = read
    pipe.start()
    app._gc_freeze()
    time.sleep(6.0)
    for name in list(BUCKETS):
        BUCKETS[name].clear()
    counts['det'] = counts['acq'] = 0
    busy0, idle0 = cpu_snapshot()
    t0 = time.monotonic()
    time.sleep(seconds)
    window = time.monotonic() - t0
    busy1, idle1 = cpu_snapshot()
    det, acq = counts['det'], counts['acq']
    pipe.stop()
    total, idle = (busy1 - busy0), (idle1 - idle0)
    cpu_pct = 100.0 * (total - idle) / max(1.0, total)
    out = {'window_s': round(window, 1), 'det_fps': round(det / window, 2),
           'acq_fps': round(acq / window, 2),
           'per_frame_ms': round(1000 * window / max(1, det), 0),
           'cpu_busy_pct_4core': round(cpu_pct, 1)}
    for name, values in sorted(BUCKETS.items()):
        if not values:
            continue
        out[name] = {'n': len(values), 'per_frame': round(len(values) / max(1, det), 2),
                     'p50_ms': round(statistics.median(values), 1),
                     'p95_ms': round(sorted(values)[int(len(values) * 0.95) - 1], 1),
                     'cpu_s_per_frame_ms': round(sum(values) / max(1, det), 0)}
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
