#!/usr/bin/env python3
"""G真实窗口性能记录：分别测采集、检测、绘图率，运行中保存，不修改判据。"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from unittest.mock import patch

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import app
from assemvision.imgio import imwrite_safe


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=0, help='0表示持续运行')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lock, stop = threading.Lock(), threading.Event()
    counters, durations, pipes, rows = [0,0,0,0], [], [], deque(maxlen=600)
    metadata = dict(scope='真实当前场景运行性能；非重新装配验收',
                    config=str(args.config.resolve()),
                    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())
    real_pipe, real_renderer = app.Pipeline, app.make_renderer

    def child_cpu():
        total, count = 0., 0
        for pipe in pipes:
            executor = getattr(pipe.observer,'_executor',None)
            processes = dict(getattr(executor,'_processes',None) or {})
            for pid in processes:
                try:
                    values = Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
                    total += (int(values[11])+int(values[12]))/os.sysconf('SC_CLK_TCK')
                    count += 1
                except (OSError,ValueError):
                    pass
        return total,count

    def system_cpu():
        path=Path('/proc/stat')
        if not path.exists():
            return None
        values=list(map(int,path.read_text().splitlines()[0].split()[1:]))
        return sum(values[:8]),values[3]+values[4]

    def pipeline(*a, **kw):
        pipe = real_pipe(*a, **kw)
        pipes.append(pipe)
        read, process, begin = pipe.source.read, pipe._d_process_once, pipe.start
        def capture(*pos, **kwargs):
            frame = read(*pos, **kwargs)
            if frame is not None:
                with lock:
                    counters[0] += 1
            return frame
        def observe():
            start = time.perf_counter()
            changed = process()
            if changed:
                with lock:
                    counters[1] += 1
                    durations.append((time.perf_counter()-start)*1000)
                    save_raw = counters[1] == 10
                if save_raw:
                    with pipe.sh.lock:
                        snapshot = pipe._d_snapshot
                    if snapshot is not None:
                        raw = args.output/'current_scene.raw.png'
                        ok = imwrite_safe(str(raw),snapshot[0].left)
                        metadata['scene_snapshot'] = dict(saved=bool(ok),frame_index=snapshot[0].index,
                            path=str(raw),scope='性能当前场景原图；不是新增独立装配试验')
            return changed
        def startup():
            begin()
            metadata.update(source=pipe.source.name,intrinsics=pipe._d_intrinsics,
                            runtime=pipe.deployment_runtime, timing=pipe.deployment_timing)
        pipe.source.read, pipe._d_process_once, pipe.start = capture, observe, startup
        return pipe

    def renderer(*a, **kw):
        ui = real_renderer(*a, **kw)
        render, show, run = ui.render, ui.show, ui.run
        sized = [False]
        def paint(state):
            before = getattr(ui, 'canvases_built', 0)
            canvas = render(state)
            with lock:
                counters[2] += 1
                counters[3] += getattr(ui, 'canvases_built', 0)-before
            return canvas
        def display(canvas):
            running = show(canvas)
            if not sized[0]:
                cv2.resizeWindow(ui.window,1000,500)
                sized[0] = True
            return running
        def loop(provider, **options):
            return run(provider,quit_flag=stop,**options)
        ui.render, ui.show, ui.run = paint, display, loop
        return ui

    def write_report(final=False):
        data = dict(metadata=metadata, samples=list(rows),final=final,
                    error=pipes[0].c_error if pipes else 'pipeline尚未构造',
                    threads_stopped=all(not t.is_alive() for p in pipes for t in p._threads))
        tmp = args.output/'report.json.tmp'
        tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        tmp.replace(args.output/'report.json')

    def monitor():
        previous, previous_cpu, previous_counts = time.monotonic(),time.process_time(),[0,0,0,0]
        previous_child,_=child_cpu()
        previous_system=system_cpu()
        started = previous
        while not stop.wait(2.):
            now, cpu = time.monotonic(),time.process_time()
            child,children=child_cpu()
            system=system_cpu()
            with lock:
                counts = list(counters)
                mean_ms = sum(durations)/len(durations) if durations else None
                durations.clear()
            row = dict(elapsed_s=now-started,
                       acquisition_fps=(counts[0]-previous_counts[0])/(now-previous),
                       detection_fps=(counts[1]-previous_counts[1])/(now-previous),
                       render_fps=(counts[2]-previous_counts[2])/(now-previous),
                       painting_fps=(counts[3]-previous_counts[3])/(now-previous),
                       process_cpu_percent=100*(cpu-previous_cpu)/(now-previous),
                       mean_detection_ms=mean_ms)
            row['child_cpu_percent']=100*max(0.,child-previous_child)/(now-previous)
            row['process_family_cpu_percent']=row['process_cpu_percent']+row['child_cpu_percent']
            row['worker_processes']=children
            if system is not None and previous_system is not None:
                row['machine_cpu_percent']=100*(1-(system[1]-previous_system[1])/max(1,system[0]-previous_system[0]))
            thermal=Path('/sys/class/thermal/thermal_zone0/temp')
            if thermal.exists():
                row['temperature_c']=int(thermal.read_text())/1000.
            rows.append(row)
            with (args.output/'samples.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')
            write_report()
            previous,previous_cpu,previous_counts=now,cpu,counts
            previous_child,previous_system=child,system
            if (args.seconds and now-started>=args.seconds) or (pipes and pipes[0].sh.quit):
                stop.set()

    worker = threading.Thread(target=monitor,name='G-performance',daemon=True)
    worker.start()
    try:
        with patch.object(app,'Pipeline',side_effect=pipeline),patch.object(app,'make_renderer',side_effect=renderer):
            return app.main(['--integration',str(args.config.resolve()),'--windowed','--display','960x540'])
    finally:
        stop.set()
        worker.join(4.)
        write_report(final=True)


if __name__=='__main__':
    raise SystemExit(main())
