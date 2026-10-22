# -*- coding: utf-8 -*-
"""ABCDE真实主程序证据：即时事件/检查点和关键同帧彩色深度，不改变流程判据。"""
from __future__ import annotations
import argparse
import copy
import dataclasses
import json
import queue
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app
from assemvision.ui import make_renderer
from assemvision.imgio import imwrite_safe
from capture_part_dataset import actual_controls
from validate_prestart_candidate import snapshot, digest, write_json


def replay(args):
    """保留原帧号/采集时刻，经真实A/C/D/E重放；只属于已参与修正的开发回归。"""
    from assemvision.config import load_config
    from assemvision.imgio import imread_safe
    from assemvision.types import Frame
    capture = json.loads((args.capture/'report.json').read_text(encoding='utf-8'))
    pipe = app.Pipeline(load_config(str(ROOT/'config/camera.yaml')),
        load_config(str(ROOT/'config/workpiece.yaml')), load_config(str(ROOT/'config/calibration.yaml')),
        source_override='images', path_override=str(args.capture), integration_path=str(args.config.resolve()))
    pipe._d_intrinsics = capture['metadata']['intrinsics']
    rows, previous = [], None
    recorded_generation = 0
    renderer = make_renderer('opencv', fullscreen=False, display=(1280,720))
    try:
        for sample in capture['samples'][:args.max_samples or None]:
            path = args.capture/sample['raw']
            assert digest(path) == sample['raw_sha256'], path
            image = imread_safe(str(path))
            now = sample['captured_at']
            with patch('assemvision.pipeline.time.monotonic', return_value=now):
                if sample.get('generation', 0) != recorded_generation:
                    pipe.reset_process()
                    recorded_generation = sample['generation']
                pipe._c_input = (Frame(sample['index'],sample['source_ts'],left=image),now)
                assert pipe._c_process_once()
                row = snapshot(pipe)
                state = pipe.build_ui_state()
                row.update(step=state.step, title=state.title, started=pipe.sequence.started,
                           done=list(pipe.sequence.done), events=copy.deepcopy(pipe.sequence.events),
                           raw=str(path.resolve()),raw_sha256=sample['raw_sha256'])
                signature = (row['step'],row['started'])
                if signature != previous:
                    ui = args.output/('ui_%06d.png'%len(rows))
                    if not imwrite_safe(str(ui),renderer.render(state)):
                        raise OSError('回放界面保存失败')
                    row.update(ui=ui.name,ui_sha256=digest(ui))
                    previous = signature
                    print(json.dumps(dict(sample=len(rows),step=state.step,index=sample['index']),ensure_ascii=False),flush=True)
                rows.append(row)
                if pipe.c_error or (args.until_four and state.step == 4):
                    break
    finally:
        renderer.close()
        pipe.stop()
        write_json(args.output/'report.json',dict(samples=rows,source=str(args.capture.resolve()),
            source_sha256=digest(args.capture/'report.json'),config_sha256=digest(args.config),
            metadata=capture['metadata'], scope='既有真实原图集成回归，非新增现场验收',error=pipe.c_error))
    print(json.dumps(dict(samples=len(rows),max_step=max(r['step'] for r in rows),
                          final_step=rows[-1]['step'],error=pipe.c_error)),flush=True)
    return int(bool(pipe.c_error))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT/'config/abcde_integration_height_3mm_candidate.yaml')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--initial-truth', default='', help='用户确认原文；目录名不充当真值')
    ap.add_argument('--capture', type=Path, help='回放本工具已有report.json目录，不改原采集时序')
    ap.add_argument('--max-samples', type=int, default=0)
    ap.add_argument('--until-four', action='store_true', help='回放首次到4/4即结束，保留原时序')
    args = ap.parse_args()
    if not args.capture and not args.initial_truth:
        ap.error('真机需先取得用户摆放确认并提供initial-truth')
    args.output.mkdir(parents=True, exist_ok=False)
    if args.capture:
        return replay(args)
    pending = queue.Queue(maxsize=16)
    rows, pipes, annotations, errors, metadata = [], [], [], [], {}
    quit_flag = threading.Event()
    confirmation = {'id': 'initial', 'truth': args.initial_truth, 'received_at': time.time()}
    annotations.append(copy.deepcopy(confirmation))
    renderer = make_renderer('opencv', fullscreen=False, display=(1280, 720))
    original = app.Pipeline

    def encode(value):
        if dataclasses.is_dataclass(value):
            return dataclasses.asdict(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    def report(code=None):
        return dict(exit=code, samples=list(rows) if code is not None else [],
            observations_count=len(rows), latest=rows[-1] if rows else None, metadata=copy.deepcopy(metadata),
            annotations=copy.deepcopy(annotations), errors=list(errors), config=str(args.config.resolve()),
            config_sha256=digest(args.config), actual_scene_requires_review=True,
            threads_stopped=all(not t.is_alive() for p in pipes for t in p._threads))

    def writer():
        while True:
            item = pending.get()
            if item is None:
                return
            if item == 'checkpoint':
                try:
                    write_json(args.output/'checkpoint.json', report())
                except Exception as exc:
                    errors.append('检查点保存失败：' + str(exc))
                continue
            row, raw, canvas, depth, measurement = item
            try:
                number = len(rows)
                path = args.output/('raw_%06d.png' % number)
                if not imwrite_safe(str(path), raw):
                    raise OSError('原图保存失败：' + str(path))
                row.update(raw=path.name, raw_sha256=digest(path))
                if canvas is not None:
                    path = args.output/('ui_%06d.png' % number)
                    if not imwrite_safe(str(path), canvas):
                        raise OSError('界面保存失败：' + str(path))
                    row.update(ui=path.name, ui_sha256=digest(path))
                if depth is not None:
                    path = args.output/('depth_%06d.npz' % number)
                    np.savez_compressed(path, **depth)
                    row.update(depth=path.name, depth_sha256=digest(path))
                if measurement is not None:
                    frame, result = measurement
                    path = args.output/('measurement_%06d.npz' % number)
                    np.savez_compressed(path, depth_mm=frame.depth_mm, points_mm=frame.depth_points_mm)
                    result['data'] = path.name
                    result['data_sha256'] = digest(path)
                    path = args.output/('measurement_%06d.png' % number)
                    if not imwrite_safe(str(path), frame.left):
                        raise OSError('测量原图保存失败')
                    result.update(raw=path.name,raw_sha256=digest(path),profile=frame.depth_profile)
                    row['measurement_evidence'] = result
                rows.append(row)
                # ★ 每条观测即时落盘，诊断可直接读JSONL，不再为导出日志关闭窗口。
                with (args.output/'observations.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, default=encode) + '\n')
                write_json(args.output/'progress.json', row)
            except Exception as exc:
                errors.append(str(exc))
                quit_flag.set()

    worker = threading.Thread(target=writer, name='E-evidence-writer')
    worker.start()

    def factory(*pos, **kw):
        pipe = original(*pos, **kw)
        pipes.append(pipe)
        process, start, reset = pipe._d_process_once, pipe.start, pipe.reset_process
        previous = [None]
        measured = [None]
        def record():
            changed = process()
            if changed:
                with pipe.sh.lock:
                    row = snapshot(pipe)
                    if row is not None:
                        state = pipe.build_ui_state()
                        row.update(started=pipe.sequence.started, done=list(pipe.sequence.done),
                                   events=copy.deepcopy(pipe.sequence.events), step=state.step,
                                   title=state.title, level=state.level, detail=state.detail,
                                   generation=pipe._e_generation, depth_profile=pipe._d_snapshot[0].depth_profile,
                                   measurement=copy.deepcopy(pipe.integration.result),
                                   confirmation=copy.deepcopy(confirmation))
                        signature = (row['step'], row['started'], row['title'],
                                     row['generation'], row['confirmation']['id'],
                                     repr(row['measurement']))
                        canvas = renderer.render(state) if signature != previous[0] else None
                        previous[0] = signature
                        frame = pipe._d_snapshot[0]
                        raw = frame.left.copy()
                        depth = None
                        if canvas is not None and frame.depth_points_mm is not None:
                            depth = dict(depth_mm=frame.depth_mm.copy(), points_mm=frame.depth_points_mm.copy())
                        measurement = None
                        candidate = pipe.integration.measurement
                        if candidate is not None:
                            key = (candidate[1]['generation'],candidate[1]['frame_index'])
                            if key != measured[0]:
                                measurement = (candidate[0],copy.deepcopy(candidate[1]))
                                measured[0] = key
                if row is not None:
                    pending.put((row, raw, canvas, depth, measurement))
            return changed
        def begin():
            start()
            metadata.update(intrinsics=pipe._d_intrinsics, source=pipe.source.name,
                            serial=getattr(pipe.source, 'dev_serial', None),
                            usb=getattr(pipe.source, 'usb', None),
                            controls_before=actual_controls(pipe.source),
                            integration_settings=pipe.integration.settings,
                            baseline_accepted=pipe.integration.camera.get('native_depth.baseline_accepted'),
                            height_assessment_accepted=pipe.integration.height_accepted,
                            height_tolerance_mm=pipe.integration.height_tolerance,
                            height_min_mm=pipe.integration.height_min,
                            height_max_mm=pipe.integration.height_max,
                            upper_limit_mm=pipe.integration.height_upper_limit)
        def resetting():
            annotations.append(dict(kind='reset', at=time.time(), confirmation=copy.deepcopy(confirmation)))
            reset()
        pipe._d_process_once, pipe.start, pipe.reset_process = record, begin, resetting
        return pipe

    class Recorder:
        def __init__(self, *pos, **kw):
            self.real = make_renderer(*pos, **kw)
            self.last_command = None
        def run(self, provider, **kw):
            def current():
                control = args.output/'control.json'
                if control.exists():
                    try:
                        command = json.loads(control.read_text(encoding='utf-8'))
                        if command['id'] != self.last_command:
                            self.last_command = command['id']
                            if command.get('action') == 'checkpoint':
                                pending.put('checkpoint')
                                return provider()
                            with pipes[0].sh.lock:
                                confirmation.clear()
                                confirmation.update(command, received_at=time.time())
                                annotations.append(copy.deepcopy(confirmation))
                            if command.get('action') == 'reset':
                                pipes[0].reset_process()
                            elif command.get('action') == 'measure':
                                pipes[0].request_inspection()
                            elif command.get('action') == 'stop':
                                quit_flag.set()
                    except (ValueError, KeyError):
                        pass  # 写入期间半个JSON稍后重读，不能影响生产流程。
                return provider()
            return self.real.run(current, quit_flag=quit_flag, **kw)
        def close(self):
            if pipes and getattr(pipes[0].source, 'profile', None) is not None:
                metadata['controls_after'] = actual_controls(pipes[0].source)
            self.real.close()

    code = None
    try:
        with patch.object(app, 'Pipeline', side_effect=factory), patch.object(app, 'make_renderer', Recorder):
            code = app.main(['--integration', str(args.config.resolve()), '--windowed', '--display', '1280x720'])
    finally:
        pending.put(None)
        worker.join()
        renderer.close()
        write_json(args.output/'report.json', report(code))
    print(json.dumps(dict(exit=code, observations=len(rows), errors=errors), ensure_ascii=False), flush=True)
    return int(code or bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
