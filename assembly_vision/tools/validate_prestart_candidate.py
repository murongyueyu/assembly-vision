# -*- coding: utf-8 -*-
"""D候选验证：真实Pipeline同帧回放或独立D窗口，原图与输出另存不覆盖。"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assemvision.config import load_config
from assemvision.imgio import imread_safe, imwrite_safe
from assemvision.pipeline import Pipeline
from assemvision.types import Frame
from assemvision.ui import make_renderer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def encode(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode), encoding='utf-8')


def snapshot(pipe):
    frame, result, observation, _, audit = pipe._d_current_snapshot()
    if frame is None:
        return None
    tracks = audit['tracks'] if audit else []
    return dict(index=frame.index, source_ts=frame.ts, captured_at=result.stamp.captured_at,
        result=dataclasses.asdict(result), observation=observation.to_dict() if observation else None,
        predictions=audit['predictions'] if audit else [],
        objects=[dict(part=o.part, confirmed=o.confirmed, current=o.current, source=o.source,
                      contour_sha256=hashlib.sha256(o.contour.tobytes()).hexdigest())
                 for o in audit['objects']] if audit else [],
        tracks=[dict(part=t.part_id, state=t.state, bbox=list(t.last_bbox), lost=t.lost_frames,
                     current=t.detail.get('visible_observed'), partial=t.detail.get('partial_observed'),
                     contour_sha256=hashlib.sha256(t.contour.tobytes()).hexdigest()
                     if t.contour is not None else None) for t in tracks],
        mask_sha256=hashlib.sha256(audit['mask'].tobytes()).hexdigest() if audit else None,
        error=pipe.c_error)


def replay(args):
    capture_path = args.capture/'capture.json'
    if not capture_path.exists():
        capture_path = args.capture/'report.json'
    capture = json.loads(capture_path.read_text(encoding='utf-8'))
    pipe = Pipeline(load_config(str(ROOT/'config/camera.yaml')),
        load_config(str(ROOT/'config/workpiece.yaml')), load_config(str(ROOT/'config/calibration.yaml')),
        source_override='images', path_override=str(args.capture), prestart_check_path=str(args.config))
    pipe._d_intrinsics = capture['metadata']['intrinsics']
    renderer = make_renderer('opencv', fullscreen=False, display=(1280, 720))
    records = []
    try:
        for i, sample in enumerate(capture['samples']):
            path = args.capture/sample['raw']
            expected_hash = sample.get('sha256', sample.get('raw_sha256'))
            assert digest(path) == expected_hash, path
            image = imread_safe(str(path))
            clock = sample['captured_at']
            with patch('assemvision.pipeline.time.monotonic', return_value=clock):
                pipe._c_input = (Frame(sample['index'], sample.get('source_ts', clock), left=image), clock)
                assert pipe._c_process_once()
                row = snapshot(pipe)
                row.update(raw=str(path.resolve()), raw_sha256=expected_hash,
                           source_ts_recorded='source_ts' in sample)
                if i in (0, 6, len(capture['samples'])-1):
                    ui = args.output/('ui_%04d.png' % i)
                    assert imwrite_safe(str(ui), renderer.render(pipe.build_ui_state()))
                    row.update(ui=ui.name, ui_sha256=digest(ui))
                records.append(row)
                if row['error']:
                    # ★ 保留源时间重复/倒退的拒绝证据，不改写原采集时间强行回放。
                    break
    finally:
        renderer.close()
        pipe.stop()
    return dict(source_capture=str(capture_path.resolve()),
                source_capture_sha256=digest(capture_path),
                declared_scene=capture.get('scene', capture.get('declared_scene')),
                actual_scene_requires_review=True, samples=records,
                source_frames=len(capture['samples']), replay_complete=len(records)==len(capture['samples']),
                error=records[-1]['error'] if records else '无采集帧')


def live(args):
    import app
    from capture_part_dataset import actual_controls
    pending, pipes, metadata = [], [], {}
    original = app.Pipeline
    renderer = make_renderer('opencv', fullscreen=False, display=(1280, 720))

    def factory(*pos, **kw):
        pipe = original(*pos, **kw)
        pipes.append(pipe)
        process, start = pipe._d_process_once, pipe.start
        def record():
            changed = process()
            if changed:
                row = snapshot(pipe)
                if row is not None:
                    frame = pipe._d_current_snapshot()[0]
                    # ★ 内存留图，结束后写盘，避免编码/写盘制造身份确认断档。
                    canvas = renderer.render(pipe.build_ui_state()) if len(pending) % 30 == 0 else None
                    pending.append((row, frame.left.copy(), canvas))
            return changed
        def begin():
            start()
            metadata.update(intrinsics=pipe._d_intrinsics, source=pipe.source.name,
                            controls_before=actual_controls(pipe.source))
        pipe._d_process_once, pipe.start = record, begin
        return pipe

    class Recorder:
        def __init__(self, *pos, **kw):
            self.real = make_renderer(*pos, **kw)
        def run(self, *pos, **kw):
            return self.real.run(*pos, **kw)
        def close(self):
            if pipes and getattr(pipes[0].source, 'profile', None) is not None:
                metadata['controls_after'] = actual_controls(pipes[0].source)
            self.real.close()

    code = None
    try:
        with patch.object(app, 'Pipeline', side_effect=factory), patch.object(app, 'make_renderer', Recorder):
            code = app.main(['--prestart-check', str(args.config), '--windowed',
                             '--max-frames', str(args.frames)])
    finally:
        renderer.close()
        rows = []
        for i, (row, image, canvas) in enumerate(pending):
            raw = args.output/('raw_%04d.png' % i)
            if not imwrite_safe(str(raw), image):
                raise OSError('原图保存失败：'+str(raw))
            row.update(raw=raw.name, raw_sha256=digest(raw))
            if canvas is not None:
                ui = args.output/('ui_%04d.png' % i)
                if not imwrite_safe(str(ui), canvas):
                    raise OSError('界面保存失败：'+str(ui))
                row.update(ui=ui.name, ui_sha256=digest(ui))
            rows.append(row)
        report = dict(exit=code, metadata=metadata, declared_scene=args.scene,
                      actual_scene_requires_review=True, samples=rows,
                      threads_stopped=all(not t.is_alive() for p in pipes for t in p._threads))
        write_json(args.output/'report.json', report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT/'config/prestart_check_wheel10_candidate.yaml')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--capture', type=Path, help='既有capture.json所在目录；不传则开启D435独立D窗口')
    ap.add_argument('--scene', default='未核对场景')
    ap.add_argument('--frames', type=int, default=300, help='现场UI轮询上限，不是采集帧数')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = replay(args) if args.capture else live(args)
    report.update(config=str(args.config.resolve()), config_sha256=digest(args.config))
    summary = dict(observations=len(report['samples']),
        states=dict(Counter(r['result']['state'] for r in report['samples'])),
        allow_start=sum(r['result']['allow_start'] for r in report['samples']))
    report['summary'] = summary
    write_json(args.output/'report.json', report)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return int(report.get('exit') or bool(report.get('error')))


if __name__ == '__main__':
    raise SystemExit(main())
