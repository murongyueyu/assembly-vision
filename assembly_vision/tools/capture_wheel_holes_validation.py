# -*- coding: utf-8 -*-
"""A 新批次现场采集：复用 A 预览原入口，内存保留原图后统一写盘。

不接入 D/E，不改正式配置；目录拒绝覆盖。每组须先由用户确认实际摆放。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from unittest.mock import patch

import cv2

from part_dataset import ROOT, read_json, write_json, save_image, file_hash
from part_track_demo import main as preview_main
from assemvision.sources import factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--frames', type=int, default=150)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = read_json(args.model)
    warmup = model['policy']['capture']['warmup_frames']
    pending, metadata = [], {}
    canvases = {}
    original = factory.make_source
    original_show = cv2.imshow

    def record_preview(name, canvas):
        number = len(pending)
        if number in (1, args.frames//2, args.frames):
            canvases[number] = canvas.copy()
        return original_show(name, canvas)

    def recorded_source(*pos, **kw):
        source = original(*pos, **kw)
        read, close = source.read, source.close
        reads = [0]
        def record(*p, **k):
            frame = read(*p, **k)
            if frame is not None:
                reads[0] += 1
                if reads[0] > warmup:
                    image = frame.left_full if frame.left_full is not None else frame.left
                    pending.append((frame.index, frame.ts, time.monotonic(), image.copy()))
            return frame
        def finish():
            try:
                metadata['intrinsics'] = source.color_intrinsics()
            finally:
                close()
                metadata['closed'] = True
        source.read, source.close = record, finish
        return source

    code = None
    try:
        with patch.object(factory, 'make_source', side_effect=recorded_source), \
                patch.object(cv2, 'imshow', side_effect=record_preview):
            code = preview_main(['--dataset-model', str(args.model), '--max-frames', str(args.frames),
                '--report', str(args.output/'preview.json'), '--diagnostics',
                '--diagnostic-slow-ms', '10000', '--diagnostic-limit', '1'])
    finally:
        records = []
        for i, (index, timestamp, captured, image) in enumerate(pending):
            path = args.output/('raw_%04d.png' % i)
            save_image(path, image)
            records.append(dict(raw=path.name, sha256=file_hash(path), index=index,
                                source_ts=timestamp, captured_at=captured))
        for number, canvas in canvases.items():
            save_image(args.output/('preview_%04d.png' % number), canvas)
        write_json(args.output/'capture.json', dict(exit=code, scene=args.scene, metadata=metadata,
            model_sha256=file_hash(args.model), samples=records,
            note='用户按提示摆放的新批次A功能验证；连续帧不算独立摆放；非完整精确率/召回率验收'))
        print('A现场原图保存成功：%d；退出码：%s' % (len(records), code), flush=True)


if __name__ == '__main__':
    main()
