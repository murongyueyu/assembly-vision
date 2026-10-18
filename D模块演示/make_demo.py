# -*- coding: utf-8 -*-
"""用本轮已核对原图生成分场景D演示视频；不改算法、配置或现场证据。"""
from pathlib import Path
import hashlib
import json
import sys
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AV = ROOT / 'assembly_vision'
EVIDENCE = ROOT / 'ai_测试工具/D模块联检/20261018_01'
sys.path.insert(0, str(AV))
from assemvision.config import load_config
from assemvision.pipeline import Pipeline
from assemvision.types import Frame
from assemvision.imgio import imread_safe, imwrite_safe, videowriter_safe
from assemvision.ui import make_renderer


def main():
    # ★ 每个片段重置A/D状态；视频明确是分场景回放，不伪装为连续操作。
    scenes = [
        ('live_recovered_01', '四件正确：独立身份与槽位匹配，稳定后允许开始'),
        ('live_missing_wheel_01', '缺轮盘：明确指出缺件，不借别槽或盘外零件补位'),
        ('live_swapped_03', '底座 / 顶盖互换：具体错件报警，同时轮盘缺件'),
        ('live_hand_box_01', '手套遮挡：只画手框，等待确认，禁止放行'),
        ('live_recovered_01', '恢复正确：重新稳定后放行；本演示不启动工序'),
    ]
    target = HERE / 'D-prestart-demo.mp4'
    if target.exists():
        raise FileExistsError('保留已有视频，重新生成请另存后再运行')
    renderer = make_renderer('opencv', fullscreen=False, display=(1280,720))
    writer = videowriter_safe(str(target), 'mp4v', 10., (1280,784))
    if writer is None:
        raise OSError('演示视频编码器无法打开')
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 24)
    manifest = []
    frames_written = 0
    try:
        for scene_index, (folder_name, caption) in enumerate(scenes, 1):
            folder = EVIDENCE / folder_name
            report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
            cam = load_config(str(AV/'config/camera.yaml'))
            wp = load_config(str(AV/'config/workpiece.yaml'))
            pipe = Pipeline(cam, wp, load_config(str(AV/'config/calibration.yaml')),
                            source_override='images', path_override=str(folder),
                            prestart_check_path=str(AV/'config/prestart_check.yaml'))
            pipe._d_intrinsics = report['metadata']['intrinsics']
            clock = [0.]
            samples = report['samples'][:60]
            records = []
            with patch('assemvision.pipeline.time.monotonic', side_effect=lambda:clock[0]):
                for sample in samples:
                    path = folder / sample['raw']
                    if hashlib.sha256(path.read_bytes()).hexdigest() != sample['raw_sha256']:
                        raise ValueError('原图hash不符')
                    image = imread_safe(str(path))
                    clock[0] = sample['captured_at']
                    frame = Frame(sample['index'], clock[0], left=image)
                    pipe._c_input = frame, clock[0]
                    pipe._c_process_once()
                    result = pipe.get_prestart_result()
                    state = pipe.build_ui_state()
                    if pipe.c_error:
                        raise RuntimeError(pipe.c_error)
                    canvas = renderer.render(state)
                    # 底部独立字幕条不覆盖原始D判断或相机画面。
                    padded = np.vstack([canvas, np.full((64,1280,3),24,'uint8')])
                    pil = Image.fromarray(cv2.cvtColor(padded,cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil)
                    draw.text((16,728), '%d/5  %s  |  分场景离线演示' % (scene_index,caption),
                              font=font,fill=(245,245,245))
                    output = cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
                    writer.write(output);frames_written += 1
                    records.append({'source':str(path.relative_to(ROOT)), 'index':frame.index,
                                    'state':result.state,'allow_start':result.allow_start})
                    if len(records) == len(samples):
                        if not imwrite_safe(str(HERE/('scene_%02d.png'%scene_index)),output):
                            raise OSError('演示预览保存失败')
            pipe.stop()
            manifest.append({'scene':scene_index,'name':folder_name,'caption':caption,'records':records})
    finally:
        writer.release();renderer.close()
    if not target.is_file() or target.stat().st_size == 0:
        raise OSError('视频未成功落盘')
    # 中文视频路径同样用ASCII临时副本验证解码，避免裸OpenCV路径失败。
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        decode_path = Path(temp)/'demo.mp4';shutil.copyfile(target,decode_path)
        reader = cv2.VideoCapture(str(decode_path))
        decoded = 0
        while True:
            ok, frame = reader.read()
            if not ok:break
            assert frame.shape[:2] == (784,1280)
            decoded += 1
        reader.release()
    assert decoded == frames_written == 300
    (HERE/'demo_manifest.json').write_text(json.dumps({'frames':decoded,'fps':10,'seconds':30,
        'video_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'scope':'本轮已核对原图分场景回放，不是连续现场操作或新增验收','scenes':manifest},
        ensure_ascii=False,indent=2),encoding='utf-8')
    print('Demo video verified: 300 frames, 30 seconds, 1280x784; ' + str(target))


if __name__ == '__main__':
    main()
