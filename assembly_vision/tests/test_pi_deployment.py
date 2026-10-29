"""G预览传输：限频缓存、JPEG编码及非法运行参数拒绝。"""
import io
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from deploy.pi.desktop_preview import Desktop


def main():
    raw = io.BytesIO()
    Image.new('RGB', (32,16), (40,140,80)).save(raw, format='PPM')
    clock = [10.]
    preview = Desktop(fps=5)
    with patch('deploy.pi.desktop_preview.subprocess.run',
               return_value=SimpleNamespace(stdout=raw.getvalue())) as capture, \
         patch('deploy.pi.desktop_preview.time.monotonic',side_effect=lambda:clock[0]):
        first = preview.capture()
        assert first[:2] == b'\xff\xd8'
        with Image.open(io.BytesIO(first)) as frame:
            assert frame.size == (32,16) and frame.format == 'JPEG'
        assert preview.capture() is first and capture.call_count == 1
        clock[0] += .21
        preview.capture()
        assert capture.call_count == 2
    for fps in (0,11,True,2.5):
        try:
            Desktop(fps)
        except ValueError:
            pass
        else:
            raise AssertionError('无效预览刷新率未拒绝')
    print('Pi预览：PPM到JPEG、5Hz限频缓存与非法频率拒绝通过')


if __name__ == '__main__':
    main()
