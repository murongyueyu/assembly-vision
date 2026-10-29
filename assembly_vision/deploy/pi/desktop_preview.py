#!/usr/bin/env python3
"""G部署：经SSH本地转发查看Pi桌面，不开放局域网端口、不改视觉算法。"""
import argparse
import io
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>树莓派实时桌面</title><style>
body{margin:0;background:#161b22;color:#e6edf3;font:16px sans-serif}
header{padding:12px 20px}img{display:block;max-width:100%;margin:auto}
small{color:#aab5c3}button{margin-left:20px;padding:6px}
</style><header>树莓派实时桌面 · 程序在Pi上运行
<button onclick="document.documentElement.requestFullscreen()">浏览器全屏</button><br>
<small id="state">正在连接；目标每秒__FPS__次。此页面仅查看画面。</small></header>
<img id="screen" alt="树莓派桌面"><script>
const img=document.getElementById('screen'),state=document.getElementById('state');
let started=0;
function refresh(){started=performance.now();img.src='/screen.jpg?t='+Date.now()}
img.onload=()=>{state.textContent='已更新 '+new Date().toLocaleTimeString()+' · 目标__FPS__fps，仅查看画面';setTimeout(refresh,Math.max(0,__PERIOD__-(performance.now()-started)))};
img.onerror=()=>{state.textContent='画面暂不可用，正在重试';setTimeout(refresh,2000)};
refresh();</script></html>'''.encode('utf-8')


class Desktop:
    def __init__(self, fps=5):
        if type(fps) is not int or not 1 <= fps <= 10:
            raise ValueError('预览刷新率必须为1～10的整数')
        self.period = 1. / fps
        self.lock = threading.Lock()
        self.image = b''
        self.captured_at = 0.

    def capture(self):
        with self.lock:
            now = time.monotonic()
            if not self.image or now - self.captured_at >= self.period:
                # Pi的grim未编译JPEG支持；PPM经现有Pillow编码，减少网络传输。
                result = subprocess.run(['grim', '-c', '-t', 'ppm', '-'],
                                        check=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, timeout=3.)
                with Image.open(io.BytesIO(result.stdout)) as frame:
                    output = io.BytesIO()
                    frame.convert('RGB').save(output, format='JPEG', quality=80)
                self.image = output.getvalue()
                self.captured_at = now
            return self.image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18478)
    parser.add_argument('--fps', type=int, default=5, choices=range(1, 11))
    args = parser.parse_args()
    os.environ.setdefault('XDG_RUNTIME_DIR', '/run/user/%d' % os.getuid())
    os.environ.setdefault('WAYLAND_DISPLAY', 'wayland-0')
    desktop = Desktop(args.fps)
    page = PAGE.replace(b'__FPS__', str(args.fps).encode()).replace(
        b'__PERIOD__', str(round(1000 / args.fps)).encode())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split('?', 1)[0]
            if route == '/':
                content, mime = page, 'text/html; charset=utf-8'
            elif route in ('/screen.jpg', '/screen.png'):
                try:
                    content, mime = desktop.capture(), 'image/jpeg'
                except (OSError, subprocess.SubprocessError):
                    self.send_error(503, 'Desktop capture unavailable')
                    return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print('Pi desktop preview listening on loopback port %d' % args.port, flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
