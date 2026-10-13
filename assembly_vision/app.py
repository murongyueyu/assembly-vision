# -*- coding: utf-8 -*-
"""入口：跑整套检测。

常用例子
    # 离线跑一段视频（相机/支架没到也能开发）
    python app.py --source video --path data/videos/test.mp4

    # 无界面模式，只打印结论（方便自动化与回归测试）
    python app.py --source images --path data/frames --no-ui --max-frames 60

    # 真实相机
    python app.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assemvision.config import load_config, merge          # noqa: E402
from assemvision.pipeline import Pipeline                  # noqa: E402
from assemvision.ui import make_renderer                   # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser("assembly_vision", add_help=True)
    ap.add_argument("--camera", default="config/camera.yaml")
    ap.add_argument("--workpiece", default="config/workpiece.yaml")
    ap.add_argument("--calibration", default="config/calibration.yaml")
    ap.add_argument("--source", default=None, help="uvc | video | images")
    ap.add_argument("--path", default=None)
    ap.add_argument("--renderer", default="opencv", help="opencv | qt")
    ap.add_argument("--no-ui", action="store_true", help="无界面，只打印结论")
    ap.add_argument("--max-frames", type=int, default=0, help="跑到指定帧数后退出（0=不限）")
    ap.add_argument("--display", default="1920x1080")
    ap.add_argument("--windowed", action="store_true", help="不占全屏")
    ap.add_argument("--save-video", default="", help="把界面录成视频，便于写报告")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))

    def _p(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(here, p)

    cam = load_config(_p(args.camera))
    wp = load_config(_p(args.workpiece))
    calib = load_config(_p(args.calibration), required=False)
    if not calib.data:
        print("[app] 没找到标定文件 %s —— 将以兜底 ROI 运行（只能看连通性）" % args.calibration)

    pipe = Pipeline(cam, wp, calib,
                    source_override=args.source, path_override=args.path)
    pipe.start()
    print("[app] 已启动：%s" % pipe.source.name)

    W, H = (int(v) for v in args.display.lower().split("x"))
    rc = 0
    try:
        if args.no_ui:
            t_last = 0.0
            frames = 0
            while not pipe.sh.quit:
                st = pipe.build_ui_state()
                now = time.time()
                if now - t_last > 0.5:
                    t_last = now
                    print("[%s] %s | 步 %d/%d | %s"
                          % (st.level, st.title, st.step, st.total,
                             " / ".join(st.detail[-2:])))
                time.sleep(0.02)
                frames += 1
                if args.max_frames and frames >= args.max_frames:
                    break
        else:
            renderer = make_renderer(
                args.renderer, window="装配流程检查",
                fullscreen=not args.windowed, display=(W, H),
                panel_ratio=float(wp.get("ui.panel_ratio", 0.20)),
                font_path=str(wp.get("ui.font_path", "")),
                colors=wp.get("ui.colors", None),
                video_fps=int(wp.get("ui.video_fps", 30)),
                actions={"inspect": pipe.request_inspection,
                         "reset": pipe.reset_process})
            rc = renderer.run(pipe.build_ui_state,
                              fps=int(wp.get("ui.video_fps", 30)),
                              save_video=args.save_video,
                              max_frames=args.max_frames)
            renderer.close()
    except KeyboardInterrupt:
        print("\n[app] 用户中断")
    finally:
        pipe.stop()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
