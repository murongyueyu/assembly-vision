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
    ap.add_argument("--tray-observer", default=None,
                    help="显式开启独立C占用观察；当前候选选config/tray_observation_v2.yaml")
    ap.add_argument("--tray-observer-runtime", default="config/tray_observer_runtime.yaml",
                    help="C显示运行配置（不改匹配/占用参数）")
    ap.add_argument('--prestart-check', default=None,
                    help='显式D开工身份与槽位联检：config/prestart_check.yaml；不启动E')
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))

    def _p(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(here, p)

    cam = load_config(_p(args.camera))
    wp = load_config(_p(args.workpiece))
    calib = load_config(_p(args.calibration), required=False)
    if not calib.data and not args.tray_observer:
        print("[app] 没找到标定文件 %s —— 将以兜底 ROI 运行（只能看连通性）" % args.calibration)

    pipe = Pipeline(cam, wp, calib,
                    source_override=args.source, path_override=args.path,
                    tray_observer_path=_p(args.tray_observer) if args.tray_observer else None,
                    tray_observer_runtime_path=_p(args.tray_observer_runtime),
                    prestart_check_path=_p(args.prestart_check) if args.prestart_check else None)
    rc = 0
    renderer = None
    try:
        W, H = (int(v) for v in args.display.lower().split("x"))
        pipe.start()
        print("[app] 已启动：%s" % pipe.source.name)
        if pipe.d_mode:
            print('[app] D开工身份/槽位联检；未启动工序、深度或终检')
        elif pipe.c_mode:
            print("[app] 独立C占用观察；模型=%s；未启用生产流程" % _p(args.tray_observer))
        if args.no_ui:
            t_last = 0.0
            frames = 0
            while not pipe.sh.quit:
                st = pipe.build_ui_state()
                now = time.time()
                if now - t_last > 0.5:
                    t_last = now
                    if pipe.c_mode:
                        print("[%s] %s | %s" % ('D' if pipe.d_mode else 'C', st.title, " / ".join(st.detail)))
                    else:
                        print("[%s] %s | 步 %d/%d | %s"
                              % (st.level, st.title, st.step, st.total,
                                 " / ".join(st.detail[-2:])))
                time.sleep(0.02)
                frames += 1
                if args.max_frames and frames >= args.max_frames:
                    break
        else:
            renderer = make_renderer(
                args.renderer, window='D Prestart Check' if pipe.d_mode else "C料盘占用观察" if pipe.c_mode else "装配流程检查",
                fullscreen=not args.windowed, display=(W, H),
                panel_ratio=float(wp.get("ui.panel_ratio", 0.20)),
                font_path=str(wp.get("ui.font_path", "")),
                colors=wp.get("ui.colors", None),
                video_fps=int(wp.get("ui.video_fps", 30)),
                actions={} if pipe.c_mode else {"inspect": pipe.request_inspection,
                                                "reset": pipe.reset_process})
            rc = renderer.run(pipe.build_ui_state,
                              fps=int(wp.get("ui.video_fps", 30)),
                              save_video=args.save_video,
                              max_frames=args.max_frames)

    except KeyboardInterrupt:
        print("\n[app] 用户中断")
    finally:
        try:
            if renderer is not None:
                renderer.close()
        finally:
            pipe.stop()
    if pipe.c_error:
        print(("[app] D模式失败：" if pipe.d_mode else "[app] C模式失败：") + pipe.c_error)
        return 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
