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
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assemvision.config import load_config, merge          # noqa: E402
from assemvision.pipeline import Pipeline                  # noqa: E402
from assemvision.ui import make_renderer                   # noqa: E402


def _detect_screen(fallback=(1920, 1080)):
    """查询当前屏幕分辨率，失败时返回 fallback。

    ★ Pi4B 实测（2026-10）：现场屏幕是 1024x768，而旧版 `--display` 固定
      1920x1080 —— 画布按 1080p 渲染后再缩小到 768p 显示，
      ABCDE 集成模式稳态只有约 14fps，而渲染成本白白多花 2.6 倍。
      改成跟随屏幕后同一配置跑到 30fps 上限（相机侧也是 30fps）。
    """
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w, h = int(root.winfo_screenwidth()), int(root.winfo_screenheight())
        root.destroy()
        if w >= 320 and h >= 240:
            return w, h
    except Exception:                       # noqa: BLE001  无 X / 无 tk 都走兜底
        pass
    return fallback


def _parse_display(text, need_ui=True):
    """`--display` 解析：支持 `auto` 与 `WxH`。"""
    s = str(text).strip().lower()
    if s == "auto":
        return _detect_screen() if need_ui else (1920, 1080)
    try:
        w, h = s.split("x")
        return int(w), int(h)
    except Exception:                       # noqa: BLE001  非法值退回默认
        print("[app] --display 取值非法（%r），按 1920x1080 处理" % text)
        return 1920, 1080


_GC_THRESHOLD_DEFAULT = (5000, 50, 50)


def _gc_tune(threshold=None):
    """把循环 GC 调成低频，压掉主循环里的长停顿。

    ★ Pi4B 实测（2026-10）：默认 GC 下帧间隔最大停顿 **144.8ms**，
      关闭 GC 后降到 **60.5ms** —— 长停顿的主因是循环回收在反复扫描
      常驻对象（每帧新建的 numpy 短命对象靠引用计数就回收了，
      根本用不上循环 GC）。
      但完全关闭有内存持续增长的风险，所以改成低频：
        · 阈值放大到 (5000, 50, 50)：gen0 从 700 提到 5000，GC 次数降到约 1/7；
        · 主循环前再 `gc.freeze()`（见 `_gc_freeze`），让 gen2 的全量扫描
          只覆盖启动后新建的存活对象，那一次扫描也就不再是长停顿。
      要回退：配置里给 `gc_threshold: [700, 10, 10]` 即可恢复 CPython 默认。

    ★★ 必须严格校验「恰好 3 项、每项 > 0」：`gc.set_threshold(1, 2)` 在
      CPython 里**不报错**，缺的那项会沿用旧值（实测得到 `(1, 2, 50)`）——
      配置少写一项就变成 gen0=1 的畸形阈值，每分配一次就回收一次，
      比默认还慢得多。宁可不认这个配置，整体退回默认。
    """
    thr = threshold or _GC_THRESHOLD_DEFAULT
    try:
        vals = [int(v) for v in thr]
    except (TypeError, ValueError):          # 不是数字序列（含 None/字符串）
        vals = []
    if len(vals) != 3 or min(vals) <= 0:
        vals = list(_GC_THRESHOLD_DEFAULT)
    try:
        gc.set_threshold(*vals)
    except (TypeError, ValueError, OverflowError):   # 兜底：绝不因配置崩启动
        gc.set_threshold(*_GC_THRESHOLD_DEFAULT)
    return gc.get_threshold()


def _gc_freeze():
    """冻结启动期常驻对象（配置/相机/模型/numpy 常量），缩小后续扫描范围。

    ★ 必须先 `gc.collect()` 再 `gc.freeze()`：先清掉启动阶段的临时垃圾，
      剩下的常驻对象才值得移出分代 —— 冻结的条目之后不再参与扫描，
      每帧的 GC 停顿随之塌下来。老 Python 没有 freeze 时静默跳过。
    """
    try:
        gc.collect()
        gc.freeze()
        return gc.get_freeze_count()
    except (AttributeError, NotImplementedError):   # 老解释器没有 freeze
        return 0


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
    ap.add_argument("--display", default="auto",
                    help="渲染分辨率：auto（跟随屏幕）或 WxH，如 1024x768")
    ap.add_argument("--windowed", action="store_true", help="不占全屏")
    ap.add_argument("--save-video", default="", help="把界面录成视频，便于写报告")
    ap.add_argument("--tray-observer", default=None,
                    help="显式开启独立C占用观察；当前候选选config/tray_observation_v2.yaml")
    ap.add_argument("--tray-observer-runtime", default="config/tray_observer_runtime.yaml",
                    help="C显示运行配置（不改匹配/占用参数）")
    ap.add_argument('--prestart-check', default=None,
                    help='显式D开工身份与槽位联检：config/prestart_check.yaml；不启动E')
    ap.add_argument('--process-sequence', default=None,
                    help='显式E取件顺序推定：config/process_sequence.yaml')
    ap.add_argument('--integration', default=None,
                    help='ABCDE集成候选：config/abcde_integration_candidate.yaml')
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
                    prestart_check_path=_p(args.prestart_check) if args.prestart_check else None,
                    process_sequence_path=_p(args.process_sequence) if args.process_sequence else None,
                    integration_path=_p(args.integration) if args.integration else None)
    rc = 0
    renderer = None
    try:
        W, H = _parse_display(args.display, need_ui=not args.no_ui)
        # ★ deployment_runtime 由部署候选配置提供（线程数/UI帧率/GC阈值）；
        #   旧配置没有这一段时按默认值跑，不能因为缺字段就崩在启动前。
        runtime = getattr(pipe, 'deployment_runtime', None) or {}
        if runtime:
            import cv2
            cv2.setNumThreads(int(runtime.get('opencv_threads', 0)))
        thr = _gc_tune(runtime.get('gc_threshold'))
        render_fps = runtime.get('ui_fps', int(wp.get('ui.video_fps', 30)))
        pipe.start()
        print("[app] 已启动：%s" % pipe.source.name)
        # ★ 此时相机/配置/模型都已就位：冻结它们，主循环的 gen2 扫描只剩新对象
        print("[app] 循环GC低频：阈值 %d/%d/%d；冻结常驻对象 %d 个"
              % (thr + (_gc_freeze(),)))
        if pipe.integration is not None:
            # ★ 2026-10-28 分阶段测量：启动横幅按配置列出每个阶段的进度与窗口，
            #   不再写死"四步后测顶盖"。2026-10-29 起 2/4 立柱改量**长度**。
            rules = '；'.join(
                '%d/4 %s%s %s' % (stage['at_step'], stage['label'],
                                  '长度' if stage.get('mode') == 'length' else '高度',
                                  pipe.integration.rule_text(stage))
                for stage in sorted(pipe.integration.stages, key=lambda s: s['at_step']))
            print('[app] ABCDE集成；2/4按I自动框，4/4按I选区；测量项=%s' % rules)
        elif pipe.e_mode:
            print('[app] E取件顺序推定；未验证物理安装、紧固或高度')
        elif pipe.d_mode:
            print('[app] D开工身份/槽位联检；未启动工序、深度或终检')
        elif pipe.c_mode:
            print("[app] 独立C占用观察；模型=%s；未启用生产流程" % _p(args.tray_observer))
        else:
            print('[app] 当前为旧主流程兼容模式；ABCDE请显式选择 --integration config/abcde_integration_height_3mm_candidate.yaml')
        if args.no_ui:
            t_last = 0.0
            frames = 0
            while not pipe.sh.quit:
                st = pipe.build_ui_state()
                now = time.time()
                if now - t_last > 0.5:
                    t_last = now
                    if pipe.c_mode:
                        print("[%s] %s | %s" % ('E' if pipe.e_mode else 'D' if pipe.d_mode else 'C', st.title, " / ".join(st.detail)))
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
                args.renderer, window='ABCDE Integration' if pipe.integration is not None else 'E Pick Sequence' if pipe.e_mode else 'D Prestart Check' if pipe.d_mode else "C料盘占用观察" if pipe.c_mode else "装配流程检查",
                fullscreen=not args.windowed, display=(W, H),
                panel_ratio=float(wp.get("ui.panel_ratio", 0.20)),
                font_path=str(wp.get("ui.font_path", "")),
                colors=wp.get("ui.colors", None),
                video_fps=int(wp.get("ui.video_fps", 30)),
                actions={"reset": pipe.reset_process, "inspect": pipe.request_inspection} if pipe.integration is not None else {"reset": pipe.reset_process} if pipe.e_mode else {} if pipe.c_mode else {"inspect": pipe.request_inspection,
                                                "reset": pipe.reset_process})
            if runtime and hasattr(renderer, 'cache_unchanged'):
                renderer.cache_unchanged = True
            rc = renderer.run(pipe.build_ui_state,
                              fps=render_fps,
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
        print(("[app] E模式失败：" if pipe.e_mode else "[app] D模式失败：" if pipe.d_mode else "[app] C模式失败：") + pipe.c_error)
        return 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
