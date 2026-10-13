# -*- coding: utf-8 -*-
"""相机实时预览 —— 排查"打不开 / 没有画面 / 帧率异常"的第一件工具。

它只做一件事：打开相机，把画面显示出来，并在画面上实时打印
实际分辨率与实测帧率。用来快速回答：
    * 画面到底有没有？（"没有画面"最常见的原因是帧率太低，窗口空白）
    * 实测帧率是多少？（驱动回报的 CAP_PROP_FPS 经常是假的，必须实测）
    * 左右眼拼接对不对？（画面上画一条中缝线）

用法
    python tools/live_preview.py                    # 用 camera.yaml 的配置
    python tools/live_preview.py --mode 1280x720    # 强制某个拼接图分辨率
    python tools/live_preview.py --raw              # 显示不拆分的原始拼接图
    python tools/live_preview.py --save one.png     # 按 s 保存当前帧到指定文件

窗口按键：s = 保存一帧，r = 重置帧率统计，q / ESC = 退出
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from assemvision.imgio import imwrite_safe                       # noqa: E402
from assemvision.sources.base import split_stereo                # noqa: E402
from assemvision.sources.uvc_stereo import (default_backend_name,  # noqa: E402
                                            resolve_backend)


def main() -> int:
    ap = argparse.ArgumentParser("live_preview")
    ap.add_argument("--device", default=0)
    ap.add_argument("--backend", default="")
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--mode", default="", help="拼接图分辨率，如 1280x720；默认读 camera.yaml")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--raw", action="store_true", help="显示未拆分的原始拼接图")
    ap.add_argument("--save", default="preview.png")
    # ── 曝光与白平衡（★ 调阈值的先决条件）──────────────────────
    # 不锁死这两项，HSV 阈值会随光照漂移；而且相机默认的自动曝光
    # 在"桌面大面积白纸"这种场景下会严重过曝（实测灰度均值 245、66% 像素顶到 255）。
    ap.add_argument("--exposure", type=float, default=None,
                    help="手动曝光值（负数越小越暗）。不给=不改动")
    ap.add_argument("--gain", type=float, default=None, help="增益")
    ap.add_argument("--wb", type=float, default=None, help="白平衡色温 K")
    ap.add_argument("--no-auto", action="store_true",
                    help="关闭自动曝光与自动白平衡（建议开启）")
    # ★ 色彩补偿：本机实测相机的红通道偏弱（白参考 R/B≈0.74），画面偏青蓝。
    #   驱动拒绝了 WB_TEMPERATURE / AUTO_WB，改不动 ISP，只能在软件层补。
    #   1.0 = 不补；建议从 1.2 试起，边看边调，以"白纸看不出青"为准。
    ap.add_argument("--rgain", type=float, default=1.0, help="红通道增益（默认 1.0 不补）")
    ap.add_argument("--bgain", type=float, default=1.0, help="蓝通道增益（默认 1.0 不补）")
    args = ap.parse_args()

    # 默认从 camera.yaml 读分辨率与拼接方式，保持与正式运行一致
    w = h = 0
    cap_mode, left_index = "side_by_side", 0
    try:
        from assemvision.config import load_config
        cam = load_config(os.path.join(ROOT, "config", "camera.yaml"))
        w = int(cam.get("capture.width", 1280))
        h = int(cam.get("capture.height", 720))
        cap_mode = str(cam.get("capture.mode", "side_by_side"))
        left_index = int(cam.get("capture.left_index", 0))
    except Exception as e:                                     # noqa: BLE001
        print("[preview] 读 camera.yaml 失败（用默认 1280x720）:", e)
        w, h = 1280, 720
    if args.mode:
        try:
            w, h = (int(v) for v in args.mode.lower().replace("*", "x").split("x"))
        except Exception:                                      # noqa: BLE001
            print("--mode 格式不对，应为 1280x720")
            return 1

    backend_name = args.backend or default_backend_name()
    dev = int(args.device) if str(args.device).isdigit() else args.device

    print("=" * 70)
    print(" 打开相机: device=%r 后端=%s  FOURCC=%s  请求 %dx%d @%.0ffps"
          % (dev, backend_name, args.fourcc, w, h, args.fps))
    print("=" * 70)

    cap = cv2.VideoCapture(dev, resolve_backend(backend_name))
    if not cap.isOpened():
        print("★ 打不开相机。检查：")
        print("   1) 是否被其它程序占用（浏览器、会议软件、Windows 相机 App、其它 Python）")
        print("   2) 拔插一次 USB；换一个 USB 口（优先主板后置口，不要用 HUB）")
        print("   3) 线材是否插紧（这类相机松动会导致协商降到很低的速率）")
        return 1
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:                                          # noqa: BLE001
        pass
    time.sleep(0.6)

    # ── 曝光与白平衡 ───────────────────────────────────────────
    # ★ 为什么必须做：相机默认自动曝光在"桌面大面积白纸"下会严重过曝
    #   （实测灰度均值 245、66% 像素顶到 255、B/G 通道 97% 饱和，画面信息基本丢光）。
    #   而且 HSV 阈值全靠光照稳定，自动曝光/白平衡一动阈值就白标了。
    auto_exp = None
    if args.no_auto:
        try:
            if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25=手动（DSHOW 约定）
            if hasattr(cv2, "CAP_PROP_AUTO_WB"):
                cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
        except Exception:                                      # noqa: BLE001
            pass
    if args.exposure is not None and hasattr(cv2, "CAP_PROP_EXPOSURE"):
        cap.set(cv2.CAP_PROP_EXPOSURE, float(args.exposure))
    if args.gain is not None and hasattr(cv2, "CAP_PROP_GAIN"):
        cap.set(cv2.CAP_PROP_GAIN, float(args.gain))
    if args.wb is not None and hasattr(cv2, "CAP_PROP_WB_TEMPERATURE"):
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(args.wb))
    exp_now = cap.get(cv2.CAP_PROP_EXPOSURE) if hasattr(cv2, "CAP_PROP_EXPOSURE") else 0.0
    time.sleep(0.4)
    # ★ 预热：改完曝光后相机要若干帧才生效，不丢几帧的话开头会是黑屏/过曝。
    #   标定脚本(_util.open_source)里同样有这一步，理由相同。
    for _ in range(12):
        cap.read()
    time.sleep(0.15)
    print("曝光: 手动=%s  当前 exposure=%s  gain=%s  WB=%s"
          % (bool(args.no_auto), exp_now,
             cap.get(cv2.CAP_PROP_GAIN) if hasattr(cv2, "CAP_PROP_GAIN") else "-",
             cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
             if hasattr(cv2, "CAP_PROP_WB_TEMPERATURE") else "-"))

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported = float(cap.get(cv2.CAP_PROP_FPS))
    print("驱动回报: %dx%d @%.1f fps（★ 这个值经常是假的，以下面实测为准）"
          % (aw, ah, reported))

    win = "live preview  (s=save  [ ] = exposure  -/= gain  r=reset  q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1280, max(640, aw)), min(800, max(360, ah)))

    n = 0
    t0 = time.time()
    fps = 0.0
    got_any = False
    exp_cur = float(args.exposure) if args.exposure is not None else -6.0
    gain_cur = float(args.gain) if args.gain is not None else 0.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            if not got_any:
                print("★ 打不开画面：read() 一直失败。拔插 USB、换口、关掉占用程序后重试。")
                break
            time.sleep(0.01)
            continue
        got_any = True
        n += 1
        # ── 软件色彩补偿（RGB 顺序：索引 2 是 R，索引 0 是 B）──
        if abs(args.rgain - 1.0) > 1e-6 or abs(args.bgain - 1.0) > 1e-6:
            frame = frame.astype(np.float32)
            frame[:, :, 2] = np.clip(frame[:, :, 2] * args.rgain, 0, 255)
            frame[:, :, 0] = np.clip(frame[:, :, 0] * args.bgain, 0, 255)
            frame = frame.astype(np.uint8)
        dt = time.time() - t0
        if dt >= 1.0:
            fps = n / dt
            n, t0 = 0, time.time()
            # ★ 低帧率必须显式提示：很多人以为"没有画面"，其实是 1 fps 的窗口还没刷新
            warn = "  ★★ 帧率异常低！" if fps < 5.0 else ""
            print("  实测 %.1f fps   实际分辨率 %dx%d%s" % (fps, frame.shape[1], frame.shape[0], warn))

        show = frame.copy()
        if not args.raw:
            # 画中缝，方便确认左右眼拼接是否正确
            if cap_mode == "side_by_side":
                cv2.line(show, (show.shape[1] // 2, 0), (show.shape[1] // 2, show.shape[0]),
                         (0, 0, 255), 1)
            else:
                cv2.line(show, (0, show.shape[0] // 2), (show.shape[1], show.shape[0] // 2),
                         (0, 0, 255), 1)
        # ── 曝光自查：过曝是调阈值前必须先解决的问题 ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_v = float(gray.mean())
        sat = float((gray >= 250).mean() * 100.0)
        warn_exp = ""
        if mean_v > 200 or sat > 30:
            warn_exp = "   ★★ 过曝！调小曝光（按 [ ）"
        txt = "%.1f fps  %dx%d  mean=%.0f sat=%.0f%%%s" % (
            fps, frame.shape[1], frame.shape[0], mean_v, sat, warn_exp)
        cv2.putText(show, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(show, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(win, show)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("s"):
            p = args.save if os.path.isabs(args.save) else os.path.join(ROOT, "data", args.save)
            print("  已保存: %s" % p if imwrite_safe(p, frame) else "  保存失败: %s" % p)
        if k == ord("r"):
            n, t0 = 0, time.time()
            print("  帧率统计已重置")
        # ★ 曝光/增益实时微调：找到不过曝的值后记下来，写进 camera.yaml
        if k == ord("["):
            exp_cur -= 1.0
            cap.set(cv2.CAP_PROP_EXPOSURE, exp_cur)
            print("  曝光 -> %.1f" % exp_cur)
        if k == ord("]"):
            exp_cur += 1.0
            cap.set(cv2.CAP_PROP_EXPOSURE, exp_cur)
            print("  曝光 -> %.1f" % exp_cur)
        if k == ord("-"):
            gain_cur = max(0.0, gain_cur - 5.0)
            cap.set(cv2.CAP_PROP_GAIN, gain_cur)
            print("  增益 -> %.1f" % gain_cur)
        if k == ord("="):
            gain_cur += 5.0
            cap.set(cv2.CAP_PROP_GAIN, gain_cur)
            print("  增益 -> %.1f" % gain_cur)

    cap.release()
    cv2.destroyAllWindows()
    if got_any and fps < 5.0:
        print()
        print("★ 结论：能出图，但实测帧率只有 %.1f fps。" % fps)
        print("  驱动回报的 %.1f fps 是假的。按项目需求（去抖/取件事件）至少要 10 fps。" % reported)
        print("  处理顺序：① 拔插 USB ② 换主板后置口、不要用 HUB ③ 换一根短线 ④ 关掉占用程序")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
