# -*- coding: utf-8 -*-
"""D435 取图 —— 已改成 RealSense SDK（原来用 cv2.VideoCapture 在这台相机上打不开）。

★ 为什么原版跑不了：
    cv2.VideoCapture(0, cv2.CAP_DSHOW) 在 D435 上**必然** False。
    D435 不是普通 UVC 摄像头，不会被 OpenCV 按「相机索引」枚举到
    （实测：MSMF / DSHOW / 默认后端 扫 0~5 全部打不开）。
    它必须由 Intel 的 SDK（pyrealsense2）直接对话硬件。

★ 原版的 1280x480 也是错的：
    那不是 D435 的任何原生模式，SDK 只会报 "Couldn't resolve requests"。
    D435 彩色只有 1920x1080 / 1280x720 / 640x480 / 424x240 这几种。

★ 本机实测（USB 2.1 连接，带宽受限，帧率上限跟着降）：
        1920x1080 -> 最高  8 fps
        1280x720  -> 最高 15 fps
        640x480   -> 最高 30 fps   ← USB2 下想要 30fps 只能用这个
    想要高分辨率 + 高帧率：插主板后置 USB 3.0（蓝色口）+ 用 D435 自带的那条线。

用法（必须用 yolov11 环境的 python）：
    PY="D:\\miniconda\\envs\\yolov11\\python.exe"
    & $PY predict.py                          # 默认 640x480@30，按 q 退出
    & $PY predict.py --width 1280 --height 720 --fps 15
    & $PY predict.py --save shot.jpg          # 启动时存一张
    & $PY predict.py --record out.mp4 --seconds 10   # 录一段视频
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

# ★ 必须用工程的安全 IO：OpenCV 在 Windows 上用窄字符 API，
#   路径里有中文（本工作区就是「视觉/用户存档」）时 cv2.imwrite 会
#   **静默失败**：不抛异常、只返回 False，看起来"已保存"其实文件不存在。
#   实测踩过：predict.py 报「已保存」但文件根本没生成。
HERE = os.path.dirname(os.path.abspath(__file__))
AV = os.path.join(os.path.dirname(HERE), "assembly_vision")
if os.path.isdir(AV):
    sys.path.insert(0, AV)

try:
    from assemvision.imgio import imread_safe, imwrite_safe, videowriter_safe
except ImportError:                                  # 工程不在旁边时的兜底
    def imread_safe(path, flags=cv2.IMREAD_COLOR):
        try:
            buf = np.fromfile(path, dtype=np.uint8)
        except OSError:
            return None
        return cv2.imdecode(buf, flags) if buf.size else None

    def imwrite_safe(path, img, params=None):
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img, list(params or []))
        if ok:
            buf.tofile(path)
        return bool(ok)

    def videowriter_safe(path, fourcc, fps, size, is_color=True):
        return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), float(fps), size, is_color)


def parse_args():
    ap = argparse.ArgumentParser("predict")
    ap.add_argument("--width", type=int, default=640, help="默认 640（USB2 下唯一能 30fps 的）")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--serial", default="", help="多台 D435 时用序列号指定")
    ap.add_argument("--save", default="stereo_spliced.jpg", help="按 s 存图的文件名")
    ap.add_argument("--save-on-start", action="store_true", help="启动就读一帧存下来（无窗口可用）")
    ap.add_argument("--record", default="", help="录制成 mp4 的路径")
    ap.add_argument("--seconds", type=float, default=0.0, help="配合 --record：录多少秒后退出")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[ERROR] 没装 pyrealsense2。执行：")
        print("  D:\\miniconda\\envs\\yolov11\\python.exe -m pip install pyrealsense2")
        return 2

    # ★ 实测：相机刚插回来时 query_devices() 会抛
    #   HResult 0x800703e3「已中止 I/O 操作」或 CreateNamedMutex Mutex_TotalFailure，
    #   等一两秒重试即可，不是相机坏了。
    devices = []
    last_err = None
    for attempt in range(6):
        try:
            devices = list(rs.context().query_devices())
            break
        except RuntimeError as e:                                # noqa: BLE001
            last_err = e
            print("[WARN] 设备枚举第 %d 次失败：%s（1.5 秒后重试）" % (attempt + 1, str(e).strip()))
            time.sleep(1.5)
    if not devices:
        print("[ERROR] RealSense SDK 看不到相机（重试 6 次都失败）。最后错误：%s" % last_err)
        print("  · 相机插好没（D435 自带的那条 USB-C 线）")
        print("  · 关掉 RealSense Viewer / Windows 相机 App —— 它们会独占设备")
        print("  · 换主板后置 USB 3.0 口（蓝色），别用 HUB")
        return 1

    dev = devices[0]
    serial = args.serial or dev.get_info(rs.camera_info.serial_number)
    try:
        usb = dev.get_info(rs.camera_info.usb_type_descriptor)
    except Exception:                                            # noqa: BLE001
        usb = "?"
    print("[INFO] %s  SN=%s  固件=%s  USB=%s"
          % (dev.get_info(rs.camera_info.name), serial,
             dev.get_info(rs.camera_info.firmware_version), usb))

    # ── 开流：先按用户要求试，失败就按 USB 实际能力降级 ──────────────
    pipe = rs.pipeline()
    cands = [(args.width, args.height, args.fps)]
    if args.width == 1280 and args.height == 720 and args.fps > 15:
        cands.append((1280, 720, 15))        # ★ USB2 下 720p 最高就是 15
    cands.append((640, 480, 30))
    profile = None
    for (w, h, f) in cands:
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
        try:
            profile = pipe.start(cfg)
            if (w, h, f) != (args.width, args.height, args.fps):
                print("[WARN] %dx%d@%d 开不了，已降级到 %dx%d@%d"
                      % (args.width, args.height, args.fps, w, h, f))
            print("[INFO] 实际模式 = %dx%d @ %d fps" % (w, h, f))
            break
        except Exception as e:                                   # noqa: BLE001
            print("[WARN] %dx%d@%d 不行（%s）" % (w, h, f, str(e).strip()))
            try:
                pipe.stop()
            except Exception:                                    # noqa: BLE001
                pass
    if profile is None:
        print("[ERROR] 所有候选模式都开不了流")
        return 1

    # 预热：丢掉前若干帧（改曝光后的前几帧还是旧的）
    for _ in range(15):
        pipe.wait_for_frames(5000)

    writer = None
    t_start = time.time()
    frames = 0
    first = True

    try:
        if args.save_on_start:
            fs = pipe.wait_for_frames(5000)
            img = np.asanyarray(fs.get_color_frame().get_data())
            ok = imwrite_safe(args.save, img)
            print("[INFO] 保存 %s -> %s  shape=%s"
                  % (args.save, "成功" if ok else "★ 失败", img.shape))
            return 0 if ok else 1

        print("[INFO] 按 q 退出，按 s 保存当前图")
        while True:
            fs = pipe.wait_for_frames(5000)
            cf = fs.get_color_frame()
            if not cf:
                print("[ERROR] 读不到彩色帧")
                break
            frame = np.asanyarray(cf.get_data())
            if frame.ndim != 3 or frame.shape[2] != 3:
                print("[ERROR] 帧格式异常 shape=%s" % (frame.shape,))
                break

            if first:
                print("[INFO] frame.shape =", frame.shape, " dtype =", frame.dtype)
                first = False

            frames += 1
            if args.record and writer is None:
                h, w = frame.shape[:2]
                # ★ 录制也走中文路径安全的 writer
                writer = videowriter_safe(args.record, "mp4v", 30.0, (w, h))
                print("[INFO] 开始录制:", args.record)
            if writer is not None:
                writer.write(frame)
            if args.seconds > 0 and time.time() - t_start >= args.seconds:
                print("[INFO] 到时，录制 %.1f 秒共 %d 帧" % (time.time() - t_start, frames))
                break

            cv2.imshow("D435", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                ok = imwrite_safe(args.save, frame)
                print("[INFO] 保存 %s -> %s" % (args.save, "成功" if ok else "★ 失败"))
    finally:
        if writer is not None:
            writer.release()
        pipe.stop()
        cv2.destroyAllWindows()

    dt = time.time() - t_start
    if dt > 0 and frames:
        print("[INFO] 平均 %.1f fps（读 %d 帧 / %.1f 秒）" % (frames / dt, frames, dt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
