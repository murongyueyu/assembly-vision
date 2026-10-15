# -*- coding: utf-8 -*-
"""枚举本机可用摄像头，并逐个后端/分辨率尝试打开，找到能用的组合。"""
import cv2

print("OpenCV 版本:", cv2.__version__)

BACKENDS = [
    ("MSMF", cv2.CAP_MSMF),
    ("DSHOW", cv2.CAP_DSHOW),
]

print("\n===== 第一步：只用 backend 的默认参数枚举 0-5 =====")
for name, be in BACKENDS:
    for idx in range(0, 6):
        cap = cv2.VideoCapture(idx, be)
        if cap.isOpened():
            ret, frame = cap.read()
            shape = frame.shape if ret and frame is not None else "读取失败"
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_s = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            print(f"  [{name}] index={idx} -> 打开成功 默认=({w:.0f}x{h:.0f}) FOURCC={fourcc_s} shape={shape}")
            cap.release()
        else:
            print(f"  [{name}] index={idx} -> 打不开")

print("\n===== 第二步：不再指定 CAP_DSHOW，用默认后端试 0-5 =====")
for idx in range(0, 6):
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"  [默认后端] index={idx} -> 打开成功 read={ret} shape={frame.shape if ret else None}")
        cap.release()
    else:
        print(f"  [默认后端] index={idx} -> 打不开")

print("\n===== 第三步：模拟你原来的写法 (DSHOW + 1280x480 + MJPG) =====")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("  VideoCapture(0, CAP_DSHOW).isOpened() =", cap.isOpened())
if cap.isOpened():
    ok1 = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    ok2 = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok3 = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    ret, frame = cap.read()
    print(f"  set返回 w={ok1} h={ok2} fourcc={ok3}; 实际=({cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}) read={ret}")
cap.release()

print("\n===== 第四步：常见分辨率逐个试 (index=0, MSMF, MJPG) =====")
for (w, h, f) in [(640, 480, "MJPG"), (1280, 720, "MJPG"), (848, 480, "MJPG"), (1920, 1080, "MJPG")]:
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("  打不开，跳过")
        break
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*f))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ret, frame = cap.read()
    print(f"  请求 {w}x{h} {f} -> 实际=({cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}) read={ret}")
    cap.release()

print("\n===== 第五步：RealSense SDK 是否可用 =====")
try:
    import pyrealsense2 as rs
    print("  pyrealsense2 已安装, 版本:", rs.__version__)
    for d in rs.context().query_devices():
        print("  设备:", d.get_info(rs.camera_info.name), "| SN:", d.get_info(rs.camera_info.serial_number))
        for s in d.sensors:
            print("     传感器:", s.get_info(rs.camera_info.name))
except ImportError:
    print("  pyrealsense2 未安装 —— 深度/双目流无法通过 OpenCV 读取，需要 pip install pyrealsense2")
