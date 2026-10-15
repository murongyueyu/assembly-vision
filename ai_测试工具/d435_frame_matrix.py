# -*- coding: utf-8 -*-
"""D435 出帧矩阵测试：逐个模式试，找出**真的能拿到帧**的最小配置。

★ 为什么需要：相机报 USB 2.1 时，"开流成功" 和 "能拿到帧" 是两件事。
   实测（固件 5.17.3.10 + USB2.1）：640x480@30 能开流但 wait_for_frames 5 秒超时。
   所以要按「分辨率 x 帧率 x 是否关自动曝光」扫一遍，找出可用的那一档。

只读测试，不改任何配置。
"""
from __future__ import annotations

import time

import numpy as np
import pyrealsense2 as rs


def enum(tries=6):
    for i in range(tries):
        try:
            return list(rs.context().query_devices())
        except RuntimeError as e:                                # noqa: BLE001
            print("  枚举第 %d 次失败: %s" % (i + 1, str(e).strip()))
            time.sleep(1.5)
    return []


devs = enum()
if not devs:
    print("✗ 看不到相机")
    raise SystemExit(1)
d0 = devs[0]
print("相机 %s SN=%s USB=%s 固件=%s"
      % (d0.get_info(rs.camera_info.name), d0.get_info(rs.camera_info.serial_number),
         d0.get_info(rs.camera_info.usb_type_descriptor),
         d0.get_info(rs.camera_info.firmware_version)))

# 尝试顺序：从最省带宽到最费
CASES = [
    (424, 240, 6, False),
    (424, 240, 6, True),
    (640, 480, 6, False),
    (640, 480, 6, True),
    (640, 480, 15, False),
    (640, 480, 30, False),
    (640, 480, 30, True),
    (1280, 720, 6, False),
    (1280, 720, 15, False),
    (1280, 720, 30, False),
]

results = []
for (w, h, f, lock_ae) in CASES:
    tag = "%dx%d@%d%s" % (w, h, f, " 锁AE" if lock_ae else "")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
    verdict = ""
    nframe = 0
    fps_real = 0.0
    try:
        profile = pipe.start(cfg)
    except Exception as e:                                       # noqa: BLE001
        verdict = "开流失败: %s" % str(e).strip()[:60]
        results.append((tag, verdict, nframe, fps_real))
        print("  %-18s %s" % (tag, verdict))
        try:
            pipe.stop()
        except Exception:                                        # noqa: BLE001
            pass
        time.sleep(0.8)
        continue

    # 关自动曝光（在开流之后改，和正式脚本一致）
    if lock_ae:
        try:
            for s in profile.get_device().sensors:
                if s.supports(rs.option.enable_auto_exposure):
                    s.set_option(rs.option.enable_auto_exposure, 0)
                if s.supports(rs.option.auto_exposure_priority):
                    s.set_option(rs.option.auto_exposure_priority, 0)
        except Exception as e:                                   # noqa: BLE001
            print("    （关自动曝光失败: %s）" % str(e).strip()[:50])

    # 先 poll 一会儿（非阻塞），再统计
    t0 = time.time()
    deadline = t0 + 8.0
    last = t0
    while time.time() < deadline:
        if not pipe.poll_for_frames():
            time.sleep(0.004)
            continue
        try:
            fs = pipe.wait_for_frames(1000)
        except RuntimeError:
            continue
        if fs.get_color_frame():
            nframe += 1
        if nframe == 1:
            print("    → 第一帧到达，用时 %.2f 秒" % (time.time() - t0))
        last = time.time()
        if nframe >= 60:
            break
    fps_real = nframe / max(1e-6, (last - t0)) if nframe > 1 else 0.0
    verdict = "✓ 拿到 %d 帧，实测 %.1f fps" % (nframe, fps_real) if nframe > 0 \
        else "✗ 开流了但 8 秒内 0 帧"
    results.append((tag, verdict, nframe, fps_real))
    print("  %-18s %s" % (tag, verdict))
    try:
        pipe.stop()
    except Exception:                                            # noqa: BLE001
        pass
    time.sleep(1.2)

print("\n" + "=" * 70)
print(" 汇总：哪些模式真的能出图")
print("=" * 70)
good = [r for r in results if r[2] > 0]
for tag, v, n, fps in results:
    print("  %-18s %s" % (tag, v))
if good:
    print("\n★ 可用模式：")
    for tag, v, n, fps in good:
        print("   %s" % tag)
    print("\n把这几个之一写进脚本参数即可。")
else:
    print("\n✗ 一个能出图的模式都没有。这不是分辨率问题，是 USB 链路问题。")
    print("  最该做的三件事（按性价比排序）：")
    print("   1. ★ 换到主板**后置 USB 3.0** 口（蓝色/带 SS 标记），别用前面板、别用 HUB")
    print("   2. 用 D435 **原装那条 USB-C 线**（D435 对线材极敏感，劣质线在 USB2 下就是不出图）")
    print("   3. 打开 RealSense Viewer 交叉验证：它能出图就是 Python 侧问题，")
    print("      它也出不了图就是链路/供电问题")
raise SystemExit(0 if good else 1)
