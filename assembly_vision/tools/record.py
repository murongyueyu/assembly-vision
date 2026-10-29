# -*- coding: utf-8 -*-
"""录一段真实素材 —— 相机到手后第一件事。

为什么先录素材：阈值、环带半径、覆盖率基线全都要从真实画面里调，
而支架没到时相机拿在手里也能录，比等硬件齐了再录省好几天。

录的内容要覆盖：
    1. 料盘四个槽都放好件（正面完整）
    2. 逐个取件、装配到装配位、装上顶盖（完整正确流程）
    3. 故意缺件 / 少辐条 / 取错件
    4. 手在画面里大幅移动、长时间遮挡零件
    5. 不同光照（开灯/关部分灯/侧向遮挡）

用法
    python tools/record.py --seconds 60                 # 录 60 秒
    python tools/record.py --seconds 120 --out data/videos/run1.mp4
按 q 或 ESC 提前结束。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assemvision.config import load_config            # noqa: E402
from assemvision.imgio import videowriter_safe        # noqa: E402
from assemvision.sources.factory import make_source   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default=os.path.join(ROOT, "config", "camera.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "videos", "run.mp4"))
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--source", default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--raw", action="store_true",
                    help="保存原始拼接画面（双目），否则保存左眼单画面")
    args = ap.parse_args()

    cam = load_config(args.camera)
    src = make_source(cam, args.source, args.path)
    if hasattr(src, "apply_locked_controls"):
        src.apply_locked_controls()          # type: ignore[attr-defined]
    src.open()

    writer = None
    t0 = time.time()
    n = 0
    print("开始录制，按 q / ESC 提前结束…")
    try:
        while time.time() - t0 < args.seconds:
            f = src.read(want_full=True)
            if f is None or f.left_full is None:
                time.sleep(0.01)
                continue
            img = f.left_full if not args.raw else None
            if args.raw:
                import numpy as np
                img = np.hstack([f.left_full, f.right_full]) \
                    if f.right_full is not None else f.left_full
            if writer is None:
                h, w = img.shape[:2]
                writer = videowriter_safe(args.out, "mp4v", args.fps, (w, h))
                print("输出 %s  %dx%d @%.0ffps" % (args.out, w, h, args.fps))
            writer.write(img)
            n += 1
            if n % 15 == 0:
                cv2.imshow("recording (q to stop)", cv2.resize(img, (0, 0), fx=0.4, fy=0.4))
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        src.close()
        cv2.destroyAllWindows()
    print("录完 %d 帧 -> %s" % (n, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
