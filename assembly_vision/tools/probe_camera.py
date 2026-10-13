# -*- coding: utf-8 -*-
"""相机能力探针 —— 把相机真正支持的分辨率 / 帧率 / 拼接方式探一遍。

为什么必须探：厂家的分辨率表只能当参考。实测这台 HBV-1714-2 S2.0：
    请求 2560x960 / 2560x720 -> 接受，但只有 3~4 fps（做不了实时）
    请求 1920x1080 / 1920x720 -> 被回落到 1280x960 / 1280x720
    请求 1280x960            -> 接受，50 fps   ← 可用模式里分辨率最高的
所以不能照着规格表猜，得让相机自己回答。

用法
    python tools/probe_camera.py                 # 探测并打印表格
    python tools/probe_camera.py --keep          # 额外把每个可用模式的样张存下来
    python tools/probe_camera.py --device 0 --backend dshow
    python tools/probe_camera.py --modes 1280x960,1280x720

输出里最后一段「建议写入 camera.yaml 的值」可以直接抄进配置文件。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from assemvision.imgio import imwrite_safe                       # noqa: E402
from assemvision.sources.uvc_stereo import (default_backend_name,  # noqa: E402
                                            resolve_backend)

# 候选模式：厂家表里出现过的 + 实测有用的。顺序即尝试顺序。
CANDIDATE_MODES: List[Tuple[int, int]] = [
    (2560, 960), (2560, 720),
    (1920, 1080), (1920, 720),
    (1280, 960), (1280, 720),
    (960, 960), (960, 480),
]

MIN_USABLE_FPS = 10.0        # 低于这个帧率就没法做实时去抖（去抖要连续 2~6 帧）


def parse_modes(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in str(text).split(","):
        part = part.strip().lower().replace("*", "x")
        if not part or "x" not in part:
            continue
        w, h = part.split("x")[:2]
        out.append((int(w), int(h)))
    return out


def measure(cap: cv2.VideoCapture, seconds: float = 1.0,
            warmup: int = 10) -> Tuple[Optional[np.ndarray], float]:
    """抓一帧并测帧率。返回 (样本帧, 实测fps)。"""
    frame = None
    for _ in range(max(1, warmup)):          # 丢掉开头几帧，等自动曝光稳定
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    if frame is None:
        return None, 0.0
    t0 = time.time()
    n = 0
    while time.time() - t0 < seconds:
        ok, _ = cap.read()
        if ok:
            n += 1
    dt = time.time() - t0
    return frame, (n / dt if dt > 0 else 0.0)


def check_stereo(frame: np.ndarray, mode: str = "side_by_side",
                 left_index: int = 0) -> Tuple[str, float, Tuple[int, int]]:
    """检查画面是不是真的左右（或上下）拼接，并给出单眼尺寸。

    ★ 用归一化互相关（NCC）而不是平均绝对差：
      本机左右眼曝光/白平衡不一致（实测右眼明显更亮更黄），
      用"平均绝对差"对齐会被亮度差带偏，真双目反而被判成"对不上"。
      NCC 对亮度与对比度的线性变化不敏感。

    判据（三个一起看，单看一个都会误判）：
      1. 中缝不连续度：真拼接在接缝处左右两列差异会明显跳高
      2. NCC 峰是否在非零位移：真双目有视差，峰不可能在 0
      3. NCC 峰是否明显高于基线相关
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape[:2]

    if mode == "top_bottom":
        a, b = g[:h // 2, :], g[h // 2:h // 2 * 2, :]
        seam = float(np.abs(g[h // 2].astype(float) - g[h // 2 - 1].astype(float)).mean())
        med = float(np.median(np.abs(np.diff(g, axis=0)).mean(axis=1)))
        span, axis = min(240, h // 2 - 5), 0
    else:
        a, b = g[:, :w // 2], g[:, w // 2:w // 2 * 2]
        seam = float(np.abs(g[:, w // 2].astype(float) - g[:, w // 2 - 1].astype(float)).mean())
        med = float(np.median(np.abs(np.diff(g, axis=1)).mean(axis=0)))
        span, axis = min(400, w // 2 - 5), 1

    if a.shape != b.shape or a.size == 0:
        return "无法判断（尺寸异常）", 0.0, (0, 0)
    eye = (w // 2, h) if mode != "top_bottom" else (w, h // 2)

    def ncc(d: int) -> float:
        if axis == 1:
            if d >= 0:
                x, y = a[:, d:], b[:, :a.shape[1] - d]
            else:
                x, y = a[:, :a.shape[1] + d], b[:, -d:]
        else:
            if d >= 0:
                x, y = a[d:, :], b[:a.shape[0] - d, :]
            else:
                x, y = a[:a.shape[0] + d, :], b[-d:, :]
        if x.shape != y.shape or x.size == 0:
            return -1.0
        x = x - x.mean()
        y = y - y.mean()
        den = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
        return float((x * y).sum() / den) if den > 1e-9 else -1.0

    # NCC 峰搜索（步长 1 较慢，先用步长 2 粗搜再细化）
    coarse = [(ncc(d), d) for d in range(-span, span + 1, 2)]
    peak_c = max(coarse)
    fine = [ncc(d) for d in range(max(-span, peak_c[1] - 3),
                                 min(span, peak_c[1] + 3) + 1)]
    best_ncc = max(fine) if fine else peak_c[0]
    best_d = max(range(max(-span, peak_c[1] - 3), min(span, peak_c[1] + 3) + 1),
                 key=ncc) if fine else peak_c[1]
    zero = ncc(0)

    seam_strong = med > 0 and seam > 2.5 * med
    if not seam_strong:
        verdict = "疑似单眼画面（中缝无跳变）"
    elif best_d == 0 and best_ncc > 0.98:
        verdict = "★ 两半完全相同（左右眼复制，非真双目）"
    elif abs(best_d) >= 2 and best_ncc > max(0.30, zero + 0.02):
        verdict = "真双目（视差 %+d px, NCC %.2f）" % (best_d, best_ncc)
    else:
        verdict = "疑似单眼画面（NCC 峰在零位移, %.2f）" % best_ncc
    _ = left_index
    return verdict, float(best_d), eye


def probe_one(dev, backend_name: str, fourcc: str, w: int, h: int,
              fps_req: float, seconds: float):
    """试一个模式。返回 dict（失败则 ok=False）。"""
    res = {"req": (w, h), "ok": False, "got": None, "fps": 0.0,
           "frame": None, "note": ""}
    backend = resolve_backend(backend_name)
    cap = cv2.VideoCapture(dev, backend)
    if not cap.isOpened() and backend != cv2.CAP_ANY:
        cap.release()
        cap = cv2.VideoCapture(dev, cv2.CAP_ANY)
        res["note"] = "回退 CAP_ANY"
    if not cap.isOpened():
        cap.release()
        res["note"] = "打不开"
        return res
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps_req)
        time.sleep(0.3)
        frame, fps = measure(cap, seconds=seconds)
        if frame is None:
            res["note"] = (res["note"] + " 无帧").strip()
            return res
        fh, fw = frame.shape[:2]
        res.update(ok=True, got=(fw, fh), fps=fps, frame=frame)
        if (fw, fh) != (w, h):
            res["note"] = (res["note"] + " 回落到 %dx%d" % (fw, fh)).strip()
    finally:
        cap.release()
    return res


def main() -> int:
    ap = argparse.ArgumentParser("probe_camera")
    ap.add_argument("--device", default=None, help="相机编号，默认读 camera.yaml 的 device")
    ap.add_argument("--backend", default="", help="v4l2 | dshow | msmf | any（默认为本平台推荐值）")
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--modes", default="", help="自定义候选模式，如 1280x960,1280x720")
    ap.add_argument("--fps", type=float, default=30.0, help="请求帧率")
    ap.add_argument("--seconds", type=float, default=1.0, help="每个模式测帧率的时长")
    ap.add_argument("--keep", action="store_true", help="把每个可用模式的样张存到 data/probe")
    args = ap.parse_args()

    # 读一下配置，拿到 device / mode / left_index 的默认值（缺文件也不报错）
    dev = 0
    cap_mode, left_index = "side_by_side", 0
    try:
        from assemvision.config import load_config
        cam = load_config(os.path.join(ROOT, "config", "camera.yaml"))
        dev = cam.get("device", 0)
        cap_mode = str(cam.get("capture.mode", "side_by_side"))
        left_index = int(cam.get("capture.left_index", 0))
    except Exception as e:                                    # noqa: BLE001
        print("[probe] 读 camera.yaml 失败（用默认值继续）:", e)
    if args.device is not None:
        dev = int(args.device) if str(args.device).isdigit() else args.device
    backend_name = args.backend or default_backend_name()

    modes = parse_modes(args.modes) or CANDIDATE_MODES

    print("=" * 78)
    print(" 相机能力探针")
    print("   device = %r   后端 = %s   FOURCC = %s   请求帧率 = %.0f"
          % (dev, backend_name, args.fourcc, args.fps))
    print("   拼接方式 = %s   left_index = %d" % (cap_mode, left_index))
    print("   Python %s   OpenCV %s   %s"
          % (sys.version.split()[0], cv2.__version__, sys.platform))
    print("=" * 78)

    results = []
    for (w, h) in modes:
        r = probe_one(dev, backend_name, args.fourcc, w, h, args.fps, args.seconds)
        results.append(r)
        if not r["ok"]:
            print("  %-11s -> %s" % ("%dx%d" % (w, h), r["note"] or "失败"))
            continue
        fw, fh = r["got"]
        flag = "✓" if (fw, fh) == (w, h) else "△"
        print("  %-11s -> %-11s %6.1f fps  %s %s"
              % ("%dx%d" % (w, h), "%dx%d" % (fw, fh), r["fps"], flag,
                 r["note"] or ""))

    usable = [r for r in results if r["ok"]]
    if not usable:
        print("\n★ 一个模式都没成功。检查：相机是否被占用、后端是否正确、USB 线材。")
        return 1

    # ── 去重：多个请求可能回落到同一个实际模式，只保留最好的那个 ──
    best_by_mode = {}
    for r in usable:
        key = r["got"]
        if key not in best_by_mode or r["fps"] > best_by_mode[key]["fps"]:
            best_by_mode[key] = r

    print("\n" + "=" * 78)
    print(" 实际拿到的模式（去重后）与双目校验")
    print("=" * 78)
    print("  %-11s %8s  %-40s %s" % ("实际模式", "帧率", "双目校验", "单眼尺寸"))
    for got, r in sorted(best_by_mode.items()):
        verdict, d, eye = check_stereo(r["frame"], cap_mode, left_index)
        print("  %-11s %6.1f fps  %-40s %dx%d"
              % ("%dx%d" % got, r["fps"], verdict, eye[0], eye[1]))

    # ── 推荐：优先"真双目 且 帧率达标"，再在其中选像素最多的 ──
    rated = []
    for got, r in best_by_mode.items():
        verdict, d, eye = check_stereo(r["frame"], cap_mode, left_index)
        rated.append((r, verdict, d, eye, verdict.startswith("真双目")))

    stereo_ok = [t for t in rated if t[4] and t[0]["fps"] >= MIN_USABLE_FPS]
    stereo_any = [t for t in rated if t[4]]
    print("\n" + "=" * 78)
    if stereo_ok:
        best_t = max(stereo_ok, key=lambda t: t[3][0] * t[3][1])
        print(" ★ 推荐：%dx%d  实测 %.1f fps（真双目，帧率达标）"
              % (best_t[0]["got"][0], best_t[0]["got"][1], best_t[0]["fps"]))
    elif stereo_any:
        best_t = max(stereo_any, key=lambda t: t[0]["fps"])
        print(" ⚠ 只有低帧率的真双目模式可用：%dx%d 仅 %.1f fps。"
              % (best_t[0]["got"][0], best_t[0]["got"][1], best_t[0]["fps"]))
        print("   低于 %.0f fps 时去抖与取件事件会明显迟滞 —— 建议把"
              " workpiece.yaml 的 check.drone_consistency 改成按时间去抖"
              "（fsm.Debouncer 的 hold_s 参数），或降低分辨率。"
              % MIN_USABLE_FPS)
    else:
        best_t = max(rated, key=lambda t: t[0]["fps"])
        print(" ✗ 没有任何模式是真双目拼接。检查 capture.mode"
              "（side_by_side / top_bottom）与 left_index 是否配错。")
    best = best_t[0]
    fw, fh = best["got"]
    eye_w, eye_h = best_t[3]
    print()
    print(" 建议写入 config/camera.yaml 的值：")
    print("     capture:")
    print("       backend: %s" % backend_name)
    print("       fourcc: %s" % args.fourcc)
    print("       width: %d" % fw)
    print("       height: %d" % fh)
    print("       mode: %s" % cap_mode)
    print("       fps: %d" % max(1, int(round(best["fps"]))))
    print("     # 单眼为 %dx%d" % (eye_w, eye_h))
    print("=" * 78)

    if args.keep:
        out = os.path.join(ROOT, "data", "probe")
        for got, r in sorted(best_by_mode.items()):
            p = os.path.join(out, "probe_%dx%d.png" % got)
            if imwrite_safe(p, r["frame"]):
                print("  样张已存: %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
