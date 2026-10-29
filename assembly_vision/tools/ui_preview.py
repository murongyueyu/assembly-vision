# -*- coding: utf-8 -*-
"""生成界面预览图 —— 不需要相机、不需要显示屏，也能看 UI 长什么样。

做法：造几个仿真 UIState（正常 / 顺序错误 / 缺件 / 等待），用 Qt 渲染器
离屏渲染成 PNG。做报告配图、或在没有显示器的机器上改界面时很有用。

用法
    python tools/ui_preview.py --out data/ui_preview --renderer qt
    python tools/ui_preview.py --out data/ui_preview --renderer opencv

Linux 无显示环境会自动设 QT_QPA_PLATFORM=offscreen。
Windows 上如果不想弹窗，先设环境变量 QT_QPA_PLATFORM=offscreen。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assemvision.config import load_config                 # noqa: E402
from assemvision.imgio import imwrite_safe                 # noqa: E402
from assemvision.types import ERROR, IDLE, OK, WARN        # noqa: E402
from assemvision.ui.base import UIState                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def synth_frame(w: int = 960, h: int = 720) -> np.ndarray:
    """造一张"像那么回事"的工作区画面：黑底料盘 + 黄铜件 + 蓝手套。"""
    img = np.full((h, w, 3), 26, np.uint8)
    # 料盘
    tx0, ty0, tx1, ty1 = 60, 40, 560, 540
    img[ty0:ty1, tx0:tx1] = (46, 46, 46)
    # 黄铜色（BGR）
    brass = (60, 165, 205)

    def mm2px(p, s=2.5, org=(tx0 + 30, ty0 + 30)):
        return int(org[0] + p[0] * s), int(org[1] + p[1] * s)

    import cv2
    # 立柱槽（斜长圆，-45°）
    c = np.array(mm2px((40, 40)), float)
    a = np.deg2rad(-45.0)
    u = np.array([np.cos(a), np.sin(a)])
    p1 = c + u * (85.2 - 15.6) / 2.0 * 2.5
    p2 = c - u * (85.2 - 15.6) / 2.0 * 2.5
    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), brass,
             int(15.6 * 2.5), cv2.LINE_AA)
    # 轮盘（外圈 + 辐条 + 球头）
    wc = mm2px((120, 40))
    cv2.circle(img, wc, int(30 * 2.5), brass, -1, cv2.LINE_AA)
    cv2.circle(img, wc, int(26 * 2.5), (34, 34, 34), -1, cv2.LINE_AA)
    for i in range(8):
        ang = np.deg2rad(i * 45 + 22.5)
        u = np.array([np.cos(ang), np.sin(ang)])
        cv2.line(img, tuple((np.array(wc) + u * 8 * 2.5).astype(int)),
                 tuple((np.array(wc) + u * 26 * 2.5).astype(int)), brass, int(6 * 2.5), cv2.LINE_AA)
        cv2.circle(img, tuple((np.array(wc) + u * 33 * 2.5).astype(int)),
                   int(4.2 * 2.5), brass, -1, cv2.LINE_AA)
    cv2.circle(img, wc, int(13 * 2.5), brass, -1, cv2.LINE_AA)
    cv2.circle(img, wc, int(4.5 * 2.5), (40, 40, 40), -1, cv2.LINE_AA)
    # 底座
    bc = mm2px((40, 120))
    cv2.circle(img, bc, int(29.65 * 2.5), brass, -1, cv2.LINE_AA)
    cv2.circle(img, bc, int(4 * 2.5), (34, 34, 34), -1, cv2.LINE_AA)
    # 顶盖
    cc = mm2px((120, 120))
    cv2.circle(img, cc, int(9.55 * 2.5), brass, -1, cv2.LINE_AA)
    # 蓝手套
    cv2.circle(img, (760, 300), 90, (185, 150, 40), -1, cv2.LINE_AA)

    # 叠加一些"检测结果"供预览
    for i, (x, y, ww, hh) in enumerate([(wc[0] - 95, wc[1] - 95, 190, 190),
                                        (bc[0] - 76, bc[1] - 76, 152, 152),
                                        (cc[0] - 26, cc[1] - 26, 52, 52)]):
        pass
    return img


def poly_circle(cx, cy, r, n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)


def build_states(frame: np.ndarray):
    """造四个有代表性的界面状态。"""
    tx0, ty0 = 90, 70
    org = np.array([tx0, ty0])

    def mm(p):
        return org[0] + p[0] * 2.5, org[1] + p[1] * 2.5

    slots = {
        "slot_column": poly_circle(*mm((40, 40)), 55),
        "slot_wheel": poly_circle(*mm((120, 40)), 100),
        "slot_base": poly_circle(*mm((40, 120)), 80),
        "slot_cap": poly_circle(*mm((120, 120)), 30),
    }
    asm = poly_circle(650, 340, 110)

    states = {}

    st = UIState()
    st.frame = frame
    st.level = OK
    st.title = "√ 步骤 3/4：轮盘套上立柱 完成"
    st.detail = ["已安装：base_present、column_present",
                 "下一步：请旋入顶盖并拧紧",
                 "辐条峰数 8/8",
                 "位姿正常（最大偏移 0.4 px）"]
    st.step, st.total = 3, 4
    st.stats = {"FPS": 28, "分割": "6ms", "辐条": "8/8", "轮盘高度": "12.3mm"}
    st.source_name = "UvcStereoSource(device=0)"
    st.polygons = [(slots["slot_wheel"], OK, 2), (slots["slot_base"], OK, 2),
                   (slots["slot_cap"], OK, 2), (slots["slot_column"], OK, 2),
                   (asm, IDLE, 2)]
    st.boxes = [((700, 210, 180, 180), "手套", WARN)]
    states["01_ok"] = st

    st2 = UIState()
    st2.frame = frame
    st2.level = ERROR
    st2.title = "× 顺序错误：顶盖旋入拧紧"
    st2.detail = ["缺失前置：wheel_present",
                  "当前应做：请将轮盘组件套到立柱上",
                  "已安装：base_present、column_present",
                  "辐条峰数 8/8"]
    st2.step, st2.total = 2, 4
    st2.stats = {"FPS": 27, "分割": "6ms", "辐条": "8/8", "轮盘高度": "-"}
    st2.source_name = "UvcStereoSource(device=0)"
    st2.polygons = [(slots["slot_wheel"], OK, 2), (slots["slot_base"], OK, 2),
                    (slots["slot_cap"], ERROR, 3), (slots["slot_column"], OK, 2),
                    (asm, WARN, 2)]
    st2.banner = st2.title
    st2.boxes = [((690, 200, 190, 190), "手套", WARN)]
    states["02_sequence_error"] = st2

    st3 = UIState()
    st3.frame = frame
    st3.level = ERROR
    st3.title = "× 出现非预期零件 / 缺件"
    st3.detail = ["缺件：slot_cap 槽内黄铜覆盖率仅 0%",
                  "缺件：slot_base 槽内黄铜覆盖率仅 2%",
                  "辐条峰数 4/8（应 8）",
                  "建议：补齐缺失零件后按空格重判"]
    st3.step, st3.total = 0, 4
    st3.stats = {"FPS": 29, "分割": "5ms", "辐条": "4/8", "轮盘高度": "-"}
    st3.source_name = "VideoFileSource(run1.mp4)"
    st3.polygons = [(slots["slot_wheel"], WARN, 2), (slots["slot_base"], ERROR, 3),
                    (slots["slot_cap"], ERROR, 3), (slots["slot_column"], ERROR, 3),
                    (asm, IDLE, 2)]
    states["03_missing"] = st3

    st4 = UIState()
    st4.frame = frame
    st4.level = IDLE
    st4.title = "请将立柱旋入底座中心孔"
    st4.detail = ["已等待 12 秒", "已安装：base_present", "辐条峰数 8/8"]
    st4.step, st4.total = 1, 4
    st4.stats = {"FPS": 30, "分割": "6ms", "辐条": "8/8", "轮盘高度": "-"}
    st4.source_name = "UvcStereoSource(device=0)"
    st4.polygons = [(k, IDLE, 1) for k in slots.values()] + [(asm, IDLE, 2)]
    states["04_waiting"] = st4
    return states


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "ui_preview"))
    ap.add_argument("--renderer", default="qt", choices=("qt", "opencv"))
    ap.add_argument("--display", default="1920x1080")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
    frame = synth_frame()
    states = build_states(frame)
    W, H = (int(v) for v in args.display.lower().split("x"))

    if args.renderer == "qt":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from assemvision.ui import make_renderer
        r = make_renderer("qt", fullscreen=False, display=(W, H),
                          panel_ratio=float(wp.get("ui.panel_ratio", 0.20)),
                          font_path=str(wp.get("ui.font_path", "")),
                          colors=wp.get("ui.colors", None))
        r.win.resize(W, H)
        r.win.show()
        for name, st in states.items():
            r.render(st)
            snap = r.snapshot()
            p = os.path.join(args.out, "%s.png" % name)
            print(("写出 %s  %dx%d" % (p, snap.shape[1], snap.shape[0]))
                  if imwrite_safe(p, snap) else "!! 写出失败 %s" % p)
        r.close()
    else:
        from assemvision.ui import make_renderer
        r = make_renderer("opencv", fullscreen=False, display=(W, H),
                          panel_ratio=float(wp.get("ui.panel_ratio", 0.20)),
                          font_path=str(wp.get("ui.font_path", "")),
                          colors=wp.get("ui.colors", None))
        for name, st in states.items():
            canvas = r.render(st)
            p = os.path.join(args.out, "%s.png" % name)
            print(("写出 %s" % p) if imwrite_safe(p, canvas) else "!! 写出失败 %s" % p)
        r.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
