# -*- coding: utf-8 -*-
"""合成测试图生成器 —— 相机和支架没到也能把整套算法跑通、回归测试。

用的是从四个 STL 反解出来的真实尺寸：
    料盘 160x160，槽心在 40/120 网格上（间距 80）
    立柱 斜圆头槽 15.6 x 85.2 @ -45°   轮盘 Ø74.4   底座 Ø59.3   顶盖 Ø19.1
比例尺默认 2.5 px/mm（≈ GSD 0.40 mm/px），与实测工作距离 40cm 接近。

三种用法
    # 1) 一组静态场景 + 对应的合成标定文件
    python tools/make_synthetic.py --out data/synthetic --calib config/calibration.synthetic.yaml

    # 2) 装配过程序列（可以端到端跑流程判断）
    python tools/make_synthetic.py --seq data/synthetic_seq

    # 3) 用它们跑无界面模式
    python app.py --source images --path data/synthetic --calibration config/calibration.synthetic.yaml --no-ui
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assemvision.imgio import imwrite_safe        # noqa: E402


# 颜色（BGR）。用 HSV 造出来，保证与 workpiece.yaml 里的阈值自洽。
def _hsv(h, s, v):
    px = np.uint8([[[h, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])


# ★ 颜色必须落在 workpiece.yaml 的阈值**区间内**，否则合成图跑不过生产分割器。
#   实测真实黄铜的 S 分布是 [52, 109]（中值 80~92），所以这里取 (22, 150, 200)：
#   既在 s_min=60 / s_max=180 之内，又比早期的 190 更贴近实物。
#   早期取 190 时，一旦把 s_max 收到 180（为挡绿幕），合成图的黄铜就整个被滤掉了。
BRASS = _hsv(22, 150, 200)        # 黄铜：偏暗的金黄（贴近实测 S 中值）
GLOVE = _hsv(110, 190, 200)       # 蓝手套（★ V 要高于 glove.v_min=130）
BG = (28, 28, 28)                 # 黑底
TRAY = (46, 46, 46)               # 料盘本体
POCKET = (34, 34, 34)             # 槽底

SLOTS = {
    "slot_column": dict(kind="capsule", center=(40.0, 40.0), length=85.2,
                        width=15.6, angle=-45.0, part="P2_column"),
    "slot_wheel":  dict(kind="circle", center=(120.0, 40.0), diameter=74.4,
                        part="P3_wheel"),
    "slot_base":   dict(kind="circle", center=(40.0, 120.0), diameter=59.3,
                        part="P1_base"),
    "slot_cap":    dict(kind="circle", center=(120.0, 120.0), diameter=19.1,
                        part="P4_cap"),
}
TRAY_SIZE = (160.0, 160.0)
ASM_CENTER = (80.0, 250.0)        # 与 workpiece.yaml 的 assembly.center 一致
ASM_RADIUS = 45.0

WHEEL_R = 37.2                    # 轮盘最大外接半径（Ø74.4）
BASE_R = 29.65                    # 底座 Ø59.3
CAP_R = 9.55                      # 顶盖 Ø19.1
COL_L, COL_D = 45.0, 15.0         # 俯视/斜俯视下立柱的投影尺寸（约 25° 倾角）


def mm2px(p_mm, s: float, origin=(0.0, 0.0)):
    return (int(round(origin[0] + p_mm[0] * s)), int(round(origin[1] + p_mm[1] * s)))


def canvas_size(px_per_mm: float, margin_mm: float, with_assembly: bool):
    w = (TRAY_SIZE[0] + 2 * margin_mm) * px_per_mm
    h_mm = (ASM_CENTER[1] + ASM_RADIUS + margin_mm) if with_assembly \
        else (TRAY_SIZE[1] + margin_mm)
    return int(round(w)), int(round(h_mm * px_per_mm))


def render_scene(px_per_mm: float = 2.5, margin_mm: float = 12.0,
                 missing=(), missing_spokes: int = 0,
                 glove_at=None, glove_r_mm: float = 28.0,
                 draw_tray: bool = True, noise: float = 3.0,
                 assembly_stage: int = 0, with_assembly: bool = False) -> np.ndarray:
    """画一张合成图。

    missing        要"拿出来"的槽 id 集合，例如 {"slot_cap"}
    missing_spokes 轮盘上故意少画几根辐条（0-8）
    glove_at       蓝手套中心（tray mm），None 表示不画
    assembly_stage 装配位完成度：1=底座 2=+立柱 3=+轮盘 4=+顶盖
    """
    W, H = canvas_size(px_per_mm, margin_mm, with_assembly or assembly_stage > 0)
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    org = (margin_mm * px_per_mm, margin_mm * px_per_mm)
    s = float(px_per_mm)

    if draw_tray:
        cv2.rectangle(img, mm2px((0, 0), s, org),
                      mm2px(TRAY_SIZE, s, org), TRAY, -1)
    for sid, spec in SLOTS.items():
        if spec["kind"] == "circle":
            cv2.circle(img, mm2px(spec["center"], s, org),
                       int(round(spec["diameter"] * s / 2.0)), POCKET, -1, cv2.LINE_AA)
        else:
            _draw_capsule(img, spec, s, org, POCKET, -1)

    for sid, spec in SLOTS.items():
        if sid in missing:
            continue
        if spec["kind"] == "circle":
            c = mm2px(spec["center"], s, org)
            r = int(round(spec["diameter"] * s / 2.0)) - max(1, int(0.15 * s))
            cv2.circle(img, c, r, BRASS, -1, cv2.LINE_AA)
            if sid == "slot_base":
                cv2.circle(img, c, max(2, int(4.0 * s)), POCKET, -1, cv2.LINE_AA)
            if sid == "slot_wheel":
                _draw_wheel(img, spec["center"], s, org, missing_spokes)
        else:
            _draw_capsule(img, spec, s, org, BRASS, -1)

    if assembly_stage > 0:
        _draw_assembly(img, assembly_stage, s, org)

    if glove_at is not None:
        cv2.circle(img, mm2px(glove_at, s, org), int(round(glove_r_mm * s)),
                   GLOVE, -1, cv2.LINE_AA)

    if noise > 0:
        n = np.random.normal(0.0, noise, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    return img


def _draw_assembly(img, stage: int, s, org):
    c = np.asarray(ASM_CENTER, float)
    if stage >= 1:                                     # 底座圆盘
        cv2.circle(img, mm2px(c, s, org), int(round(BASE_R * s)), BRASS, -1, cv2.LINE_AA)
        cv2.circle(img, mm2px(c, s, org), int(round(4.5 * s)), POCKET, -1, cv2.LINE_AA)
    if stage >= 2:                                     # 立柱（斜俯视下的投影：一段细长）
        quad = np.array([c + [-COL_D / 2, -COL_L / 2], c + [COL_D / 2, -COL_L / 2],
                         c + [COL_D / 2, COL_L / 2], c + [-COL_D / 2, COL_L / 2]])
        cv2.fillPoly(img, [np.asarray([mm2px(q, s, org) for q in quad], np.int32)], BRASS)
    if stage >= 3:                                     # 轮盘（会盖住立柱大部分）
        _draw_wheel(img, c, s, org, 0)
    if stage >= 4:                                     # 顶盖
        cv2.circle(img, mm2px(c, s, org), int(round(CAP_R * s)), BRASS, -1, cv2.LINE_AA)


def _draw_capsule(img, spec, s, org, color, thickness):
    a = np.deg2rad(spec["angle"])
    u = np.array([np.cos(a), np.sin(a)])
    v = np.array([-np.sin(a), np.cos(a)])
    c = np.asarray(spec["center"], float)
    half = max(0.0, (spec["length"] - spec["width"]) / 2.0)
    w2 = spec["width"] / 2.0
    pts = []
    for sign in (1.0, -1.0):
        p = c + sign * half * u
        t0, t1 = (-np.pi / 2, np.pi / 2) if sign > 0 else (np.pi / 2, 3 * np.pi / 2)
        for t in np.linspace(t0, t1, 24):
            q = p + w2 * (np.cos(t) * u + np.sin(t) * v)
            pts.append(mm2px(q, s, org))
    cv2.fillPoly(img, [np.asarray(pts, np.int32)], color)


def _draw_wheel(img, center_mm, s, org, missing_spokes: int = 0):
    """按真实结构画轮盘：中心毂 + 8 根辐条臂 + 外圈 + 球头手柄。

    辐条"臂"在 r 8~26，外圈在 r 26~30，球头手柄伸出到 37.2。
    所以数辐条的环带要取在臂上（12~24），不能取到外圈上。
    """
    c = np.asarray(center_mm, float)
    ring_in, ring_out = 26.0, 30.0
    arm_in, arm_out, arm_w = 8.0, ring_in, 6.0
    ball_r, ball_at = 4.2, 33.0
    n = 8

    cv2.circle(img, mm2px(c, s, org), int(round(ring_out * s)), BRASS, -1, cv2.LINE_AA)
    cv2.circle(img, mm2px(c, s, org), int(round(ring_in * s)), POCKET, -1, cv2.LINE_AA)

    for i in range(n):
        if i < int(missing_spokes):
            continue
        ang = np.deg2rad(i * 360.0 / n + 22.5)
        u = np.array([np.cos(ang), np.sin(ang)])
        v = np.array([-np.sin(ang), np.cos(ang)])
        quad = np.array([c + u * arm_in + v * (arm_w / 2), c + u * arm_out + v * (arm_w / 2),
                         c + u * arm_out - v * (arm_w / 2), c + u * arm_in - v * (arm_w / 2)])
        cv2.fillPoly(img, [np.asarray([mm2px(q, s, org) for q in quad], np.int32)], BRASS)
        neck = np.array([c + u * ring_out + v * 1.2, c + u * ball_at + v * 1.2,
                         c + u * ball_at - v * 1.2, c + u * ring_out - v * 1.2])
        cv2.fillPoly(img, [np.asarray([mm2px(q, s, org) for q in neck], np.int32)], BRASS)
        cv2.circle(img, mm2px(c + u * ball_at, s, org), int(round(ball_r * s)),
                   BRASS, -1, cv2.LINE_AA)

    cv2.circle(img, mm2px(c, s, org), int(round(13.0 * s)), BRASS, -1, cv2.LINE_AA)
    cv2.circle(img, mm2px(c, s, org), int(round(4.5 * s)), (40, 40, 40), -1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────
def write_synthetic_calib(path: str, px_per_mm: float, margin_mm: float) -> bool:
    """写一份与合成图完全匹配的标定文件，让离线全链路能跑。"""
    ox = oy = margin_mm * px_per_mm
    text = (
        "# 由 tools/make_synthetic.py 自动生成的合成图标定（仅用于离线开发与回归测试）\n"
        "meta:\n"
        "  note: synthetic\n"
        "  px_per_mm: %r\n"
        "fov:\n"
        "  work_distance_mm: 400.0\n"
        "  gsd_mm_per_px: %r\n"
        "tray_map:\n"
        "  homography: [[%r, 0.0, %r], [0.0, %r, %r], [0.0, 0.0, 1.0]]\n"
        "  scale: 1.0\n"
        "  on_rectified: false\n"
        "  reprojection_rms_px: 0.0\n"
        % (px_per_mm, 1.0 / px_per_mm, px_per_mm, ox, px_per_mm, oy))
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return True


def build_sequence(px_per_mm: float, margin_mm: float):
    """脚本化的装配过程：正确顺序走一遍，再插几种错误。"""
    steps = []
    A = lambda **kw: steps.append(render_scene(px_per_mm=px_per_mm, margin_mm=margin_mm,
                                               with_assembly=True, **kw))
    hold = 3                                    # 每步重复几帧，满足去抖
    hold_final = 30                             # 终态保持帧数（终检要抓全分辨率图）

    # 上料：四个件都在料盘上
    for _ in range(hold):
        A(assembly_stage=0)
    # op1 取底座 → 放到装配位
    for _ in range(hold):
        A(missing={"slot_base"}, assembly_stage=1)
    # op2 取立柱 → 旋入
    for _ in range(hold):
        A(missing={"slot_base", "slot_column"}, assembly_stage=2)
    # op3 取轮盘 → 套上
    for _ in range(hold):
        A(missing={"slot_base", "slot_column", "slot_wheel"}, assembly_stage=3)
    # op4 取顶盖 → 拧上（完成）
    for _ in range(hold):
        A(missing={"slot_base", "slot_column", "slot_wheel", "slot_cap"},
          assembly_stage=4)
    # ★ 终态要多保持一会儿：终检要抓一帧全分辨率图，太快翻篇会抓到空装配位
    for _ in range(hold_final):
        A(missing={"slot_base", "slot_column", "slot_wheel", "slot_cap"},
          assembly_stage=4)
    return steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--seq", default="")
    ap.add_argument("--calib", default="")
    ap.add_argument("--px-per-mm", type=float, default=2.5)
    ap.add_argument("--margin-mm", type=float, default=12.0)
    args = ap.parse_args()

    if args.out:
        scenes = {
            "full":           dict(),
            "missing_cap":    dict(missing={"slot_cap"}),
            "missing_base":   dict(missing={"slot_base"}),
            "missing_column": dict(missing={"slot_column"}),
            "spokes_7":       dict(missing_spokes=1),
            "spokes_4":       dict(missing_spokes=4),
            "glove_on_wheel": dict(glove_at=(120.0, 40.0)),
            "all_missing":    dict(missing=set(SLOTS.keys())),
            "assembly_done":  dict(missing=set(SLOTS.keys()), assembly_stage=4),
            "assembly_no_cap": dict(missing=set(SLOTS.keys()), assembly_stage=3),
        }
        for name, kw in scenes.items():
            img = render_scene(px_per_mm=args.px_per_mm, margin_mm=args.margin_mm, **kw)
            p = os.path.join(args.out, "%s.png" % name)
            print(("写出 %s  %dx%d" % (p, img.shape[1], img.shape[0]))
                  if imwrite_safe(p, img) else "!! 写出失败: %s" % p)

    if args.seq:
        frames = build_sequence(args.px_per_mm, args.margin_mm)
        for i, img in enumerate(frames):
            p = os.path.join(args.seq, "f%03d.png" % i)
            imwrite_safe(p, img)
        print("装配过程序列: %d 帧 -> %s" % (len(frames), args.seq))

    if args.calib:
        write_synthetic_calib(args.calib, args.px_per_mm, args.margin_mm)
        print("合成标定文件 -> %s" % args.calib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
