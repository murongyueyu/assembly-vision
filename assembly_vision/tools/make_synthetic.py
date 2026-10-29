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

# ══════════════════════════════════════════════════════════════════════════
# ★★ 槽位几何（2026-10 按实物实测修正）
#
# ★ 这个字典必须与 config/workpiece.yaml 的 slots 保持一致！
#   测试的做法是「用本文件渲染合成图 -> 用 workpiece.yaml 的槽位算覆盖率」，
#   两边不一致就会误判（实测踩过：改了 yaml 但没改这里，
#   test_segment / test_polar 立刻挂掉，报「四个槽都判为有件」失败）。
#   改动任一处都要同步另一处。
#
# ★ 实测值来源：用 c3 单应把空料盘图正交化成 160x160mm 正视图（4 px/mm），
#   再做凹陷检测拟合 minAreaRect。实测与零件名义尺寸交叉吻合：
#     轮盘 Ø74.4 ↔ 槽 Ø75；底座 Ø59.3 ↔ 槽 Ø65；立柱长 85 ↔ 槽 91
# ★ 2026-10-18 C模块参考姿态（用户确认、空槽拟合）：
#   左上=底座 / 右上=顶盖 / 左下=立柱 / 右下=轮盘；运行时料盘仍可移动。
#   中心与长轴方向同步本轮workpiece.yaml，尺寸仍保留既有ROI/零件渲染口径。
# ══════════════════════════════════════════════════════════════════════════
SLOTS = {
    "slot_wheel":  dict(kind="circle", center=(120.7, 119.4), diameter=75.0,
                        part="P3_wheel"),
    # ★ 立柱：真实零件是约 85x15 的细杆，槽是 91x28 的胶囊（有富余）。
    #   早期把立柱画成 91x28（正好等于槽尺寸），结果**一旋转就超出槽口被截断**，
    #   形状变形 -> 特征识别失败（实测踩过：转 23°/45°/67°/203° 时认不出立柱）。
    "slot_column": dict(kind="capsule", center=(40.2, 119.6), length=85.0,
                        width=16.0, angle=44.4, part="P2_column"),
    "slot_cap":    dict(kind="circle", center=(120.3, 39.8), diameter=25.0,
                        part="P4_cap"),
    "slot_base":   dict(kind="circle", center=(40.1, 39.9), diameter=65.0,
                        part="P1_base"),
}
TRAY_SIZE = (160.0, 160.0)
ASM_CENTER = (-65.0, 80.0)        # ★ 与 workpiece.yaml 的 assembly.center 一致
                                  #   （料盘左侧 —— 设备夹桌右侧，工人在左侧作业）
ASM_RADIUS = 45.0

WHEEL_R = 37.2                    # 轮盘最大外接半径（Ø74.4）
BASE_R = 29.65                    # 底座 Ø59.3
CAP_R = 9.55                      # 顶盖 Ø19.1
COL_L, COL_D = 45.0, 15.0         # 俯视/斜俯视下立柱的投影尺寸（约 25° 倾角）


def mm2px(p_mm, s: float, origin=(0.0, 0.0)):
    return (int(round(origin[0] + p_mm[0] * s)), int(round(origin[1] + p_mm[1] * s)))


def canvas_origin(px_per_mm: float, margin_mm: float):
    """画布原点（tray-mm 原点 (0,0) 对应的像素位置）。

    ★★ 原点必须**只由内容包围盒决定，与"这次画不画装配位"无关**。
      踩过的坑：一开始把 origin 算进 canvas_size(..., with_assembly) 里，
      于是"带装配位"和"不带装配位"两种调用得到**不同的原点**，
      测试里 make_env 与 render_scene 各用一个，ROI 整体错位 553px。

    包围盒按"完整场景"算（料盘 + 装配位），这样任何一次渲染都用同一个原点，
    只是画布大小随内容变化。
    """
    xs = [0.0, TRAY_SIZE[0], ASM_CENTER[0] - ASM_RADIUS, ASM_CENTER[0] + ASM_RADIUS]
    ys = [0.0, TRAY_SIZE[1], ASM_CENTER[1] - ASM_RADIUS, ASM_CENTER[1] + ASM_RADIUS]
    x0 = min(xs) - margin_mm
    y0 = min(ys) - margin_mm
    return (-x0 * px_per_mm, -y0 * px_per_mm)


def canvas_size(px_per_mm: float, margin_mm: float, with_assembly: bool = True):
    """画布尺寸 + 原点偏移。返回 (w_px, h_px, origin_px)。

    ★ 装配位在 tray-mm 的**左侧（x 为负）**，所以画布原点不再是 (0,0) 对应的
      margin 位置 —— 必须按内容包围盒平移，否则装配位会被画到画面外。

    ★★ 画布**恒定按完整场景**（料盘 + 装配位）算，`with_assembly` 参数保留只为
      兼容旧调用，**不再影响尺寸**。踩过的坑：一开始让尺寸随 with_assembly 变，
      导致"带装配位"和"不带"两种调用得到不同的画布与原点，
      测试里 make_env 与 render_scene 各用一个，ROI 整体错位 553px。

      恒定尺寸的代价只是"不带装配位时右侧多一点空白"，换来的是
      **任何调用路径的画布与原点都完全一致**，值。
    """
    xs = [0.0, TRAY_SIZE[0], ASM_CENTER[0] - ASM_RADIUS, ASM_CENTER[0] + ASM_RADIUS]
    ys = [0.0, TRAY_SIZE[1], ASM_CENTER[1] - ASM_RADIUS, ASM_CENTER[1] + ASM_RADIUS]
    x1 = max(xs) + margin_mm
    y1 = max(ys) + margin_mm
    ox, oy = canvas_origin(px_per_mm, margin_mm)
    w = int(round(x1 * px_per_mm + ox))
    h = int(round(y1 * px_per_mm + oy))
    return w, h, (ox, oy)


def _rot_mm(pts_mm, center_mm, deg: float):
    """把一组 tray-mm 点绕 center_mm 旋转 deg 度（纯几何，无插值）。

    ★ 为什么要几何旋转而不是旋转位图：位图旋转会在离散网格上产生
      孔洞合并/边缘锯齿等**插值伪影**，用它测"特征是否旋转不变"测不准
      （实测踩过：用 warpAffine 转掩膜，137°/180° 就认错，但那是伪影导致的，
        不是特征本身的问题）。几何旋转出的形状才是干净的。
    """
    a = np.deg2rad(float(deg))
    ca, sa = np.cos(a), np.sin(a)
    p = np.asarray(pts_mm, dtype=np.float64) - np.asarray(center_mm, dtype=np.float64)
    x = p[:, 0] * ca - p[:, 1] * sa
    y = p[:, 0] * sa + p[:, 1] * ca
    return np.stack([x, y], axis=1) + np.asarray(center_mm, dtype=np.float64)


def render_scene(px_per_mm: float = 2.5, margin_mm: float = 12.0,
                 missing=(), missing_spokes: int = 0,
                 glove_at=None, glove_r_mm: float = 28.0,
                 draw_tray: bool = True, noise: float = 3.0,
                 assembly_stage: int = 0, with_assembly: bool = False,
                 rotate_deg: float = 0.0, wheel_rotate_deg: float = 0.0) -> np.ndarray:
    """画一张合成图。

    missing        要"拿出来"的槽 id 集合，例如 {"slot_cap"}
    missing_spokes 轮盘上故意少画几根辐条（0-8）
    glove_at       蓝手套中心（tray mm），None 表示不画
    assembly_stage 装配位完成度：1=底座 2=+立柱 3=+轮盘 4=+顶盖
    rotate_deg     ★ 把**立柱**绕槽心几何旋转这么多度。
                   ⚠️ 注意：真实工位里**槽是固定的**，把零件转出槽口范围会被槽壁
                   截断，覆盖率必然下降。所以这个参数主要用于研究几何，
                   不要拿它当"朝向鲁棒性"的判据（用 wheel_rotate_deg）。
    wheel_rotate_deg ★ 把**轮盘辐条图案**绕轮心几何旋转这么多度。
                   辐条图案是唯一朝向敏感的特征，所以朝向鲁棒性测这个。
                   8 根辐条下 45° 是一个周期。
    """
    W, H, org = canvas_size(px_per_mm, margin_mm, with_assembly or assembly_stage > 0)
    img = np.full((H, W, 3), BG, dtype=np.uint8)
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
                _draw_wheel(img, spec["center"], s, org, missing_spokes,
                            rotate_deg=wheel_rotate_deg)
        else:
            # 立柱是胶囊，本身就带朝向 —— 一并转，验证朝向无关性
            sp = dict(spec)
            sp["angle"] = float(spec["angle"]) + float(rotate_deg)
            _draw_capsule(img, sp, s, org, BRASS, -1)

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


def _draw_wheel(img, center_mm, s, org, missing_spokes: int = 0,
                rotate_deg: float = 0.0):
    """按真实结构画轮盘：中心毂 + 8 根辐条臂 + 外圈 + 球头手柄。

    辐条"臂"在 r 8~26，外圈在 r 26~30，球头手柄伸出到 37.2。
    所以数辐条的环带要取在臂上（12~24），不能取到外圈上。

    rotate_deg  把辐条图案整体绕轮心转这么多度（几何旋转，无插值伪影）。
                用来验证"轮盘朝向变了，特征识别与辐条计数还准不准"。
                注意：8 根辐条下 45° 是一个周期（转 45° 图形与自身重合）。
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
        ang = np.deg2rad(i * 360.0 / n + 22.5 + float(rotate_deg))
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
    # ★★ 注意：这里画了中心通孔，于是 8 个辐条空隙**完全被外圈包住**，
    #   `findContours(RETR_CCOMP)` 把它们（连同这个通孔）数成 `holes=8~9`。
    #   而**真机轮盘实测 `holes=0`** —— 真机辐条空隙是**豁口**（与背景连通）、
    #   中心毂也不透孔（见 docs/踩坑实录.md §2.30）。
    #   => 合成图的 `holes` 天生比真机高，**两个域不可能用同一个区间**。
    #      所以 `part_id.default_templates()` 里轮盘的 `holes` 取 `(0, 9)`，
    #      把两个域都覆盖住，而不是为合成图编一个假区间。
    #   ★ 试过"把中心通孔填成黄铜"来抹平差异：结果是辐条空隙仍从豁口变回
    #     封闭空腔、`holes` 只从 9 降到 8，白改一场 —— 已回退。


# ──────────────────────────────────────────────────────────────
def write_synthetic_calib(path: str, px_per_mm: float, margin_mm: float) -> bool:
    """写一份与合成图完全匹配的标定文件，让离线全链路能跑。

    ★ 原点要和 render_scene 用的一致 —— 从 canvas_size 取，别再自己算
      （装配位在 tray-mm 左侧后，原点不再是 margin，写死会整体错位）。
    """
    _, _, org = canvas_size(px_per_mm, margin_mm, True)
    ox, oy = float(org[0]), float(org[1])
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
            # ★ 轮盘槽实测在左上 (42,43)（原写 (120,40)，那是旧布局）
            "glove_on_wheel": dict(glove_at=SLOTS["slot_wheel"]["center"]),
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
