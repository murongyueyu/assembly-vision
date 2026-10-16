# -*- coding: utf-8 -*-
"""★★ 料盘槽位**实时检测 + 现场调参**（v2，2026-10 rev.3r-11）。

═══ 为什么有 v2 ═══
v1 的问题：我只给了**相机/图像/检测**三组滑条，**没给"料盘检测"这组** ——
而真帧对比图（用户标注蓝色真值）显示：
    · 底座（右下角）**完全准**；
    · 轮盘（左上）、立柱（右上）、顶盖（左下）**全都往中心缩**，偏差随
      "离右下角的距离"**单调增大**。
★ 这个规律**不是单个槽的搜索问题，而是坐标映射本身的问题**：
  拉正用的**四个角点偏了**（多半把阴影算进料盘里了）⇒
  拉正后的坐标系在远离参考点的地方**被压缩**。
⇒ v2 把**「料盘检测」独立成一组滑条**（掩膜阈值 + 角点内缩），
  并画出**设计位置的蓝圈参考**，偏多少一眼就能看见、当场就能调。

═══ 三组滑条（别搞混）═══
 【料盘检测】← ★ 本轮新增，修"整体往中心缩"就靠它
     黑阈值 V/S、闭运算核、**角点内缩 px**、面积下限
 【相机】曝光/增益/白平衡/自动/激光 +（可能不支持的）brightness/contrast/…
 【图像】软件对比度：局部对比 clahe / 伽马 / 亮度 / 平滑
 【检测】Canny 阈值、凹性权重、半径正则、搜索范围、抹细碎

═══ 按键 ═══
  q 退出 | s 存参数到 config/pocket_tune.yaml | f 存当前帧 | b 切换蓝圈参考

★ 本工具只做检测给人看，不参与判定链；参数存成 yaml 后由我接进正式代码。
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
sys.path.insert(0, os.path.join(ROOT, "assembly_vision"))

from assemvision.imgio import imread_safe, imwrite_safe      # noqa: E402

OUT_DIR = os.path.join(HERE, "工位帧", "原图")
TUNE_PATH = os.path.join(ROOT, "assembly_vision", "config", "pocket_tune.yaml")

# ══════════════════════════════════════════════════════════════
# ★★ 槽的**设计几何**（料盘 mm，左上原点）—— 就是用户图里画的蓝圈。
#   轮盘 Ø88 / 立柱 91×28 胶囊 / 顶盖 Ø25 / 底座 Ø65
#   ★ 「轮盘比底座大、且几乎与料盘边相切」⇒ r=44（到边净距 ~5mm）
SLOTS = [("slot_wheel", 49.4, 49.1, 44.0, "circle"),
         ("slot_column", 124.9, 41.2, 14.0, "capsule"),
         ("slot_cap", 53.4, 130.6, 12.5, "circle"),
         ("slot_base", 119.6, 123.0, 32.5, "circle")]
COLORS = {"slot_wheel": (0, 0, 255), "slot_column": (0, 255, 255),
          "slot_cap": (255, 0, 255), "slot_base": (0, 255, 0)}
BLUE = (235, 160, 60)          # ★ 设计位置参考圈（与用户标注同色系）
ZH = {"slot_wheel": "wheel", "slot_column": "column",
      "slot_cap": "cap", "slot_base": "base"}

DEF = dict(
    # ── 料盘检测 ← ★ 本轮新增：修"整体往中心缩" ──
    tray_v=110, tray_s=120, tray_close=9, corner_inset=0, tray_min=20000,
    # ── 相机：曝光/增益/白平衡 ──
    exposure=126, gain=16, wb=6300, auto=0, laser=1,
    # ── 相机：画质（D435 可能不支持，开机报清单）──
    cam_bright=0, cam_contrast=50, cam_sat=64, cam_sharp=50,
    cam_gamma=300, cam_hue=0, cam_blc=0,
    # ── 图像：软件对比度 ──
    clahe=30, gamma=100, bright=0, blur=5,
    # ── 检测 ──
    canny_lo=25, canny_hi=80, conc_w=100, pos_mm=40, r_pct=140, erase=0, reg_w=60,
)


# ══════════════════════════════════════════════════════════════
def enhance(gray: np.ndarray, p: dict) -> np.ndarray:
    """图像层：局部对比 + 伽马 + 亮度。★ 槽沿是**局部**特征，CLAHE 最有效。"""
    g = gray
    if p["clahe"] > 0:
        cl = cv2.createCLAHE(clipLimit=max(0.5, p["clahe"] / 10.0),
                             tileGridSize=(8, 8))
        g = cl.apply(g)
    if p["gamma"] != 100:
        inv = 1.0 / max(0.05, p["gamma"] / 100.0)
        lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], np.uint8)
        g = cv2.LUT(g, lut)
    if p["bright"] != 0:
        g = cv2.convertScaleAbs(g, alpha=1.0, beta=float(p["bright"]))
    k = int(p["blur"]) | 1
    if k >= 3:
        g = cv2.GaussianBlur(g, (k, k), 0)
    return g


def tray_corners(bgr: np.ndarray, p: dict):
    """★ 四个**角点**定料盘位姿。

    ★★ 为什么把阈值做成滑条（v2 的关键）：
      真帧对比显示"只有底座准、离右下越远越往中心缩" —— 这是**坐标映射被压缩**
      的典型症状，根源是**拉正用的四角偏了**（阴影被算进料盘 ⇒ 四角外扩 ⇒
      内部点被压向中心）。所以必须让人**现场调**黑阈值/闭运算/内缩，
      一边调一边看蓝圈与彩圈是否重合。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.int32)
    s = hsv[:, :, 1].astype(np.int32)
    m = ((v < int(p["tray_v"])) & (s < int(p["tray_s"]))).astype(np.uint8) * 255
    k = int(p["tray_close"]) | 1
    if k >= 3:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _c = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None, m
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[big, cv2.CC_STAT_AREA] < int(p["tray_min"]):
        return None, m
    mask = (lab == big).astype(np.uint8) * 255
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    cnt = max(cnts, key=cv2.contourArea)
    P = cnt.reshape(-1, 2).astype(np.float64)

    # 四条边拟合直线（用 minAreaRect 的四条边当"提名"，再各自拟合）
    box = cv2.boxPoints(cv2.minAreaRect(cnt))
    ss, dd = box.sum(axis=1), np.diff(box, axis=1).ravel()
    box = np.array([box[np.argmin(ss)], box[np.argmin(dd)],
                    box[np.argmax(ss)], box[np.argmax(dd)]], np.float64)
    nxt = np.roll(box, -1, axis=0)
    edges = []
    for i in range(4):
        a, b = box[i], nxt[i]
        ab = b - a
        L = float(np.hypot(*ab))
        if L < 10:
            return None, mask
        rel = P - a
        t = (rel @ ab) / (L * L)
        nx, ny = -ab[1] / L, ab[0] / L
        dist = np.abs(rel[:, 0] * nx + rel[:, 1] * ny)
        sel = (t > 0.12) & (t < 0.88) & (dist < max(6.0, 0.10 * L))
        pts = P[sel]
        if len(pts) < 20:
            return None, mask
        c0 = pts.mean(axis=0)
        _u, _s, vt = np.linalg.svd(pts - c0, full_matrices=False)
        edges.append((c0, vt[0]))
    corners = []
    for i in range(4):
        c1, d1 = edges[(i - 1) % 4]
        c2, d2 = edges[i]
        A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
        if abs(np.linalg.det(A)) < 1e-9:
            return None, mask
        t, _ = np.linalg.solve(A, c2 - c1)
        corners.append(c1 + t * d1)
    corners = np.array(corners)
    # ★ 角点内缩：把四角沿对角线朝中心挪 inset 像素。
    #   物理含义 = "料盘真实边缘在检出边缘的里面"（阴影让它看起来更大）。
    ins = float(p["corner_inset"])
    if ins != 0.0:
        ctr = corners.mean(axis=0)
        d = corners - ctr
        nrm = np.linalg.norm(d, axis=1, keepdims=True)
        corners = corners - ins * d / np.maximum(nrm, 1e-6)
    ss, dd = corners.sum(axis=1), np.diff(corners, axis=1).ravel()
    corners = np.array([corners[np.argmin(ss)], corners[np.argmin(dd)],
                        corners[np.argmax(ss)], corners[np.argmax(dd)]])
    return corners, mask


def ring_pts(c, r, kind, px_per_mm, ang_deg=44.0):
    if kind != "capsule":
        t = np.linspace(0.0, 2.0 * np.pi, 240)
        return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], axis=1)
    # ★★ 胶囊：圆端半径 r、**两圆心距 63mm**、长轴 **+44°**（左上→右下）。
    #   ★ sep = 91 − 28 = 63（**不是 91**：91 是总长，sep 是两圆心距）
    a = np.radians(ang_deg)
    ca, sa = np.cos(a), np.sin(a)
    L = 63.0 * px_per_mm / 2.0
    out = []
    for sgn in (+1, -1):
        cx, cy = c[0] + sgn * L * ca, c[1] + sgn * L * sa
        t = np.linspace(0.0, np.pi, 90)
        out.append(np.stack([cx + r * np.cos(t) * ca - r * np.sin(t) * sa,
                             cy + r * np.cos(t) * sa + r * np.sin(t) * ca], axis=1))
    return np.vstack(out)


def _ring_kernel(r_px: float, width: float = 2.0, n: int = 256) -> np.ndarray:
    """★ 造一个"环"卷积核（霍夫式投票用）。"""
    s = int(np.ceil(r_px * 1.6)) * 2 + 3
    k = np.zeros((s, s), np.float32)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    for rr in np.arange(r_px - width, r_px + width + 1e-9, 0.5):
        xs = np.round(s / 2.0 + rr * np.cos(t)).astype(int)
        ys = np.round(s / 2.0 + rr * np.sin(t)).astype(int)
        ok = (xs >= 0) & (xs < s) & (ys >= 0) & (ys < s)
        k[ys[ok], xs[ok]] = 1.0
    nz = float((k > 0).sum())
    return k / (nz if nz > 0 else 1.0)


def hough_center(edg: np.ndarray, c0, r_px: float, span_px: float):
    """★★ **霍夫式投票**找圆心：把"环"当卷积核在边缘图上相关，取峰值。

    ★★ 为什么换成这个（v1 的教训）：
      v1 是"逐点穷举 + 边缘密度(二值)"⇒ 分数在最优解附近**是个平台**
      （很多相邻位置命中同样多的边缘像素）⇒ 最优解**在跳**，
      所以用户看到同一个顶盖一会儿"偏高"、一会儿"偏右偏低"。
      ⇒ 用**卷积**（每个边缘像素为"可能的圆心"投票）得到一个**平滑的单峰**，
        峰的位置就是圆心，**抗纹理、抗少量边缘缺失**（这正是霍夫变换的性质）。
    """
    if r_px < 3:
        return None
    k = _ring_kernel(r_px)
    h, w = k.shape
    if h >= edg.shape[0] or w >= edg.shape[1]:
        return None
    # 只在候选点附近算（省时间）：裁一块 ROI 做相关
    x0 = int(max(0, c0[0] - span_px - w))
    x1 = int(min(edg.shape[1], c0[0] + span_px + w))
    y0 = int(max(0, c0[1] - span_px - h))
    y1 = int(min(edg.shape[0], c0[1] + span_px + h))
    roi = edg[y0:y1, x0:x1].astype(np.float32)
    if roi.shape[0] < h or roi.shape[1] < w:
        return None
    resp = cv2.filter2D(roi, -1, k, borderType=cv2.BORDER_CONSTANT)
    _mn, mx, _ml, ml = cv2.minMaxLoc(resp)
    px = x0 + ml[0] + w // 2
    py = y0 + ml[1] + h // 2
    return float(px), float(py), float(mx)


def detect(front_enh: np.ndarray, px_per_mm: float, p: dict):
    """在**正视料盘**上找四个槽：霍夫投票定圆心 + 扫一圈定半径。"""
    g = cv2.cvtColor(front_enh, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edg = cv2.Canny(g.astype(np.uint8), int(p["canny_lo"]), int(p["canny_hi"]))
    if p["erase"] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(p["erase"]) | 1,) * 2)
        edg = cv2.morphologyEx(edg, cv2.MORPH_OPEN, k)
    H, W = g.shape
    m = int(round(1.0 * px_per_mm))
    span_px = float(p["pos_mm"]) * px_per_mm
    reg = float(p["reg_w"]) / 100.0

    res = []
    for cid, cx_mm, cy_mm, r_mm, kind in SLOTS:
        c0 = (cx_mm * px_per_mm, cy_mm * px_per_mm)
        r0 = r_mm * px_per_mm
        rlo = r0 * 0.55
        rhi = r0 * float(p["r_pct"]) / 100.0
        best = None
        angs = ([34.0, 39.0, 44.0, 49.0, 54.0] if kind == "capsule" else [44.0])
        for ang in angs:
            # ★ 胶囊不能直接用环卷积核（它是长条）⇒ 用旋转后的椭圆核近似：
            #   这里退化为"逐点抽样"，但半径用**多次投票**平滑
            if kind == "capsule":
                step = max(1.0, 1.0 * px_per_mm)
                for r in np.arange(rlo, rhi + 1e-9, 0.5 * px_per_mm):
                    for dy in np.arange(-span_px, span_px + 1e-9, step):
                        for dx in np.arange(-span_px, span_px + 1e-9, step):
                            c = (c0[0] + dx, c0[1] + dy)
                            pts = ring_pts(c, r, kind, px_per_mm, ang)
                            xs = np.round(pts[:, 0]).astype(int)
                            ys = np.round(pts[:, 1]).astype(int)
                            if ((xs < m).any() or (xs >= W - m).any()
                                    or (ys < m).any() or (ys >= H - m).any()):
                                continue
                            # ★ 用**软**投票（blur 后的边缘）而不是 0/1 硬计数
                            dens = float(edg[ys, xs].mean()) / 255.0
                            if dens < 0.03:
                                continue
                            sc = dens - reg * abs(r - r0) / max(1e-6, r0)
                            if best is None or sc > best[0]:
                                best = (sc, c[0], c[1], r, ang)
                continue
            # ---- 圆：霍夫投票 ----
            for r in np.arange(rlo, rhi + 1e-9, max(0.5, 0.5 * px_per_mm)):
                got = hough_center(edg, c0, r, span_px)
                if got is None:
                    continue
                px, py, val = got
                if not (m <= px < W - m and m <= py < H - m):
                    continue
                sc = val - reg * abs(r - r0) / max(1e-6, r0)
                if best is None or sc > best[0]:
                    best = (sc, px, py, r, 44.0)
        if best is not None:
            res.append((cid, (best[1], best[2]), best[3], kind, best[4]))
    return res, edg


def draw(front: np.ndarray, res, px_per_mm: float, show_ref: bool,
         quad_dev=None):
    out = front.copy()
    # ★ 先画**设计位置**的蓝圈（参考）—— 偏多少一眼可见
    if show_ref:
        for cid, cx_mm, cy_mm, r_mm, kind in SLOTS:
            pts = ring_pts((cx_mm * px_per_mm, cy_mm * px_per_mm), r_mm * px_per_mm,
                           kind, px_per_mm)
            cv2.polylines(out, [pts.astype(np.int32)], True, BLUE, 2, cv2.LINE_AA)
    for cid, c, r, kind, ang in res:
        pts = ring_pts(c, r, kind, px_per_mm, ang)
        cv2.polylines(out, [pts.astype(np.int32)], True, COLORS[cid], 2, cv2.LINE_AA)
        txt = "%s d=(%+.1f,%+.1f) r=%.1f" % (
            ZH[cid], c[0] / px_per_mm - dict(
                (s[0], s[1]) for s in SLOTS)[cid],
            c[1] / px_per_mm - dict((s[0], s[2]) for s in SLOTS)[cid],
            r / px_per_mm)
        o = (max(2, int(c[0]) - 70), int(c[1]))
        cv2.putText(out, txt, o, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, o, cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS[cid], 1, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════
class CamControls:
    """D435 的彩色流能改哪些项 —— **开机现场探测**，不靠猜。"""

    def __init__(self, rs, profile) -> None:
        self.rs = rs
        self.sensors = []
        try:
            self.sensors = list(profile.get_device().sensors)
        except Exception:                                          # noqa: BLE001
            pass
        self.ok = bool(self.sensors)

    def _opts(self):
        rs = self.rs
        return {k: getattr(rs.option, k, None) for k in (
            "exposure", "gain", "white_balance", "enable_auto_exposure",
            "enable_auto_white_balance", "emitter_enabled",
            "brightness", "contrast", "saturation", "sharpness",
            "gamma", "hue", "backlight_compensation")}

    def probe(self) -> dict:
        out = {}
        for s in self.sensors:
            nm = "?"
            try:
                nm = s.get_info(self.rs.camera_info.name)
            except Exception:                                      # noqa: BLE001
                pass
            lst = []
            for k, o in self._opts().items():
                if o is None:
                    continue
                try:
                    cur = s.get_option(o)
                    rng = s.get_option_range(o)
                    lst.append("%-24s 现%8.1f  范围[%.0f, %.0f]"
                               % (k, cur, rng.min, rng.max))
                except Exception:                                  # noqa: BLE001
                    pass
            out[nm] = lst
        return out

    def apply(self, p: dict) -> list:
        if not self.ok:
            return []
        opt = self._opts()
        done = []
        for s in self.sensors:
            for key in ("enable_auto_exposure", "enable_auto_white_balance"):
                o = opt.get(key)
                if o is None:
                    continue
                try:
                    s.set_option(o, 0.0 if p["auto"] == 0 else 1.0)
                except Exception:                                  # noqa: BLE001
                    pass
            pairs = [("emitter_enabled", p["laser"])]
            if p["auto"] == 0:
                pairs += [("exposure", p["exposure"]), ("gain", p["gain"]),
                          ("white_balance", p["wb"])]
            pairs += [("brightness", p["cam_bright"]), ("contrast", p["cam_contrast"]),
                      ("saturation", p["cam_sat"]), ("sharpness", p["cam_sharp"]),
                      ("gamma", p["cam_gamma"]), ("hue", p["cam_hue"]),
                      ("backlight_compensation", p["cam_blc"])]
            for key, v in pairs:
                o = opt.get(key)
                if o is None:
                    continue
                try:
                    rng = s.get_option_range(o)
                    s.set_option(o, float(np.clip(v, rng.min, rng.max)))
                    done.append(key)
                except Exception:                                  # noqa: BLE001
                    pass
        return sorted(set(done))


def make_trackbars(p: dict) -> None:
    win = "调参"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 480, 760)

    def tb(name, key, lo, hi):
        cv2.createTrackbar(name, win, int(p[key]) - lo, hi - lo,
                           lambda v, k=key, l=lo: p.__setitem__(k, v + l))

    cv2.createTrackbar("═══ 料盘检测（修「往中心缩」）═══", win, 0, 0, lambda v: None)
    tb("黑阈值 V", "tray_v", 20, 200)
    tb("黑阈值 S", "tray_s", 10, 255)
    tb("闭运算核", "tray_close", 1, 31)
    tb("★角点内缩 px", "corner_inset", -30, 30)
    tb("面积下限", "tray_min", 2000, 80000)
    cv2.createTrackbar("═══ 相机：曝光/增益 ═══", win, 0, 0, lambda v: None)
    tb("曝光 exposure", "exposure", 1, 10000)
    tb("增益 gain", "gain", 0, 128)
    tb("白平衡 wb", "wb", 2800, 6500)
    tb("自动曝光 0/1", "auto", 0, 1)
    tb("红外激光 0/1", "laser", 0, 1)
    cv2.createTrackbar("═══ 相机：画质（可能不支持）═══", win, 0, 0, lambda v: None)
    tb("brightness", "cam_bright", 0, 255)
    tb("contrast", "cam_contrast", 0, 100)
    tb("saturation", "cam_sat", 0, 100)
    tb("sharpness", "cam_sharp", 0, 100)
    tb("camera gamma", "cam_gamma", 100, 500)
    tb("hue", "cam_hue", -180, 180)
    tb("逆光补偿", "cam_blc", 0, 1)
    cv2.createTrackbar("═══ 图像：软件对比度 ═══", win, 0, 0, lambda v: None)
    tb("局部对比 clahe", "clahe", 0, 100)
    tb("伽马 gamma", "gamma", 30, 300)
    tb("亮度偏移", "bright", -100, 100)
    tb("平滑 blur", "blur", 1, 15)
    cv2.createTrackbar("═══ 检测参数 ═══", win, 0, 0, lambda v: None)
    tb("Canny 低", "canny_lo", 1, 200)
    tb("Canny 高", "canny_hi", 5, 400)
    tb("凹性权重", "conc_w", 0, 500)
    tb("半径正则", "reg_w", 0, 300)
    tb("位置搜索 mm", "pos_mm", 5, 150)
    tb("半径上限 %", "r_pct", 100, 200)
    tb("抹细碎 px", "erase", 0, 9)


def save_tune(p: dict) -> None:
    os.makedirs(os.path.dirname(TUNE_PATH), exist_ok=True)
    with open(TUNE_PATH, "w", encoding="utf-8") as fh:
        fh.write("# ★ 由 ai_测试工具/pocket_tune_live.py 生成（按 s 保存）\n")
        fh.write("# 现场调出来的**检测参数**（不是相机标定），供正式代码读取。\n")
        fh.write("pocket_tune:\n")
        for k, v in p.items():
            fh.write("  %s: %s\n" % (k, v))


def process(bgr: np.ndarray, p: dict, show_ref: bool):
    """返回 (vis_src, vis_enh, front_vis, msg)。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    enh = enhance(gray, p)
    corners, mask = tray_corners(bgr, p)
    vis = bgr.copy()
    enh_bgr = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
    edg = cv2.Canny(enh, int(p["canny_lo"]), int(p["canny_hi"]))
    vis_enh = np.hstack([enh_bgr, cv2.cvtColor(edg, cv2.COLOR_GRAY2BGR)])
    if corners is None:
        cv2.putText(vis, "★ 找不到料盘（调「料盘检测」那组）", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return vis, vis_enh, None, "★ 找不到料盘 —— 请调「料盘检测」那组滑条"
    cv2.polylines(vis, [corners.astype(np.int32)], True, (255, 180, 0), 2, cv2.LINE_AA)
    for i, c in enumerate(corners):
        cv2.circle(vis, (int(c[0]), int(c[1])), 4, (0, 165, 255), -1)
        cv2.putText(vis, str(i), (int(c[0]) + 5, int(c[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    side = 640
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1],
                    [0, side - 1]], np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    front = cv2.warpPerspective(bgr, M, (side, side))
    front_e = cv2.warpPerspective(enh_bgr, M, (side, side))
    ppm = side / 160.0
    res, _e = detect(front_e, ppm, p)
    front_vis = draw(front, res, ppm, show_ref)
    Minv = np.linalg.inv(M)

    def back(pts):
        return cv2.perspectiveTransform(
            pts.astype(np.float32).reshape(-1, 1, 2), Minv).astype(np.int32)

    if show_ref:
        for cid, cx_mm, cy_mm, r_mm, kind in SLOTS:
            cv2.polylines(vis, [back(ring_pts((cx_mm * ppm, cy_mm * ppm),
                                              r_mm * ppm, kind, ppm))],
                          True, BLUE, 2, cv2.LINE_AA)
    for cid, c, r, kind, ang in res:
        cv2.polylines(vis, [back(ring_pts(c, r, kind, ppm, ang))],
                      True, COLORS[cid], 2, cv2.LINE_AA)
    Ls = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
    msg = ("料盘边长 %s px = %s mm | 检出 %d/4 | 偏差 %s"
           % (np.round(Ls, 0).astype(int).tolist(),
              np.round([v * 0.7287 for v in Ls], 1).tolist(), len(res),
              ", ".join("%s(%+.1f,%+.1f)" % (
                  ZH[cid],
                  c[0] / ppm - dict((s[0], s[1]) for s in SLOTS)[cid],
                  c[1] / ppm - dict((s[0], s[2]) for s in SLOTS)[cid])
                  for cid, c, _r, _k, _a in res) or "—"))
    return vis, vis_enh, front_vis, msg


def run_live(p: dict, show_ref: bool) -> int:
    try:
        import pyrealsense2 as rs
    except Exception as e:                                         # noqa: BLE001
        print("★ 打不开 pyrealsense2：%s" % e)
        return 2
    W, H, FPS = 640, 480, 30
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    prof = pipe.start(cfg)
    ctl = CamControls(rs, prof)
    print("相机已开：%dx%d@%d" % (W, H, FPS))
    print("已写入：%s" % (", ".join(ctl.apply(p)) or "（无）"))
    print("\n★ 相机**实际支持**的项（不在清单里的滑条 = 拖了没用）：")
    for nm, lst in (ctl.probe() or {}).items():
        print("   [%s]" % nm)
        for it in lst:
            print("      %s" % it)
    make_trackbars(p)
    print("\n按键：q 退出 | s 存参数 | f 存帧 | b 切换蓝圈参考\n")
    last_ctl = dict(p)
    ctl_keys = ("exposure", "gain", "wb", "auto", "laser", "cam_bright",
                "cam_contrast", "cam_sat", "cam_sharp", "cam_gamma",
                "cam_hue", "cam_blc")
    while True:
        try:
            fs = pipe.wait_for_frames(2000)
        except Exception as e:                                     # noqa: BLE001
            print("取帧失败：%s" % e)
            break
        cf = fs.get_color_frame()
        if not cf:
            continue
        bgr = np.asanyarray(cf.get_data())
        if any(p[k] != last_ctl[k] for k in ctl_keys):
            ctl.apply(p)
            last_ctl = dict(p)
        vis, vis_enh, front_vis, msg = process(bgr, p, show_ref)
        bar = np.zeros((30, vis.shape[1], 3), np.uint8)
        cv2.putText(bar, msg[:150], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("原始（+检测叠加）", np.vstack([vis, bar]))
        cv2.imshow("增强 / 边缘", vis_enh)
        if front_vis is not None:
            cv2.imshow("正视（蓝=设计位置，彩=检出）", front_vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('s'):
            save_tune(p)
            print("★ 已存：%s" % TUNE_PATH)
        if k == ord('b'):
            show_ref = not show_ref
        if k == ord('f'):
            ts = time.strftime("%H%M%S")
            for nm, im in (("color", bgr), ("front", front_vis)):
                if im is None:
                    continue
                q = os.path.join(OUT_DIR, "tune_%s_%s.png" % (ts, nm))
                print("   %s：%s" % (q, "成功" if imwrite_safe(q, im) else "★失败"))
    pipe.stop()
    cv2.destroyAllWindows()
    return 0


def run_offline(path: str, p: dict, show_ref: bool) -> int:
    bgr = imread_safe(path)
    if bgr is None:
        print("读不到 %s" % path)
        return 1
    print("离线：%s %s（不接相机，滑条一样能调）" % (path, bgr.shape))
    make_trackbars(p)
    print("按键：q 退出 | s 存参数 | b 切换蓝圈参考")
    while True:
        vis, vis_enh, front_vis, msg = process(bgr, p, show_ref)
        bar = np.zeros((30, vis.shape[1], 3), np.uint8)
        cv2.putText(bar, msg[:150], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("原始（+检测叠加）", np.vstack([vis, bar]))
        cv2.imshow("增强 / 边缘", vis_enh)
        if front_vis is not None:
            cv2.imshow("正视（蓝=设计位置，彩=检出）", front_vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            break
        if k == ord('s'):
            save_tune(p)
            print("★ 已存：%s" % TUNE_PATH)
        if k == ord('b'):
            show_ref = not show_ref
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="料盘槽位实时检测 + 现场调参（v2）")
    ap.add_argument("--image", help="离线模式：拿一张已存的图调参（不接相机）")
    ap.add_argument("--no-ref", action="store_true", help="不画设计位置的蓝圈")
    a = ap.parse_args()
    p = dict(DEF)
    if a.image:
        return run_offline(a.image, p, not a.no_ref)
    return run_live(p, not a.no_ref)


if __name__ == "__main__":
    raise SystemExit(main())
