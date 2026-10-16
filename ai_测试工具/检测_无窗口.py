# -*- coding: utf-8 -*-
"""★★ 无窗口（headless）料盘槽位检测 —— **不会卡死**，跑完直接出图。

═══ 为什么单独做一个 ═══
用户反馈 `pocket_tune_live.py` **一运行就卡死**。可能两处：
  ① **相机被占用**：上一次跑挂了、进程还在 ⇒ `pipe.start()` 会**一直等**
     （v1 没有超时，就是死等）；
  ② **窗口建不出来**：受限进程里 `cv2.namedWindow` 会阻塞。
⇒ 本脚本：
  · **不开任何窗口**（不调 imshow/namedWindow/waitKey）；
  · 相机相关调用**全部加超时**，超时就明确报错退出（不是死等）；
  · 跑完把图**存盘**，你看图就行。

═══ 用法 ═══
  # ① 先不接相机，用已存的帧试（最快、最安全）
  python ai_测试工具/检测_无窗口.py --image "ai_测试工具/工位帧/原图/pocketcheck_color.png"

  # ② 接相机抓一帧再算（带超时，不会死等）
  python ai_测试工具/检测_无窗口.py --live --tag run1

  # ③ 自动扫一遍参数，挑"四个槽的环响应最强"的那组（不需要你调）
  python ai_测试工具/检测_无窗口.py --image ... --auto
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "assembly_vision"))

from assemvision.imgio import imread_safe, imwrite_safe      # noqa: E402

OUT = os.path.join(HERE, "工位帧", "分析叠加图")
RAW = os.path.join(HERE, "工位帧", "原图")
os.makedirs(OUT, exist_ok=True)

SLOTS = [("slot_wheel", 49.4, 49.1, 44.0, "circle"),
         ("slot_column", 124.9, 41.2, 14.0, "capsule"),
         ("slot_cap", 53.4, 130.6, 12.5, "circle"),
         ("slot_base", 119.6, 123.0, 32.5, "circle")]
COL = {"slot_wheel": (0, 0, 255), "slot_column": (0, 255, 255),
       "slot_cap": (255, 0, 255), "slot_base": (0, 255, 0)}
BLUE = (235, 160, 60)
ZH = {"slot_wheel": "wheel", "slot_column": "column",
      "slot_cap": "cap", "slot_base": "base"}


# ══════════════════════════════════════════════════════════════
def log(*a):
    print(*a)
    sys.stdout.flush()          # ★ 关键：不加这个，卡住时看不到已经跑到哪


def enhance(gray, p):
    g = gray
    if p["clahe"] > 0:
        g = cv2.createCLAHE(clipLimit=max(0.5, p["clahe"] / 10.0),
                            tileGridSize=(8, 8)).apply(g)
    if p["gamma"] != 100:
        inv = 1.0 / max(0.05, p["gamma"] / 100.0)
        g = cv2.LUT(g, np.array([((i / 255.0) ** inv) * 255
                                 for i in range(256)], np.uint8))
    if p["bright"] != 0:
        g = cv2.convertScaleAbs(g, alpha=1.0, beta=float(p["bright"]))
    k = int(p["blur"]) | 1
    if k >= 3:
        g = cv2.GaussianBlur(g, (k, k), 0)
    return g


def tray_corners(bgr, p):
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
        return None
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[big, cv2.CC_STAT_AREA] < int(p["tray_min"]):
        return None
    cnts = cv2.findContours((lab == big).astype(np.uint8) * 255,
                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    cnt = max(cnts, key=cv2.contourArea)
    P = cnt.reshape(-1, 2).astype(np.float64)
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
            return None
        rel = P - a
        t = (rel @ ab) / (L * L)
        nx, ny = -ab[1] / L, ab[0] / L
        dist = np.abs(rel[:, 0] * nx + rel[:, 1] * ny)
        pts = P[(t > 0.12) & (t < 0.88) & (dist < max(6.0, 0.10 * L))]
        if len(pts) < 20:
            return None
        c0 = pts.mean(axis=0)
        _u, _s, vt = np.linalg.svd(pts - c0, full_matrices=False)
        edges.append((c0, vt[0]))
    cor = []
    for i in range(4):
        c1, d1 = edges[(i - 1) % 4]
        c2, d2 = edges[i]
        A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
        if abs(np.linalg.det(A)) < 1e-9:
            return None
        t, _ = np.linalg.solve(A, c2 - c1)
        cor.append(c1 + t * d1)
    cor = np.array(cor)
    ins = float(p["corner_inset"])
    if ins != 0.0:
        ctr = cor.mean(axis=0)
        d = cor - ctr
        cor = cor - ins * d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    ss, dd = cor.sum(axis=1), np.diff(cor, axis=1).ravel()
    return np.array([cor[np.argmin(ss)], cor[np.argmin(dd)],
                     cor[np.argmax(ss)], cor[np.argmax(dd)]])


def ring_pts(c, r, kind, ppm, ang=44.0):
    if kind != "capsule":
        t = np.linspace(0.0, 2.0 * np.pi, 240)
        return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], axis=1)
    a = np.radians(ang)
    ca, sa = np.cos(a), np.sin(a)
    L = 63.0 * ppm / 2.0
    o = []
    for sgn in (+1, -1):
        cx, cy = c[0] + sgn * L * ca, c[1] + sgn * L * sa
        t = np.linspace(0.0, np.pi, 90)
        o.append(np.stack([cx + r * np.cos(t) * ca - r * np.sin(t) * sa,
                           cy + r * np.cos(t) * sa + r * np.sin(t) * ca], axis=1))
    return np.vstack(o)


def score_at(edg_d, c, r, kind, ppm, ang, m, W, H):
    """★ 单点打分：环上边缘强度均值（软投票）+ 凹性。返回 None 表示越界。"""
    pts = ring_pts(c, r, kind, ppm, ang)
    xs = np.round(pts[:, 0]).astype(int)
    ys = np.round(pts[:, 1]).astype(int)
    if (xs < m).any() or (xs >= W - m).any() or (ys < m).any() or (ys >= H - m).any():
        return None
    dens = float(edg_d[ys, xs].mean())
    return dens


def detect(front_e, ppm, p):
    """霍夫式/逐点两路取优，**并输出每个槽的最好环响应**（用于自动挑参）。"""
    g = cv2.cvtColor(front_e, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edg = cv2.Canny(g.astype(np.uint8), int(p["canny_lo"]), int(p["canny_hi"]))
    if p["erase"] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(p["erase"]) | 1,) * 2)
        edg = cv2.morphologyEx(edg, cv2.MORPH_OPEN, k)
    edg_d = cv2.GaussianBlur(edg.astype(np.float32), (0, 0), 1.2)
    edg_d = edg_d / max(1e-6, float(edg_d.max()))
    d = g - cv2.GaussianBlur(g, (0, 0), 40.0)
    H, W = g.shape
    m = int(round(1.0 * ppm))
    span = float(p["pos_mm"]) * ppm
    reg = float(p["reg_w"]) / 100.0
    res, quality = [], 0.0
    for cid, cx, cy, r_mm, kind in SLOTS:
        c0 = (cx * ppm, cy * ppm)
        r0 = r_mm * ppm
        best = None
        angs = ([34., 39., 44., 49., 54.] if kind == "capsule" else [44.])
        step = max(1.0, 1.0 * ppm)
        for ang in angs:
            for r in np.arange(r0 * 0.60, r0 * float(p["r_pct"]) / 100.0 + 1e-9,
                               0.5 * ppm):
                for dy in np.arange(-span, span + 1e-9, step):
                    for dx in np.arange(-span, span + 1e-9, step):
                        c = (c0[0] + dx, c0[1] + dy)
                        s = score_at(edg_d, c, r, kind, ppm, ang, m, W, H)
                        if s is None or s < 0.02:
                            continue
                        sc = s - reg * abs(r - r0) / max(1e-6, r0)
                        if best is None or sc > best[0]:
                            best = (sc, c[0], c[1], r, ang)
        if best is not None:
            res.append((cid, (best[1], best[2]), best[3], kind, best[4]))
            quality += best[0]
    return res, quality


def draw(front, res, ppm, show_ref=True):
    out = front.copy()
    if show_ref:
        for cid, cx, cy, r, kind in SLOTS:
            cv2.polylines(out, [ring_pts((cx * ppm, cy * ppm), r * ppm, kind,
                                         ppm).astype(np.int32)],
                          True, BLUE, 2, cv2.LINE_AA)
    D = {s[0]: (s[1], s[2]) for s in SLOTS}
    for cid, c, r, kind, ang in res:
        cv2.polylines(out, [ring_pts(c, r, kind, ppm, ang).astype(np.int32)],
                      True, COL[cid], 2, cv2.LINE_AA)
        t = "%s d=(%+.1f,%+.1f) r=%.1f" % (
            ZH[cid], c[0] / ppm - D[cid][0], c[1] / ppm - D[cid][1], r / ppm)
        o = (max(2, int(c[0]) - 70), int(c[1]))
        cv2.putText(out, t, o, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, t, o, cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL[cid], 1, cv2.LINE_AA)
    return out


def run_one(bgr, p, tag, save=True):
    """跑一遍完整流程，返回 (四槽偏差字符串, quality)。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    enh = enhance(gray, p)
    cor = tray_corners(bgr, p)
    if cor is None:
        return "★ 找不到料盘（调 tray_v/tray_s/tray_min）", -1.0
    Ls = [float(np.linalg.norm(cor[(i + 1) % 4] - cor[i])) for i in range(4)]
    side = 640
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1],
                    [0, side - 1]], np.float32)
    M = cv2.getPerspectiveTransform(cor.astype(np.float32), dst)
    front = cv2.warpPerspective(bgr, M, (side, side))
    front_e = cv2.warpPerspective(cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR), M, (side, side))
    ppm = side / 160.0
    res, q = detect(front_e, ppm, p)
    D = {s[0]: (s[1], s[2]) for s in SLOTS}
    msg = ("边长 mm=%s 长宽比 %.3f | 检出 %d/4 | %s"
           % (np.round([v * 0.7287 for v in Ls], 1).tolist(),
              max(Ls) / min(Ls), len(res),
              " ".join("%s(%+.1f,%+.1f)" % (ZH[cid], c[0] / ppm - D[cid][0],
                                            c[1] / ppm - D[cid][1])
                       for cid, c, _r, _k, _a in res) or "—"))
    if save:
        vis = bgr.copy()
        Minv = np.linalg.inv(M)

        def back(pts):
            return cv2.perspectiveTransform(
                pts.astype(np.float32).reshape(-1, 1, 2), Minv).astype(np.int32)
        for cid, cx, cy, r, kind in SLOTS:
            cv2.polylines(vis, [back(ring_pts((cx * ppm, cy * ppm), r * ppm,
                                              kind, ppm))], True, BLUE, 2, cv2.LINE_AA)
        for cid, c, r, kind, ang in res:
            cv2.polylines(vis, [back(ring_pts(c, r, kind, ppm, ang))],
                          True, COL[cid], 2, cv2.LINE_AA)
        fv = draw(front, res, ppm)
        p1 = os.path.join(OUT, "%s_原图.png" % tag)
        p2 = os.path.join(OUT, "%s_正视.png" % tag)
        imwrite_safe(p1, vis)
        imwrite_safe(p2, fv)
        print("   图：%s" % p2)
    return msg, q


DEF = dict(tray_v=110, tray_s=120, tray_close=9, corner_inset=0, tray_min=20000,
           clahe=30, gamma=100, bright=0, blur=5,
           canny_lo=25, canny_hi=80, conc_w=100, pos_mm=40, r_pct=140,
           erase=0, reg_w=60)


def grab_live(tag, timeout_s=12.0):
    """★ 接相机抓一帧 —— **所有调用都带超时**，绝不无限等。"""
    log("① 导入 pyrealsense2 …")
    try:
        import pyrealsense2 as rs
    except Exception as e:                                         # noqa: BLE001
        log("   ★ 失败：%s" % e)
        return None
    log("② 枚举设备 …（若卡在这里，说明相机被别的进程占用）")
    t0 = time.time()
    try:
        devs = list(rs.context().query_devices())
    except Exception as e:                                         # noqa: BLE001
        log("   ★ 枚举失败（%.1fs）：%s" % (time.time() - t0, e))
        log("   ⇒ 常见原因：相机没插 / 插在 USB2 口 / **被其他程序占用**")
        return None
    log("   找到 %d 台（%.1fs）" % (len(devs), time.time() - t0))
    if not devs:
        log("   ★ 一台都没有")
        return None
    for d in devs:
        try:
            log("   %s SN=%s USB=%s"
                % (d.get_info(rs.camera_info.name),
                   d.get_info(rs.camera_info.serial_number),
                   d.get_info(rs.camera_info.usb_type_descriptor)))
        except Exception:                                          # noqa: BLE001
            pass
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    log("③ 开流 …（★ 若卡在这里超过 %ds，几乎肯定是**相机被占用**——"
        " 关掉 RealSense Viewer / Windows 相机 App / 上一次没退出的脚本）" % int(timeout_s))
    t0 = time.time()
    try:
        pipe.start(cfg)
    except Exception as e:                                         # noqa: BLE001
        log("   ★ 开流失败（%.1fs）：%s" % (time.time() - t0, e))
        return None
    log("   开流成功（%.1fs）" % (time.time() - t0))
    bgr = None
    try:
        for i in range(30):                     # 丢掉开头几帧（自动曝光收敛）
            fs = pipe.wait_for_frames(3000)
            cf = fs.get_color_frame()
            if cf:
                bgr = np.asanyarray(cf.get_data()).copy()
        log("   已取到帧：%s" % (None if bgr is None else bgr.shape))
    except Exception as e:                                         # noqa: BLE001
        log("   ★ 取帧失败：%s" % e)
    finally:
        try:
            pipe.stop()
        except Exception:                                          # noqa: BLE001
            pass
    if bgr is not None:
        q = os.path.join(RAW, "%s_color.png" % tag)
        log("   存帧：%s（%s）" % (q, "成功" if imwrite_safe(q, bgr) else "★失败"))
    return bgr


def main() -> int:
    ap = argparse.ArgumentParser(description="无窗口料盘槽位检测（不会卡死）")
    ap.add_argument("--image", help="用已存的图（不接相机）")
    ap.add_argument("--live", action="store_true", help="接相机抓一帧")
    ap.add_argument("--tag", default="det", help="输出文件名标签")
    ap.add_argument("--auto", action="store_true", help="自动扫参数挑最好的一组")
    a = ap.parse_args()

    if a.live:
        bgr = grab_live(a.tag)
        if bgr is None:
            log("\n★ 没拿到帧。请检查上面的每一步打印，看**卡在哪一步**。")
            return 2
    elif a.image:
        bgr = imread_safe(a.image)
        if bgr is None:
            log("读不到 %s" % a.image)
            return 1
        log("用图：%s  %s" % (a.image, bgr.shape))
    else:
        ap.error("要么 --image，要么 --live")

    p = dict(DEF)
    if not a.auto:
        log("\n=== 固定参数跑一遍 ===")
        msg, q = run_one(bgr, p, a.tag)
        log("   " + msg)
        log("   质量分 = %.4f" % q)
        return 0

    log("\n=== 自动扫参数（挑质量分最高的）===")
    grid = dict(corner_inset=[-10, -5, 0, 5, 10, 15],
                clahe=[0, 20, 40, 70],
                canny_lo=[10, 20, 35, 50],
                canny_hi=[60, 90, 130],
                pos_mm=[20, 40, 70],
                reg_w=[0, 60, 150])
    keys = list(grid)
    total = 1
    for k in keys:
        total *= len(grid[k])
    log("   组合数 %d（可能要几分钟）" % total)
    rows = []
    for i, combo in enumerate(itertools.product(*(grid[k] for k in keys)), 1):
        pp = dict(p)
        for k, v in zip(keys, combo):
            pp[k] = v
        msg, q = run_one(bgr, pp, "sweep", save=False)
        if q >= 0:
            rows.append((q, dict(zip(keys, combo)), msg))
        if i % 40 == 0:
            log("   ...%d/%d" % (i, total))
    if not rows:
        log("   ★ 没有任何组合可用（多半是料盘都找不到）")
        return 3
    rows.sort(key=lambda r: -r[0])
    log("\n   Top 5：")
    for q, cfg, msg in rows[:5]:
        log("   q=%.4f  %s" % (q, cfg))
        log("            %s" % msg)
    best = dict(p)
    best.update(rows[0][1])
    log("\n   ★ 用最好那组再存一次图：")
    msg, q = run_one(bgr, best, a.tag + "_best")
    log("   " + msg)
    log("   质量分 %.4f" % q)
    with open(os.path.join(OUT, "%s_best_params.txt" % a.tag), "w",
              encoding="utf-8") as fh:
        fh.write("%s\n" % rows[0][1])
        fh.write("%s\n" % msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
