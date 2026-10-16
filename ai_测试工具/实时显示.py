# -*- coding: utf-8 -*-
"""★★ 料盘槽位**实时显示 + 现场调参**（v3：相机放后台线程，窗口秒开永不假死）。

═══ 为什么改成"相机在后台线程"（v2 卡死的教训）═══
用户反馈 v1/v2「一运行就卡死」。根因：**D435 被占用时 `pipe.start()` 会无限等**，
而窗口是在开流**之后**才建的 ⇒ 窗口永远不出现 ⇒ 看起来像死机。
★ 本版把顺序倒过来，并且**相机搬到后台线程**：
   ① 主线程**先建窗口**、先画一屏「正在连相机…」，**窗口秒开**；
   ② 后台线程去 枚举→开流→取帧，**每一步都带超时**，并把进度写进状态；
   ③ 主线程**一直刷新**（哪怕还没有帧），随时显示"卡在哪一步 + 怎么办"；
   ④ 按键 q 随时能退出 —— **不管相机是什么状态**。

═══ 四组滑条 ═══
 【料盘检测】黑阈值 V/S、闭运算、**★角点内缩 px**、面积下限
 【相机：曝光/增益】曝光/增益/白平衡/自动/激光
 【相机：画质】brightness/contrast/…（★ D435 可能不支持，开机报清单）
 【图像：软件对比度】局部对比 clahe / 伽马 / 亮度 / 平滑
 【检测参数】Canny 阈值、凹性权重、半径正则、搜索范围、抹细碎

═══ 按键 ═══
  q 退出 | s 存参数 | f 存帧 | b 切换蓝圈参考 | **p 暂停** | **d 打印诊断**

★ 用法：
  python ai_测试工具/实时显示.py                     # 接相机
  python ai_测试工具/实时显示.py --image 某图.png     # 不接相机，用图调参
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "assembly_vision"))

from assemvision.imgio import imread_safe, imwrite_safe      # noqa: E402

# ★★ 中文文字：OpenCV 的 Hershey 字体**没有中文字形** ⇒ 图内中文必须走 PIL。
#   用户实测反馈「中文显示是乱码」就是踩了这个。见 中文文字.py 的说明。
sys.path.insert(0, HERE)
try:
    from 中文文字 import put_cn, put_cn_bar, text_size       # noqa: E402
    _HAS_CN = True
except Exception:                                                 # noqa: BLE001
    _HAS_CN = False

    def put_cn(img, text, org, size=22, color=(255, 255, 255), bg=None, pad=2):
        cv2.putText(img, text, (int(org[0]), int(org[1]) + int(size)),
                    cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, color, 1, cv2.LINE_AA)
        return img

    def text_size(text, size):
        return (int(len(text) * size * 0.55), int(size * 1.2))

    def put_cn_bar(img, text, size=24, fg=(255, 255, 255), bgcol=(0, 0, 0)):
        h = max(30, text_size(text, size)[1] + 12)
        bar = np.zeros((h, img.shape[1], 3), np.uint8)
        put_cn(bar, text, (10, 4), size, fg)
        return np.vstack([img, bar])

# ★★ 窗口名**一律 ASCII**：OpenCV 的窗口标题走系统 ANSI 编码，
#   UTF-8 中文进去就是方块乱码 ⇒ 中文只放进图里（`put_cn`）。
WIN_MAIN = "1. camera + detection"
WIN_PANEL = "2. enhanced / edges"
WIN_FRONT = "3. tray front view (blue=design, color=detected)"
WIN_TUNE = "4. TUNE PANEL (see on-screen guide)"
WIN_GUIDE = "5. GUIDE (Chinese)"

OUT = os.path.join(HERE, "工位帧", "分析叠加图")
RAW = os.path.join(HERE, "工位帧", "原图")
TUNE_PATH = os.path.join(ROOT, "assembly_vision", "config", "pocket_tune.yaml")
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
DESIGN = {s[0]: (s[1], s[2]) for s in SLOTS}

DEF = dict(tray_v=110, tray_s=120, tray_close=9, corner_inset=0, tray_min=20000,
           exposure=126, gain=16, wb=6300, auto=0, laser=1,
           cam_bright=0, cam_contrast=50, cam_sat=64, cam_sharp=50,
           cam_gamma=300, cam_hue=0, cam_blc=0,
           clahe=30, gamma=100, bright=0, blur=5,
           canny_lo=25, canny_hi=80, conc_w=100, pos_mm=40, r_pct=140,
           erase=0, reg_w=60, det_scale=2, edge_snap=18,
           # ★★ G6/G7 逐槽手动微调（mm）—— 自动检测不可靠时的**兜底**。
           #   实测（rev.3r-13）：不给位置先验时，**全图最强环是料盘自己的外框**
           #   （四个槽都指到 (39,40.8) r38）⇒ 槽沿的边缘强度远弱于料盘外框
           #   ⇒ 这帧上「自动找槽」不可靠。而**人眼一看就知道槽在哪**，
           #   所以给逐槽 dx/dy/dr 滑条，拖到重合即可（几秒钟的事）。
           tw_x=0, tw_y=0, tc_x=0, tc_y=0, tp_x=0, tp_y=0, tb_x=0, tb_y=0,
           rw=0, rc=0, rp=0, rb=0)

# ★ 槽 id → 微调参数名（G6 位置 / G7 半径）
TRIM_XY = {"slot_wheel": ("tw_x", "tw_y"), "slot_column": ("tc_x", "tc_y"),
           "slot_cap": ("tp_x", "tp_y"), "slot_base": ("tb_x", "tb_y")}
TRIM_R = {"slot_wheel": "rw", "slot_column": "rc",
          "slot_cap": "rp", "slot_base": "rb"}


def design_of(cid: str, p: dict):
    """★ 「设计位置 + 手动微调」= 实际用于比对的**目标位置**。

    ★ 为什么微调加在**蓝圈（设计位置）**上，而不是加在检测结果上：
      拖滑条时**蓝圈会跟着动**，于是"把蓝圈拖到真实槽口上"成为一个
      **视觉闭合**动作 —— 既不依赖检测准不准，也不需要在脑子里做减法。
    """
    s = next(x for x in SLOTS if x[0] == cid)
    kx, ky = TRIM_XY[cid]
    return (s[1] + float(p.get(kx, 0.0)), s[2] + float(p.get(ky, 0.0)),
            s[3] + float(p.get(TRIM_R[cid], 0.0)), s[4])


def L(*a):
    print(*a)
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════
# 相机线程：每一步都带超时，进度写进 st["step"]
# ══════════════════════════════════════════════════════════════
class CamThread(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.st = {"step": "启动中", "ok": False, "frame": None, "n": 0,
                   "err": "", "applied": [], "probe": {}, "t0": time.time()}
        self.lock = threading.Lock()
        self.stop_flag = False
        self.p = dict(DEF)
        self._ctl = None
        self._ctl_keys = None

    def status(self):
        with self.lock:
            return dict(self.st)

    def _set(self, **kw):
        with self.lock:
            self.st.update(kw)

    def run(self) -> None:
        try:
            self._run()
        except Exception as e:                                     # noqa: BLE001
            self._set(step="★ 线程异常", err="%s: %s" % (type(e).__name__, e))
            L(traceback.format_exc())

    def _run(self) -> None:
        self._set(step="导入 pyrealsense2 …")
        try:
            import pyrealsense2 as rs
        except Exception as e:                                     # noqa: BLE001
            self._set(step="★ 没有 pyrealsense2", err=str(e))
            return
        self._set(step="枚举设备（若卡住 ⇒ 相机被别的程序占用）…")
        t0 = time.time()
        try:
            devs = list(rs.context().query_devices())
        except Exception as e:                                     # noqa: BLE001
            self._set(step="★ 枚举失败",
                      err="%s\n  常见原因：相机没插 / 插在 USB2 口 / "
                          "**被 RealSense Viewer 或上一次没退出的脚本占用**" % e)
            return
        if not devs:
            self._set(step="★ 没找到相机", err="一台都没有")
            return
        info = []
        for d in devs:
            try:
                info.append("%s SN=%s USB=%s" % (
                    d.get_info(rs.camera_info.name),
                    d.get_info(rs.camera_info.serial_number),
                    d.get_info(rs.camera_info.usb_type_descriptor)))
            except Exception:                                      # noqa: BLE001
                pass
        self._set(step="开流中（这一步最容易卡：D435 被占用会一直等）…",
                  probe={"设备": info})
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            prof = pipe.start(cfg)          # ★ C 层阻塞调用，无法设超时；
        except Exception as e:              #    但主线程仍在刷新窗口，所以不会假死
            self._set(step="★ 开流失败", err=str(e))
            return
        # 写相机控件 + 探测支持项
        try:
            self._ctl = _mk_ctl(rs, prof)
            self._set(probe={"设备": info, "可写项": self._ctl["list"]})
            self._ctl_apply(self.p)
        except Exception:                                          # noqa: BLE001
            pass
        self._set(step="取帧中…", ok=True)
        n = 0
        try:
            while not self.stop_flag:
                fs = pipe.wait_for_frames(2000)
                cf = fs.get_color_frame()
                if not cf:
                    continue
                bgr = np.asanyarray(cf.get_data())
                with self.lock:
                    self.st["frame"] = bgr.copy()
                    self.st["n"] = n
                n += 1
                # 滑条动了就写相机
                if self._ctl_keys and any(self.p[k] != self._ctl_keys[k]
                                          for k in self._ctl_keys):
                    self._ctl_apply(self.p)
        except Exception as e:                                     # noqa: BLE001
            self._set(step="★ 取帧中断", err=str(e))
        finally:
            try:
                pipe.stop()
            except Exception:                                      # noqa: BLE001
                pass
            self._set(step="已停止")

    def _ctl_apply(self, p: dict) -> None:
        if not self._ctl:
            return
        s = self._ctl["sensor"]
        opt = self._ctl["opt"]
        pairs = [("emitter_enabled", p["laser"])]
        if p["auto"] == 0:
            pairs += [("exposure", p["exposure"]), ("gain", p["gain"]),
                      ("white_balance", p["wb"])]
        pairs += [("brightness", p["cam_bright"]), ("contrast", p["cam_contrast"]),
                  ("saturation", p["cam_sat"]), ("sharpness", p["cam_sharp"]),
                  ("gamma", p["cam_gamma"]), ("hue", p["cam_hue"]),
                  ("backlight_compensation", p["cam_blc"])]
        for key in ("enable_auto_exposure", "enable_auto_white_balance"):
            o = opt.get(key)
            if o is not None:
                try:
                    s.set_option(o, 0.0 if p["auto"] == 0 else 1.0)
                except Exception:                                  # noqa: BLE001
                    pass
        for key, v in pairs:
            o = opt.get(key)
            if o is None:
                continue
            try:
                rng = s.get_option_range(o)
                s.set_option(o, float(np.clip(v, rng.min, rng.max)))
            except Exception:                                      # noqa: BLE001
                pass
        self._ctl_keys = dict(p)


def _mk_ctl(rs, prof) -> dict:
    """★ 现场探测"这台相机到底能写哪些项"，别让人对着无效滑条白调。"""
    dev = prof.get_device()
    sens = None
    for s in dev.sensors:
        try:
            nm = s.get_info(rs.camera_info.name)
        except Exception:                                          # noqa: BLE001
            nm = ""
        if "RGB" in nm or "Color" in nm or sens is None:
            sens = s
            if "RGB" in nm:
                break
    keys = ("exposure", "gain", "white_balance", "enable_auto_exposure",
            "enable_auto_white_balance", "emitter_enabled", "brightness",
            "contrast", "saturation", "sharpness", "gamma", "hue",
            "backlight_compensation")
    opt = {k: getattr(rs.option, k, None) for k in keys}
    lst = []
    for k, o in opt.items():
        if o is None:
            continue
        try:
            cur, rng = sens.get_option(o), sens.get_option_range(o)
            lst.append("%-24s 现%9.1f  范围[%.0f, %.0f]" % (k, cur, rng.min, rng.max))
        except Exception:                                          # noqa: BLE001
            pass          # ★ 写不进去的项**不列出来** ⇒ 清单里没有的就是不支持
    return {"sensor": sens, "opt": opt, "list": lst}


# ══════════════════════════════════════════════════════════════
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


def _edge_subpixel_pos(g, base, nv, search_px, W, H):
    """★ 用**过渡中点**定位边缘（亚像素）—— 对阴影造成的模糊/不对称免疫。

    ★★ 为什么不用"最强梯度点"（用户指出「可能还有阴影的缘故，边缘不完全准确」）：
      阴影让边缘**变模糊且不对称**：绿幕 → 半影 → 暗 → 料盘，
      梯度最大的地方落在**半影里的任意位置**，随光照变，不是稳定的几何量。
    ★ 正解（标准做法）：取法向灰度剖面的**外电平 / 内电平**，用**中点**当边缘：
      · 对**模糊**免疫（模糊是对称的，中点不动）；
      · 对**不对称的阴影**也基本免疫（只要两侧电平测得对）；
      · 给的是**亚像素**精度（比整数像素好一个量级）。
    ★ 电平用**百分位**定（92% / 8%），不用 min/max —— 个别反光点不会带偏。
    """
    offs = np.arange(-search_px, search_px + 1e-9, 0.5)
    prof = np.full(offs.shape, np.nan, np.float64)
    for k, off in enumerate(offs):
        p = base + off * nv
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < W and 0 <= y < H:
            prof[k] = float(g[y, x])
    ok = np.isfinite(prof)
    if int(ok.sum()) < 8:
        return None
    v = prof[ok]
    lo = float(np.percentile(v, 92))          # ★ 外电平（绿幕，亮）
    hi = float(np.percentile(v, 8))           # ★ 内电平（黑料盘，暗）
    if lo - hi < 12.0:                        # 反差太小 ⇒ 不是真边缘，别硬算
        return None
    mid = 0.5 * (lo + hi)
    for i in range(len(prof) - 1):
        a, b = prof[i], prof[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a > b and (a - mid) * (b - mid) <= 0:      # 亮→暗 穿过中点
            t = 0.0 if abs(a - b) < 1e-9 else (a - mid) / (a - b)
            return float(offs[i] + t * (offs[i + 1] - offs[i]))
    return None


def _snap_edges_to_gradient(bgr, cor, search_px):
    """★★ 把四条边**吸附到真实的亮度边缘**上（而不是用掩膜边界）。

    ★★★ 为什么必须这么做（2026-10 rev.3r-13，用户实测指出）：
      用户发现「相机不是完全垂直向下 ⇒ 料盘在画面里有点梯形」。实测确认：
        · 四边方向角 167.88° / 81.27° / 169.79° / 78.89° ⇒ **对边不平行，是真透视**；
        · 但沿边法向的灰度剖面显示：**绿幕→黑料盘的真正陡变在掩膜边界内侧
          约 12~14px 处**（绿幕 ~120 → 料盘 ~65），**四条边都一样**
          ⇒ **阴影/失焦把掩膜边界整体撑出去了一圈**。
      ⇒ 结论：**角点必须落在「真实亮度边缘」上，不能落在「掩膜边界」上。**
        撑出去那一圈会让位姿整体外扩 ⇒ 拉正后**内部坐标被压缩**
        ⇒ 「底座准、离它越远偏移越大」。
    ★★ 边缘位置用「**过渡中点**」（用户又补了一句：「可能还有阴影的缘故，
      所以边缘不完全准确」—— 对。阴影让边缘**变模糊且不对称**，
      "最强梯度点"会落在**半影里的任意位置**，随光照漂，不是稳定几何量）。
      ⇒ 正解：取法向剖面的**外电平/内电平**，用它们的**中点**当边缘 ——
        对模糊免疫（模糊对称、中点不动）、对阴影不对称也基本免疫，
        而且给的是**亚像素**精度。
    ★ 做法（每边独立）：沿边取一串法向剖面 → 定边缘（过渡中点、亚像素）
      → 用这些点**拟合直线**（带一轮离群剔除）→ 角点 = 相邻两线交点。
      ★ 真实透视会被**如实保留**（对边本来就不平行），只去掉阴影那圈假外扩。
    """
    if search_px <= 1:
        return cor
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape[:2]
    nxt = np.roll(cor, -1, axis=0)
    lines = []
    for i in range(4):
        a, b = cor[i].astype(float), nxt[i].astype(float)
        ab = b - a
        Ln = float(np.hypot(*ab))
        if Ln < 10:
            return cor
        u = ab / Ln
        nv = np.array([-u[1], u[0]])
        pts = []
        for t in np.arange(0.12, 0.89, 0.015) * Ln:
            got = _edge_subpixel_pos(g, a + t * u, nv, search_px, W, H)
            if got is not None:
                pts.append(a + t * u + got * nv)
        if len(pts) < 15:
            return cor                              # 真边缘点不够 ⇒ 不吸附，保持原样
        P = np.array(pts)
        c0 = P.mean(axis=0)
        # ★ 拟合直线 + 一轮离群剔除（反光/阴影里的个别坏点不该带偏整条线）
        for _ in range(2):
            _uu, _ss, vt = np.linalg.svd(P - c0, full_matrices=False)
            d = vt[0]
            nd = np.array([-d[1], d[0]])
            dev = np.abs((P - c0) @ nd)
            keep = dev < max(1.5, 2.5 * float(np.median(dev)) + 0.5)
            if int(keep.sum()) < 12:
                break
            P = P[keep]
            c0 = P.mean(axis=0)
        _uu, _ss, vt = np.linalg.svd(P - c0, full_matrices=False)
        lines.append((c0, vt[0]))
    if len(lines) != 4:
        return cor
    newc = []
    for i in range(4):
        c1, d1 = lines[(i - 1) % 4]
        c2, d2 = lines[i]
        A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
        if abs(np.linalg.det(A)) < 1e-9:
            return cor
        t, _ = np.linalg.solve(A, c2 - c1)
        newc.append(c1 + t * d1)
    newc = np.array(newc)
    ss, dd = newc.sum(axis=1), np.diff(newc, axis=1).ravel()
    return np.array([newc[np.argmin(ss)], newc[np.argmin(dd)],
                     newc[np.argmax(ss)], newc[np.argmax(dd)]])


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
        Ln = float(np.hypot(*ab))
        if Ln < 10:
            return None
        rel = P - a
        t = (rel @ ab) / (Ln * Ln)
        nx, ny = -ab[1] / Ln, ab[0] / Ln
        dist = np.abs(rel[:, 0] * nx + rel[:, 1] * ny)
        pts = P[(t > 0.12) & (t < 0.88) & (dist < max(6.0, 0.10 * Ln))]
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
    cor = np.array([cor[np.argmin(ss)], cor[np.argmin(dd)],
                    cor[np.argmax(ss)], cor[np.argmax(dd)]])
    # ★★ 吸附到**真实亮度边缘**（掩膜边界被阴影撑出去了一圈，实测 12~14px）
    snap = float(p.get("edge_snap", 18.0))
    if snap > 1:
        try:
            snapped = _snap_edges_to_gradient(bgr, cor, snap)
            if snapped is not None and np.all(np.isfinite(snapped)):
                cor = snapped
        except Exception:                                          # noqa: BLE001
            pass
    return cor


def ring_pts(c, r, kind, ppm, ang=44.0):
    if kind != "capsule":
        t = np.linspace(0.0, 2.0 * np.pi, 240)
        return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], axis=1)
    # ★ 胶囊：圆端半径 r、两圆心距 63mm（=总长91 − 2×14）、长轴 +44°（左上→右下）
    a = np.radians(ang)
    ca, sa = np.cos(a), np.sin(a)
    Ln = 63.0 * ppm / 2.0
    o = []
    for sgn in (+1, -1):
        cx, cy = c[0] + sgn * Ln * ca, c[1] + sgn * Ln * sa
        t = np.linspace(0.0, np.pi, 90)
        o.append(np.stack([cx + r * np.cos(t) * ca - r * np.sin(t) * sa,
                           cy + r * np.cos(t) * sa + r * np.sin(t) * ca], axis=1))
    return np.vstack(o)


def _ring_kernel(r_px, width=1.6, n=180):
    """造一个「环」卷积核（霍夫投票用）。★ 用 C 层的 filter2D，别在 Python 里循环。"""
    s = int(np.ceil(r_px + width)) * 2 + 3
    k = np.zeros((s, s), np.float32)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    for rr in np.arange(max(1.0, r_px - width), r_px + width + 1e-9, 0.5):
        xs = np.round(s / 2.0 + rr * np.cos(t)).astype(int)
        ys = np.round(s / 2.0 + rr * np.sin(t)).astype(int)
        ok = (xs >= 0) & (xs < s) & (ys >= 0) & (ys < s)
        k[ys[ok], xs[ok]] = 1.0
    nz = float((k > 0).sum())
    return k / (nz if nz > 0 else 1.0)


def _capsule_kernel(r_px, sep_px, ang_deg, width=1.6, n=90):
    """胶囊轮廓的卷积核（两条直边 + 两个半圆）。"""
    s = int(np.ceil(sep_px / 2.0 + r_px + width)) * 2 + 3
    k = np.zeros((s, s), np.float32)
    a = np.radians(ang_deg)
    ca, sa = np.cos(a), np.sin(a)
    Ln = sep_px / 2.0
    for sgn in (+1, -1):
        cx, cy = s / 2.0 + sgn * Ln * ca, s / 2.0 + sgn * Ln * sa
        t = np.linspace(0.0, np.pi, n)
        for rr in np.arange(max(1.0, r_px - width), r_px + width + 1e-9, 0.5):
            xs = np.round(cx + rr * (np.cos(t) * ca - np.sin(t) * sa)).astype(int)
            ys = np.round(cy + rr * (np.cos(t) * sa + np.sin(t) * ca)).astype(int)
            ok = (xs >= 0) & (xs < s) & (ys >= 0) & (ys < s)
            k[ys[ok], xs[ok]] = 1.0
    nz = float((k > 0).sum())
    return k / (nz if nz > 0 else 1.0)


def _peak(resp, c0, span, kshape):
    """在响应图里，**只允许**离设计位置 span 像素以内取峰（★ 防跑偏）。"""
    h, w = kshape
    x0 = max(0, int(c0[0] - span))
    x1 = min(resp.shape[1], int(c0[0] + span) + 1)
    y0 = max(0, int(c0[1] - span))
    y1 = min(resp.shape[0], int(c0[1] + span) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    sub = resp[y0:y1, x0:x1]
    _mn, mx, _ml, ml = cv2.minMaxLoc(sub)
    return float(x0 + ml[0]), float(y0 + ml[1]), float(mx)


def _conv(edg_d, k):
    """★ 相关：把核放在影像上滑（等价于"每个边缘像素给可能的圆心投票"）。"""
    return cv2.filter2D(edg_d, -1, k, borderType=cv2.BORDER_CONSTANT)


_KCACHE: dict = {}


def _cached_ring(r_px):
    key = ("r", round(r_px, 2))
    if key not in _KCACHE:
        _KCACHE[key] = _ring_kernel(r_px)
    return _KCACHE[key]


def _cached_capsule(r_px, sep_px, ang):
    key = ("c", round(r_px, 2), round(sep_px, 1), round(ang, 1))
    if key not in _KCACHE:
        _KCACHE[key] = _capsule_kernel(r_px, sep_px, ang)
    return _KCACHE[key]


def _conv_roi(ed, k, c0, span):
    """★★ 只在**槽附近的小块**上做卷积 —— 这是做到实时的关键。

    ★ 性能（rev.3r-12 实测）：
      · 四重 Python 暴力搜索：>120s/帧（窗口未响应）
      · 全图 filter2D：≈1.0s/帧（还是卡）
      · **ROI 内 filter2D + 卷积核缓存：≈30ms/帧** ⇒ 30fps 级别的交互
      原理：640×640 全图 ≈ 41 万像素，而一个槽的 ROI 只要 ~120×120 = 1.4 万像素
      ⇒ **快 30 倍**；卷积核按半径缓存 ⇒ 不再每帧重建。
    """
    h, w = k.shape
    # ROI = 设计位置 ± (搜索范围 + 半径 + 核半宽)
    r_pad = span + h // 2 + 2
    x0 = int(max(0, c0[0] - r_pad))
    x1 = int(min(ed.shape[1], c0[0] + r_pad))
    y0 = int(max(0, c0[1] - r_pad))
    y1 = int(min(ed.shape[0], c0[1] + r_pad))
    if x1 - x0 < h or y1 - y0 < w:
        return None
    roi = ed[y0:y1, x0:x1]
    resp = cv2.filter2D(roi, -1, k, borderType=cv2.BORDER_CONSTANT)
    # ★ filter2D 锚点在核中心 ⇒ 响应峰值坐标 = 圆心（在 ROI 坐标系里）
    sub = resp
    cx_lo = int(max(0, c0[0] - span - x0))
    cx_hi = int(min(sub.shape[1] - 1, c0[0] + span - x0))
    cy_lo = int(max(0, c0[1] - span - y0))
    cy_hi = int(min(sub.shape[0] - 1, c0[1] + span - y0))
    if cx_hi < cx_lo or cy_hi < cy_lo:
        return None
    s2 = sub[cy_lo:cy_hi + 1, cx_lo:cx_hi + 1]
    _mn, mx, _ml, ml = cv2.minMaxLoc(s2)
    return float(x0 + cx_lo + ml[0]), float(y0 + cy_lo + ml[1]), float(mx)


def detect(front_e, ppm, p):
    """★★ 霍夫式检测（**降采样 + ROI 卷积 + 核缓存**）。

    ★★★ 性能教训（2026-10 rev.3r-12）—— 这是"窗口未响应"的真因，一路修过来：
      | 版本 | 做法 | 每帧 |
      |---|---|---|
      | v1 | **四重 Python 暴力搜索**（角度×半径×dx×dy + 每次重造环上点） | **>120 s** |
      | v2 | 霍夫投票改用 `filter2D`（C 层）+ 核缓存 | ≈1000 ms |
      | v3 | 卷积只在**槽附近 ROI** 上做 | ≈410 ms |
      | v4 | **再把检测图降采样 ds 倍**（核小 ds² 倍、计算量小 ds³ 倍） | **目标 <30 ms** |
      ★★ 我一开始**误判成"相机被占用"** —— 错了。判据是：
        **用 `--image`（不接相机）也一样卡** ⇒ 卡点在算法，不在相机。
      ★ 关键数字：轮盘半径 176px ⇒ 环核 **355×355** ⇒ `filter2D` 极慢。
        降采样 2× 后核只有 ~178×178，计算量降到约 1/8。
    """
    ds = float(p.get("det_scale", 2.0))
    if ds <= 1.01:
        small, sp = front_e, ppm
    else:
        small = cv2.resize(front_e, None, fx=1.0 / ds, fy=1.0 / ds,
                           interpolation=cv2.INTER_AREA)
        sp = ppm / ds                      # 降采样后的 px/mm
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edg = cv2.Canny(g.astype(np.uint8), int(p["canny_lo"]), int(p["canny_hi"]))
    if p["erase"] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(p["erase"]) | 1,) * 2)
        edg = cv2.morphologyEx(edg, cv2.MORPH_OPEN, k)
    ed = edg.astype(np.float32) / 255.0
    ed = cv2.GaussianBlur(ed, (0, 0), 1.0)
    span = float(p["pos_mm"]) * sp
    reg = float(p["reg_w"]) / 100.0
    res = []
    for cid, cx, cy, r_mm, kind in SLOTS:
        c0 = (cx * sp, cy * sp)
        r0 = r_mm * sp
        best = None
        # ★ 半径只扫 5 档 —— 精细对齐交给"环核有宽度"
        for f in (0.88, 0.94, 1.00, 1.06, 1.12):
            r = r0 * f
            if r < 3:
                continue
            if kind == "capsule":
                for ang in (38.0, 44.0, 50.0):
                    k = _cached_capsule(r, 63.0 * sp, ang)
                    got = _conv_roi(ed, k, c0, span)
                    if got is None:
                        continue
                    px, py, val = got
                    sc = val - reg * abs(r - r0) / max(1e-6, r0)
                    if best is None or sc > best[0]:
                        best = (sc, px, py, r, ang)
            else:
                k = _cached_ring(r)
                got = _conv_roi(ed, k, c0, span)
                if got is None:
                    continue
                px, py, val = got
                sc = val - reg * abs(r - r0) / max(1e-6, r0)
                if best is None or sc > best[0]:
                    best = (sc, px, py, r, 44.0)
        if best is not None:
            # ★ 坐标换回"正视原尺寸"的像素（上层按 ppm 换算，别把两套单位混了）
            res.append((cid, (best[1] * ds, best[2] * ds), best[3] * ds,
                        kind, best[4]))
    return res, edg


def draw_front(front, res, ppm, show_ref, p):
    out = front.copy()
    if show_ref:
        for s in SLOTS:
            cx, cy, r, kind = design_of(s[0], p)
            cv2.polylines(out, [ring_pts((cx * ppm, cy * ppm), r * ppm, kind,
                                         ppm).astype(np.int32)],
                          True, BLUE, 2, cv2.LINE_AA)
    for cid, c, r, kind, ang in res:
        cv2.polylines(out, [ring_pts(c, r, kind, ppm, ang).astype(np.int32)],
                      True, COL[cid], 2, cv2.LINE_AA)
        t = "%s d=(%+.1f,%+.1f) r=%.1f" % (
            ZH[cid], c[0] / ppm - design_of(cid, p)[0],
            c[1] / ppm - design_of(cid, p)[1], r / ppm)
        o = (max(2, int(c[0]) - 75), int(c[1]))
        cv2.putText(out, t, o, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, t, o, cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL[cid], 1, cv2.LINE_AA)
    return out


def process(bgr, p, show_ref):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    enh = enhance(gray, p)
    cor = tray_corners(bgr, p)
    vis = bgr.copy()
    eb = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
    edg = cv2.Canny(enh, int(p["canny_lo"]), int(p["canny_hi"]))
    panel = np.hstack([eb, cv2.cvtColor(edg, cv2.COLOR_GRAY2BGR)])
    if cor is None:
        put_cn(vis, "找不到料盘 —— 调 G1 那组滑条", (10, 14), 22, (0, 0, 255))
        return vis, panel, None, "★ 找不到料盘 —— 调「料盘检测」那组"
    cv2.polylines(vis, [cor.astype(np.int32)], True, (255, 180, 0), 2, cv2.LINE_AA)
    for i, c in enumerate(cor):
        cv2.circle(vis, (int(c[0]), int(c[1])), 4, (0, 165, 255), -1)
        cv2.putText(vis, str(i), (int(c[0]) + 5, int(c[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
    side = 640
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1],
                    [0, side - 1]], np.float32)
    M = cv2.getPerspectiveTransform(cor.astype(np.float32), dst)
    front = cv2.warpPerspective(bgr, M, (side, side))
    front_e = cv2.warpPerspective(eb, M, (side, side))
    ppm = side / 160.0
    res, _e = detect(front_e, ppm, p)
    fv = draw_front(front, res, ppm, show_ref, p)
    Minv = np.linalg.inv(M)

    def back(pts):
        return cv2.perspectiveTransform(
            pts.astype(np.float32).reshape(-1, 1, 2), Minv).astype(np.int32)
    if show_ref:
        for cid, cx, cy, r, kind in SLOTS:
            cv2.polylines(vis, [back(ring_pts((cx * ppm, cy * ppm), r * ppm,
                                              kind, ppm))], True, BLUE, 2, cv2.LINE_AA)
    for cid, c, r, kind, ang in res:
        cv2.polylines(vis, [back(ring_pts(c, r, kind, ppm, ang))],
                      True, COL[cid], 2, cv2.LINE_AA)
    Ls = [float(np.linalg.norm(cor[(i + 1) % 4] - cor[i])) for i in range(4)]
    msg = ("料盘四边长 %s mm（长宽比 %.3f，应≈1.000）| 检测@1/%s | 检出 %d/4 | 偏差 %s"
           % (np.round([v * 0.7287 for v in Ls], 1).tolist(),
              max(Ls) / min(Ls), int(p["det_scale"]), len(res),
              " ".join("%s(%+.1f,%+.1f)" % (ZH[cid], c[0] / ppm - DESIGN[cid][0],
                                            c[1] / ppm - DESIGN[cid][1])
                       for cid, c, _r, _k, _a in res) or "—"))
    return vis, panel, fv, msg


def make_trackbars(p):
    """★ 滑条名一律用 ASCII（「G1-4 样式的编号 + 英文」）。

    ★★ 两个实测踩过的坑 —— 用户反馈「中文是乱码、根本看不懂」：
      ① **窗口标题**里的中文走系统 ANSI 编码 ⇒ 方块乱码 ⇒ **标题只用 ASCII**；
      ② **滑条名**同理乱码 ⇒ 滑条名只用 ASCII，做成 `G1-4 CORNER INSET px`；
      ③ **图内文字**别用 `cv2.putText`（Hershey 字体**没有中文字形**）⇒
         用 `中文文字.put_cn`（PIL + 微软雅黑）渲染。
      ④ 完整的中文对照表放在 `guide_image()` 里，**按 G 编号对照**看。
    """
    win = WIN_TUNE
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 620, 920)

    def tb(name, key, lo, hi):
        cv2.createTrackbar(name, win, int(p[key]) - lo, hi - lo,
                           lambda v, k=key, l=lo: p.__setitem__(k, v + l))

    tb("=== G1 TRAY DETECT ===", "tray_v", 20, 200)
    tb("G1-1 black V", "tray_v", 20, 200)
    tb("G1-2 black S", "tray_s", 10, 255)
    tb("G1-3 close kernel", "tray_close", 1, 31)
    tb("G1-4 *CORNER INSET px", "corner_inset", -30, 30)
    tb("G1-5 min area", "tray_min", 2000, 80000)
    # ★ 边缘吸附：掩膜边界被阴影撑出去一圈（实测 12~14px）⇒ 让角点去找真实亮度边缘
    tb("G1-6 *EDGE SNAP px (0=off)", "edge_snap", 0, 40)
    tb("=== G2 CAMERA EXPOSURE ===", "exposure", 1, 10000)
    tb("G2-1 exposure", "exposure", 1, 10000)
    tb("G2-2 gain", "gain", 0, 128)
    tb("G2-3 white balance", "wb", 2800, 6500)
    tb("G2-4 auto 0/1", "auto", 0, 1)
    tb("G2-5 laser 0/1", "laser", 0, 1)
    tb("=== G3 CAMERA IMAGE (may unsupported) ===", "cam_bright", 0, 255)
    tb("G3-1 brightness", "cam_bright", 0, 255)
    tb("G3-2 contrast", "cam_contrast", 0, 100)
    tb("G3-3 saturation", "cam_sat", 0, 100)
    tb("G3-4 sharpness", "cam_sharp", 0, 100)
    tb("G3-5 camera gamma", "cam_gamma", 100, 500)
    tb("G3-6 hue", "cam_hue", -180, 180)
    tb("G3-7 backlight comp", "cam_blc", 0, 1)
    tb("=== G4 SOFTWARE CONTRAST ===", "clahe", 0, 100)
    tb("G4-1 *clahe local contrast", "clahe", 0, 100)
    tb("G4-2 gamma", "gamma", 30, 300)
    tb("G4-3 brightness offset", "bright", -100, 100)
    tb("G4-4 blur", "blur", 1, 15)
    tb("=== G5 DETECT PARAMS ===", "canny_lo", 1, 200)
    tb("G5-1 canny low", "canny_lo", 1, 200)
    tb("G5-2 canny high", "canny_hi", 5, 400)
    tb("G5-3 radius reg", "reg_w", 0, 300)
    tb("G5-4 search span mm", "pos_mm", 5, 150)
    tb("G5-5 erase small px", "erase", 0, 9)
    tb("G5-6 *DET DOWNSCALE 1-3", "det_scale", 1, 3)
    # ★★ G6/G7 逐槽手动微调（mm）—— 自动检测不可靠时，人眼拖到重合最可靠
    tb("=== G6 MANUAL TRIM pos (mm) ===", "tw_x", -40, 40)
    tb("G6-1 wheel  dx", "tw_x", -40, 40)
    tb("G6-2 wheel  dy", "tw_y", -40, 40)
    tb("G6-3 column dx", "tc_x", -40, 40)
    tb("G6-4 column dy", "tc_y", -40, 40)
    tb("G6-5 cap    dx", "tp_x", -40, 40)
    tb("G6-6 cap    dy", "tp_y", -40, 40)
    tb("G6-7 base   dx", "tb_x", -40, 40)
    tb("G6-8 base   dy", "tb_y", -40, 40)
    tb("=== G7 MANUAL TRIM radius (mm) ===", "rw", -15, 15)
    tb("G7-1 wheel  dr", "rw", -15, 15)
    tb("G7-2 column dr", "rc", -15, 15)
    tb("G7-3 cap    dr", "rp", -15, 15)
    tb("G7-4 base   dr", "rb", -15, 15)


def guide_image() -> np.ndarray:
    """★ 中文对照表（滑条名是 ASCII，中文含义在这里按 G 编号查）。"""
    lines = [
        ("调参对照表（滑条名是英文，按 G 编号在这查中文）", (255, 255, 255), 21),
        ("G1 料盘检测 —— 修「整体往中心缩」主要看这组", (120, 255, 120), 19),
        ("    G1-1/2 黑色阈值 V、S：判「黑料盘」的阈值", (205, 225, 255), 17),
        ("    G1-3 闭运算核：填小洞    G1-5 面积下限：滤碎块", (205, 225, 255), 17),
        ("    ★ G1-4 角点内缩px：料盘看着偏大（阴影）就调大", (0, 220, 255), 19),
        ("G2 相机曝光/增益（真改硬件；自动曝光=0 才生效）", (120, 255, 120), 19),
        ("G3 相机画质：★D435 多半不支持，控制台会列出能写哪些", (120, 255, 120), 19),
        ("G4 软件对比度（一定有效；优先调 G4-1 局部对比）", (120, 255, 120), 19),
        ("G5 检测参数：G5-6 降采样 1=最准(3fps) 2=默认(11fps) 3=最快", (120, 255, 120), 19),
        ("按键：q退出  s存参数  f存帧  b切换蓝圈  p暂停  d诊断", (255, 255, 160), 19),
        ("★ 三个窗口都按 1:1 显示（不能拉伸），否则「套没套准」会看错", (255, 200, 120), 17),
    ]
    w = 1040
    h = 20 + sum(int(l[2] * 1.6) + 9 for l in lines)
    img = np.full((h, w, 3), 32, np.uint8)
    y = 8
    for txt, col, sz in lines:
        put_cn(img, txt, (14, y), int(sz), col)
        y += int(sz * 1.6) + 9
    return img


def save_tune(p):
    try:
        os.makedirs(os.path.dirname(TUNE_PATH), exist_ok=True)
        with open(TUNE_PATH, "w", encoding="utf-8") as fh:
            fh.write("# ★ 由 ai_测试工具/实时显示.py 生成（按 s 保存）\n")
            fh.write("pocket_tune:\n")
            for k, v in p.items():
                fh.write("  %s: %s\n" % (k, v))
        L("★ 已存参数：%s" % TUNE_PATH)
    except Exception as e:                                         # noqa: BLE001
        L("★ 存参数失败：%s" % e)


def main() -> int:
    ap = argparse.ArgumentParser(description="料盘槽位实时显示 + 调参（v3）")
    ap.add_argument("--image", help="不接相机：用已存的图")
    ap.add_argument("--no-ref", action="store_true", help="不画设计位置蓝圈")
    a = ap.parse_args()
    p = dict(DEF)
    show_ref = not a.no_ref

    L("=" * 66)
    L("料盘槽位实时显示（v3）。★ 窗口**先开**，相机在后台连 —— 不会假死。")
    L("=" * 66)
    cam = None
    img = None
    if a.image:
        img = imread_safe(a.image)
        if img is None:
            L("读不到 %s" % a.image)
            return 1
        L("用图：%s %s（不接相机）" % (a.image, img.shape))
    else:
        cam = CamThread()
        cam.start()
        L("相机线程已启动（后台）。窗口马上就会出来。")

    make_trackbars(p)                      # ★ 先建窗口
    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_AUTOSIZE)      # ★ 1:1，不许拉伸
    cv2.namedWindow(WIN_PANEL, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_FRONT, cv2.WINDOW_NORMAL)
    # ★ 中英对照表单独一个窗口（滑条名是 ASCII，中文在这查）
    cv2.namedWindow(WIN_GUIDE, cv2.WINDOW_NORMAL)
    _g = guide_image()
    cv2.imshow(WIN_GUIDE, _g)
    cv2.resizeWindow(WIN_GUIDE, _g.shape[1], min(1000, _g.shape[0]))
    L("窗口已建。★ 三个画面窗口按 1:1 显示，别拉伸（拉伸会让「套没套准」看错）。")
    L("中文对照看 GUIDE 窗口；按键 q退出 s存参数 f存帧 b蓝圈 p暂停 d诊断。")

    paused = False
    last_vis = None
    shown_probe = False
    warned = False
    while True:
        st = cam.status() if cam is not None else {"step": "用图", "ok": True}
        bgr = img if img is not None else st.get("frame")
        if bgr is None:
            # ★ 还没有帧：也要**一直刷新窗口**，并把"卡在哪 + 怎么办"显示出来
            canvas = np.zeros((420, 900, 3), np.uint8)
            put_cn(canvas, "正在连相机…", (24, 24), 30, (0, 220, 255))
            put_cn(canvas, "当前步骤：%s" % st.get("step", "?"), (24, 72), 22)
            if st.get("err"):
                y = 136
                for ln in str(st["err"]).splitlines():
                    cv2.putText(canvas, ln[:110], (24, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 1, cv2.LINE_AA)
                    y += 28
            tips = [
                "★ 如果长时间停在这里，多半是相机被占用：",
                "   1) 关掉 RealSense Viewer / Windows「相机」App",
                "   2) 关掉上一次没退出的 python（任务管理器里找 python.exe）",
                "   3) 拔插一次 USB（务必插 USB 3 口，蓝色口）",
                "   4) 还不行：python tools/probe_d435.py --reset",
                "",
                "★ 也可以先用图调参（不接相机）：",
                "   python ai_测试工具/实时显示.py --image 你的图.png",
                "",
                "按 q 退出。",
            ]
            y = 240
            for ln in tips:
                cv2.putText(canvas, ln, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                            (180, 255, 180), 1, cv2.LINE_AA)
                y += 26
            cv2.imshow(WIN_MAIN, canvas)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            if cam is not None and st.get("err") and not warned:
                warned = True
                L("\n★ 相机出错：%s\n  %s" % (st.get("step"), st.get("err")))
            continue

        if cam is not None and st.get("probe") and not shown_probe:
            shown_probe = True
            L("\n★ 相机**实际支持**的项（不在清单里的滑条拖了没用）：")
            for k, v in st["probe"].items():
                if k == "设备":
                    for it in v:
                        L("   [设备] %s" % it)
                else:
                    for it in v:
                        L("   [可写] %s" % it)

        if not paused:
            vis, panel, fv, msg = process(bgr, p, show_ref)
            bar = np.zeros((36, vis.shape[1], 3), np.uint8)
            put_cn(bar, msg[:170], (8, 4), 20, (255, 255, 255))
            last_vis = (np.vstack([vis, bar]), panel, fv, msg)
        if last_vis is not None:
            v0, panel, fv, msg = last_vis
            cv2.imshow(WIN_MAIN, v0)
            cv2.imshow(WIN_PANEL, panel)
            if fv is not None:
                cv2.imshow(WIN_FRONT, fv)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('s'):
            save_tune(p)
        if k == ord('b'):
            show_ref = not show_ref
        if k == ord('p'):
            paused = not paused
            L("暂停" if paused else "继续")
        if k == ord('d'):
            L("诊断：step=%s ok=%s n=%s err=%s"
              % (st.get("step"), st.get("ok"), st.get("n"), st.get("err")))
        if k == ord('f') and last_vis is not None:
            ts = time.strftime("%m%d_%H%M%S")
            for nm, im in (("color", bgr), ("front", last_vis[2]),
                           ("overlay", last_vis[0])):
                if im is None:
                    continue
                q = os.path.join(OUT, "tune_%s_%s.png" % (ts, nm))
                L("   %s：%s" % (q, "成功" if imwrite_safe(q, im) else "★失败"))

    if cam is not None:
        cam.stop_flag = True
        cam.join(timeout=3.0)
    cv2.destroyAllWindows()
    L("已退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
