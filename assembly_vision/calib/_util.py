# -*- coding: utf-8 -*-
"""标定脚本公用工具。"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from assemvision.config import Config, load_config, merge   # noqa: E402
from assemvision.imgio import imread_safe, imwrite_safe      # noqa: E402


def paths():
    return dict(
        root=ROOT,
        camera=os.path.join(ROOT, "config", "camera.yaml"),
        workpiece=os.path.join(ROOT, "config", "workpiece.yaml"),
        calib=os.path.join(ROOT, "config", "calibration.yaml"),
        refs=os.path.join(ROOT, "data", "refs"),
    )


def load_all(args_camera: str = "", args_workpiece: str = "") -> tuple:
    p = paths()
    cam = load_config(args_camera or p["camera"])
    wp = load_config(args_workpiece or p["workpiece"])
    calib = load_config(p["calib"], required=False)
    if not calib.data:
        calib = Config({"meta": {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}}, p["calib"])
    return cam, wp, calib


def save_calib(calib: Config, out: str = "") -> str:
    p = paths()
    return calib.save(out or p["calib"])


# ── 分段预览：让标定脚本"看得见" ─────────────────────────────
# ★★ 为什么必须有（实测踩过，代价很大）：
#   c5 之前是"盲跑"—— 用户在终端按回车，脚本自己去抓帧判断，
#   中途**完全看不到画面**。结果一次严重过曝（曝光 -5.0 在当前光照下
#   近全白像素 8%、轮盘的黄铜面积从 3200px 塌到 626px）把整条判据链弄废，
#   而界面上只是"看着亮一点"，终端只报一句"扫优失败"，无从下手。
#   所以标定的每一步都必须能**看到画面 + 看到过曝量 + 看到分割结果**。
_PREVIEW_WIN = "标定预览  [空格/回车=继续  q/ESC=中止]"
# 供标定脚本在预览结束后关掉窗口用
PREVIEW_WIN_NAME = _PREVIEW_WIN


def preview_until_key(src, q=None, seg=None, scale: float = 1.0,
                      title_extra: str = "", fps_cap: float = 20.0,
                      guide: str = "") -> str:
    """实时预览窗口，直到用户按键。返回 "ok" | "quit"。

    画面上会叠加：
      - 红色多边形        各槽位 ROI + 装配位 ROI（若传了 q）
      - 绿色粗圈          `guide` 指定的那个槽/装配位（"东西该放这里"）
      - 黄色              被切成"黄铜"的像素
      - 品红              被切成"金属"的像素（不靠色相那条通路）
      - 左上角文字        过曝量 / V 均值 / 分辨率 / 帧率
      - 左下角文字        各槽覆盖率 + 识别结果 + 判定（让判据可见，不是黑盒）
      - ★ 过曝量 > 0.5% 时**整屏描红边 + 大字报警**（这正是踩过的坑）

    参数：
      src    图像来源（要有 .read(want_full=...)）
      q      Qualifier，用来画 ROI、覆盖率与识别结果（可选）
      seg    Segmenter，用来做分割叠加（可选，不传就只显示原图）
      scale  传给 segment 的缩放
      guide  "slot_wheel" / "assembly" 等，指定要**高亮提示**的目标
    """
    win = _PREVIEW_WIN
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except Exception:                                            # noqa: BLE001
        pass
    t_last = 0.0
    fps = 0.0
    while True:
        f = src.read(want_full=False)
        if f is None:
            time.sleep(0.02)
            continue
        bgr = f.left if getattr(f, "left", None) is not None else f.left_full
        if bgr is None:
            continue
        vis = bgr.copy()
        info: list = []
        tail: list = []
        blown = 0.0
        res = None
        if seg is not None:
            try:
                res = seg.segment(bgr, scale=scale)
                ov = bgr.copy()
                # ★ 黄铜用**半透明填充**（它就是要看的主体），
                #   金属掩膜只画**轮廓** —— 实测它在真机帧上占 28%，
                #   填充后整个画面糊成品红，反而看不见零件长什么样。
                if res.brass_mask is not None:
                    ov[res.brass_mask > 0] = (0, 255, 255)      # 黄 = 黄铜
                vis = cv2.addWeighted(vis, 0.6, ov, 0.4, 0)
                mm_ = getattr(res, "metal_mask", None)
                if mm_ is not None and int(np.count_nonzero(mm_)) > 200:
                    cnts, _ = cv2.findContours(mm_, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    big = [c for c in cnts if cv2.contourArea(c) > 40]
                    cv2.drawContours(vis, big, -1, (255, 0, 255), 2)   # 品红轮廓
                blown = float(res.detail.get("blown_ratio", 0.0))
                info.append("blown %.2f%%" % (100 * blown))
                info.append("V %.0f" % res.detail.get("v_mean", 0.0))
            except Exception as e:                               # noqa: BLE001
                info.append("分段失败: %s" % e)
        if q is not None:
            try:
                # 画所有槽位 ROI + 装配位 ROI（TrayMap 没有整体的"料盘多边形"）
                targets = dict(q.tray.slot_polygons_px(scale))
                ap = q.tray.assembly_polygon_px(scale)
                if ap is not None:
                    targets["assembly"] = ap
                for key, poly in targets.items():
                    if poly is None or len(poly) < 3:
                        continue
                    pts = np.round(np.asarray(poly)).astype(np.int32).reshape(-1, 1, 2)
                    if guide and key == guide:
                        cv2.polylines(vis, [pts], True, (0, 255, 0), 5)   # 绿圈=该放这里
                    else:
                        cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
            except Exception:                                    # noqa: BLE001
                pass

        # 左下角：覆盖率 + 识别结果 + 判定（让判据可见，不是黑盒）
        if q is not None and res is not None and res.brass_mask is not None:
            try:
                for s in q.check_slots(res, scale, bgr=bgr):
                    idp = "?"
                    try:
                        idp = s.features.get("id_part") or "?"
                    except Exception:                            # noqa: BLE001
                        pass
                    tail.append("%-7s cov %.2f  %-10s %s"
                                % (s.slot_id.replace("slot_", ""), s.coverage,
                                   idp, s.verdict))
            except Exception:                                    # noqa: BLE001
                pass

        now = time.time()
        if t_last > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, now - t_last))
        t_last = now
        lines = ["%dx%d  %.1f fps" % (bgr.shape[1], bgr.shape[0], fps)] + info
        if title_extra:
            lines.append(title_extra)
        for i, s in enumerate(lines):
            y = 22 + 20 * i
            cv2.putText(vis, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3)
            cv2.putText(vis, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)
        for i, s in enumerate(reversed(tail)):
            y = vis.shape[0] - 16 - 18 * i
            cv2.putText(vis, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 3)
            cv2.putText(vis, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
        if blown > 0.005:
            # ★ 过曝报警：这是把 c5 弄废的那个原因，必须刺眼
            cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1),
                          (0, 0, 255), 8)
            msg = ("!! 过曝 %.1f%% -> 黄铜会被判成绿垫，判据会失效 !! 调低曝光"
                   % (100 * blown))
            yy = 34 + 20 * len(lines)
            cv2.putText(vis, msg, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 4)
            cv2.putText(vis, msg, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

        cv2.imshow(win, vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            return "quit"
        if k in (13, 32, 10):        # 回车 / 空格
            return "ok"
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return "quit"
        except Exception:                                        # noqa: BLE001
            pass


def ask_or_preview(src, prompt: str, q=None, seg=None, scale: float = 1.0,
                   title_extra: str = "", preview: bool = True,
                   guide: str = "") -> bool:
    """把"按回车继续"和"实时预览"合成一步。返回 True=继续 / False=中止。

    preview=False 时退化成原来的 `ask()`（无窗口环境仍可用）。
    guide 指定"这一步该放东西的位置"（如 "slot_wheel" / "assembly"），
    预览里会用**绿色粗圈**高亮它。
    """
    print(prompt)
    if not preview:
        ask("准备好按回车…")
        return True
    print("  ★ 预览窗口已打开（绿圈 = 这一步该放东西的位置）：")
    print("      看着画面调好摆放/曝光，按【空格或回车】继续，按 q/ESC 中止。")
    try:
        return preview_until_key(src, q=q, seg=seg, scale=scale,
                                 title_extra=title_extra, guide=guide) == "ok"
    except Exception as e:                                       # noqa: BLE001
        print("  （预览不可用，退回按回车：%s）" % e)
        ask("准备好按回车…")
        return True


def warmup(src, seconds: float = 1.5, max_frames: int = 40) -> None:
    """预热：改完曝光/白平衡后连续读帧丢弃，等画面稳定。

    ★ 为什么必须做：相机的曝光值改变后需要若干帧才生效。
      实测踩过这个坑 —— c1 在 apply_locked_controls() 之后**立刻抓第一帧**，
      拿到的还是改曝光之前的黑帧，于是窗口一片黑，看起来像"相机坏了"。
      （tools/live_preview.py 是连续读帧显示的，所以它不受影响。）
    """
    t0 = time.time()
    n = 0
    while time.time() - t0 < seconds and n < max_frames:
        f = src.read(want_full=False)
        if f is not None:
            n += 1
        time.sleep(0.02)
    if n:
        time.sleep(0.15)          # 让最后一帧真正落地


def open_source(cam: Config, args):
    """按命令行参数打开图像来源（相机或离线文件）。

    ★ 顺序很重要：必须先 open() 再 apply_locked_controls()，最后 warmup()。
      反过来会得到「相机未打开，跳过控件锁定」—— 曝光与白平衡根本没写进去，
      于是标定脚本看到的画面和正式运行( pipeline )看到的画面不一致，
      标出来的阈值到了正式跑就不成立。曾经踩过这个坑。
    """
    from assemvision.sources.factory import make_source
    src = make_source(cam, getattr(args, "source", None), getattr(args, "path", None))
    src.open()
    if hasattr(src, "apply_locked_controls"):
        src.apply_locked_controls()      # type: ignore[attr-defined]
        warmup(src)                      # ★ 别省：否则抓到的是改曝光前的黑帧
    return src


def grab_full(src, timeout_s: float = 5.0):
    """抓一帧（带全分辨率左眼）。返回 (left, right)。"""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        f = src.read(want_full=True)
        if f is not None and f.left_full is not None:
            return f.left_full, (f.right_full if f.right_full is not None else f.right)
        time.sleep(0.02)
    return None, None


def grab_small(src, timeout_s: float = 5.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        f = src.read(want_full=False)
        if f is not None and f.left is not None:
            return f.left, f
        time.sleep(0.02)
    return None, None


def ask(prompt: str, default: str = "") -> str:
    try:
        s = input(prompt).strip()
    except EOFError:
        s = ""
    return s or default


def show_and_pick(image, n: int = 2, window: str = "pick",
                  src=None, interval: float = 0.02, want_full: bool = True,
                  show_both: bool = True) -> list:
    """实时预览 + 在画面上点 n 个点。返回像素坐标列表。

    参数
      image      起始画面（连不上相机时退回用它）
      src        图像来源；给了就连续取流
      interval   取流间隔（秒）
      want_full  取全分辨率帧（点像素坐标必须用全分辨率）
      show_both  ★ True = 显示**左右眼合并**的完整拼接图（默认）
                    False = 只显示左眼
                 为什么要合并显示：标定时你需要看到相机的完整视野来判断摆放，
                 只给一半画面会让人以为"视野被裁了"。而**点的坐标仍然只取左半**，
                 所以返回的像素坐标与"只显示左眼"时完全一致。

    ★ 算法本身全程只用左眼（分割/覆盖率/辐条都在左眼上做），
      所以左眼必须能看全工件 —— 这点不会因为显示方式而改变。

    按键：鼠标左键=取点（只能点在左半），r=重来，回车/空格=确认，ESC=取消
    """
    pts = []
    disp = image.copy()
    latest = {"img": None}
    # 点击边界：合并显示时只允许点左半，避免误点到右眼造成坐标错乱
    bound = {"x_max": None}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(pts) >= n:
            return
        xm = bound["x_max"]
        if xm is not None and x > xm:
            print("  ★ 请点在【左眼】那一半（黄线左边）—— 算法用的是左眼")
            return
        pts.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    # ★ 窗口尺寸必须把"标题栏 + 边框"算进去，否则底部会被屏幕切掉
    try:
        h0, w0 = disp.shape[:2]
        scale = max(0.5, min(1100.0 / max(1, w0), 780.0 / max(1, h0)))
        cv2.resizeWindow(window, int(w0 * scale), int(h0 * scale))
    except Exception:                                          # noqa: BLE001
        cv2.resizeWindow(window, 800, 700)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        if src is not None:
            f = src.read(want_full=want_full) if hasattr(src, "read") else None
            img = None
            eye_w = None
            if f is not None:
                left = getattr(f, "left", None)
                right = getattr(f, "right", None)
                if want_full and getattr(f, "left_full", None) is not None:
                    left = f.left_full
                    right = getattr(f, "right_full", None)
                if left is not None:
                    if show_both and right is not None and right.shape[:2] == left.shape[:2]:
                        img = np.hstack([left, right])
                        eye_w = left.shape[1]
                    else:
                        img = left
                        eye_w = None
            if img is not None:
                latest["img"] = img
                bound["x_max"] = eye_w
            if latest["img"] is not None:
                disp = latest["img"].copy()
                # 合并显示时：画中缝 + 标注哪一半是算法用的
                if bound["x_max"] is not None:
                    xm = bound["x_max"]
                    cv2.line(disp, (xm, 0), (xm, disp.shape[0]), (0, 255, 255), 2)
                    cv2.putText(disp, "LEFT (used by algorithm) - click here",
                                (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(disp, "LEFT (used by algorithm) - click here",
                                (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2, cv2.LINE_AA)
                for i, (px, py) in enumerate(pts):
                    cv2.circle(disp, (px, py), 5, (0, 0, 255), -1)
                    cv2.circle(disp, (px, py), 9, (255, 255, 255), 1)
                    cv2.putText(disp, "P%d" % (i + 1), (px + 10, py - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(disp, "P%d" % (i + 1), (px + 10, py - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        # ★ 提示必须用 ASCII：OpenCV 自带字体画不了中文，会显示成一串 ????
        tip = "picked %d/%d  |  L-click=pick  r=reset  ENTER=ok  ESC=cancel" % (len(pts), n)
        h, w = disp.shape[:2]
        cv2.rectangle(disp, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(disp, tip, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)
        if bound["x_max"] is not None:
            side = "LEFT half = the eye the algorithm uses"
        else:
            side = "single eye view"
        cv2.putText(disp, side, (w - 420, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)
        # 底部提示条（不贴最底边，窗口被裁也不会只剩一半）
        if len(pts) >= n:
            bot = ">> press ENTER to confirm, r to redo"
            col = (0, 255, 255)
        else:
            bot = ">> click point %d of %d  (left half only)" % (len(pts) + 1, n)
            col = (200, 200, 200)
        cv2.rectangle(disp, (0, h - 46), (w, h), (0, 0, 0), -1)
        cv2.putText(disp, bot, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    col, 2, cv2.LINE_AA)
        cv2.imshow(window, disp)
        k = cv2.waitKey(1) & 0xFF
        if len(pts) >= n and k in (13, 32):     # 回车 / 空格
            break
        if k == 27:                             # ESC
            pts = []
            break
        if k == ord("r"):
            pts = []
        if src is not None and interval > 0:
            time.sleep(interval)
    cv2.destroyWindow(window)
    return pts
