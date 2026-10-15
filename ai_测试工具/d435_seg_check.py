# -*- coding: utf-8 -*-
"""D435 分割验证实时窗口 —— 「换相机后分割还对不对」的检查工具。

★ 核心认知（这个工具存在的前提）：
    cv2.VideoCapture(x)  只管**取图**；cv2.imshow 只管**显示**。两者毫无关系。
    pyrealsense2 取出来的是标准 numpy BGR 数组，和 cap.read() 给的完全一样，
    所以分割算法、ROI 叠加、cv2.imshow 全部照用，只有取图那 3 行要换。

★ 这个工具刻意**不改 assembly_vision/ 里任何文件**：
    它是一个「只读消费者」—— import 工程的 Segmenter 来验证分割，
    默认还把 work_roi 关掉，好让你先看清**全局分割**长什么样。
    等你确认阈值要改什么，再决定要不要动工程里的配置。

画面上能看到什么：
    黄色   被切成「黄铜」的像素（半透明填充）
    品红   被切成「金属掩膜」的像素（只描轮廓，填充会糊住画面）
    青色   被切成「蓝手套」的像素
    左上   分辨率 / 帧率 / 过曝% / V均值 / 画面里最大黄铜块的等效直径
    右上   窗口中心那个小方框区域的 HSV 实测值 ← 用来重新定阈值
    红边   过曝 > 0.5% 时整屏报警（黄铜会被判成绿垫，判据静默失效）

按键：
    q / ESC  退出
    s        存图（原图 + 叠加图各一张）
    h        鼠标指向哪、右上角就显示哪的 HSV（默认显示窗口中心）
    g        显示/隐藏 蓝手套 叠加
    c        把当前画面中心设为「工位裁剪框」的基准（先用 --crop 调好大小）
    d        调试：把 黄铜金属手套 三个掩膜并排显示出来

用法（必须用 yolov11 环境的 python）：
    PY="D:\\miniconda\\envs\\yolov11\\python.exe"
    & $PY "D:\\university resource\\大三上\\视觉\\ai_测试工具\\d435_seg_check.py"
    & $PY ... --width 640 --height 480 --fps 30          # 出图更稳（USB2 下 30fps 只有这个分辨率）
    & $PY ... --crop 40 60 560 420                       # 只判工位区域（见下）
    & $PY ... --roi                                      # 打开工程的 work_roi 闸门（会因旧标定而错位）
    & $PY ... --lock-exposure 120                        # 锁手动曝光（RealSense 量纲 0~10000 左右）

★ 为什么要 --crop：
    D435 彩色是 **86°×57°** 视场，比 HBV 宽得多，画面里会多出大片木桌/墙。
    而工程文档记着：木桌的 H/S 与黄铜**本就重叠**，靠调阈值挡不住
    （实测木桌整块被判黄铜 13269mm²，真零件最大才 3281mm²）——
    正规解法是 work_roi「位置即身份」。
    但 work_roi 要用 c3 料盘单应算，而 c3 在新相机下已过期。
    所以先用 --crop 手框一个工位范围当临时替身，**先把「零件 vs 绿幕」这条
    最关键的判据看清楚**，木桌误检等重新标定后自然解决。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

# ── 把 assembly_vision 挂到 sys.path（只读引用，不改它任何文件）────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # 视觉/
AV = os.path.join(ROOT, "assembly_vision")
if not os.path.isdir(AV):
    print("✗ 找不到 %s" % AV)
    raise SystemExit(2)
sys.path.insert(0, AV)

from assemvision.config import load_config                     # noqa: E402
from assemvision.core.segment import Segmenter                # noqa: E402
from assemvision.imgio import imread_safe, imwrite_safe         # noqa: E402

WIN = "D435 分割检查  [q=退出 s=存图 h=HSV取样 g=手套 c=设中心 d=掩膜]"
SAMPLE = 9          # 取样方框边长（像素）


def draw_text(vis, s, x, y, scale=0.5, color=(255, 255, 255)):
    cv2.putText(vis, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(vis, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def hsv_at(bgr, x, y, r=SAMPLE):
    """取 (x,y) 周围 r 像素方框的 HSV 中位数（中位数比均值抗噪）。"""
    h, w = bgr.shape[:2]
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    patch = bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    med = np.median(hsv, axis=0)
    bgr_med = np.median(patch.reshape(-1, 3), axis=0)
    return float(med[0]), float(med[1]), float(med[2]), bgr_med


def overlay(seg, bgr, res, crop=None, show_glove=True):
    """把分割结果画成叠加图（和实时窗口用的是同一套画法）。"""
    vis = bgr.copy()
    if res.brass_mask is not None:
        ov = bgr.copy()
        ov[res.brass_mask > 0] = (0, 255, 255)          # 黄 = 黄铜
        vis = cv2.addWeighted(vis, 0.6, ov, 0.4, 0)
    mm = getattr(res, "metal_mask", None)
    if mm is not None and int(np.count_nonzero(mm)) > 200:
        cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in cnts if cv2.contourArea(c) > 40]
        cv2.drawContours(vis, big, -1, (255, 0, 255), 2)   # 品红 = 金属
    if show_glove and res.glove_mask is not None and int(np.count_nonzero(res.glove_mask)) > 0:
        ov2 = bgr.copy()
        ov2[res.glove_mask > 0] = (255, 255, 0)          # 青 = 手套
        vis = cv2.addWeighted(vis, 0.75, ov2, 0.25, 0)
    if crop:
        x, y, w, h = crop
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return vis


def analyze_images(seg, wp, src_dir, out_dir, crop=None, show_glove=True,
                   max_frames=20) -> int:
    """离线批处理：对一批图跑分割、存叠加图、打印统计。

    ★ 为什么要有这个模式：实时窗口需要人盯着，而且**留不下证据**。
      离线模式产出「原图 + 叠加图 + 数字」三件套，可以事后复盘、可以对比改动前后。
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    if os.path.isfile(src_dir):
        files = [src_dir]
    else:
        files = sorted(os.path.join(src_dir, f) for f in os.listdir(src_dir)
                       if f.lower().endswith(exts))[:max_frames]
    if not files:
        print("✗ %s 里没有图片" % src_dir)
        return 1
    os.makedirs(out_dir, exist_ok=True)
    print("离线分析 %d 张 -> %s" % (len(files), out_dir))

    rows = []
    for i, p in enumerate(files):
        bgr_full = imread_safe(p)
        if bgr_full is None:
            print("  读不了: %s" % p)
            continue
        bgr = bgr_full.copy()
        if crop:
            x, y, w, h = crop
            m = np.zeros(bgr.shape[:2], np.uint8)
            m[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = 255
            bgr[m == 0] = (0, 0, 0)
        res = seg.segment(bgr, scale=1.0)
        vis = overlay(seg, bgr, res, crop=crop, show_glove=show_glove)
        base = os.path.splitext(os.path.basename(p))[0]
        imwrite_safe(os.path.join(out_dir, "%s_overlay.png" % base), vis)
        imwrite_safe(os.path.join(out_dir, "%s_raw.png" % base), bgr_full)

        n_brass = int(np.count_nonzero(res.brass_mask)) if res.brass_mask is not None else 0
        n_glove = int(np.count_nonzero(res.glove_mask)) if res.glove_mask is not None else 0
        n_metal = int(np.count_nonzero(res.metal_mask)) if res.metal_mask is not None else 0
        diam = 0.0
        nblk = 0
        if n_brass:
            cnts, _ = cv2.findContours(res.brass_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            big = [c for c in cnts if cv2.contourArea(c) > 40]
            nblk = len(big)
            if big:
                diam = 2.0 * (max(cv2.contourArea(c) for c in big) / np.pi) ** 0.5
        rows.append(dict(f=os.path.basename(p), h=bgr_full.shape[0], w=bgr_full.shape[1],
                         brass=n_brass, blk=nblk, diam=diam, glove=n_glove, metal=n_metal,
                         blown=float(res.detail.get("blown_ratio", 0.0)),
                         v=float(res.detail.get("v_mean", 0.0))))
        print("  %-14s %dx%d  黄铜 %7d (%4.1f%%, %d 块, Ø%.0fpx)  金属 %7d  手套 %6d  过曝 %.2f%%"
              % (os.path.basename(p), bgr_full.shape[1], bgr_full.shape[0],
                 n_brass, 100.0 * n_brass / bgr_full.shape[0] / bgr_full.shape[1],
                 nblk, diam, n_metal, n_glove, 100 * rows[-1]["blown"]))
    if rows:
        import statistics as st
        print("\n  中位：黄铜 %d px（占 %.1f%%） 最大块 Ø%.0fpx  金属 %d  手套 %d  过曝 %.3f%%"
              % (st.median(r["brass"] for r in rows),
                 100.0 * st.median(r["brass"] for r in rows) / rows[0]["h"] / rows[0]["w"],
                 st.median(r["diam"] for r in rows), st.median(r["metal"] for r in rows),
                 st.median(r["glove"] for r in rows),
                 100 * st.median(r["blown"] for r in rows)))
        print("\n  ★ 判读：真零件最大 3300px / Ø64.5mm（HBV 实测），木桌 31795px（Ø201）。")
        print("     黄铜占比远大于零件量级 -> 背景被误判，必须靠 work_roi / 尺寸闸门，调阈值挡不住。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser("d435_seg_check")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--serial", default="")
    ap.add_argument("--workpiece", default=os.path.join(AV, "config", "workpiece.yaml"))
    ap.add_argument("--crop", type=float, nargs=4, metavar=("X", "Y", "W", "H"),
                    default=None, help="工位裁剪框（像素，左上角+宽高）")
    ap.add_argument("--roi", action="store_true",
                    help="打开工程的 work_roi 全局闸门（★ 旧标定，大概率错位，仅作对照）")
    ap.add_argument("--lock-exposure", type=int, default=None,
                    help="锁手动曝光（RealSense 量程约 1~10000；不给则保持自动曝光）")
    ap.add_argument("--gain", type=int, default=None, help="手动增益（配合 --lock-exposure）")
    ap.add_argument("--no-glove", action="store_true", help="不显示手套叠加")
    ap.add_argument("--dry-run", action="store_true",
                    help="只验证配置/分割器能否加载并打印阈值，不碰相机（自检用）")
    ap.add_argument("--analyze", default="",
                    help="离线批处理：对一个图片目录（或单张图）跑分割并存叠加图，不开窗口")
    ap.add_argument("--out", default=os.path.join(HERE, "d435_seg_out"),
                    help="--analyze 的输出目录")
    ap.add_argument("--save", default="", help="按 s 存图的前缀，默认存到本目录")
    args = ap.parse_args()

    # ── 0. 自检：先确认配置与分割器这条链是通的，再谈相机 ──────────────
    wp = load_config(args.workpiece)
    print("工件配置: %s" % wp.path)
    print("  黄铜阈值: h∈[%s,%s] s∈[%s,%s] v∈[%s,%s]"
          % tuple(wp.get("segment.brass.%s" % k, "?") for k in
                  ("h_min", "h_max", "s_min", "s_max", "v_min", "v_max")))
    print("  手套阈值: h∈[%s,%s] s≥%s v≥%s"
          % tuple(wp.get("segment.glove.%s" % k, "?") for k in
                  ("h_min", "h_max", "s_min", "v_min")))
    print("  金属掩膜: v≥%s s≤%s" % (wp.get("segment.metal.v_min", "?"),
                                     wp.get("segment.metal.s_max", "?")))
    seg = Segmenter(wp)
    print("  Segmenter 构造成功（work_roi 由 --roi 控制，默认不用）")
    if args.dry_run:
        print("\n✓ dry-run 结束：配置链、分割器、import 全部正常。")
        print("  插上相机后去掉 --dry-run 即可看实时分割。")
        return 0

    if args.analyze:
        crop0 = tuple(int(v) for v in args.crop) if args.crop else None
        if crop0:
            print("工位裁剪框 = x%d y%d w%d h%d" % crop0)
        return analyze_images(seg, wp, args.analyze, args.out, crop=crop0,
                              show_glove=not args.no_glove)

    # ── 1. 打开 D435（这就是「换掉 VideoCapture」的那部分）────────────────
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("✗ 没装 pyrealsense2：")
        print('   D:\\miniconda\\envs\\yolov11\\python.exe -m pip install pyrealsense2')
        return 2

    devs = []
    last_err = None
    for attempt in range(6):
        try:
            devs = list(rs.context().query_devices())
            break
        except RuntimeError as e:                                # noqa: BLE001
            last_err = e
            # ★ 实测：相机刚插回来时 query_devices 会抛 HResult 0x800703e3
            #   「已中止 I/O 操作」或 CreateNamedMutex Mutex_TotalFailure，重试即可。
            print("设备枚举第 %d 次失败：%s（1.5 秒后重试）" % (attempt + 1, str(e).strip()))
            time.sleep(1.5)
    if not devs:
        print("✗ RealSense SDK 看不到相机（重试 6 次都失败）。最后错误：%s" % last_err)
        print("  检查：插好没 / 关掉 RealSense Viewer 和 Windows 相机 App / 换 USB3 口")
        return 1
    serial = args.serial or devs[0].get_info(rs.camera_info.serial_number)
    try:
        usb = devs[0].get_info(rs.camera_info.usb_type_descriptor)
    except Exception:                                          # noqa: BLE001
        usb = "?"
    print("相机: %s  SN=%s  USB=%s" % (devs[0].get_info(rs.camera_info.name), serial, usb))
    if not str(usb).startswith("3"):
        print("★ USB %s：带宽受限，640x480 才有 30fps；1280x720 最高 15fps、1080p 只有 8fps。" % usb)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    try:
        profile = pipe.start(cfg)
    except Exception as e:                                     # noqa: BLE001
        print("✗ 开流失败: %s" % e)
        print("   试试 --width 640 --height 480 --fps 30")
        return 1

    # 曝光：D435 必须先关自动再给值，和 HBV 同一个道理
    if args.lock_exposure is not None:
        try:
            for s in profile.get_device().sensors:
                if s.supports(rs.option.enable_auto_exposure):
                    s.set_option(rs.option.enable_auto_exposure, 0)
                if args.gain is not None and s.supports(rs.option.gain):
                    s.set_option(rs.option.gain, float(args.gain))
                if s.supports(rs.option.exposure):
                    s.set_option(rs.option.exposure, float(args.lock_exposure))
            print("已锁手动曝光 = %s" % args.lock_exposure)
        except Exception as e:                                 # noqa: BLE001
            print("曝光设置失败:", e)

    # ── 2. 分割器已在上面的自检里构造好 ──────────────────────────────
    if args.roi:
        print("★ --roi 已开：work_roi 用的是**旧相机的标定**，在 D435 上一定错位，只当对照看。")
        print("  （真正要用它得先重跑 c3 料盘单应）")

    crop = tuple(int(v) for v in args.crop) if args.crop else None
    if crop:
        print("工位裁剪框（像素）= x%d y%d w%d h%d" % crop)

    # ── 3. 预热，丢掉前几帧（改完曝光要若干帧才生效，工程里踩过这个坑）──
    print("\n预热中…（按 q 退出）")
    for _ in range(15):
        pipe.wait_for_frames(5000)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    mouse = {"x": -1, "y": -1}

    def on_mouse(event, x, y, flags, param):
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            mouse["x"], mouse["y"] = x, y

    cv2.setMouseCallback(WIN, on_mouse)

    show_glove = not args.no_glove
    dbg = False
    t_last = time.time()
    fps = 0.0
    shots = 0

    try:
        while True:
            fs = pipe.wait_for_frames(5000)
            cf = fs.get_color_frame()
            if not cf:
                continue
            bgr_full = np.asanyarray(cf.get_data())
            if bgr_full.ndim != 3 or bgr_full.shape[2] != 3:
                print("✗ 拿到的不是 3 通道 BGR，shape=%s" % (bgr_full.shape,))
                break

            # 工位裁剪：范围外涂黑，模拟「位置即身份」的临时替身
            bgr = bgr_full.copy()
            if crop:
                x, y, w, h = crop
                m = np.zeros(bgr.shape[:2], np.uint8)
                m[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = 255
                bgr[m == 0] = (0, 0, 0)

            now = time.time()
            if now - t_last > 1e-6:
                fps = 0.9 * fps + 0.1 * (1.0 / (now - t_last))
            t_last = now

            # ── 分割：注意这里传的就是工程原样的 Segmenter ─────────────
            roi_mask = None
            if args.roi:
                try:
                    from assemvision.core.checks import Qualifier
                    from assemvision.core.roi import TrayMap
                    cal = load_config(os.path.join(AV, "config", "calibration.yaml"),
                                      required=False)
                    if cal.data and cal.get("tray_map.homography", None) is not None:
                        tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"),
                                                                 np.float64),
                                       ref_scale=float(cal.get("tray_map.scale", 1.0)))
                        qq = Qualifier(wp, tray, cal)
                        roi_mask = qq.work_roi_mask(1.0)
                except Exception as e:                          # noqa: BLE001
                    print("work_roi 算不出来（意料之中）:", e)

            res = seg.segment(bgr, scale=1.0, work_roi=roi_mask)

            # 叠加：黄铜填充 / 金属描边 / 手套填充
            vis = bgr.copy()
            ov = bgr.copy()
            if res.brass_mask is not None:
                ov[res.brass_mask > 0] = (0, 255, 255)
                vis = cv2.addWeighted(vis, 0.6, ov, 0.4, 0)
            mm = getattr(res, "metal_mask", None)
            if mm is not None and int(np.count_nonzero(mm)) > 200:
                cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                big = [c for c in cnts if cv2.contourArea(c) > 40]
                cv2.drawContours(vis, big, -1, (255, 0, 255), 2)
            if show_glove and res.glove_mask is not None:
                gm = res.glove_mask
                if int(np.count_nonzero(gm)) > 0:
                    ov2 = bgr.copy()
                    ov2[gm > 0] = (255, 255, 0)
                    vis = cv2.addWeighted(vis, 0.75, ov2, 0.25, 0)

            # ── 读数：过曝%、V 均值、黄铜最大块等效直径（换相机后最该看的三个数）──
            blown = float(res.detail.get("blown_ratio", 0.0))
            v_mean = float(res.detail.get("v_mean", 0.0))
            n_brass = int(np.count_nonzero(res.brass_mask)) if res.brass_mask is not None else 0
            max_d_mm = 0.0
            diam_px = 0.0
            if res.brass_mask is not None and n_brass > 0:
                cnts, _ = cv2.findContours(res.brass_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    a = max(cv2.contourArea(c) for c in cnts)
                    diam_px = 2.0 * (a / np.pi) ** 0.5
                    gsd = None
                    try:
                        gsd = float(wp.get("segment.gsd_mm_per_px", 0.0)) or None
                    except Exception:                            # noqa: BLE001
                        gsd = None
                    if gsd:
                        max_d_mm = diam_px * gsd

            h, w = vis.shape[:2]
            mx, my = mouse["x"], mouse["y"]
            if mx < 0 or mx >= w or my < 0 or my >= h:
                mx, my = w // 2, h // 2
            draw_text(vis, "%dx%d  %.1f fps  USB %s" % (w, h, fps, usb), 8, 22, 0.55)
            col = (0, 0, 255) if blown > 0.005 else (255, 255, 255)
            draw_text(vis, "过曝 %.2f%%   V均值 %.0f   %s"
                      % (100 * blown, v_mean,
                         "★ 过曝！判据会静默失效" if blown > 0.005 else "曝光正常"),
                      8, 44, 0.55, col)
            draw_text(vis, "黄铜像素 %d   最大块 Ø%.0fpx%s"
                      % (n_brass, diam_px,
                         (" = %.1fmm" % max_d_mm) if max_d_mm else ""), 8, 66, 0.55)
            if crop:
                draw_text(vis, "工位裁剪 x%d y%d w%d h%d（范围外已涂黑）" % crop,
                          8, 88, 0.5, (200, 200, 200))

            # 右上角 HSV 取样 —— 换相机后重新定阈值就靠它
            smp = hsv_at(bgr_full, mx, my)
            if smp is not None:
                hh, ss, vv, bgr_med = smp
                draw_text(vis, "取样 (%d,%d)  H %.0f  S %.0f  V %.0f"
                          % (mx, my, hh, ss, vv), w - 285, 22, 0.5, (0, 255, 255))
                draw_text(vis, "BGR %.0f/%.0f/%.0f   黄铜阈值 H<=%s S<=%s"
                          % (bgr_med[0], bgr_med[1], bgr_med[2],
                             wp.get("segment.brass.h_max", "?"),
                             wp.get("segment.brass.s_max", "?")),
                          w - 285, 44, 0.5, (0, 255, 255))
                cv2.circle(vis, (mx, my), SAMPLE, (0, 255, 255), 1)
            draw_text(vis, "q退出  s存图  h取样  g手套  c设中心  d掩膜",
                      w - 340, h - 12, 0.45, (200, 200, 200))

            if blown > 0.005:
                cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

            if dbg:
                def m3(m):
                    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) if m is not None \
                        else np.zeros_like(bgr)
                row = np.hstack([m3(res.brass_mask), m3(mm), m3(res.glove_mask)])
                cv2.imshow("掩膜: 黄铜 | 金属 | 手套", cv2.resize(row, (w, int(row.shape[0] * w / row.shape[1]))))
            else:
                try:
                    cv2.destroyWindow("掩膜: 黄铜 | 金属 | 手套")
                except Exception:                                # noqa: BLE001
                    pass

            cv2.imshow(WIN, vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                shots += 1
                base = args.save or os.path.join(HERE, "d435_seg")
                p1 = "%s_%02d_raw.png" % (base, shots)
                p2 = "%s_%02d_overlay.png" % (base, shots)
                imwrite_safe(p1, bgr_full)
                imwrite_safe(p2, vis)
                print("已存: %s\n      %s" % (p1, p2))
            if k == ord("g"):
                show_glove = not show_glove
                print("手套叠加:", "开" if show_glove else "关")
            if k == ord("d"):
                dbg = not dbg
            if k == ord("c"):
                # 以当前鼠标位置为中心，保持当前裁剪框大小
                if crop:
                    cw, ch = crop[2], crop[3]
                else:
                    cw, ch = w // 2, h // 2
                mouse["cx"], mouse["cy"] = mx, my
                crop = (max(0, mx - cw // 2), max(0, my - ch // 2), cw, ch)
                print("裁剪框中心 -> (%d,%d)，现在 = x%d y%d w%d h%d"
                      % (mx, my, crop[0], crop[1], crop[2], crop[3]))
            try:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:                                    # noqa: BLE001
                pass
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
