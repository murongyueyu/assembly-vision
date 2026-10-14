# -*- coding: utf-8 -*-
"""零件检测实时演示 —— 看得见画面、看得见**形状轮廓**、带追踪与去抖。

用户要求（逐条对应实现）：
  ① 「要能让我看到实时画面」          -> cv2 窗口，本工具
  ② 「要能看到标注轮廓，是形状轮廓而不是一个方框」
        -> `PartTracker.draw()` 用 `cv2.drawContours` 画**真实轮廓** + 顶点，
           方框只在按 `b` 时作为调试叠加
  ③ 「我手要是拿起来（戴蓝手套）就算被遮挡一部分最好也要能识别」
        -> 用**追踪框当 ROI 重新分割**（实测遮挡 70% 仍能取回轮廓）；
           遮挡超阈值时**冻结身份**不重判（残缺件拿去认必然认错）
  ④ 「帧率要比较高」                  -> **检测 + 追踪**：每 N 帧才全图分割
       （实测 640×480 全分割 9.9ms，而 MOSSE 0.25ms / KCF 2.0ms）
  ⑤ 「而且需要去抖」                  -> 卡尔曼平滑质心 + 状态机（detected/tracked/…）
  ⑥ 「最好能用上卡尔曼滤波追踪或者 KCF 追踪或者各种追踪算法」
        -> 卡尔曼恒开 + `--backend kcf|mosse|mil`。
           ★★ **实测只有 KCF 可靠**（别按"常见做法"选后端）：
             后端        单次update   结论
             kcf           5.5 ms    ★ 唯一可靠，默认。4 目标约 32ms/帧
             mosse         0.02 ms   ✗ `update` **恒返回 False**（OpenCV 已知问题），
                                        实测 40/40 全"丢"，**不可用**
             medianflow    1.8 ms    ✗ 会崩（lkpyramid 断言前后帧尺寸一致）
             csrt         22.7 ms    ✗ 太慢（Pi 估 136ms）
           ★ 而 `segment` 的耗时与 ROI 大小**几乎无关**（全图 10.5ms vs
             小框 9.2ms）—— 所以"给小 ROI 就快"是错的。
           ★ 试过"裁剪小图再跑 KCF"：**只快 14%，但静止漂移从 2.76px 崩到 131px**
             （裁剪窗滑动逼着反复 init），**已放弃**。
           -> 结论：**省算力只能靠少跟几帧**，而实测调大 `--detect-every`
             几乎不提帧率（26~29fps），因为瓶颈是 KCF 的每目标固定成本。
  ⑦ 「需要考虑后续延伸到边缘算力也就是树莓派上跑同样能够使用的代码」
        -> 追踪器**两套 API 都试**（OpenCV 4.x 只有 `*_create()`）；
           `--bench` 报逐段耗时与树莓派 6x 估算

用法：
    python tools/part_track_demo.py                     # 真机实时
    python tools/part_track_demo.py --backend mosse     # 树莓派推荐（2ms）
    python tools/part_track_demo.py --source images --path data/synthetic_seq
    python tools/part_track_demo.py --bench --max-frames 200   # 只测帧率不显示

按键：
    空格/回车 暂停/继续    b 方框开关    d 调试信息    q/ESC 退出
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cv2                                                        # noqa: E402
from assemvision.config import load_config                        # noqa: E402
from assemvision.core.checks import Qualifier                     # noqa: E402
from assemvision.core.part_tracker import PartTracker, make_tracker   # noqa: E402
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402

WIN = "零件检测 · 轮廓 + 追踪"


def _read_frame(src, images, idx):
    """统一取一帧：真机源或图片序列。

    ★★ 真机路径踩过的坑（用户实跑报「处理帧数 0」）：
      ① `UvcStereoSource.__init__` **只设 `self.cap = None`，不自动开相机** ——
         必须显式 `src.open()`（`pipeline.py` 就是这么做的，见其第 145 行）。
         漏了它 -> `read()` 立刻 `return None` -> 第一帧就退出，**一帧都处理不了**。
      ② 相机刚开时 `read()` **正常会返回 None**（DSHOW 要预热几帧），
         不能把"某一帧读到 None"当成"序列结束"。
      所以这里区分：真机源连续读到 None 由调用方**重试**，图片序列读到末尾才算结束。
    """
    if images is not None:
        if idx >= len(images):
            return None
        return imread_safe(images[idx])
    f = src.read(want_full=False)
    if f is None:
        return None
    # 帧对象可能只有 left（单目）—— 用 getattr 兜住，别让属性名变化直接崩
    return getattr(f, "left", None)


def main() -> int:
    ap = argparse.ArgumentParser("part_track_demo")
    ap.add_argument("--workpiece", default=os.path.join(ROOT, "config", "workpiece.yaml"))
    ap.add_argument("--calibration", default=os.path.join(ROOT, "config", "calibration.yaml"))
    ap.add_argument("--camera", default=os.path.join(ROOT, "config", "camera.yaml"))
    ap.add_argument("--source", default=None, help="uvc | video | images")
    ap.add_argument("--path", default=None, help="images 模式的目录")
    ap.add_argument("--backend", default="kcf", choices=["kcf", "mosse", "mil"],
                    help="只有 kcf 可靠（实测）；mosse 的 update 恒返回 False，"
                         "medianflow 会崩，csrt 太慢 —— 见文件头的实测表")
    ap.add_argument("--detect-every", type=int, default=8,
                    help="每 N 帧做一次全图检测；其余帧只追踪。"
                         "★ 实测调大它**几乎不提帧率**（26~29fps 之间），"
                         "因为瓶颈是 KCF 的每目标成本、不是分割（见文件头）")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--bench", action="store_true", help="只测帧率，不开窗口")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--save", default="", help="把带标注的画面录成视频")
    args = ap.parse_args()

    wp = load_config(args.workpiece)
    cal = load_config(args.calibration)
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    q = Qualifier(wp, tray, cal)
    seg = Segmenter(wp)
    clf = q._classifier()
    mmpp = tray.mm_per_px(1.0)

    tcfg = dict((wp.get("tracking", {}) or {}))
    tcfg["backend"] = args.backend
    if args.detect_every:
        tcfg["detect_every_n"] = args.detect_every
    tracker = PartTracker(tcfg)

    # 源
    images = None
    src = None
    if args.source == "images" or (args.source is None and args.path and not args.bench):
        d = args.path or os.path.join(ROOT, "data", "synthetic_seq")
        images = [os.path.join(d, f) for f in sorted(os.listdir(d))
                  if f.lower().endswith(".png")]
        source_desc = "images:%s（%d 帧）" % (d, len(images))
    else:
        from assemvision.sources.factory import make_source       # noqa: E402
        cam = load_config(args.camera)
        # ★ 用 make_source 的**真实签名**（cam_cfg, source, path）——
        #   我第一版往 cam.data 里塞 "kind" 是臆造的字段，不生效。
        src = make_source(cam, args.source, args.path)
        try:
            src.open()                       # ★★ 必须显式开！见 _read_frame 的说明
            if hasattr(src, "apply_locked_controls"):
                src.apply_locked_controls()  # 锁曝光等（与 pipeline.py 同做法）
        except Exception as e:                                     # noqa: BLE001
            print("  ★ 相机打开失败: %s: %s" % (type(e).__name__, e))
            print("     排查顺序：① 相机是否被别的程序占用（关掉 app.py / 预览窗口）")
            print("               ② 跑 python tools/probe_camera.py 确认能出图")
            print("               ③ 换 USB 口 / 确认 dshow 后端可用")
            return 2
        source_desc = "%s（已 open + 锁参数）" % src.name

    tr_probe, api = make_tracker(args.backend)
    print("═" * 78)
    print("  零件检测实时演示")
    print("═" * 78)
    print("  源            : %s" % source_desc)
    print("  追踪后端      : %s  ->  %s" % (args.backend, api if tr_probe else "★ 不可用"))
    print("  检测频率      : 每 %d 帧一次全图分割（其余帧只追踪）" % tracker.detect_every)
    print("  OpenCV        : %s" % cv2.__version__)
    print("  按键          : 空格=暂停  b=方框  d=调试  q/ESC=退出")
    print("  ★ 后端        : 只有 kcf 可靠 —— mosse 的 update 恒返回 False、")
    print("                  medianflow 会崩、csrt 太慢（见文件头实测表）")
    print()

    def classify(feats):
        f = dict(feats)
        f["area_mm2"] = float(f.get("area_px", 0.0)) * mmpp * mmpp
        rw, rh = float(f.get("rect_w", 0)), float(f.get("rect_h", 0))
        f["length_mm"] = max(rw, rh) * mmpp
        f["short_side_mm"] = min(rw, rh) * mmpp
        pid, sc, _ = clf.classify(f)
        return pid, sc

    paused = False
    show_bbox = False
    debug = False
    idx = 0
    frames = 0
    blank = 0
    last_detect_frame = -999
    t_start = time.time()
    fps_times = []
    writer = None
    seg_ms = trk_ms = 0.0

    try:
        while True:
            if not paused:
                frame = _read_frame(src, images, idx)
                if frame is None:
                    if images is not None:
                        break                       # 图片序列：读完就结束
                    # ★ 真机：read() 返回 None 多是**预热/偶发丢帧**，不能退出。
                    #   连续读不到才判失败（DSHOW 开流通常要几帧才出图）。
                    blank += 1
                    if blank == 1:
                        print("  ★ 相机前几帧没出图（DSHOW 预热中），继续等…")
                    if blank > 90:
                        print("  ★ 连续 %d 帧读不到图 -> 放弃" % blank)
                        print("     跑 python tools/probe_camera.py 看相机是否正常")
                        break
                    time.sleep(0.02)
                    continue
                blank = 0
                idx += 1
                frames += 1
                t0 = time.perf_counter()

                # ★ 检测节奏**本身要去抖**（否则帧率会周期性地掉）：
                #   `or not tracker.tracks` 在"零件全被拿走"时会变成**每帧都全图检测**，
                #   帧率立刻塌。所以：
                #     · 强制补检最多每 `recheck_every` 帧一次（默认 2×检测周期）
                #     · 传 detect 时把"有没有轨迹"交给 PartTracker 自己判断
                every = max(1, tracker.detect_every)
                stale = (frames - last_detect_frame) >= every * 2
                need_detect = (frames % every == 1) or (not tracker.tracks and stale)
                blobs = []
                if need_detect:
                    last_detect_frame = frames
                    ts = time.perf_counter()
                    # ★ 只在检测帧跑全图分割；用 work_roi 挡木桌
                    roi = q.work_roi_mask(frame.shape, 1.0,
                                          pose=getattr(seg, "_last_pose", None),
                                          margin_mm=float(wp.get("segment.work_roi_margin_mm",
                                                                 25.0)))
                    res = seg.segment(frame, scale=1.0, work_roi=roi)
                    blobs = list(res.brass_blobs)
                    seg_ms = (time.perf_counter() - ts) * 1000.0

                ts = time.perf_counter()
                tracks = tracker.update(frame, blobs, detect=need_detect,
                                        classify=classify)
                trk_ms = (time.perf_counter() - ts) * 1000.0

                # ── 遮挡处理：用追踪框当 ROI **重新分割**，取回轮廓 ──
                #    实测遮挡 70% 仍能取回（面积剩 20%、轮廓完整）
                if not need_detect:
                    for t in tracks:
                        if t.state in ("occluded", "tracked"):
                            box = tracker.roi_of(t)
                            if box is None:
                                continue
                            m = np.zeros(frame.shape[:2], np.uint8)
                            x0, y0, x1, y1 = box
                            m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 255
                            r2 = seg.segment(frame, scale=1.0, work_roi=m)
                            cand = [b for b in r2.brass_blobs
                                    if tracker.min_area_px <= b.area_px <= tracker.max_area_px]
                            if cand:
                                # ★ 按**离预测位置最近**选块，不是"取面积最大" ——
                                #   实测踩过：遮挡到只剩一点点时，目标块的面积会**小于**
                                #   追踪框里混进来的别的零件，于是"取最大"选错了对象
                                #   （85% 遮挡时选中了质心距 87px 的另一块）。
                                b = min(cand, key=lambda z: float(np.hypot(
                                    z.centroid[0] - t.centroid[0],
                                    z.centroid[1] - t.centroid[1])))
                                # 太远就别要（追踪框漂了）
                                if float(np.hypot(b.centroid[0] - t.centroid[0],
                                                  b.centroid[1] - t.centroid[1])) > 40:
                                    continue
                                t.contour = b.contour
                                t.area_px = float(b.area_px)
                                t.visible_ratio = b.area_px / max(1e-6, t.area_ref)
                                t.bbox = cv2.boundingRect(b.contour)
                                if t.visible_ratio >= tracker.full_ratio:
                                    t.state = "detected"
                                    pid, sc = classify(dict(b.features or {}))
                                    if pid:
                                        t.part_id, t.id_score = pid, float(sc)
                                elif t.visible_ratio <= tracker.occluded_ratio:
                                    t.state = "occluded"
                                else:
                                    t.state = "tracked"

                canvas = frame.copy()
                tracker.draw(canvas, show_bbox=show_bbox)

                # 面板
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 60), (28, 28, 28), -1)
                n_ok = sum(1 for t in tracks if t.visible)
                n_occ = sum(1 for t in tracks if t.state == "occluded")
                n_lost = sum(1 for t in tracks if t.state == "lost")
                cv2.putText(canvas,
                            "零件 %d（可用 %d / 遮挡 %d / 丢失 %d）"
                            % (len(tracks), n_ok, n_occ, n_lost),
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1,
                            cv2.LINE_AA)
                cv2.putText(canvas,
                            "FPS %.1f | %s | 检测每 %d 帧 (%.1fms) | 追踪 %.1fms"
                            % (fps_times[-1] if fps_times else 0.0, args.backend,
                               tracker.detect_every, seg_ms, trk_ms),
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 230, 170), 1,
                            cv2.LINE_AA)
                if debug:
                    lines = ["状态明细:",
                             "  tracker API = %s" % (api or "-"),
                             "  tracks=%d  检测帧=%d  纯追踪帧=%d"
                             % (len(tracks), tracker.stats["detects"],
                                tracker.stats["track_only"]),
                             "  检测耗时 %.1fms  追踪耗时 %.1fms"
                             % (tracker.last_detect_ms, tracker.last_track_ms)]
                    for i, s in enumerate(lines):
                        cv2.putText(canvas, s, (10, 84 + i * 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (200, 200, 200), 1, cv2.LINE_AA)

                if args.save and canvas is not None:
                    if writer is None:
                        from assemvision.imgio import videowriter_safe   # noqa: E402
                        h, w = canvas.shape[:2]
                        writer = videowriter_safe(args.save, "mp4v", 30.0, (w, h))
                    writer.write(canvas)

                if not args.bench and not args.no_preview:
                    cv2.imshow(WIN, canvas)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (27, ord("q")):
                        break
                    if k in (32, 13):
                        paused = not paused
                    elif k == ord("b"):
                        show_bbox = not show_bbox
                    elif k == ord("d"):
                        debug = not debug

                dt = time.perf_counter() - t0
                fps_times.append(1.0 / dt if dt > 0 else 0.0)
                if len(fps_times) > 60:
                    fps_times.pop(0)
                if args.max_frames and frames >= args.max_frames:
                    break
            else:
                if not args.bench:
                    cv2.waitKey(50)
    except KeyboardInterrupt:
        print("\n  用户中断")
    finally:
        if src is not None:
            try:
                src.close()
            except Exception:                                      # noqa: BLE001
                pass
        if writer is not None:
            writer.release()
            print("  已保存 -> %s" % args.save)
        if not args.bench:
            cv2.destroyAllWindows()

    el = time.time() - t_start
    print()
    print("═" * 78)
    print("  实测结果")
    print("═" * 78)
    print("  处理帧数       : %d（%.1f 秒）" % (frames, el))
    # ★★ 报**中位帧率**，不报平均 —— 实测踩过：
    #   相机 DSHOW 开流要预热，前几帧很慢（一次实测均值 6.8fps、中位 31.7fps）。
    #   均值被那几个慢帧拖垮，**看着像"只有 7fps"，其实已经贴着相机 31fps 上限**。
    #   同理 `--max-frames` 计时也含预热，所以把前 `WARMUP` 帧单独剔掉。
    WARMUP = 10
    steady = fps_times[WARMUP:] if len(fps_times) > WARMUP else fps_times
    if steady:
        med = float(np.median(steady))
        print("  ★ 帧率（中位）  : %.1f fps" % med)
        print("     稳态范围     : %.1f ~ %.1f fps（剔掉前 %d 帧预热）"
              % (min(steady), max(steady), WARMUP))
    if frames and el > 0:
        print("     全程均值     : %.1f fps（含预热，仅供参考）" % (frames / el))
    n_det = tracker.stats["detects"]
    n_trk = tracker.stats["track_only"]
    print("  检测帧 / 纯追踪: %d / %d" % (n_det, n_trk))
    print("  全图分割(检测帧): %.2f ms      追踪 update/帧: %.2f ms" % (seg_ms, trk_ms))
    # ★ 每帧均摊 = 分割/检测周期 + 追踪。这是判断"能不能跟上相机"的正确指标。
    per_frame = seg_ms / max(1, tracker.detect_every) + trk_ms
    print("  ★ 实测每帧均摊  : %.2f ms  -> 上限 %.1f fps（相机 31 fps）"
          % (per_frame, 1000.0 / max(1e-6, per_frame)))
    print("     其中分割被摊薄到 1/%d（每 %d 帧才跑一次全图分割）"
          % (tracker.detect_every, tracker.detect_every))
    if per_frame <= 32.0:
        print("     ✓ 够用：每帧均摊 <= 32ms，能贴住相机 31fps")
    else:
        print("     ★ 不够：每帧均摊 > 32ms，会掉帧 -> 把 --detect-every 调大")
    per_frame_pi = (seg_ms * 6.0) / max(1, tracker.detect_every) + trk_ms * 6.0
    print("  树莓派估（6x）  : 每帧约 %.0f ms -> 上限约 %.0f fps"
          % (per_frame_pi, 1000.0 / max(1e-6, per_frame_pi)))
    print()
    print("  ── 后端选择（实测，别按「常见做法」选）──")
    print("     kcf       5.5ms/目标  ★ 唯一可靠的，默认")
    print("     mosse     0.02ms      ✗ update 恒返回 False（OpenCV 已知问题），不可用")
    print("     medianflow 1.8ms      ✗ 会崩（lkpyramid 要求前后帧尺寸一致）")
    print("     csrt      22.7ms      ✗ 太慢（Pi 估 136ms）")
    print("     -> 想提帧率只能调大 --detect-every（少跟几帧），不是换后端")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
