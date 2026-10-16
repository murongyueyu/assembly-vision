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
from assemvision.core.part_id import classify_features               # noqa: E402
from assemvision.core.part_tracker import PartTracker, make_tracker   # noqa: E402
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter, glove_hits         # noqa: E402
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
    ap.add_argument("--no-hand", action="store_true",
                    help="不画手（蓝手套）的检测框；默认画")
    ap.add_argument("--window-scale", type=float, default=1.0,
                    help="显示放大倍数。★ 相机配置里 scale.realtime=0.5，"
                         "会在 640x480 窗口里塞一张 320x240 的图 -> 四周全是黑边（实测踩过）")
    ap.add_argument("--allow-new-inset-px", type=int, default=6,
                    help="新轨迹的质心必须在工位 ROI **内缩**这么多像素以内"
                         "（work_roi 本身是料盘外扩 25mm 的，不内缩挡不住料盘外的噪声）")
    ap.add_argument("--new-track-min-score", type=float, default=0.60,
                    help="新建轨迹时，那个块**必须被认成某种零件**且得分不低于此值"
                         "（噪声不会被认成零件，这是比位置更强的一道闸）")
    ap.add_argument("--allow-new-anywhere", action="store_true",
                    help="允许在画面任何地方建新轨迹"
                         "（默认只允许在工位 ROI 内，"
                         "防止料盘外的噪声变成幽灵）")
    ap.add_argument("--full-search-every", type=int, default=30,
                    help="每 N 帧做一次**全图**检测（供判丢用）。"
                         "★ 判丢要求全图搜索，不能只搜原位附近；"
                         "全图的代价是去掉了挡木桌的工位 ROI，"
                         "所以按这个周期跑（默认 30 帧 ≈ 1 秒）")
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

    def prepare(blob, img=None, mask=None, roi_area_px=None):
        """把**分类器要用的特征**算好挂到 `blob.features` 上。

        ★★ 为什么必须这样分工（2026-10 修，本模块一个真 bug）：
          `PartTracker` 的 `classify` 回调收到的是**一个特征字典**（不是 blob、不是图），
          所以"算特征"这件事只能在**调用方**做。原来这里做错了 —— 只补了
          `area_mm2` / `length_mm` / `short_side_mm` 三个尺度特征，
          **没有色域特征、也没有 `area_frac`**；而 c6 标定与 `checks.py` 判定走的是
          完整的 `features_from_blob(bgr=, mask=)`。
          ⇒ 同一个零件在「界面显示的身份」与「槽位判定的身份」上**得分不一样**，
            而且缺的正是**权重合计 3.2/18.1 的色域特征** ——
            **标定了一份特征、推理时扔掉了**。

        ★ 现在统一走 `part_id.classify_features()`（全项目唯一入口）：
            img / mask  给了才算色域特征；不给则退回"只有形状+尺度"的老行为（不抛异常）
            roi_area_px 给了才算 `area_frac`
          ⇒ 调用方**手里有什么就传什么**，不再各自补特征。
        """
        blob.features = classify_features(blob, mm_per_px=mmpp, bgr=img, mask=mask,
                                          roi_area_px=roi_area_px)
        return blob

    def classify(feats):
        """`PartTracker` 的回调：只吃特征字典（特征由 `prepare()` 预先挂好）。"""
        pid, sc, _ = clf.classify(dict(feats))
        return pid, sc

    # ★ 槽位 -> 该放的零件 id（"缺件"提示要用它，见下面的说明）
    slot_parts = {}
    for _s in (wp.get("slots") or []):
        if _s.get("part"):
            slot_parts[_s["id"]] = str(_s["part"])
    # ★★ 防幽灵的最后一道闸：**新轨迹只允许建在工位范围内**。
    #   全图检测会把画面任何地方的像零件的块都拿进来（这是**有意**的，
    #   判丢要全图搜），但「建新轨迹」应该只在工位内 —— 否则料盘外的噪声/手边缘
    #   会变成一个个「全图搜」空心圈幽灵（用户截图里就是这个）。
    #   实测：work_roi 只占画面 24.8%，而**四个真零件全在里面**。
    #   ★ 用 --allow-new-anywhere 可关掉这道闸（例如以后要支持「开放位置」）。
    _roi_cache = {}
    # ★ 最近一次算出的料盘位姿 + 它的来源（"outline" / "cached" / "nominal"）。
    #   检测帧上更新；`_ensure_roi` 用它把"新轨迹只准建在工位内"那道闸也对齐料盘。
    _roi_state = {"pose": None, "src": "nominal"}
    if args.allow_new_anywhere:
        def _allow_new(_b):
            return True
    else:
        def _allow_new(_b):
            # 两道闸：位置（内缩 ROI 内）+ 身份（能被认成某种零件）。
            if not _in_inset_roi(_b):
                return False
            # ★ 先把特征补齐（含色域）再判身份 —— 见 prepare() 的说明
            _pid, _sc = classify(prepare(_b, frame,
                                         res.brass_mask if res is not None else None).features)
            if not _pid:
                return False
            return float(_sc) >= float(args.new_track_min_score)

    def _ensure_roi(_h, _w):
        """确保内缩 ROI 已算好（两个谓词共用同一份缓存）。

        ★★ 这里也必须**跟着料盘位姿走**：它决定"新轨迹只准建在工位内"。
          用标称 ROI 的话，料盘一挪，零件就全在"工位外"⇒ 一个轨迹都建不起来
          （症状与识别失败长得一样，但根因在防幽灵这道闸上）。
        ★ 检测帧上会重算位姿，所以这里取 `_roi_state` 里最近一次的结果；
          没跑过检测帧（工具刚启动）时退回标称坐标。
        """
        key = (_h, _w)
        m = _roi_cache.get(key)
        if m is None:
            m = q.work_roi_mask(frame.shape, 1.0,
                                pose=_roi_state.get("pose"),
                                margin_mm=float(wp.get("segment.work_roi_margin_mm", 25.0)))
            ins = int(args.allow_new_inset_px)
            if ins > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ins * 2 + 1,) * 2)
                m = cv2.erode(m, k)
            _roi_cache[key] = m
        return m

    def _in_inset_roi(_b):
        """质心是否在**内缩后**的工位 ROI 内（用于筛掉料盘外的噪声块）。"""
        m = _ensure_roi(frame.shape[0], frame.shape[1])
        cx, cy = int(_b.centroid[0]), int(_b.centroid[1])
        if not (0 <= cy < m.shape[0] and 0 <= cx < m.shape[1]):
            return False
        return m[cy, cx] > 0

    # ★ 预先把槽位多边形算好（`glove_hits` 要它判断"手压在哪个槽"）。
    #   用**标称**位置即可 —— 手压判断只需要大致区域，不必跟位姿。
    slot_polys = {}
    for _s in (wp.get("slots") or []):
        try:
            _p = tray.slot_polygon_px(_s["id"], 1.0)
            if _p is not None:
                slot_polys[_s["id"]] = _p
        except Exception:                                          # noqa: BLE001
            pass

    paused = False
    show_bbox = False
    show_hand = not args.no_hand
    last_hand = (None, 0.0, None)
    debug = False
    idx = 0
    frames = 0
    blank = 0
    last_detect_frame = -999
    last_full_search = -999
    last_seen_ids = set()
    last_missing = []
    full_search_every = max(1, int(args.full_search_every))
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
                res = None
                if need_detect:
                    last_detect_frame = frames
                    ts = time.perf_counter()
                    # ★★ ROI 必须**跟着料盘走** —— 见 `Qualifier.tracking_roi()` 的说明。
                    #   ★ 原来这里写的是 `pose=getattr(seg, "_last_pose", None)`，
                    #     而 `seg` 是 `Segmenter` —— **它永远不会设置 `_last_pose`**，
                    #     所以那个 pose **恒为 None**，"位姿修正"从来没生效过。
                    #     实测后果（料盘挪了 222mm）：标称 ROI 黄铜覆盖 **0.5%**、
                    #     4 个零件**一个都不在 ROI 里** ⇒ 一个都识别不到。
                    #   ★ 现在改走 `tracking_roi()`：**第一依据是料盘自身轮廓**
                    #     （不依赖零件、也不依赖标称 ROI），实测料盘挪 222mm 后
                    #     仍能圈住 **4/4** 个零件。
                    roi, _roi_pose, _roi_src = q.tracking_roi(frame, 1.0, detect=True)
                    # ★★ 位姿变了就得把 ROI 缓存作废 —— 否则 `_ensure_roi` 会一直
                    #   用旧料盘位置算出来的内缩 ROI，"新轨迹只准建在工位内"那道闸
                    #   又会把料盘挪动后的零件全挡在外面。
                    if _roi_state.get("src") != _roi_src or _roi_state.get("pose") is not _roi_pose:
                        _roi_cache.clear()
                    _roi_state["pose"] = _roi_pose
                    _roi_state["src"] = _roi_src
                    res = seg.segment(frame, scale=1.0, work_roi=roi, mm_per_px=mmpp)
                    # ★★ 传给追踪器的块必须**限在工位内**（内缩后）。
                    #   不能只靠 allow_new 拦：先前这里传的是**外扩 25mm** 的 ROI 块，
                    #   料盘边沿外那条噪声依然能进来（实测 now_raw.png 里
                    #   (610,474) 那块甚至被认成 P2_column 0.89）。
                    #   在这里就把它挡掉，**根上不让它进追踪器**。
                    #   ★ 而全图搜索那一份（下面 r_full）**必须保持全画面** ——
                    #     判丢要全图搜，不能只看工位内。两份用途不同，不能混。
                    blobs = [b for b in res.brass_blobs if _in_inset_roi(b)]
                    seg_ms = (time.perf_counter() - ts) * 1000.0

                ts = time.perf_counter()
                # ★★ 判丢要**全图搜索**（用户明确要求）—— 所以 `seen_ids` 必须来自
                #   **覆盖全画面**的检测，不能只用工位 ROI 内的那一份（那就退化成
                #   "范围搜索"了）。全图分割比带 ROI 略贵（实测 8.9 vs 10.5ms），
                #   所以按 `full_search_every` 控制频率（默认约每秒一次）。
                seen_ids = None
                unidentified = False
                occl = res.glove_mask if res is not None else None
                # ★ 有轨迹进入"遮挡物已移开、正在全图找"的状态时，**加密全图搜索**，
                #   否则 2 秒宽限窗口内可能一次全图都没跑到（全图比带 ROI 贵，
                #   所以平时按 full_search_every 稀疏跑）。
                searching = any(
                    t.state == "occluded" and not t.covered and t.lost_frames > 0
                    for t in tracker.tracks)
                gap = max(1, min(full_search_every, 10) if searching else full_search_every)
                if (res is not None and frames - last_full_search >= gap):
                    last_full_search = frames
                    r_full = seg.segment(frame, scale=1.0, mm_per_px=mmpp)  # ★ 不带 work_roi = 全图
                    # ★★ 判丢只需要"**哪些零件 id 在画面里可见**"，
                    #   所以**直接对全图分割出的块做识别**就行 ——
                    #   不要跑 `check_slots`（那要算覆盖率+特征+逐槽判定，
                    #   实测把每帧均摊从 32ms 抬到 40ms）。
                    #   只对**面积像零件**的块识别（400~8000px），噪声块直接跳过。
                    try:
                        ids = set()
                        unid = False
                        for _b in r_full.brass_blobs:
                            if not (tracker.min_area_px <= _b.area_px
                                    <= tracker.max_area_px):
                                continue
                            # ★ 先把特征补齐（含色域）再判身份 —— 见 prepare() 的说明
                            _pid, _sc = classify(prepare(_b, frame, r_full.brass_mask).features)
                            if _pid:
                                ids.add(_pid)
                            else:
                                # ★★ 有“像零件但认不出身份”的块 -> 位置上有东西，
                                #    只是没认出来。“认不出” ≠ “不在”，**不能判丢**。
                                unid = True
                        seen_ids = ids
                        unidentified = unid
                    except Exception:                              # noqa: BLE001
                        seen_ids = None
                        unidentified = False
                    if occl is None:
                        occl = r_full.glove_mask

                # ★★ 关键一步：**在交给追踪器之前**把分类器要用的特征补齐（含色域）。
                #   为什么必须在这里做：`PartTracker` 内部调的是 `classify(b.features)`
                #   （只吃特征字典），它手上没有图与掩膜。
                #   而 `segment()` 出来的块只带 `contour_features()` 的结果
                #   （形状 + `area_px`），**没有 `area_mm2`/`length_mm`/色域** ——
                #   不补的话追踪器那条路径就只能拿形状分，与 c6 标定/`checks.py`
                #   那条路径**得分不一致**（缺的正是权重 3.2/18.1 的色域特征）。
                for _b in blobs:
                    prepare(_b, frame, res.brass_mask if res is not None else None)
                tracks = tracker.update(frame, blobs, detect=need_detect,
                                        classify=classify,
                                        seen_ids=seen_ids, occluder=occl,
                                        unidentified=unidentified,
                                        allow_new=_allow_new)
                trk_ms = (time.perf_counter() - ts) * 1000.0

                # ── 手（蓝手套）检测 ──────────────────────────────
                #   ★ 用户要求：「还要能识别工人的手，手用方框框就可以了」。
                #   手**只在检测帧算一次**（`_mask` 要跑 HSV + 连通域，占几个 ms），
                #   非检测帧沿用上一帧的手位 —— 手移动慢，且装配时本来就该少动。
                hand_box, hand_area, hand_on = None, 0.0, None
                if res is not None:
                    hand_box = res.glove_bbox
                    hand_area = res.glove_area_px
                    # ★★ `Segmenter.segment()` **不填 `hand_on`** ——
                    #   它是 `pipeline.py` 另调 `glove_hits(mask, polygons)` 算的
                    #   （实测踩过：直接读 `res.hand_on` 永远是空字典）。
                    #   这里照 pipeline 的做法补上，否则"手压在哪个槽"永远不显示。
                    try:
                        hand_on = glove_hits(res.glove_mask, slot_polys)
                    except Exception:                              # noqa: BLE001
                        hand_on = {}
                    last_hand = (hand_box, hand_area, hand_on)
                elif show_hand:
                    hand_box, hand_area, hand_on = last_hand

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
                            r2 = seg.segment(frame, scale=1.0, work_roi=m, mm_per_px=mmpp)
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
                                # ★★ 太远就别要（追踪框漂了）。
                                #   这个闸门原来**硬编码 40 px**，与
                                #   `tracking.dist_match_px` 重复却没读配置 ——
                                #   违反「阈值不要硬编码在算法里，一律走配置」。
                                #   ★ 2026-10 换 D435 后 GSD 从 0.8623 变 0.7287 mm/px，
                                #     40 px 的物理含义从 34.5mm 缩到 29.1mm；
                                #     现在统一读 `tracker.dist_match`（配置里已定 60 px = 43.7mm）。
                                _gate = float(getattr(tracker, "dist_match", 40.0))
                                if float(np.hypot(b.centroid[0] - t.centroid[0],
                                                  b.centroid[1] - t.centroid[1])) > _gate:
                                    continue
                                # ★★ 交给 `PartTracker.absorb()` 去做状态机 ——
                                #   **不要在这里自己改 `t.state` / 直接赋 `t.part_id`**。
                                #   原来这段自己写了一遍状态机，与 core 的 `_adopt` 不一致：
                                #     · 没更新 `area_ref`
                                #     · 直接赋值 -> 绕过 `apply_id()`，`score_ema` 永不更新
                                #       -> `reid_ok()` 的"分数余量"闸门形同虚设
                                #     · 没走形状对称闸 -> 非检测帧上更容易换错身份
                                #     · 没重置 `lost_frames`/`clear_frames`/`covered`/`miss_streak`
                                #   ⇒ 现在两边共用同一套（见 `absorb()` 的说明）。
                                #   ★ `reattach=False`：非检测帧只是**临时补一次轮廓与位置**，
                                #     重建 KCF 会把刚跟稳的追踪器状态打散，还多花一次 init。
                                tracker.absorb(frame, t, b, cv2.boundingRect(b.contour),
                                               float(b.area_px), classify=classify,
                                               reattach=False)

                canvas = frame.copy()
                # 手**先画**（方框在下层），零件轮廓后画（压在上面，不被手框盖住）
                if show_hand:
                    tracker.draw_hand(canvas, hand_box, hand_area,
                                      hand_on=hand_on, mm_per_px=mmpp)
                tracker.draw(canvas, show_bbox=show_bbox)

                # 面板
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 60), (28, 28, 28), -1)
                # ★★ 「缺件」必须用**全图检测的权威结果** `seen_ids`，
                #   不能拿“有没有轨迹落在这个槽里”来判断 ——
                #   实测踩过：幽灵轨迹正好压在底座上，于是**底座已经没了却报“有件”**。
                #   `seen_ids` 只在跑过全图搜索的帧才有值；没值的帧沿用上一次结果，
                #   这样“缺件”不会因为搜索周期而闪。
                #   ★ 用户明确：**缺什么还是要在面板上显示**（我之前误删，已恢复）；
                #     要去掉的是**画面上那个橙色的「全图搜 N 帧」标记**。
                if seen_ids is not None:
                    last_seen_ids = set(seen_ids)
                    last_missing = [
                        k for k, part in slot_parts.items()
                        if part and part not in last_seen_ids]
                missing = last_missing
                miss_txt = ("   缺: " + ",".join(k.replace("slot_", "") for k in missing)
                            if missing else "")
                n_ok = sum(1 for t in tracks if t.visible)
                n_occ = sum(1 for t in tracks if t.state == "occluded")
                # ★ 「久不可见」与「已放弃」分开报 —— 两者都会让人想去看一眼，
                #   但含义不同（前者手移开能恢复，后者已经不管了）。
                n_lost = sum(1 for t in tracks if t.state == "lost")
                n_gone = sum(1 for t in tracks if t.state == "gone")
                _san = tracker.sanity()
                dup_txt = (("  ⚠ 重复:" + ",".join(_san["duplicated"]))
                           if _san["duplicated"] else "")
                hand_txt = ("手: 有（压在 %s）"
                            % ",".join(k.replace("slot_", "")
                                       for k, v in (hand_on or {}).items() if v)
                            if (show_hand and hand_box is not None and
                                any((hand_on or {}).values()))
                            else ("手: 有" if (show_hand and hand_box is not None)
                                  else ("手: 无" if show_hand else "手: 关")))
                cv2.putText(canvas,
                            "零件 %d（可用 %d / 遮挡 %d / 久不可见 %d / 放弃 %d）  |  %s"
                            % (len(tracks), n_ok, n_occ, n_lost, n_gone,
                               hand_txt + dup_txt + miss_txt),
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1,
                            cv2.LINE_AA)
                cv2.putText(canvas,
                            "FPS %.1f | %s | 检测每 %d 帧 (%.1fms) | 追踪 %.1fms"
                            % (fps_times[-1] if fps_times else 0.0, args.backend,
                               tracker.detect_every, seg_ms, trk_ms),
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 230, 170), 1,
                            cv2.LINE_AA)
                # ★★ 「缺件」提示必须用**全图检测的权威结果** `seen_ids`，
                #   不能拿"有没有轨迹落在这个槽里"来判断 ——
                #   实测踩过：幽灵轨迹正好压在底座上，于是**底座已经没了却报"有件"**。
                #   `seen_ids` 只在跑过全图搜索的帧才有值；没值的帧沿用上一次的结果，

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
                    # ★ 画面以 `scale.realtime` 缩过（相机配置里是 0.5），
                    #   直接 imshow 会在窗口里留一大圈黑边（实测踩过：截图里
                    #   640x480 的窗口只用了左上 320x240）。这里按需放大回满窗。
                    view = canvas
                    if args.window_scale and abs(args.window_scale - 1.0) > 1e-6:
                        view = cv2.resize(
                            canvas, None, fx=args.window_scale, fy=args.window_scale,
                            interpolation=cv2.INTER_NEAREST)
                    cv2.imshow(WIN, view)
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
    if frames == 0:
        # ★★ 一帧都没读到 —— 最常见的两个原因，直接说清，别让用户猜。
        #   （实测踩过：相机被另一个进程占着时会静默读到 0 帧。）
        print("  ★★ 一帧都没读到（%.1f 秒）" % el)
        print("     最可能的原因：**相机被别的程序占用**。")
        print("       ① 先关掉其它正在用相机的窗口（app.py --windowed / 预览窗口 / 会议软件）")
        print("       ② 跑 python tools/probe_camera.py 确认相机本身能出图")
        print("       ③ 还不行就拔插一次 USB（DSHOW 偶尔要重新枚举）")
        return 2
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
    if seg_ms > 0 or trk_ms > 0:
        per_frame = seg_ms / max(1, tracker.detect_every) + trk_ms
        print("  ★ 实测每帧均摊  : %.2f ms  -> 上限 %.1f fps（相机 31 fps）"
              % (per_frame, 1000.0 / max(1e-6, per_frame)))
    else:
        per_frame = 0.0
        print("  ★ 实测每帧均摊  : 未测到（没有检测帧/追踪帧）")
    print("     其中分割被摊薄到 1/%d（每 %d 帧才跑一次全图分割）"
          % (tracker.detect_every, tracker.detect_every))
    if per_frame and per_frame <= 32.0:
        print("     ✓ 够用：每帧均摊 <= 32ms，能贴住相机 31fps")
    elif per_frame:
        print("     ★ 不够：每帧均摊 > 32ms，会掉帧 -> 把 --detect-every 调大")
    if per_frame:
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
