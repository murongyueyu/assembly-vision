# -*- coding: utf-8 -*-
"""零件追踪：卡尔曼平滑 + 相关滤波追踪 + 遮挡处理。

★ 为什么需要这个模块（而不是每帧全图重检测）：
    实测 640×480 全图分割 **9.88 ms**（PC）-> 树莓派 4B 按 6 倍估 **≈59 ms**
    也就是 Pi 上"每帧全分割"最多 **17 fps**，还要留给识别/追踪/绘制 -> 不够。
    而追踪器 update 只要 **MOSSE 0.25ms / KCF 2.0ms**（Pi 估 2 / 12 ms）。
    -> **检测 + 追踪**：检测每 N 帧跑一次（或按需），中间帧靠追踪器。

★ 帧率设计（在 config 里可调，Pi 上自动降档）：
    `every_n_frames` 决定检测频率；其余帧只跑追踪 + 绘制。
    实测（PC）：分割 9.9ms + 4 个 KCF 8ms + 绘制 ~1ms ≈ 19ms -> **~50 fps**
                （每 5 帧检测一次 -> 平均 ~50*... 见 tools/part_track_demo.py 的实测输出）

★ 遮挡处理（用户明确要求：「手拿起来就算遮一部分也要能认」）：
    实测（`tools/_probe_track_quality.py`）：用**追踪框当 ROI 重新分割**，
    遮挡 30% 保留 76% 轮廓、50% 保留 46%、**70% 仍保留 20% 且轮廓形状完整**
    -> 所以遮挡不是"丢了"，而是"面积变小"。
    处理策略：
      · 面积 >= full_ratio * 历史中值  -> **正常识别**（重算特征、重判身份）
      · visible_ratio <= lost_ratio    -> **暂时遮蔽/LOST**，冻结身份不重判（避免拿残缺件去认）
      · 中间地带                       -> 只更新位置与轮廓，**身份沿用上次**
    ★ 关键取舍：**遮挡时绝不重判身份** —— 残缺件的面积/长度都会变小，
      拿去认必然认错（实测底座被切到 1542mm² 就曾被认成轮盘）。

★ 树莓派适配（用户要求"延伸到边缘算力"）：
    · 追踪器**两套 API 都试**：5.x 的 `cv2.TrackerKCF()` 与 4.x 的
      `cv2.TrackerKCF_create()` —— Pi 上装的常是 OpenCV 4.x，只有后者。
    · `backend` 可选 mosse/kcf/csrt：Pi 上建议 `mosse`（2ms）或 `kcf`（12ms）。
      **CSRT 实测 22.7ms（Pi 估 136ms）—— 不要用。**
    · 全 numpy/OpenCV，无新依赖。

★ 分层：本模块只吃 numpy 与数据类，**不碰相机、不碰界面**（遵守 §3 分层铁律）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..types import Blob


# ──────────────────────────────────────────────────────────────
class KalmanXY:
    """匀速模型卡尔曼：状态 [x, y, vx, vy]，观测 [x, y]。

    ★ 为什么需要它（而不只用追踪器框）：
      追踪器框会**逐帧抖动**（框的边缘在像素间跳），而下游要的是
      "零件在不在这个槽里""质心稳不稳"。卡尔曼把抖动压掉，并给出
      **预测位置** —— 遮挡时追踪器框会漂，预测值反而更可信。
    """

    def __init__(self, x: float, y: float,
                 process_noise: float = 1e-3, meas_noise: float = 3.0) -> None:
        """默认参数由**实测扫参**定（`tools/_probe_kalman.py`）：

        场景 = 真值匀速直线 + 观测噪声 sigma=3px（贴近"分割质心抖动"）：

            轨迹误差(原)  轨迹误差(滤后)   逐帧抖动(原)  逐帧抖动(滤后)
              2.312        1.930          2.902        1.958     <- 旧默认 0.03/0.12
              2.312        1.286          2.902        1.044     <- 现默认 1e-3/3.0

        ★ 两个参数的**物理含义**（调参时按这个想，别乱试）：
            process_noise  = 「我多信模型预测」
            meas_noise     = 「我多信观测」
          零件在帧间几乎不动（料盘是静的，手慢），所以 process_noise 应当**很小**；
          而分割质心的抖动量级是 1~3px，meas_noise 应与它同量级。
        """
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * float(process_noise)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(meas_noise)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], np.float32)

    def predict(self) -> Tuple[float, float]:
        p = self.kf.predict()
        return float(p[0, 0]), float(p[1, 0])

    def correct(self, x: float, y: float) -> Tuple[float, float]:
        c = self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))
        return float(c[0, 0]), float(c[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        s = self.kf.statePost
        return float(s[2, 0]), float(s[3, 0])


def make_tracker(kind: str = "kcf"):
    """造一个追踪器。**两套 API 都试** —— 树莓派上装的常是 OpenCV 4.x。

    返回 (追踪器实例, 实际用到的 API 名)；都不可用返回 (None, 原因)。
    """
    kind = (kind or "kcf").lower()
    table = {
        "kcf": [("cv2.TrackerKCF()", lambda: cv2.TrackerKCF()),
                ("cv2.TrackerKCF_create()", lambda: cv2.TrackerKCF_create()),
                ("cv2.legacy.TrackerKCF_create()",
                 lambda: cv2.legacy.TrackerKCF_create())],
        "csrt": [("cv2.TrackerCSRT()", lambda: cv2.TrackerCSRT()),
                 ("cv2.TrackerCSRT_create()", lambda: cv2.TrackerCSRT_create())],
        "mosse": [("cv2.legacy.TrackerMOSSE_create()",
                   lambda: cv2.legacy.TrackerMOSSE_create()),
                  ("cv2.TrackerMOSSE_create()", lambda: cv2.TrackerMOSSE_create())],
        "medianflow": [("cv2.legacy.TrackerMedianFlow_create()",
                        lambda: cv2.legacy.TrackerMedianFlow_create())],
        "mil": [("cv2.TrackerMIL_create()", lambda: cv2.TrackerMIL_create())],
    }
    # 用一张小图探一下能不能真构造出来（只看"有没有这个属性"不够 ——
    # 实测 OpenCV 5.0 里 `cv2.TrackerKCF` 存在但直接构造会抛异常）
    probe = np.full((64, 64, 3), 128, np.uint8)
    for name, f in table.get(kind, []):
        try:
            t = f()
            t.init(probe, (8, 8, 32, 32))
            return t, name
        except Exception:                                          # noqa: BLE001
            continue
    return None, "没有可用的 %s 追踪器（试过 %s）" % (kind, [n for n, _ in table.get(kind, [])])


# ──────────────────────────────────────────────────────────────
@dataclass
class TrackState:
    """一个零件的追踪状态。"""
    part_id: str = ""
    id_score: float = 0.0
    contour: Optional[np.ndarray] = None       # 形状轮廓（像素），用于画与量
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    centroid: Tuple[float, float] = (0.0, 0.0)
    area_px: float = 0.0
    area_ref: float = 0.0                      # 历史中值面积（判断"被遮了多少"）
    visible_ratio: float = 1.0
    state: str = "detected"                    # detected | tracked | occluded | lost
    age: int = 0
    lost_frames: int = 0
    is_glove: bool = False
    detail: Dict[str, float] = field(default_factory=dict)
    id_margin: float = 0.12                    # 换标签需要比现有分数高出这么多
    score_ema: float = 0.0                     # 历史最好分数（指数平滑）

    def reid_ok(self, new_id: str, new_sc: float, symmetric: bool = True) -> bool:
        """要不要接受这个新身份？

        ★ 两道闸门（缺一不可）—— 因为"一次误判就永久污染身份"很危险：
          ① **分数要有明显优势**：新标签分数必须比现有分数高 `id_margin`，
             否则保留旧标签。避免两个零件分数接近时来回跳。
          ② **形状要够对称**：被遮成半边的零件，`elongation`/`aspect` 会失真，
             拿去和"完整的柱子"比反而更像柱状物 —— 实测就发生过
             「轮盘被遮一半 -> 认出 P2_column」。
             判据：可见度低时要求 `aspect` 接近 1（不细长）才允许改标签。
        """
        if not self.part_id:
            return True                        # 还没有身份，第一次认什么都接受
        if not symmetric:
            return False                       # 形状已失真 -> 冻结身份
        return new_sc > self.score_ema + self.id_margin

    def apply_id(self, new_id: str, new_sc: float) -> None:
        self.part_id, self.id_score = new_id, float(new_sc)
        self.score_ema = float(new_sc)

    @property
    def visible(self) -> bool:
        """能不能拿它去**判定**（被遮太多就不该拿去认身份）。"""
        return self.state in ("detected", "tracked")

    def label(self) -> str:
        if self.state == "occluded":
            return "遮挡 %d%%" % int(round(100 * (1.0 - self.visible_ratio)))
        if self.state == "lost":
            return "丢失 %d 帧" % self.lost_frames
        name = self.part_id or "?"
        return "%s %.2f" % (name, self.id_score) if self.id_score else name


class PartTracker:
    """多零件追踪：卡尔曼平滑 + 相关滤波 + 遮挡降级。

    用法（每帧）：
        tr = PartTracker(cfg)
        tracks = tr.update(frame, seg_result, detect=need_detect)
    其中 `detect=True` 的那一帧会做"分割结果 -> 框 -> 匹配/新建/删除"，
    其余帧只跑追踪器 update（快）。
    """

    def __init__(self, cfg: Optional[dict] = None) -> None:
        c = dict(cfg or {})
        self.backend = str(c.get("backend", "kcf"))
        self.detect_every = int(c.get("detect_every_n", 5))
        self.max_lost = int(c.get("max_lost_frames", 12))
        self.min_area_px = float(c.get("min_area_px", 400))
        self.max_area_px = float(c.get("max_area_px", 8000))
        self.iou_match = float(c.get("iou_match", 0.15))
        self.dist_match = float(c.get("dist_match_px", 40.0))
        self.occluded_ratio = float(c.get("occluded_ratio", 0.55))
        self.full_ratio = float(c.get("full_ratio", 0.80))
        self.area_rate = float(c.get("area_ref_rate", 0.15))
        # ★ KCF 的耗时是**每目标固定成本**（实测 5.5ms/目标，4 目标 35ms/帧），
        #   与整帧尺寸无关。曾试过「裁剪出小图再跑 KCF」，实测**只快 14% 但
        #   静止漂移从 2.76px 崩到 131px**（裁剪窗滑动会逼着反复 init，
        #   而 init 既不快又破坏追踪器状态）—— 已**放弃该方案**，
        #   详见 docs/踩坑实录.md §2.20。
        #   所以省算力只能靠「少跟几帧」（见 detect_every_n 与 pipeline 的用法）。
        self.tracks: List[TrackState] = []
        self._trackers: Dict[int, object] = {}
        self._kf: Dict[int, KalmanXY] = {}
        self._frame_ref: Dict[int, int] = {}
        self._frame_ref_n = 0
        self._next_key = 0
        self.last_detect_ms = 0.0
        self.last_track_ms = 0.0
        self.stats: Dict[str, float] = {"detects": 0, "track_only": 0}
        self.api_name = ""

    # ── 对外 ────────────────────────────────────────────────
    def update(self, frame: np.ndarray, blobs: Optional[Sequence[Blob]] = None,
               detect: bool = True,
               classify=None) -> List[TrackState]:
        """跑一帧。`classify(features)->(part,score)` 可选，用于给新框打身份。"""
        t0 = time.perf_counter()
        self._frame_ref_n += 1
        if detect:
            self._detect(frame, blobs or [], classify)
            self._run_trackers(frame)          # 检测帧也 update 一次，保持一致
            self._prune()
            self.stats["detects"] += 1
            self.last_detect_ms = (time.perf_counter() - t0) * 1000.0
        else:
            self._run_trackers(frame)
            self._prune()
            self.stats["track_only"] += 1
            self.last_track_ms = (time.perf_counter() - t0) * 1000.0
        self.age_all()
        return self.tracks

    # ── 内部 ────────────────────────────────────────────────
    def age_all(self) -> None:
        for t in self.tracks:
            t.age += 1

    def _detect(self, frame, blobs, classify) -> None:
        """把这一帧的分割结果与已有轨迹关联：匹配上的更新，没匹配上的新建。"""
        cands = []
        for b in blobs:
            a = float(b.area_px)
            if not (self.min_area_px <= a <= self.max_area_px):
                continue
            x, y, w, h = cv2.boundingRect(b.contour)
            cands.append((b, (int(x), int(y), int(w), int(h)), a))

        used = set()
        for t in list(self.tracks):
            best, best_iou = None, 0.0
            for i, (b, bb, a) in enumerate(cands):
                if i in used:
                    continue
                iou = _iou(t.bbox, bb)
                cx, cy = b.centroid
                d = float(np.hypot(cx - t.centroid[0], cy - t.centroid[1]))
                score = iou if iou > 0 else (self.dist_match - d) / self.dist_match * 0.1
                if (iou >= self.iou_match or d <= self.dist_match) and score > best_iou:
                    best, best_iou = i, score
            if best is None:
                continue
            used.add(best)
            b, bb, a = cands[best]
            self._adopt(frame, t, b, bb, a, classify)

        # 新目标
        for i, (b, bb, a) in enumerate(cands):
            if i in used:
                continue
            t = TrackState(bbox=bb, area_px=a, area_ref=a, contour=b.contour,
                           centroid=(float(b.centroid[0]), float(b.centroid[1])),
                           state="detected")
            self._attach(frame, t)
            if classify is not None:
                pid, sc = classify(dict(b.features or {}))
                if pid:
                    # ★ 必须走 `apply_id`（不是直接赋值）—— 它会把 `score_ema`
                    #   一起设上。第一版这里直接赋值，导致 ema 恒为 0，
                    #   于是后续"换标签要有分数余量"的闸门**形同虚设**
                    #   （实测：新目标第一次识别后 ema=0，第二次任何标签都能顶掉它）。
                    t.apply_id(pid, float(sc))
            self.tracks.append(t)

    def _adopt(self, frame, t: TrackState, b, bb, a: float, classify) -> None:
        """已有轨迹吸收一次新检测。

        ★★ 这里的顺序很关键（我第一版写反了，实测出假数据）：
            先 `area_ref` 更新再算 `visible_ratio` -> 参考值被**遮挡后的小面积**拉低
            -> `visible_ratio` 恒为 1.0 -> 状态永远是 `detected`
            -> **每帧都拿"被切掉一半的残缺件"去重新识别** -> 认错
               （实测 遮挡 50% 时 P3_wheel 被认成 P2_column）
            正确顺序：**先用旧参考算可见比例 -> 再按状态决定要不要更新参考**。
        """
        # ① 先算可见比例（用**旧**参考值）
        vis = a / max(1e-6, t.area_ref)
        t.bbox, t.area_px, t.contour = bb, a, b.contour
        t.centroid = (float(b.centroid[0]), float(b.centroid[1]))
        t.lost_frames = 0
        t.visible_ratio = vis

        if vis >= self.full_ratio:
            # ② 只有"看得够全"才动参考值，且**只允许缓慢上升/有限下降**
            #    —— 否则被遮挡几帧就把参考拉低，之后再也判不出遮挡
            t.area_ref = min((1 - self.area_rate) * t.area_ref + self.area_rate * a,
                             t.area_ref * (1.0 + self.area_rate))
            t.state = "detected"
            # ★ 只在"看得够全"时重判身份 —— 残缺件拿去认必然认错
            if classify is not None:
                feats = dict(b.features or {})
                pid, sc = classify(feats)
                if pid:
                    # ★ 再套两道闸门：分数要有优势 + 形状要够对称（见 TrackState.reid_ok）
                    asp = float(feats.get("aspect", feats.get("elongation", 1.0)) or 1.0)
                    sym = 0.45 <= asp <= 2.2
                    if t.reid_ok(pid, float(sc), symmetric=sym):
                        t.apply_id(pid, float(sc))
        elif vis <= self.occluded_ratio:
            # ③ 遮挡重：**不改参考、不重判身份**，只更新轮廓与位置
            t.state = "occluded"
        else:
            t.state = "tracked"
        self._attach(frame, t)

    def _attach(self, frame, t: TrackState) -> None:
        tr, api = make_tracker(self.backend)
        if tr is None:
            return
        self.api_name = api
        try:
            tr.init(frame, t.bbox)
        except Exception:                                          # noqa: BLE001
            return
        key = id(t)
        self._trackers[key] = tr
        if key not in self._kf:
            self._kf[key] = KalmanXY(t.centroid[0], t.centroid[1])
        self._frame_ref[key] = self._frame_ref_n

    def _run_trackers(self, frame) -> None:
        for t in self.tracks:
            key = id(t)
            tr = self._trackers.get(key)
            kf = self._kf.get(key)
            ok, box = (False, t.bbox)
            if tr is not None:
                try:
                    ok, box = tr.update(frame)
                except Exception:                                  # noqa: BLE001
                    ok = False
            if kf is not None:
                px, py = kf.predict()
                if ok:
                    x, y, w, h = (int(v) for v in box)
                    cx, cy = x + w / 2.0, y + h / 2.0
                    # 追踪器跳太远（多半漂了）-> 不信它，用预测值兜
                    if abs(cx - px) <= 60 and abs(cy - py) <= 60:
                        sx, sy = kf.correct(cx, cy)
                        t.bbox = (x, y, w, h)
                        t.centroid = (sx, sy)
                    else:
                        t.centroid = (px, py)
                else:
                    t.centroid = (px, py)
            if not ok:
                t.lost_frames += 1
                # ★★ 追踪器 update 失败 = 目标**完全看不见了**（被手/别的东西盖死）。
                #   此时必须把可见度**降下去**，否则 `visible_ratio` 会一直停在
                #   上一次的值（实测：手套把零件完全盖住时仍显示「可见 100%」）。
                #   ★ `lost_frames` 的清零只由 `_adopt`（真的看到检测）负责 ——
                #     **不要**在 `ok=True` 时清零：追踪器在完全遮挡时也可能报 ok，
                #     那样计数永远涨不上去，`state` 就降不下来。
                t.visible_ratio = 0.0
                t.state = "occluded" if t.lost_frames <= self.max_lost else "lost"

    def _prune(self) -> None:
        keep = []
        for t in self.tracks:
            if t.state == "lost" and t.lost_frames > self.max_lost:
                self._trackers.pop(id(t), None)
                self._kf.pop(id(t), None)
                continue
            keep.append(t)
        self.tracks = keep

    def roi_of(self, t: TrackState, pad: int = 10) -> Optional[np.ndarray]:
        """用追踪框生成 ROI 掩膜 —— 给"遮挡时重新分割"用。

        实测（tools/_probe_track_quality.py）：拿它当 work_roi 重新分割，
        遮挡 70% 仍能取回轮廓（面积剩 20%，点数 189）。
        """
        x, y, w, h = t.bbox
        if w <= 0 or h <= 0:
            return None
        return (x - pad, y - pad, x + w + pad, y + h + pad)

    def draw_hand(self, canvas: np.ndarray,
                  bbox: Optional[Sequence[int]],
                  area_px: float = 0.0,
                  hand_on: Optional[Dict[str, bool]] = None,
                  mm_per_px: float = 1.0) -> bool:
        """把**工人的手（蓝手套）用方框**标出来。

        ★ 用户要求：「还要能识别工人的手，工人的手用方框框就可以了（戴蓝手套）」
          所以**手画方框**（而零件画形状轮廓）—— 两者刻意不同，别统一。
        ★ 为什么手用方框就够：手是非刚体、形状多变，画轮廓既不准也没意义；
          而下游要的是「手在哪、有没有压到某个槽」（`hand_on`），方框足够。

        `bbox` 来自 `SegResult.glove_bbox`（`Segmenter.segment()` 已经算好，
        并把非手套的蓝色块剔除了）。返回是否画了。
        """
        if bbox is None:
            return False
        x, y, w, h = (int(v) for v in bbox)
        if w <= 0 or h <= 0:
            return False
        # 手用**青色**，与零件的绿/黄/橙/红全部区分开
        COL = (255, 220, 60)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COL, 2)
        # 四角加粗（一眼看出是"检测框"而不是随便一个矩形）
        L = max(6, min(w, h) // 5)
        for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1),
                                 (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            cv2.line(canvas, (px, py), (px + dx * L, py), COL, 3)
            cv2.line(canvas, (px, py), (px, py + dy * L), COL, 3)
        # 标签：手 + 面积（换算成 mm²，便于判断是不是误检的小蓝块）
        mm2 = float(area_px) * float(mm_per_px) * float(mm_per_px)
        txt = "手 %.0fmm²" % mm2
        if hand_on:
            hit = [k.replace("slot_", "") for k, v in hand_on.items() if v]
            if hit:
                txt += " 压在:" + ",".join(hit)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x, y - th - 8), (x + tw + 8, y), COL, -1)
        cv2.putText(canvas, txt, (x + 4, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        return True

    def draw(self, canvas: np.ndarray, show_bbox: bool = False) -> None:
        """画**形状轮廓**（不是方框）+ 卡尔曼平滑后的质心。

        ★ 用户明确要求"要能看到标注轮廓，是形状轮廓而不是一个方框"。
          方框只作为可选调试信息（`show_bbox=True`）。
        """
        COLOR = {"detected": (80, 230, 80),      # 绿：正常
                 "tracked": (80, 220, 240),      # 黄：跟得住但没重判
                 "occluded": (0, 165, 255),      # 橙：被遮（身份冻结）
                 "lost": (90, 90, 220)}          # 红：丢了
        for t in self.tracks:
            col = COLOR.get(t.state, (200, 200, 200))
            if t.contour is not None and len(t.contour) >= 3:
                cv2.drawContours(canvas, [t.contour], -1, col, 2)
                # 轮廓顶点（形状特征的直观体现）
                for p in t.contour.reshape(-1, 2):
                    cv2.circle(canvas, (int(p[0]), int(p[1])), 1, col, -1)
            if show_bbox:
                x, y, w, h = t.bbox
                cv2.rectangle(canvas, (x, y), (x + w, y + h), col, 1)
            cx, cy = int(t.centroid[0]), int(t.centroid[1])
            cv2.drawMarker(canvas, (cx, cy), col, cv2.MARKER_CROSS, 10, 2)
            cv2.putText(canvas, t.label(), (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = float((x1 - x0) * (y1 - y0))
    union = float(aw * ah + bw * bh) - inter
    return inter / union if union > 0 else 0.0
