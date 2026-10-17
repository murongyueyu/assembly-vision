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


class IdentityConfirmation:
    """单件工位的可选连续帧确认；只接收检测记录，不接触相机或文件。"""

    def __init__(self, config=None):
        if config is not None and not isinstance(config, dict):
            raise ValueError("连续帧确认配置必须为字典")
        self.config = dict(config or {})
        self.enabled = bool(self.config)
        self.pending = {}
        self.streaks = []
        if not self.enabled:
            return
        if not {"frames", "min_score", "max_gap_seconds", "dist_match_px", "iou_match", "area_ratio"}.issubset(self.config):
            raise ValueError("连续帧确认缺少必要配置")
        self.frames = self.config["frames"]
        self.min_score = float(self.config["min_score"])
        self.max_gap = float(self.config["max_gap_seconds"])
        self.distance = float(self.config["dist_match_px"])
        self.iou = float(self.config["iou_match"])
        self.area_ratio = tuple(self.config["area_ratio"])
        if (type(self.frames) is not int or self.frames < 1
                or not np.isfinite([self.min_score, self.max_gap, self.distance, self.iou]).all()
                or not 0 <= self.min_score <= 1 or self.max_gap <= 0 or self.distance <= 0
                or not 0 <= self.iou <= 1 or len(self.area_ratio) != 2
                or not np.isfinite(self.area_ratio).all()
                or not 0 < self.area_ratio[0] <= 1 <= self.area_ratio[1]):
            raise ValueError("连续帧确认的帧数、分数、时间、位置或面积配置无效")

    def update(self, predictions, elapsed_s):
        """同类只有一个可靠候选且连续帧位置/面积一致才确认；缺测/冲突立即重计。"""
        if not self.enabled:
            self.streaks = [1] * len(predictions)
            return [True] * len(predictions)
        if not np.isfinite(elapsed_s):
            raise ValueError("连续帧确认需要有限的源时间")
        counts = {}
        for p in predictions:
            if p["part"]:
                counts[p["part"]] = counts.get(p["part"], 0) + 1
        current, accepted, self.streaks = {}, [], []
        for p in predictions:
            pid, score, bbox, area = p["part"], p["score"], p["bbox"], p["area_px"]
            valid = (pid and counts[pid] == 1 and np.isfinite([score, area]).all()
                     and score >= self.min_score and area > 0)
            streak, reference = 0, area
            if valid:
                previous = self.pending.get(pid)
                if previous:
                    oldbox, oldarea, oldcount, oldtime = previous
                    dx = bbox[0]+bbox[2]/2 - oldbox[0]-oldbox[2]/2
                    dy = bbox[1]+bbox[3]/2 - oldbox[1]-oldbox[3]/2
                    consistent = (0 < elapsed_s-oldtime <= self.max_gap
                        and (_iou(oldbox, bbox) >= self.iou or np.hypot(dx, dy) <= self.distance)
                        and self.area_ratio[0] <= area / oldarea <= self.area_ratio[1])
                    if consistent:
                        streak, reference = min(oldcount, self.frames-1), oldarea
                streak += 1
                current[pid] = (tuple(bbox), reference, streak, elapsed_s)
            self.streaks.append(streak)
            accepted.append(streak >= self.frames)
        self.pending = current
        return accepted


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
    # ★★ `miss_streak`：连续多少帧**没有拿到检测结果**（不管追踪器报什么）。
    #   用它决定「还值不值得跑 KCF」—— 因为 `lost_frames` 只在**真检测到**时清零，
    #   如果用它做这个判断，那些被保留的 `gone` 轨迹会**永远在跑 KCF**
    #   （实测踩过：保留 60 秒后帧率从 28fps 掉到 **9fps**）。
    miss_streak: int = 0
    # ★★ 「从未真正跟踪成功过」的帧数 —— 幻影轨迹就是这样：
    #   它是被一个检测建出来的（所以 `ever_seen=True`），但 `tr.update()` **一次都没成功过**。
    #   实测踩过：只把它从画面上藏起来不够 —— 它会活到 `gone_keep_seconds`(60s)，
    #   期间**每帧都在跑 KCF**。所以给它一个短命上限。
    #   实测踩过：**不能用追踪器的返回值判幽灵** ——
    #   在一帧纯灰背景上（根本没有零件）`cv2.TrackerKCF.update()` **依然返回 True**，
    #   它会自己找个东西盯上。-> 只能靠“检测有没有匹配上”。
    # ── 丢失判据用的三个计时（用户明确定义，见下方 `PartTracker._judge_lost`）──
    #   ★ 语义：**不是"看不见多久就怀疑"，而是"遮挡物移开后再给一小段时间
    #     全图搜不到"才算丢失**（用户原话）。
    #     occluded_frames  当前**被遮挡物盖着**的帧数（这期间永不判丢）
    #     clear_frames     遮挡物**已移开**之后过了多少帧（判丢的计时器）
    #     ever_seen        是否曾经真的检测到过（新建后连一次都没看到的不算"丢"）
    occluded_frames: int = 0
    clear_frames: int = 0                        # 兼容旧字段（帧数，仅调试用）
    clear_since: float = 0.0                     # 遮挡物移开的那一刻（perf_counter）
    clear_seconds: float = 0.0                   # 已过去多少**秒**（判丢用它）
    ever_seen: bool = False
    covered: bool = False                      # 本帧是否被遮挡物盖着
    gone_since: float = 0.0                    # 进入 gone 的时刻（用于保留多久）
    # ★★ 「最后一次真正检测到」的位置与轮廓 —— **不要**用卡尔曼预测值做匹配。
    #   实测踩过：追踪失败后卡尔曼会漂移，而匹配用的是 `centroid`（预测值），
    #   配上宽松的 `dist_match_px=40`，一条轨迹会**抢到别处的块**
    #   -> 那个块本该新建轨迹却没有，而轨迹数虚增 -> 界面上出现
    #      "全图搜 N帧" 的**幽灵轮廓**（形状扭曲、位置在料盘外）。
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    last_centroid: Tuple[float, float] = (0.0, 0.0)
    # 遮挡时**只冻结位置、保留轮廓**：轮廓用于显示"它本来在哪、有多大"，
    # 但部位不动。★ 只在 `ever_seen`（真的看到过）时才有意义。
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
        """界面上显示的文字。

        ★★ 三档要**分得清**（用户明确要求）：
            occluded  零件还在原位，只是被挡（含**完全被挡**）
            lost      久到可疑，但**还没删**，手移开能恢复
            gone      很久都看不见，已放弃
          实测踩过：原来"被完全挡住 0.4 秒"就报 lost，与"真被拿走"分不开。
        """
        if self.state == "occluded":
            if self.covered:
                # ★ 遮挡物**还盖着** —— 永不判丢（用户明确要求）
                return "被挡 %.1fs" % (self.occluded_frames / 30.0)
            if self.visible_ratio <= 0.02:
                # ★ 按**秒**显示，不按帧 —— 帧数在掉帧时与真实时间脱节
                #   （实测用户"等了一分钟"而计数器只走 64 帧）。
                return "全图搜 %.1fs" % self.clear_seconds
            return "遮挡 %d%%" % int(round(100 * (1.0 - self.visible_ratio)))
        if self.state == "lost":
            return "丢失(全图搜 %.1fs 无果)" % self.clear_seconds
        if self.state == "gone":
            return "已放弃"
        name = self.part_id or "?"
        return "%s %.2f" % (name, self.id_score) if self.id_score else name


class PartTracker:
    """多零件追踪：卡尔曼平滑 + 相关滤波 + 遮挡降级。

    用法（每帧）：
        tr = PartTracker(cfg)
        tracks = tr.update(frame, seg_result, detect=need_detect)
    其中 `detect=True` 的那一帧会做"分割结果 -> 框 -> 匹配/新建/删除"，
    其余帧只跑追踪器 update（快）。

    backend="detection"时每帧必须提供检测结果，不创建相关滤波器；缺观测仍按
    遮挡物和真实时钟判丢。identity_relink仅供每类单件的工位显式开启，需配置
    分数及面积比例闸门；同身份有歧义时保留历史，不新建重复身份轨迹。
    """

    def __init__(self, cfg: Optional[dict] = None, *, clock=None) -> None:
        c = dict(cfg or {})
        # 离线回放可注入录制时钟；真机默认仍使用perf_counter。
        self._clock = clock
        self.backend = str(c.get("backend", "kcf"))
        self.detect_every = int(c.get("detect_every_n", 5))
        if self.backend == "detection" and self.detect_every != 1:
            raise ValueError("逐帧检测关联模式必须配置detect_every_n=1")
        # ── 丢失判据（用户明确定义，不要再用"看不见多久"来判）──
        #   ★★ 用户原话：
        #     「不是持续很久才被怀疑，而是最后一次还检测到有零件的位置的时候，
        #       到遮挡物已经完全消失了之后的一小段时间（暂定2秒）还是没有检测到
        #       这个零件，算作这个零件丢失。而且找这个零件是**全图搜索**而不是范围搜索。」
        #   -> 三件事：
        #     ① 遮挡物**还盖着**的时候，**永不判丢**（挡多久都不算）
        #     ② 计时从"遮挡物**完全移开**"那一刻开始，默认 2 秒
        #     ③ 到期前必须做**全图搜索**（不是只在原位附近找）
        # ★★ 宽限期**以秒为准**（用户：「暂定2秒」）。
        #   `lost_grace_frames` 保留作为兼容字段，内部一律用秒。
        #   实测踩过：用户感觉「等了一分钟」，而帧数只走了 64 ——
        #   帧数在掉帧/暂停时与真实时间脱节。
        self.lost_grace_seconds = float(c.get("lost_grace_seconds", 2.0))
        self.lost_grace_frames = int(c.get("lost_grace_frames",
                                           round(self.lost_grace_seconds
                                                 * float(c.get("assumed_fps", 30.0)))))
        # ★ 判丢按**秒**算，所以要有个帧率假设把 `lost_grace_frames` 换算成秒。
        #   相机是 31fps；用 30 作缺省。
        self.assumed_fps = float(c.get("assumed_fps", 30.0))
        # ★★「已放弃」的轨迹**也要留够久**，因为零件可能被拿走又放回来 ——
        #   实测用户问「如果零件回去了，该怎么识别」：留着它，等同一块回来时
        #   `_detect` 能按位置重新关联上，**身份与历史都恢复**。
        #   ★ 而且这也是防幽灵的关键：若一个回来的块不能关联到原轨迹，
        #     它会**新建一条** -> 同一零件两条轨迹 -> 界面上 "零件 8"。
        self.gone_keep_seconds = float(c.get("gone_keep_seconds", 60.0))
        # ★★ 幻影轨迹的删除门槛：**已 gone** 且**未被遮挡** 且
        #   **连续这么多帧没拿到检测结果**。
        #   实测踩过：不能用“从未被匹配过”做判据 —— 轨迹就是检测建出来的，
        #   那个条件**永远不成立**（删除逻辑从来没跑到过）。
        #   也不能在遮挡期间计数 —— 否则“手盖住真零件”会把轨迹删掉。
        # ★ 默认从宽限期推：**至少 3 倍宽限期，且不少于 90 帧**（约 3 秒）。
        #   宽限2s、假设30fps时是180帧（约6秒）；短宽限配置会收紧到90帧下限，
        #   不会因为一个写死的数字而让测试永远跑不到那个条件。
        self.phantom_miss_frames = int(c.get(
            "phantom_miss_frames", max(90, self.lost_grace_frames * 3)))   # 60 帧 ≈ 2 秒
        self.cover_overlap_ratio = float(c.get("cover_overlap_ratio", 0.25))
        self.min_area_px = float(c.get("min_area_px", 400))
        self.max_area_px = float(c.get("max_area_px", 8000))
        self.iou_match = float(c.get("iou_match", 0.15))
        self.dist_match = float(c.get("dist_match_px", 40.0))
        self.occluded_ratio = float(c.get("occluded_ratio", 0.55))
        self.full_ratio = float(c.get("full_ratio", 0.80))
        self.area_rate = float(c.get("area_ref_rate", 0.15))
        # ★★ `id_margin`：换标签的分数余量。**2026-10 修**：
        #   它原来只是 `TrackState` 的 dataclass 默认值（`:180`），
        #   `PartTracker.__init__` **从不读配置** —— 也就是说
        #   `workpiece.yaml` 里写的 `tracking.id_margin: 0.12` **完全不生效**。
        #   而 `workpiece.yaml` 与 `tests/test_part_id_tol.py` 都**拿它当依据**
        #   （「底座 0.65 vs 轮盘 0.70 贴住，id_margin=0.12 就不安全了」）——
        #   配置与代码脱节，调配置没用。
        #   现在读进来并**下发到每条轨迹**（见 `_detect` 建新轨迹处）。
        self.id_margin = float(c.get("id_margin", 0.12))
        # ★ 单件工位的可选远距离重关联：仅对唯一身份且面积接近的完整候选生效。
        # 默认关闭，先在独立开发回放验证，不能把同类多件场景套进来。
        self.identity_relink = bool(c.get("identity_relink", False))
        if self.identity_relink:
            self.identity_relink_min_score = float(c["identity_relink_min_score"])
            self.identity_relink_area_ratio = tuple(c["identity_relink_area_ratio"])
            if (len(self.identity_relink_area_ratio) != 2
                    or not 0 < self.identity_relink_area_ratio[0] <= self.identity_relink_area_ratio[1]
                    or not 0 <= self.identity_relink_min_score <= 1):
                raise ValueError("身份重关联的分数/面积比例配置无效")
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
        self.api_name = "detection association" if self.backend == "detection" else ""

    # ── 对外 ────────────────────────────────────────────────
    def update(self, frame: np.ndarray, blobs: Optional[Sequence[Blob]] = None,
               detect: bool = True,
               classify=None,
               seen_ids: Optional[set] = None,
               occluder: Optional[np.ndarray] = None,
               unidentified: bool = False,
               allow_new=None) -> List[TrackState]:
        """跑一帧。

        参数
        ----
        blobs     : 本帧分割出的零件块。★ 判丢要求**全图搜索**，
                    所以检测帧请传**全画面**的块（别只传工位 ROI 内的）。
        classify  : `classify(features) -> (part_id, score)`，给新框打身份。
        seen_ids  : 本帧**全图**识别到的零件 id 集合 —— 判丢用它（用户要求全图搜索）。
        occluder  : 遮挡物掩膜（手套）。有它才能判断"遮挡物是否已移开"。
        """
        if self.backend == "detection" and not detect:
            raise ValueError("逐帧检测关联模式不能跳过检测帧")
        t0 = time.perf_counter()
        self._frame_ref_n += 1
        if detect:
            if self.backend == "detection":
                # 未匹配的历史轨迹先记缺观测；真正匹配上的由absorb统一重置。
                for track in self.tracks:
                    track.lost_frames += 1
                    track.miss_streak += 1
                    track.state = "occluded"
                    track.visible_ratio = 0.0
            self._detect(frame, blobs or [], classify, allow_new=allow_new)
            if self.backend != "detection":
                self._run_trackers(frame)      # 相关滤波模式保持原路径
            self.judge_lost(seen_ids, occluder, unidentified=unidentified)
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

    def _detect(self, frame, blobs, classify, allow_new=None) -> None:
        """把这一帧的分割结果与已有轨迹关联：匹配上的更新，没匹配上的新建。

        ★★ 对全图候选按配对分数做一对一认领，而不是"按轨迹列表顺序认领"：
           贪心时先遍历到的轨迹会抢走后面轨迹更该拿的块，
           结果是"一条轨迹吃掉别人的块 + 那个块不再新建" -> 轨迹错配、数量虚增。
           做法：把所有 (轨迹, 块) 的合法配对按打分排序，从高到低依次认领，
           **每条轨迹与每个块都只能用一次**；这不是求总权重最大值的算法。
        ★★ 匹配位置用 `last_centroid`（**最后一次真正检测到**的位置），
           不用卡尔曼预测值 —— 见 `TrackState.last_centroid` 的说明。
        """
        cands = []
        for b in blobs:
            a = float(b.area_px)
            if not (self.min_area_px <= a <= self.max_area_px):
                continue
            x, y, w, h = cv2.boundingRect(b.contour)
            cands.append((b, (int(x), int(y), int(w), int(h)), a))

        # 单件身份可靠时先认回原轨迹，防止跨越位置闸门后建第二条同身份轨迹。
        # 身份/面积有歧义时不做远距离猜测，继续走原空间匹配。
        t_used, b_used = set(), set()
        if self.identity_relink and classify is not None:
            labels = [classify(dict(b.features or {})) for b, _, _ in cands]
            selected = identity_relink_pairs(
                [(t.part_id if t.ever_seen else None, t.area_ref) for t in self.tracks],
                [(pid, score, area) for (pid, score), (_, _, area) in zip(labels, cands)],
                self.identity_relink_min_score, self.identity_relink_area_ratio)
            for ti, bi in selected:
                b, bb, area = cands[bi]
                self._adopt(frame, self.tracks[ti], b, bb, area,
                            lambda _features, label=labels[bi]: label)
                t_used.add(ti)
                b_used.add(bi)

        # ① 收集剩余的合法空间配对并打分
        pairs = []
        for ti, t in enumerate(self.tracks):
            if not t.ever_seen or ti in t_used:
                continue                       # 从没看到过的轨迹不参与匹配
            rx, ry = t.last_centroid
            for bi, (b, _bb, _a) in enumerate(cands):
                if bi in b_used:
                    continue
                iou = _iou(t.last_bbox, _bb)
                cx, cy = b.centroid
                d = float(np.hypot(cx - rx, cy - ry))
                if iou >= self.iou_match:
                    pairs.append((iou + 1.0, ti, bi))       # IoU 命中，优先
                elif d <= self.dist_match:
                    # 距离命中：按接近程度打分（分数远低于 IoU 命中）
                    pairs.append((1.0 - d / max(1e-6, self.dist_match), ti, bi))
        pairs.sort(key=lambda p: -p[0])

        # ② 从高到低认领，轨迹与块都只能用一次
        for _score, ti, bi in pairs:
            if ti in t_used or bi in b_used:
                continue
            t_used.add(ti)
            b_used.add(bi)
            t = self.tracks[ti]
            b, bb, a = cands[bi]
            self._adopt(frame, t, b, bb, a, classify)

        # ③ 剩下没用上的块 -> 新建轨迹
        #   ★★ llow_new 是**防幽灵的最后一道闸**：
        #     全图检测会把画面任何地方的像零件的块都拿进来（是有意的，
        #     判丢要全图搜），但**建新轨迹**应该只在工位范围内 ——
        #     否则料盘外的噪声会变成一个个「全图搜」空心圈幽灵。
        for i, (b, bb, a) in enumerate(cands):
            if i in b_used:
                continue
            if self.identity_relink and classify is not None:
                pid, score = labels[i]
                # 单件假设下同身份不建第二条；歧义时保留历史，不能按面积随便挑。
                if pid and (score < self.identity_relink_min_score
                            or sum(label[0] == pid for label in labels) != 1
                            or any(t.part_id == pid for t in self.tracks)):
                    continue
            if allow_new is not None and not allow_new(b):
                continue
            t = TrackState(bbox=bb, area_px=a, area_ref=a, contour=b.contour,
                           centroid=(float(b.centroid[0]), float(b.centroid[1])),
                           last_bbox=bb,
                           last_centroid=(float(b.centroid[0]), float(b.centroid[1])),
                           # ★ 把配置里的 `tracking.id_margin` 下发到轨迹上
                           #   （不传的话会用 dataclass 默认值，配置就白写了）
                           id_margin=self.id_margin,
                           state="detected")
            # ★★ 新建轨迹就是"第一次真的看到它" —— 必须置 `ever_seen`。
            #   实测踩过：只在这里漏了，于是 `ever_seen` 恒为 False，
            #   `judge_lost()` 直接把它当成"误检"清掉（`state=gone`），
            #   表现是**轨迹莫名消失、判丢逻辑完全不生效**。
            #   （与上面 `score_ema` 那个坑同一类：新建路径与更新路径必须一致。）
            t.ever_seen = True
            # ★★ 新建轨迹就是“被一次检测匹配上了” —— **必须一并置上**。
            #   实测踩过（本模块第五次同类）：只在 _adopt 里设了 dopted_once，
            #   新建路径漏了 -> **每条轨迹一出生就被当幻影**（ntf 从 1 开始），
            #   表现是合法轨迹被当幻影删掉。
            #   ★ 与 ever_seen / score_ema / 状态覆盖 同一类：**新建与更新路径必须做同样的初始化**。
            t.occluded_frames = 0
            t.clear_frames = 0
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

    def absorb(self, frame, t: TrackState, b, bb, a: float, classify=None,
               reattach: bool = True) -> None:
        """**已有轨迹吸收一次检测** —— 状态机的唯一入口。

        ★★ 为什么要把它抽成公开方法（2026-10 修，本模块的一个真 bug）：
          原来这套状态机只存在于 `_detect` -> `_adopt` 这条内部路径里，
          而 `tools/part_track_demo.py` 在**非检测帧**的"遮挡重分割"路径上
          **自己又写了一遍**，而且写得不一样：
            · 没更新 `area_ref`（参考面积不再跟着走）
            · 直接 `t.part_id, t.id_score = pid, sc` —— **绕过 `apply_id()`**，
              于是 `score_ema` 永远不更新，`reid_ok()` 的"换标签要有分数余量"
              这道闸门**形同虚设**（与本文件 `_detect` 里记过的那个坑同型）
            · 完全没走 `reid_ok()` 的**形状对称闸** —— 非检测帧上更容易换错身份
            · 没重置 `lost_frames` / `clear_frames` / `covered` / `miss_streak`
          后果：同一个零件在"检测帧"与"非检测帧"上走**两套不同的身份逻辑**，
          表现就是身份偶尔跳变、判丢偶发误报 —— 而这两条路各自单独看都"像是对的"。
        ⇒ 现在两边**共用本方法**，差异只剩"怎么把块匹配到轨迹"
          （内部是一对一配对认领，demo 是当帧最近的块）。

        参数
        ----
        b, bb, a   Blob / bbox / area_px（`bb` 单独传是为了省一次 `boundingRect`）
        classify   `classify(features) -> (part_id, score)`；None = 不重判身份
        reattach   是否重建 KCF 追踪器（`_attach`）。★ demo 的非检测帧**不要**重建 ——
                   它只是临时补一次轮廓/位置，重建会把刚跟稳的追踪器状态打散，
                   而且 `_attach` 每次都要 `make_tracker()` + `init()`，是有成本的。
        """
        # ① 先算可见比例（用**旧**参考值）—— 顺序铁律，见 `_adopt` 的说明
        vis = a / max(1e-6, t.area_ref)
        t.bbox, t.area_px, t.contour = bb, a, b.contour
        t.centroid = (float(b.centroid[0]), float(b.centroid[1]))
        t.lost_frames = 0
        t.visible_ratio = vis
        t.ever_seen = True
        t.clear_frames = 0
        t.clear_since = 0.0
        t.clear_seconds = 0.0
        t.covered = False
        t.miss_streak = 0
        t.last_bbox = bb
        t.last_centroid = (float(b.centroid[0]), float(b.centroid[1]))

        if vis >= self.full_ratio:
            # ② 只有"看得够全"才动参考值，且只允许缓慢上升/有限下降
            t.area_ref = min((1 - self.area_rate) * t.area_ref + self.area_rate * a,
                             t.area_ref * (1.0 + self.area_rate))
            t.state = "detected"
            if classify is not None:
                feats = dict(b.features or {})
                pid, sc = classify(feats)
                if pid:
                    asp = float(feats.get("aspect", feats.get("elongation", 1.0)) or 1.0)
                    sym = 0.45 <= asp <= 2.2
                    if t.reid_ok(pid, float(sc), symmetric=sym):
                        t.apply_id(pid, float(sc))
        elif vis <= self.occluded_ratio:
            # ③ 遮挡重：不改参考、不重判身份，只更新轮廓与位置
            t.state = "occluded"
        else:
            t.state = "tracked"
        if reattach:
            self._attach(frame, t)

    def _adopt(self, frame, t: TrackState, b, bb, a: float, classify) -> None:
        """已有轨迹吸收一次检测（内部路径）。★ 现在只是 `absorb()` 的薄封装 ——
        状态机本体已抽到 `absorb()`，好让 `tools/part_track_demo.py` 共用同一套
        （原来那两套不一致，见 `absorb()` 的说明）。"""
        self.absorb(frame, t, b, bb, a, classify=classify, reattach=True)

    def _attach(self, frame, t: TrackState) -> None:
        if self.backend == "detection":
            return                            # 由当帧轮廓定位，不创建相关滤波器
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
        t.miss_streak = 0      # 追踪器刚重建，重新计数

    def _run_trackers(self, frame) -> None:
        for t in self.tracks:
            key = id(t)
            tr = self._trackers.get(key)
            kf = self._kf.get(key)
            ok, box = (False, t.bbox)
            # ★★ 长时间看不见的轨迹**不再跑 KCF** —— 每目标约 8ms（实测），
            #   而它已经"久不可见"，继续跑没有收益。只保留卡尔曼预测，
            #   等全图搜索真找到它时 `_adopt` 会重建追踪器。
            #   （实测踩过：影子轨迹白白吃掉预算，每帧均摊从 32ms 抬到 40ms。）
            # ★ 只在「还相信它在」时跑 KCF：
            #   miss_streak 小 =　刚看到过 / 刚被挡 —— 值得跟；
            #   miss_streak 大 =　久不可见 —— 跑了也白跑（每目标约 8ms）。
            if tr is not None and t.miss_streak <= self.phantom_miss_frames:
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
                t.miss_streak += 1
                # 追踪器 update 失败 = 目标**看不见了**。
                # ★ 注意：`lost_frames` 只是"连续多少次追踪失败"，
                #   **不再用它判 lost** —— 判 lost 由 `_judge_lost()` 按
                #   "遮挡物移开后再等 2 秒"来做（用户明确要求）。
                #   ★ `lost_frames` 的清零只由 `_adopt`（真的看到检测）负责 ——
                #     **不要**在 `ok=True` 时清零：追踪器在完全遮挡时也可能报 ok，
                #     那样计数永远涨不上去。
                t.visible_ratio = 0.0

    def judge_lost(self, seen_ids: Optional[set] = None,
                   occluder: Optional[np.ndarray] = None,
                   unidentified: bool = False) -> None:
        """判定 `occluded` / `lost` / `gone` —— **按用户定义的语义**。

        ★★ 用户原话（这是唯一权威定义，别再自己发明）：
            「不是持续很久才被怀疑，而是最后一次还检测到有零件的位置的时候，
             到**遮挡物已经完全消失了之后的一小段时间（暂定2秒）**还是没有检测到
             这个零件，算作这个零件丢失。而且找这个零件是**全图搜索**而不是范围搜索。」

        逻辑：
          ① `covered`  —— 遮挡物**还盖着**：`occluded`，**永不判丢**（挡多久都不算）
          ② 遮挡物**已移开**：开始计时（按**秒**），窗口内做**全图搜索**，
                           这段时间仍是 `occluded`（正在找，先别惊动）
          ③ 窗口过期**还没找到** -> `lost`；再久 -> `gone`（允许清理）

        参数
        ----
        seen_ids    : 本帧**全图**识别到的零件 id 集合。
                      ★ 必须来自“覆盖全画面的检测”，否则就退化成范围搜索了。
        occluder    : 遮挡物掩膜（手套）；None 表示没有遮挡物。
        unidentified: 画面里**有没有“像零件但认不出身份”的块**。
                      ★★ 这一条很关键（实测踩过）：如果那个位置**有东西**
                         （只是分类器没给出 id），那就**不能判“丢失”** ——
                         “认不出” ≠ “不在”。零件明明在那儿却被报成丢了，
                         这是最误导的一种错。

        ★ 计时用**真实时间**而不是帧数：实测用户感觉“等了一分钟”，
          而帧数计数器只走了 64 —— 帧数在掉帧/暂停时与真实时间脱节。
        """
        now = self._clock() if self._clock is not None else time.perf_counter()
        for t in self.tracks:
            if not t.ever_seen:
                # 新建后**一次都没真看到**过：不算“丢”，过一会儿清掉（多是误检）
                if t.age > self.lost_grace_frames:
                    t.state = "gone"
                continue

            # ① 遮挡物还盖着吗？
            t.covered = self._is_covered(t, occluder)
            if t.covered:
                t.occluded_frames += 1
                t.clear_since = 0.0
                t.clear_seconds = 0.0
                # ★★ 遮挡期间**把 `miss_streak` 清零** —— 实测踩过：
                #   不清零的话，“手盖 60 帧”会把它堆到 60，
                #   手一移开再过 30 帧就超过幻影阈值 ——
                #   于是**一个真实存在的零件被当幻影删掉**。
                #   ★ 又是“一个计数器被两种语义抢用”（参 §2.24 / §2.28）：
                #     它既要表示“多久没检测到”，又要表示“多久真的丢了”。
                #     **被遮挡不是不存在的证据**，不能计入。
                t.miss_streak = 0
                t.state = "occluded"          # ★ 挡着就永远是“遮挡”，不管多久
                continue

            # ② 本帧追踪器**成功了** -> 它就在那儿，
            #    **不要**改写 `_adopt` / `_run_trackers` 已算好的状态。
            #    （实测踩过：这里无条件覆盖，导致“完全可见(0%)”也被标 occluded。）
            if t.lost_frames == 0:
                t.clear_since = 0.0
                t.clear_seconds = 0.0
                continue

            # ③ 遮挡物已移开、且本帧没看到它 -> 开始/继续计时
            if t.clear_since <= 0.0:
                t.clear_since = now
            t.clear_seconds = now - t.clear_since

            found = bool(seen_ids and t.part_id and t.part_id in seen_ids)
            if found:
                # 全图搜到了 -> 它还在（只是刚才被挡），重置
                t.clear_since = 0.0
                t.clear_seconds = 0.0
                continue
            if unidentified:
                # ★★ 画面里有“像零件但认不出”的块 -> 位置上**有东西**。
                #    “认不出” ≠ “不在”，**不能判丢**。
                t.state = "occluded"
                continue

            grace_s = self.lost_grace_seconds
            if t.clear_seconds <= grace_s:
                t.state = "occluded"          # 还在窗口内：正在全图搜
            elif t.clear_seconds <= grace_s * 10.0:
                t.state = "lost"              # 窗口过了还没找到 -> 丢
            else:
                t.state = "gone"              # 再过久 -> 放弃（允许清理）

    def sanity(self) -> Dict[str, object]:
        """把匹配质量暴露出来，便于诊断“幽灵轨迹”。

        实测踩过（用户截图）：界面上出现 `零件 8` 而物理上只有 4 个零件 ——
        说明**同一零件被建了多条轨迹**。原因通常是：
          · 分割/识别漏了某帧 -> 那块没匹配上 -> 新建一条
          · 匹配阈值太松 -> 一条轨迹抢到别处的块
        这里统计“同一 part_id 有几条轨迹”，`max_per_id > 1` 就是有问题。
        """
        per: Dict[str, int] = {}
        for t in self.tracks:
            if t.part_id:
                per[t.part_id] = per.get(t.part_id, 0) + 1
        return {"tracks": len(self.tracks), "per_id": per,
                "max_per_id": max(per.values()) if per else 0,
                "duplicated": [k for k, v in per.items() if v > 1]}


    def _is_covered(self, t: TrackState, occluder: Optional[np.ndarray]) -> bool:
        """这个零件**当前**被遮挡物盖着吗（按它最后已知的位置/轮廓判定）。

        判据：遮挡物掩膜与"零件上次所在区域"的重叠比例 > `cover_overlap_ratio`。
        ★ 用**上次所在位置**而不是当前位置 —— 位置正是我们不知道的东西。
        """
        if occluder is None or t.contour is None:
            return False
        try:
            cm = np.zeros(occluder.shape[:2], np.uint8)
            cv2.drawContours(cm, [t.contour], -1, 255, -1)
            a = float(np.count_nonzero(cm))
            if a <= 0:
                return False
            inter = float(np.count_nonzero(cv2.bitwise_and(occluder, cm)))
            return (inter / a) >= self.cover_overlap_ratio
        except Exception:                                          # noqa: BLE001
            return False

    def _prune(self) -> None:
        """清理轨迹。**只在 gone 且已过 gone_keep_seconds 之后**才真删。

        ★★ 为什么不能早删（用户问「如果零件回去了，该怎么识别」）：
          零件可能**被拿走又放回原位**。只要轨迹还在，回来的那块就能被
          _detect 按位置重新关联上，**身份与历史都恢复**。
          ★ 而且这也是**防幽灵的关键**：若回来的块不能关联到原轨迹，
            它会**新建一条** -> 同一零件两条轨迹 -> 界面上出现「零件 8」。
          实测踩过：原来超过宽限窗口就删，于是「拿走 -> 放回」就变成两条。
        """
        now = self._clock() if self._clock is not None else time.perf_counter()
        keep = []
        for t in self.tracks:
            # ★ 幻影：从未跟踪成功过的轨迹，活过短限就删（不让它白吃 KCF）
            # ★ 幻影：**已 gone + 未被遮挡 + 长时间没拿到检测结果**
            #   三个条件同时成立才删 —— 保证“挡一下就能恢复”的轨迹不会被误删。
            # ★ 逐帧检测模式的已确认身份可能只是被拿离画面，不能按帧数当幻影删。
            # 保留其lost/gone历史，直到下面的gone_keep_seconds按真实时间到期。
            retained_identity = self.backend == "detection" and t.ever_seen and bool(t.part_id)
            if not retained_identity and not t.covered and t.miss_streak > self.phantom_miss_frames:
                self._trackers.pop(id(t), None)
                self._kf.pop(id(t), None)
                self._frame_ref.pop(id(t), None)
                continue
            if t.state == "gone":
                if t.gone_since <= 0.0:
                    t.gone_since = now
                    keep.append(t)
                    continue
                if now - t.gone_since > self.gone_keep_seconds:
                    self._trackers.pop(id(t), None)
                    self._kf.pop(id(t), None)
                    self._frame_ref.pop(id(t), None)
                    continue
            else:
                t.gone_since = 0.0
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
                 "occluded": (0, 165, 255),      # 橙：被遮（含**完全被遮**，身份冻结）
                 "lost": (60, 120, 255),         # 橙红：久不可见，但还留着（手移开能恢复）
                 "gone": (90, 90, 220)}          # 红：已放弃
        for t in self.tracks:
            col = COLOR.get(t.state, (200, 200, 200))
            # ★★ 幻影：从来没真正跟住过的轨迹（`miss_streak == 0` 且从未跟丢）
            #   **完全不画** —— 它本身就是噪声，画出来就是用户看到的
            #   「黑色料盘外的全图搜 0 帧」那个橙色标记。★ 用户明确不要看到它。
            #   ★ 用 	racked_once（追踪器从未成功过）而不是看计数器 ——
            #     这才是“从噪声里生出来”的准确定义。
            #   实测踩过：不能用“从未被匹配过”（轨迹就是检测建的），
            #   也不能看 `tracked_once`（KCF 在纯灰背景上也返回 True）。
            #   用 `miss_streak == 0`：这一帧刚诞生、还没跑过追踪器 —— 正是用户看到的
            #   「全图搜 0 帧」那个橙色标记。下一帧它就会被判定，不再是幻影。
            phantom = (t.miss_streak == 0 and t.lost_frames == 0)
            if phantom:
                continue
            # ★★ 没有可见部分的轨迹（`visible_ratio == 0`）**不要画轮廓** ——
            #   那个 `contour` 是**旧的**，画出来是一个形状扭曲、位置在料盘外的
            #   "**幽灵**"，比不画还误导（实测用户直接问"这是什么东西"）。
            #   -> 只画一个十字 + 文字，明确表示"它曾在这、现在看不到"。
            solid = t.visible_ratio > 0.02
            if solid and t.contour is not None and len(t.contour) >= 3:
                cv2.drawContours(canvas, [t.contour], -1, col, 2)
                # 轮廓顶点（形状特征的直观体现）
                for p in t.contour.reshape(-1, 2):
                    cv2.circle(canvas, (int(p[0]), int(p[1])), 1, col, -1)
            if show_bbox and solid:
                x, y, w, h = t.bbox
                cv2.rectangle(canvas, (x, y), (x + w, y + h), col, 1)
            # 质心：用 `last_centroid`（最后一次真看到的位置）比卡尔曼预测更可信
            lx, ly = t.last_centroid if t.ever_seen else t.centroid
            cx, cy = int(lx), int(ly)
            if solid:
                cv2.drawMarker(canvas, (cx, cy), col, cv2.MARKER_CROSS, 10, 2)
            else:
                # 空心的圈 + 放射线，一眼看出"这是记忆位置，不是当前观测"
                cv2.circle(canvas, (cx, cy), 12, col, 1)
                for ang in range(0, 360, 90):
                    a = np.radians(ang)
                    cv2.line(canvas, (cx, cy),
                             (int(cx + 18 * np.cos(a)), int(cy + 18 * np.sin(a))), col, 1)
            cv2.putText(canvas, t.label(), (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def identity_relink_pairs(tracks, candidates, min_score, area_ratio):
    """单件身份的保守重关联；输入仅含身份/分数/面积，不依赖图像或位置。

    tracks为(part_id, area_ref)，candidates为(part_id, score, area)。
    同身份多轨迹或多候选均不猜；部分可见或尺寸突变也留给原空间匹配。
    """
    track_ids, candidate_ids = {}, {}
    for index, (pid, _) in enumerate(tracks):
        if pid:
            track_ids.setdefault(pid, []).append(index)
    for index, (pid, _, _) in enumerate(candidates):
        if pid:
            candidate_ids.setdefault(pid, []).append(index)
    out = []
    for pid, indices in track_ids.items():
        found = candidate_ids.get(pid, [])
        if len(indices) != 1 or len(found) != 1:
            continue
        ti, bi = indices[0], found[0]
        _, reference = tracks[ti]
        _, score, area = candidates[bi]
        if not all(np.isfinite(x) for x in (reference, score, area)) or reference <= 0:
            continue
        if score >= min_score and area_ratio[0] <= area / reference <= area_ratio[1]:
            out.append((ti, bi))
    return out


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
