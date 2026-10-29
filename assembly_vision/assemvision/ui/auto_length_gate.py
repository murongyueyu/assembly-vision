"""F检测框联调：只负责稳定两秒触发，不改变长度算法或质量门槛。

2026-10-29 手抖容忍调整（用户要求“检测稳定不要太严”）：
- 计时锚点允许更大的中心/轮廓长度漂移，轻微手抖不再从零开始；
- 2 秒保持与最少有效帧数不变；
- 明显移动、空框、真正断帧仍然撤销计时；
- 短暂丢失只冻结计时，不把丢失的时间累计成保持时长。
"""
import math


class PlacementGate:
    def __init__(self, settings):
        self.settings = settings
        self.hold = float(settings['hold_seconds'])
        self.gap = float(settings['max_frame_gap_seconds'])
        self.min_frames = int(settings['min_frames'])
        self.max_lost = float(settings.get('max_lost_seconds', 0.0))
        if (not all(math.isfinite(v) and v > 0 for v in (self.hold, self.gap))
                or self.min_frames < 2 or not math.isfinite(self.max_lost)
                or self.max_lost < 0):
            raise ValueError('稳定计时参数无效')
        self.reset()

    def reset(self):
        self.start = self.last = self.anchor = None
        self.last_frame = None
        self.frames = 0
        self.observed = 0.0
        self.lost_since = None
        self.triggered = False
        self.empty_since = None

    def _clear_attempt(self):
        self.start = self.last = self.anchor = None
        self.last_frame = None
        self.frames = 0
        self.observed = 0.0
        self.lost_since = None

    def _begin(self, now, metrics):
        self.start = self.last = self.last_frame = now
        self.anchor = metrics
        self.frames = 1
        self.observed = 0.0
        self.lost_since = None

    def _missing(self, now, prev_frame, empty):
        # 空框说明组件已移走；没有进行中的计时；或宽限为 0：立即撤销。
        if empty or self.anchor is None or self.last is None or self.max_lost <= 0:
            self._clear_attempt()
            return dict(color='yellow', elapsed=0., trigger=False, measured=False)
        # 真正断帧（帧间隔超限）不享受短暂丢失宽限。
        if prev_frame is not None and now-prev_frame > self.gap:
            self._clear_attempt()
            return dict(color='yellow', elapsed=0., trigger=False, measured=False)
        if self.lost_since is None:
            self.lost_since = now
        if now-self.lost_since > self.max_lost:
            self._clear_attempt()
            return dict(color='yellow', elapsed=0., trigger=False, measured=False)
        # 短暂丢失：冻结计时，不把这段空白算进保持时长。
        return dict(color='yellow', elapsed=self.observed, trigger=False,
                    measured=False, lost=True)

    def update(self, now, metrics=None, empty=False, hard=False):
        prev_frame = self.last_frame
        self.last_frame = now
        if self.triggered:
            if empty:
                if self.empty_since is None:
                    self.empty_since = now
                if now-self.empty_since+1e-9 >= float(self.settings['rearm_empty_seconds']):
                    self.reset()
            else:
                self.empty_since = None
            if self.triggered:
                return dict(color='green', elapsed=self.hold, trigger=False, measured=True)
        if hard:
            self._clear_attempt()
            return dict(color='yellow', elapsed=0., trigger=False, measured=False)
        if metrics is None:
            return self._missing(now, prev_frame, empty)
        metrics = tuple(float(v) for v in metrics)
        if not all(math.isfinite(v) for v in metrics):
            return self._missing(now, prev_frame, empty)
        limits = (float(self.settings['max_center_drift_px']),)*2 + (
            float(self.settings['max_length_drift_px']),)
        if self.anchor is None:
            self._begin(now, metrics)
        elif prev_frame is not None and now-prev_frame > self.gap:
            # 帧间隔过大：不能用旧帧续计时。
            self._begin(now, metrics)
        elif any(abs(a-b) > limit for a, b, limit in zip(metrics, self.anchor, limits)):
            # 明显移动：从零重新计时。
            self._begin(now, metrics)
        else:
            if self.lost_since is None:
                self.observed += max(0., now-self.last)
            self.last = now
            self.frames += 1
            self.lost_since = None
        fire = self.observed+1e-9 >= self.hold and self.frames >= self.min_frames
        if fire:
            self.triggered = True
        return dict(color='green', elapsed=self.observed, trigger=fire,
                    measured=self.triggered)
