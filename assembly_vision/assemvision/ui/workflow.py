"""六步显示编排；不改变E四次取件或B测量判据。"""
from __future__ import annotations

import math
import time

LABELS = ('底座', '立柱', '长度检测', '轮盘', '顶盖', '顶盖深度检测')
ACTIONS = ('请取底座', '请取立柱', '按 I 自动测量立柱长度', '请取轮盘', '请取顶盖', '按 I 检测顶盖高度')


def duration_text(seconds):
    if seconds is None:
        return '--:--'
    ticks = max(0, int(seconds * 10))
    minutes, rem = divmod(ticks, 600)
    if minutes >= 60:
        return '%02d:%02d:%02d' % (minutes // 60, minutes % 60, rem // 10)
    return '%02d:%02d.%d' % (minutes, rem // 10, rem % 10)


class SixStepWorkflow:
    def __init__(self, settings, clock=time.monotonic):
        self.clock = clock
        self.hold = float(settings.get('completion_seconds', 5.5))
        if not math.isfinite(self.hold) or not 5 <= self.hold <= 6:
            raise ValueError('完成提示时长必须在5～6秒')
        self.reset()

    def reset(self):
        self.generation = None
        self.started_at = self.finished_at = None
        self.count = 0
        self.complete_serial = 0
        self.announced = False
        self.last_checks = (False, False)

    def update(self, count, started, results, generation, level='warn', now=None):
        now = self.clock() if now is None else now
        if self.generation != generation:
            self.reset()
            self.generation = generation
        count = max(0, min(4, int(count)))
        # ★ 放回立柱/顶盖后，旧测量不再证明当前装配状态。
        if count < self.count:
            if count < 2:
                results.pop('column', None)
            if count < 4:
                results.pop('cap', None)
            self.finished_at = None
        self.count = count
        if started and self.started_at is None:
            self.started_at = now

        def passed(name, step):
            r = results.get(name, {})
            return (r.get('generation') == generation and r.get('step') == step
                    and r.get('ok') is True and r.get('height_conforms') is True)

        length_ok, cap_ok = passed('column', 2), passed('cap', 4)
        facts = (count >= 1, count >= 2, length_ok, count >= 3, count >= 4, cap_ok)
        completed = 0
        for fact in facts:
            if not fact:
                break
            completed += 1
        ready = started and completed == 6
        if ready and self.finished_at is None and level != 'error':
            self.finished_at = now
            if not self.announced:
                self.complete_serial += 1
                self.announced = True
        if not ready:
            self.finished_at = None
        active = min(completed, 5)
        statuses = ['done' if i < completed else 'current' if i == completed else 'pending'
                    for i in range(6)]
        check = results.get('column' if active == 2 else 'cap' if active == 5 else '', {})
        if check and (not check.get('ok') or check.get('height_conforms') is False):
            statuses[active] = 'failed'
        elapsed = None if self.started_at is None else max(
            0., (self.finished_at if self.finished_at is not None else now) - self.started_at)
        return dict(labels=LABELS, statuses=tuple(statuses), completed=completed,
                    action=('流程异常，请处理告警' if level == 'error' else
                            ('本轮检测完成' if ready else ACTIONS[active])) if started else '四件归位，等待开工',
                    elapsed=duration_text(elapsed), complete=ready and level != 'error',
                    completion_id=(generation, self.complete_serial),
                    completion_seconds=self.hold, generation=generation,
                    early_pick=count > 2 and not length_ok,
                    measurement=check, fault=level == 'error')
