# -*- coding: utf-8 -*-
"""流程判断（要求③）：装配状态谓词 + 前置条件表。

设计要点
  * 不做动作识别，只做「装配状态观测 + 前置条件表」
  * 合法性完全由 pre 集合表达：把两个 op 的 pre 写成同一个，它们就自动可互换
  * 去抖：连续 N 帧成立才算状态改变
  * 遮挡冻结：手套压在该 ROI 上时冻结对应观测量，防止"手一晃就报错"
"""
from __future__ import annotations

import time
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..config import Config
from ..types import ERROR, IDLE, OK, WARN, Verdict


class PickSequence:
    """E显式顺序推定；不修改旧谓词FSM，不把取件称为实际安装验证。"""

    def __init__(self, wp_cfg, settings):
        from .events import ConfirmedPickDetector
        self.ops = list(wp_cfg.get('operations', []))
        self.expected = {s['id']: s['part'] for s in wp_cfg.get('slots', [])}
        bindings = settings['operations']
        if (len(self.ops) != 4 or len(self.expected) != 4
                or set(bindings) != {op['id'] for op in self.ops}
                or set(bindings.values()) != set(self.expected)):
            raise ValueError('E必须一对一绑定实际四工序与四槽')
        self.order = [bindings[op['id']] for op in self.ops]
        effects = set()
        for op in self.ops:
            if not set(op.get('pre', [])) <= effects or not op.get('eff'):
                raise ValueError('E顺序不符合实际工序前置条件')
            effects.update(op['eff'])
        self.detector = ConfirmedPickDetector(self.expected, settings['confirmation'])
        self.tracking = None
        if 'tracking_departure' in settings:
            from .events import TrackingSlotEvidence
            self.tracking = TrackingSlotEvidence(self.expected, settings['tracking_departure'], self.detector)
        self.timeout = float(settings['timeout_s'])
        if not self.timeout > 0 or not math.isfinite(self.timeout):
            raise ValueError('E等待时间必须为有限正数')
        self.reset()

    def reset(self):
        self.started = False
        self.done = []
        self.events = []
        self.last_progress = None
        self.detector.reset()
        if self.tracking is not None:
            self.tracking.reset()
        self.verdict = self._view(WARN, 'E：等待D四槽匹配放行')

    def _view(self, level, title, detail=None):
        lines = list(detail or [])
        if len(self.done) < len(self.ops):
            op = self.ops[len(self.done)]
            lines.append('下一步取件：' + op['name'])
        lines.extend(['进度仅按取件顺序推定', '未验证安装、紧固或高度'])
        return Verdict(level=level, title=title, detail=lines,
                       step=len(self.done), total=len(self.ops))

    def interrupt(self, reason):
        self.detector.interrupt()
        if self.tracking is not None:
            self.tracking.reset()
        self.verdict = self._view(WARN, 'E：暂停，等待可靠当前观测', [reason])
        return self.verdict

    def update(self, result, now, tracking=None):
        if not self.started:
            if not result.allow_start:
                self.verdict = self._view(ERROR if result.state == 'alarm' else WARN,
                                          'E：等待D四槽匹配放行', [result.reason] if result.reason else [])
                return self.verdict
            # 放行来自D稳定门禁；仍验证当前来源有效，不能用历史allow_start启动。
            self.detector.update(result, now)
            if not result.valid or not 0 <= now-result.stamp.captured_at <= self.detector.max_age:
                return self.interrupt('开工信号过期或无效')
            if (set(result.slots) != set(self.expected)
                    or any(row.state != 'match' for row in result.slots.values())):
                return self.interrupt('开工四槽证据不完整')
            self.started = True
            if self.tracking is not None:
                projected = self.tracking.project(result, now, tracking, [])
                if not projected.valid or any(row.state != 'match' for row in projected.slots.values()):
                    self.started = False
                    return self.interrupt('开工需同帧四件当前可见追踪')
            self.last_progress = result.stamp.captured_at
            self.detector.stable = {sid: 'match' for sid in self.expected}
            self.verdict = self._view(OK, 'E：已开工，请按顺序取件')
            return self.verdict
        if self.tracking is not None:
            result = self.tracking.project(result, now, tracking, self.done)
        events = self.detector.update(result, now)
        if self.tracking is not None:
            for event in events:
                event.detail['evidence'] = 'current_tracking'
                if event.slot in self.tracking.departed:
                    event.detail['departure'] = dict(self.tracking.departed[event.slot])
        self.events = (self.events + events)[-100:]
        if not result.valid or not 0 <= now-result.stamp.captured_at <= self.detector.max_age:
            return self.interrupt(result.reason or '观测失效')
        # 正确放回即可撤销该步和后续推定，即使别槽仍遮挡也不保留虚假完成。
        returned = [i for i, sid in enumerate(self.done)
                    if self.detector.stable.get(sid) == 'match' and result.slots[sid].state == 'match']
        if returned:
            index = min(returned)
            self.done = self.done[:index]
            self.detector.interrupt()
            self.last_progress = result.stamp.captured_at
            self.verdict = self._view(WARN, 'E：放回零件，已回退工序', ['后续零件须放回正确槽位重新确认'])
            return self.verdict
        wrong = [sid for sid, row in result.slots.items() if row.state == 'wrong']
        if wrong:
            self.verdict = self._view(ERROR, 'E：槽内错件，暂停推进', wrong)
            return self.verdict
        if not self.detector.settled:
            self.verdict = self._view(WARN, 'E：遮挡、未知或取放待确认')
            return self.verdict
        removed = [sid for sid in self.order if sid not in self.done
                   and self.detector.stable.get(sid) == 'missing']
        if removed:
            target = self.order[len(self.done)]
            if len(removed) != 1 or removed[0] != target:
                self.verdict = self._view(ERROR, 'E：越序或多件取走，不推进', ['请将未确认取件放回正确槽位后重试'])
            elif target not in self.detector.picked:
                self.verdict = self._view(WARN, 'E：缺少连续取件证据', ['请放回正确槽位，确认后重新取件'])
            else:
                self.done.append(target)
                self.last_progress = result.stamp.captured_at
                self.verdict = self._view(OK, 'E：按取件顺序推定完成' if len(self.done) == 4
                                          else 'E：已确认取件 %d/4' % len(self.done))
            return self.verdict
        if len(self.done) == 4:
            self.verdict = self._view(OK, 'E：按取件顺序推定完成')
        else:
            elapsed = result.stamp.captured_at - self.last_progress
            self.verdict = self._view(WARN if elapsed > self.timeout else IDLE,
                                      'E：等待下一步取件', ['已等待 %.1f 秒' % elapsed])
        return self.verdict


class Debouncer:
    """观测量去抖：连续 n 帧成立才认为状态真的变了。

    ★ 用『时间』而不是『帧数』做判据更稳 —— 帧率一变，帧数阈值就悄悄失效了。
    """

    def __init__(self, n_frames: int = 6, hold_s: Optional[float] = None) -> None:
        self.n = int(max(1, n_frames))
        self.hold_s = hold_s
        self._count: Dict[str, int] = {}
        self._since: Dict[str, float] = {}
        self._state: Dict[str, bool] = {}

    def update(self, values: Dict[str, bool], ts: float) -> Dict[str, bool]:
        for k, v in values.items():
            if v:
                self._count[k] = self._count.get(k, 0) + 1
                self._since.setdefault(k, ts)
            else:
                self._count[k] = 0
                self._since.pop(k, None)
                self._state[k] = False

            if self.hold_s is not None:
                held = (ts - self._since.get(k, ts)) >= float(self.hold_s)
                self._state[k] = bool(v) and held
            else:
                self._state[k] = self._count.get(k, 0) >= self.n
        # 补齐本次没出现的键（保持上次状态，避免误清）
        for k in list(self._state.keys()):
            if k in values:
                continue
        return dict(self._state)

    def reset(self) -> None:
        self._count.clear()
        self._since.clear()
        self._state.clear()


class PreconditionNet:
    """前置条件表引擎。

    installed : 已达成效果的集合（= 装配状态记忆）
    done      : 已判定的操作 id
    """

    def __init__(self, wp_cfg: Config, timeout_s: float = 15.0) -> None:
        self.ops: List[dict] = list(wp_cfg.get("operations", []) or [])
        self.total: int = len(self.ops)
        self.timeout_s = float(timeout_s)
        self.installed: Set[str] = set()
        self.done: Set[str] = set()
        self.last_progress_ts: float = time.time()

    # ---------- 查询 ----------
    def enabled(self) -> List[dict]:
        """当前可执行的操作集合。合法性就由这个集合决定。"""
        return [op for op in self.ops
                if op["id"] not in self.done and set(op.get("pre", [])) <= self.installed]

    def next_op(self) -> Optional[dict]:
        en = self.enabled()
        if not en:
            return None
        return sorted(en, key=lambda o: len(o.get("pre", []) or []))[0]

    def all_done(self) -> bool:
        return len(self.done) == self.total

    def reset(self) -> None:
        self.installed.clear()
        self.done.clear()
        self.last_progress_ts = time.time()

    # ---------- 主循环 ----------
    def step(self, preds: Dict[str, bool], ts: float,
             frozen: bool = False) -> Verdict:
        """喂入（已去抖的）谓词，返回给界面看的结论。"""
        if self.total == 0:
            return Verdict(level=IDLE, title="未配置装配流程", step=0, total=0)

        if frozen:
            # 遮挡冻结：不更新任何状态，但把"正在检测"告诉界面
            return Verdict(level=WARN, title="检测中…（手遮住了工件）",
                           detail=["等手套离开画面后再判定"],
                           step=len(self.done), total=self.total)

        # 找出「效果全部达成但尚未记录」的操作
        for op in self.ops:
            if op["id"] in self.done:
                continue
            eff = list(op.get("eff", []) or [])
            if not eff or not all(preds.get(e, False) for e in eff):
                continue

            missing = sorted(set(op.get("pre", []) or []) - self.installed)
            if missing:
                # 顺序错误：记录但标记违规，并明确说出缺哪一条前置
                self.done.add(op["id"])
                self.installed.update(eff)
                self.last_progress_ts = ts
                nxt = self.next_op()
                detail = ["缺失前置：%s" % "、".join(missing)]
                if nxt:
                    detail.append("当前应做：%s" % nxt.get("prompt", nxt["name"]))
                return Verdict(level=ERROR,
                               title="× 顺序错误：%s" % op["name"],
                               detail=detail,
                               step=len(self.done), total=self.total)

            self.done.add(op["id"])
            self.installed.update(eff)
            self.last_progress_ts = ts
            nxt = self.next_op()
            detail = ["已安装：%s" % "、".join(sorted(self.installed))]
            if nxt:
                detail.append("下一步：%s" % nxt.get("prompt", nxt["name"]))
            else:
                detail.append("全部步骤已完成，开始终检")
            return Verdict(level=OK,
                           title="√ %d/%d  %s"
                                 % (len(self.done), self.total, op["name"]),
                           detail=detail,
                           step=len(self.done), total=self.total)

        if self.all_done():
            return Verdict(level=OK, title="√ 装配流程全部完成",
                           detail=["正在进行齐套性检查"],
                           step=self.total, total=self.total)

        if (ts - self.last_progress_ts) > self.timeout_s:
            nxt = self.next_op()
            if nxt:
                return Verdict(level=WARN,
                               title="等待：%s" % nxt.get("prompt", nxt["name"]),
                               detail=["已等待 %.0f 秒" % (ts - self.last_progress_ts)],
                               step=len(self.done), total=self.total)

        nxt = self.next_op()
        return Verdict(level=IDLE,
                       title=nxt.get("prompt", nxt["name"]) if nxt else "等待",
                       detail=[], step=len(self.done), total=self.total)


def evaluate_predicates(preds_cfg: Dict[str, str],
                        ctx: Dict[str, object]) -> Dict[str, bool]:
    """把 config 里的谓词表达式求值成布尔。

    表达式是可控的小语法（slot_present('x') / column_visible() / a or b），
    用 eval 但限定内置函数，避免依赖与安全隐患。
    """
    env: Dict[str, object] = dict(ctx)
    env.setdefault("__builtins__", {})
    out: Dict[str, bool] = {}
    for k, expr in (preds_cfg or {}).items():
        try:
            out[k] = bool(eval(str(expr), env))  # noqa: S307 - 表达式来自本项目的配置文件
        except Exception:  # noqa: BLE001
            out[k] = False
    return out
