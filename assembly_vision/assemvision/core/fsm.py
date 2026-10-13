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
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..config import Config
from ..types import ERROR, IDLE, OK, WARN, Verdict


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
