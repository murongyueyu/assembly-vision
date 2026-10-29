# -*- coding: utf-8 -*-
"""流程判断测试：前置条件表 + 错误流程用例。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check, wp_config                        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.core.fsm import Debouncer, PreconditionNet  # noqa: E402


def _run(net: PreconditionNet, preds: dict, ts: float) -> str:
    v = net.step(preds, ts)
    return "%s|%s" % (v.level, v.title)


def main() -> int:
    wp = wp_config()
    print("== 前置条件表：配置的流程 ==")
    for op in wp.get("operations"):
        print("   %-5s %-14s pre=%-34s eff=%s"
              % (op["id"], op["name"], str(op.get("pre")), str(op.get("eff"))))

    print("\n== 用例 1：正确顺序 ==")
    net = PreconditionNet(wp)
    ts = 0.0
    seq = [
        ({"base_present": True}, 1.0),
        ({"base_present": True, "column_present": True}, 2.0),
        ({"base_present": True, "column_present": True, "wheel_present": True}, 3.0),
        ({"base_present": True, "column_present": True, "wheel_present": True,
          "cap_present": True}, 4.0),
    ]
    levels = []
    for preds, t in seq:
        v = net.step(preds, t)
        levels.append(v.level)
        print("   t=%.0f %s | %s" % (t, v.level, v.title))
    check(all(l == "ok" for l in levels), "正确顺序全程判 OK")
    check(net.all_done(), "四个操作全部完成")

    print("\n== 用例 2：跳步（跳过轮盘直接拧顶盖）==")
    net = PreconditionNet(wp)
    net.step({"base_present": True}, 1.0)
    net.step({"base_present": True, "column_present": True}, 2.0)
    v = net.step({"base_present": True, "column_present": True,
                  "cap_present": True}, 3.0)
    print("   %s | %s | %s" % (v.level, v.title, v.detail))
    check(v.level == "error", "跳过轮盘被判 ERROR")
    check(any("wheel_present" in d for d in v.detail), "明确指出缺失前置 wheel_present")

    print("\n== 用例 3：倒序（先装轮盘再装立柱）==")
    net = PreconditionNet(wp)
    net.step({"base_present": True}, 1.0)
    v = net.step({"base_present": True, "wheel_present": True}, 2.0)
    print("   %s | %s | %s" % (v.level, v.title, v.detail))
    check(v.level == "error", "倒序被判 ERROR")
    check(any("column_present" in d for d in v.detail), "明确指出缺失前置 column_present")

    print("\n== 用例 4：长时间无进展 → 黄灯等待 ==")
    net = PreconditionNet(wp)
    net.step({"base_present": True}, 1.0)
    v = net.step({"base_present": True}, 100.0)
    print("   %s | %s | %s" % (v.level, v.title, v.detail))
    check(v.level == "warn", "超时未操作判 WARN（黄灯）")

    print("\n== 用例 5：遮挡冻结时不更新状态 ==")
    net = PreconditionNet(wp)
    net.step({"base_present": True}, 1.0)
    before = set(net.done)
    v = net.step({"base_present": True, "column_present": True}, 2.0, frozen=True)
    print("   %s | %s" % (v.level, v.title))
    check(set(net.done) == before, "冻结时 done 集合不变")
    check("检测中" in v.title, "冻结时提示『检测中』")

    print("\n== 用例 6：可换序的写法（把两个 op 的 pre 写成同一个）==")
    wp2 = wp_config()
    ops = wp2.get("operations")
    ops[2]["pre"] = ["column_present"]
    ops[3]["pre"] = ["column_present"]
    net2 = PreconditionNet(wp2)
    net2.step({"base_present": True}, 1.0)
    net2.step({"base_present": True, "column_present": True}, 2.0)
    v_a = net2.step({"base_present": True, "column_present": True,
                     "wheel_present": True}, 3.0)
    v_b = net2.step({"base_present": True, "column_present": True,
                     "cap_present": True}, 4.0)
    print("   先轮盘后顶盖: %s | %s" % (v_a.level, v_a.title))
    print("   先顶盖后轮盘: %s | %s" % (v_b.level, v_b.title))
    check(v_a.level == "ok" and v_b.level == "ok", "两种顺序都判 OK（可换序）")

    print("\n== 用例 7：去抖（连续帧数 / 保持时间）==")
    db = Debouncer(n_frames=3)
    st = db.update({"x": True}, 0.0)
    check(st.get("x") is False, "第 1 帧还不算成立")
    db.update({"x": True}, 0.1)
    st = db.update({"x": True}, 0.2)
    check(st.get("x") is True, "第 3 帧才算成立")
    st = db.update({"x": False}, 0.3)
    check(st.get("x") is False, "一帧不成立立刻回落")
    db2 = Debouncer(hold_s=0.3)
    db2.update({"y": True}, 0.0)
    st = db2.update({"y": True}, 0.2)
    check(st.get("y") is False, "保持时间未到不算成立")
    st = db2.update({"y": True}, 0.35)
    check(st.get("y") is True, "保持时间到就算成立")

    print("\n流程判断测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
