# -*- coding: utf-8 -*-
"""一次跑完所有离线测试（不需要相机、不需要支架）。

用法：
    python tests/run_all.py
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

# ★★ 控制台编码兜底（2026-10 实测踩过，会伪装成"代码坏了"）：
#   本机默认控制台是 **GBK**，而测试输出里用了 `✓`(U+2713) / `²`(U+00B2) /
#   `Ø`(U+00D8) —— GBK 里没有这几个字符，`print()` 会直接抛 UnicodeEncodeError，
#   于是 test_part_id / test_tray_pose / test_assembly 三个模块**整块假失败**（exit 1），
#   而它们其实全过了。实测：默认控制台 exit 1，设 PYTHONIOENCODING=utf-8 后 12/12。
#   ★ 这里只把"编不出的字符"降级成 `?`（errors="replace"），**不改编码** ——
#     改成 utf-8 反而会让 GBK 控制台满屏乱码。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")                  # type: ignore[attr-defined]
    except Exception:                                          # noqa: BLE001
        pass

MODULES = ["test_features", "test_segment", "test_d435_source", "test_polar",
           "test_fsm", "test_part_id", "test_part_id_tol", "test_tray_pose",
           "test_work_roi", "test_size_gate", "test_assembly", "test_ui_renderer",
           "test_tracker"]


def main() -> int:
    failed = []
    for name in MODULES:
        print("\n" + "=" * 74)
        print("### %s" % name)
        print("=" * 74)
        try:
            mod = importlib.import_module(name)
            mod.main()
        except Exception:                      # noqa: BLE001
            failed.append(name)
            traceback.print_exc()
    print("\n" + "=" * 74)
    if failed:
        print("失败: %s" % ", ".join(failed))
        return 1
    print("全部通过 ✓  （%d 个测试模块）" % len(MODULES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
