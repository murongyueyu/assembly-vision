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

MODULES = ["test_features", "test_segment", "test_polar", "test_fsm",
           "test_part_id", "test_tray_pose", "test_work_roi", "test_assembly",
           "test_ui_renderer"]


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
