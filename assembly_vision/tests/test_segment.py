# -*- coding: utf-8 -*-
"""分割 + 槽位覆盖率（要求①的核心判据）离线测试。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check, make_env                     # noqa: E402
from make_synthetic import SLOTS, render_scene           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assemvision.core.segment import glove_hits          # noqa: E402


def main() -> int:
    wp, tray, seg, q = make_env()
    print("== 场景 1：全部就位 ==")
    img = render_scene()
    res = seg.segment(img)
    slots = q.check_slots(res)
    covs = {s.slot_id: s.coverage for s in slots}
    print("   覆盖率:", {k: round(v, 3) for k, v in covs.items()})
    check(all(c > 0.45 for c in covs.values()), "四个槽都判为『有件』")
    check(all(s.verdict == "OK" for s in slots), "四个槽判定都是 OK")

    print("== 场景 2：全部拿掉 ==")
    img0 = render_scene(missing=set(SLOTS.keys()))
    res0 = seg.segment(img0)
    slots0 = q.check_slots(res0)
    covs0 = {s.slot_id: s.coverage for s in slots0}
    print("   覆盖率:", {k: round(v, 3) for k, v in covs0.items()})
    check(all(c < 0.15 for c in covs0.values()), "四个槽都判为『缺件』")
    check(all(s.verdict == "NG" for s in slots0), "四个槽判定都是 NG")

    print("== 场景 3：只缺顶盖 ==")
    img1 = render_scene(missing={"slot_cap"})
    res1 = seg.segment(img1)
    m = {s.slot_id: s.verdict for s in q.check_slots(res1)}
    print("   判定:", m)
    check(m["slot_cap"] == "NG", "顶盖槽判 NG")
    check(m["slot_wheel"] == "OK" and m["slot_base"] == "OK"
          and m["slot_column"] == "OK", "其余三槽仍判 OK")

    print("== 场景 4：蓝手套压在轮盘上 ==")
    img2 = render_scene(glove_at=(120.0, 40.0))
    res2 = seg.segment(img2)
    check(res2.glove_bbox is not None, "检测到手套")
    hits = glove_hits(res2.glove_mask, tray.slot_polygons_px(1.0))
    print("   hand_on:", hits)
    check(hits.get("slot_wheel", False), "hand_on(slot_wheel) 为真")

    print("\n分割与覆盖率测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
