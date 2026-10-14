# -*- coding: utf-8 -*-
"""色域分割的**两件真正的用途** —— 按用户澄清的问题边界来测。

用户澄清（这条必须记进 AGENTS.md）：
    色域分割的真正用途是
      ① **区分零件和背景**（黄铜 vs 绿幕/木桌）
      ② **区分零件和手套**（黄铜 vs 蓝手套）
    **不是**区分零件与零件（四个零件是同一种黄铜）。

本工具测三件事，每件都给"有它 / 没它"的对照：
  A. 零件 vs 背景：`brass.s_max` 拿掉会怎样
  B. 零件 vs 手套：手套压住槽时 `glove_hits` 能不能报出来
  C. `metal` 掩膜（数辐条用）会不会被蓝手套污染
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from assemvision.config import load_config                        # noqa: E402
from assemvision.core.checks import Qualifier                     # noqa: E402
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter, glove_hits        # noqa: E402


def main() -> int:
    from make_synthetic import render_scene                       # noqa: E402

    wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
    cal = load_config(os.path.join(ROOT, "config", "calibration.synthetic.yaml"))
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    q = Qualifier(wp, tray, cal)
    seg = Segmenter(wp)
    mmpp = tray.mm_per_px(1.0)
    brass = dict((wp.get("segment", {}) or {}).get("brass", {}) or {})

    print("═" * 80)
    print("  色域分割的两件真正用途（零件 vs 背景 / 零件 vs 手套）")
    print("═" * 80)
    print("  当前 brass 阈值: h %s~%s   s %s~%s   v>=%s"
          % (brass.get("h_min"), brass.get("h_max"),
             brass.get("s_min"), brass.get("s_max"), brass.get("v_min")))
    print("  判据: 黄铜靠 **h 区间** 定位色相 + **s 区间** 分开绿幕(≈255)与高光(≈20)")
    print()

    # ── A. 零件 vs 背景：s_max 的作用 ──────────────────────────
    print("  ── A. 零件 vs 背景：`s_max` 是不是必需？ ──")
    print("  %-24s %-11s %-14s %s" % ("配置", "黄铜总px", "最大块mm²", "说明"))
    ref = None
    for label, smax in (("s_max=180（现值）", 180), ("s_max=255（等于不设）", 255),
                        ("s_max=90（过严）", 90)):
        d = dict(wp.data)                                     # ★ Config 是 .data
        d["segment"] = dict(d.get("segment", {}) or {})
        d["segment"]["brass"] = dict(d["segment"].get("brass", {}) or {})
        d["segment"]["brass"]["s_max"] = smax
        s = Segmenter(d)
        r = s.segment(render_scene(), scale=1.0)
        tot = int(np.count_nonzero(r.brass_mask))
        big = max((b.area_px for b in r.brass_blobs), default=0) * mmpp * mmpp
        if ref is None:
            ref = tot
        note = ""
        if smax == 255:
            note = ("绿幕 S≈255 灌进来 -> 翻 %+.0f%%" % (100.0 * (tot - ref) / max(1, ref))
                    if abs(tot - ref) > 0.05 * ref else
                    "(本合成图绿幕未达 S 上限；真机上会灌)")
        elif smax == 90:
            note = "削掉零件自身高光（实测零件 S 上限 109）"
        print("  %-24s %-11d %-14.0f %s" % (label, tot, big, note))

    # ── B. 零件 vs 手套 ────────────────────────────────────────
    print()
    print("  ── B. 零件 vs 手套：手套压住槽时能不能报出来？ ──")
    polys = {}
    for s in (wp.get("slots") or []):
        p = tray.slot_polygon_px(s["id"], 1.0)
        if p is not None:
            polys[s["id"]] = p
    print("  %-16s %-9s %-10s %-34s %s"
          % ("场景", "手套px", "轮盘覆盖", "glove_hits（各槽是否被手压）", "轮盘判定"))
    for label, gkw in (("无手套", dict()),
                       ("手套压轮盘", dict(glove_at=(24.0, 53.0))),
                       ("手套压立柱", dict(glove_at=(105.0, 54.0)))):
        img = render_scene(**gkw)
        res = seg.segment(img, scale=1.0)
        gpx = int(np.count_nonzero(res.glove_mask)) if res.glove_mask is not None else 0
        hits = glove_hits(res.glove_mask, polys, 0.10)
        slots = q.check_slots(res, 1.0, bgr=img)
        sw = next((s for s in slots if s.slot_id == "slot_wheel"), None)
        hit_txt = " ".join("%s=%s" % (k.replace("slot_", ""), "手" if v else "-")
                           for k, v in hits.items())
        print("  %-16s %-9d %-10s %-34s %s"
              % (label, gpx, "%.2f" % (sw.coverage if sw else 0),
                 hit_txt, sw.verdict if sw else "-"))
    print("     ★ 判据：手套压住哪个槽，`glove_hits` 就该在那个槽为 True；")
    print("       上层据此**冻结该观测量**（fsm 的『遮挡冻结』），避免「手一晃就报缺件」。")

    # ── C. metal 掩膜 vs 蓝手套 ────────────────────────────────
    print()
    print("  ── C. `metal` 掩膜（数辐条用）会不会被蓝手套污染？ ──")
    m = dict((wp.get("segment", {}) or {}).get("metal", {}) or {})
    print("     判据: metal = (V>=%s) & (S<=%s) & (~green[h %s~%s & S>%s])"
          % (m.get("v_min"), m.get("s_max"), m.get("green_h_min"),
             m.get("green_h_max"), m.get("green_s_min")))
    res = seg.segment(render_scene(glove_at=(24.0, 53.0)), scale=1.0)
    if res.glove_mask is not None and res.metal_mask is not None:
        gpx = int(np.count_nonzero(res.glove_mask))
        inter = int(np.count_nonzero(
            np.logical_and(res.glove_mask > 0, res.metal_mask > 0)))
        print("     手套像素 %d，其中被判成 metal 的 %d（%.1f%%）-> %s"
              % (gpx, inter, 100.0 * inter / max(1, gpx),
                 "✓ 手套不污染数辐条" if inter < 0.05 * max(1, gpx)
                 else "★ 会污染！"))
        print("     依据：蓝手套 S≈134~246 **远超** s_max=%s -> 被挡掉；"
              % m.get("s_max"))
        print("           且蓝手套 H≈92~99 不在 green 区间(%s~%s) -> "
              "**不是靠 green 挡的，是靠 S 上限**"
              % (m.get("green_h_min"), m.get("green_h_max")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
