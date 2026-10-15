# -*- coding: utf-8 -*-
r"""Verify the glove fix: the hand in the user's screenshot must now be detected.

ASCII-only on every line containing quotes, because this session has hit the
"Chinese string + ASCII quote" SyntaxError six times (see AGENTS.md §3（改代码时的约定）).
"""
import os
import sys

import numpy as np

ROOT = r"D:\university resource\大三上\视觉\assembly_vision"
sys.path.insert(0, ROOT)

import cv2                                                        # noqa: E402
from assemvision.config import load_config                        # noqa: E402
from assemvision.core.checks import Qualifier                     # noqa: E402
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter, glove_hits        # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402

wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
cal = load_config(os.path.join(ROOT, "config", "calibration.yaml"))
tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
               ref_scale=float(cal.get("tray_map.scale", 1.0)))
q = Qualifier(wp, tray, cal)
seg = Segmenter(wp)
mmpp = tray.mm_per_px(1.0)

polys = {}
for s in (wp.get("slots") or []):
    p = tray.slot_polygon_px(s["id"], 1.0)
    if p is not None:
        polys[s["id"]] = p

g = dict((wp.get("segment", {}) or {}).get("glove", {}) or {})
print("  glove: morph_close=%s  min_area_px=%s" % (g.get("morph_close"),
                                                   g.get("min_area_px")))
print()

img = imread_safe(os.path.join(ROOT, "data", "refs", "glove_miss.png"))
img = cv2.resize(img, (640, int(round(img.shape[0] * 640.0 / img.shape[1]))),
                 interpolation=cv2.INTER_AREA)
roi = q.work_roi_mask(img.shape, 1.0, margin_mm=25.0)
r = seg.segment(img, scale=1.0, work_roi=roi)
hits = glove_hits(r.glove_mask, polys)
pressed = [k.replace("slot_", "") for k, v in hits.items() if v]
print("  HAND FRAME (user screenshot, resized to 640 wide)")
print("     glove_area = %d px   glove_bbox = %s" % (r.glove_area_px, r.glove_bbox))
print("     glove_mm2  = %.0f" % (r.glove_area_px * mmpp * mmpp))
print("     pressed_on = %s" % (pressed or "none"))
ok_hand = r.glove_bbox is not None
print("     -> %s" % ("DETECTED (fix works)" if ok_hand else "STILL MISSED"))
print()

print("  CLEAN FRAMES (no hand; all must be 0)")
worst = 0
for n in ("assembly_choice.png", "now_raw.png", "tray_empty.png",
          "slot_check.png", "asm_raw.png", "assembly_left_check.png"):
    p = os.path.join(ROOT, "data", "refs", n)
    im = imread_safe(p)
    if im is None:
        continue
    rr = seg.segment(im, scale=1.0, work_roi=roi)
    worst = max(worst, int(rr.glove_area_px))
    print("     %-24s glove_area=%d  bbox=%s" % (n, rr.glove_area_px, rr.glove_bbox))
print()
print("  RESULT: hand frame -> %s ; worst clean frame -> %d px (gate = %s)"
      % ("OK" if ok_hand else "MISS", worst, g.get("min_area_px")))
print("  %s" % ("PASS: detects the hand and no false positives"
                if ok_hand and worst == 0 else "FAIL"))
