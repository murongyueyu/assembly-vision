# -*- coding: utf-8 -*-
r"""Measure the glove colour-gamut headroom.

The user's point: the glove IS clearly blue, so framing the hand is a
colour-gamut segmentation job.  Correct -- `_mask(hsv, glove)` already is exactly
that.  The question this tool answers is: how much margin does the current
hue/sat/val band have, i.e. can we widen it to survive lighting drift?

We measure, on the real hand frame:
  1. the glove's own H / S / V distribution (where is the mass?)
  2. the H / S / V distribution of everything ELSE in the frame
     (what would we catch if we widened the band?)
  3. for a sweep of widened bands: glove pixels kept vs false positives
     (false positives measured on 6 clean no-hand frames)

ASCII-only on lines containing quotes.
"""
import os
import sys

import numpy as np

ROOT = r"D:\university resource\大三上\视觉\assembly_vision"
sys.path.insert(0, ROOT)

import cv2                                                        # noqa: E402
from assemvision.config import load_config                        # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402

wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
g = dict((wp.get("segment", {}) or {}).get("glove", {}) or {})
print("  current glove band: h %s~%s  s>=%s  v>=%s"
      % (g["h_min"], g["h_max"], g["s_min"], g["v_min"]))


def load640(name):
    img = imread_safe(os.path.join(ROOT, "data", "refs", name))
    if img is None:
        return None
    if img.shape[1] != 640:
        img = cv2.resize(img, (640, int(round(img.shape[0] * 640.0 / img.shape[1]))),
                         interpolation=cv2.INTER_AREA)
    return img


hand = load640("glove_miss.png")
CLEAN = [x for x in (load640(n) for n in (
    "assembly_choice.png", "now_raw.png", "tray_empty.png",
    "slot_check.png", "asm_raw.png", "assembly_left_check.png")) if x is not None]


def hsv_of(img):
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return h[:, :, 0].astype(int), h[:, :, 1].astype(int), h[:, :, 2].astype(int)


print()
print("=" * 84)
print("  1. Glove pixels: where is the colour mass? (loose band h 85~140 s>=50 v>=60)")
print("=" * 84)
H, S, V = hsv_of(hand)
loose = (H >= 85) & (H <= 140) & (S >= 50) & (V >= 60)
print("     loose-band pixel count = %d" % int(loose.sum()))
if loose.sum():
    for nm, arr in (("H", H[loose]), ("S", S[loose]), ("V", V[loose])):
        print("     %s: p1 %5.0f  p10 %5.0f  median %5.0f  p90 %5.0f  p99 %5.0f  (min %d max %d)"
              % (nm, np.percentile(arr, 1), np.percentile(arr, 10), np.median(arr),
                 np.percentile(arr, 90), np.percentile(arr, 99), arr.min(), arr.max()))

print()
print("=" * 84)
print("  2. Everything ELSE in the frame (what a widened band would catch)")
print("=" * 84)
print("     %-12s %-14s %-14s %s" % ("frame", "blue-ish px", "of which h85-140", "note"))
for img, nm in [(hand, "HAND (glove inside)")] + [(c, "clean") for c in CLEAN]:
    Hh, Ss, Vv = hsv_of(img)
    blue = (Hh >= 85) & (Hh <= 140) & (Ss >= 50) & (Vv >= 60)
    print("     %-24s %-14d" % (nm, int(blue.sum())))

print()
print("=" * 84)
print("  3. Widening sweep: glove kept on the hand frame vs false positives on clean frames")
print("=" * 84)
print("     %-22s %-14s %-18s %s"
      % ("band", "hand kept px", "worst clean px", "verdict"))


def biggest(img, hmin, hmax, smin, vmin, close_k, min_area):
    Hh, Ss, Vv = hsv_of(img)
    m = ((Hh >= hmin) & (Hh <= hmax) & (Ss >= smin) & (Vv >= vmin)).astype(np.uint8) * 255
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, _l, stats, _c = cv2.connectedComponentsWithStats(m, 8)
    best = 0
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= min_area and a > best:
            best = a
    return best


for label, (h0, h1, s0, v0) in (
        ("current h95-130 s80 v130", (95, 130, 80, 130)),
        ("widen v: v110", (95, 130, 80, 110)),
        ("widen v: v90", (95, 130, 80, 90)),
        ("widen s: s60", (95, 130, 60, 130)),
        ("widen h: h90-135", (90, 135, 80, 130)),
        ("wide h90-135 s60 v90", (90, 135, 60, 90))):
    kept = biggest(hand, h0, h1, s0, v0, 21, 500)
    worst = 0
    for c in CLEAN:
        worst = max(worst, biggest(c, h0, h1, s0, v0, 21, 500))
    verdict = "OK" if (kept > 0 and worst == 0) else ("MISS" if kept == 0 else "FALSE POS")
    print("     %-22s %-14d %-18d %s" % (label, kept, worst, verdict))

print()
print("=" * 84)
print("  4. FINE sweep on h_min (the tight one) -- keep glove, reject clean")
print("=" * 84)
print("  Why h_min is the one that matters:")
print("     glove      H p10/med/p90 = 90 / 94 / 95   (half the glove is BELOW 95!)")
print("     now_raw    H p10/med/p90 = 85 / 86 / 88, S 182, V 255  <- 150211 px of")
print("                cyan-ish blue that the CURRENT h_min=95 happens to exclude")
print("     -> widening h_min below ~90 would walk straight into it.")
print()
print("     %-30s %-12s %-16s %s" % ("band", "hand kept", "worst clean", "verdict"))
print("     " + "-" * 74)
for hmin in (88, 89, 90, 91, 92, 93, 94, 95):
    kept = biggest(hand, hmin, 130, 80, 110, 21, 500)
    worst = 0
    for c in CLEAN:
        worst = max(worst, biggest(c, hmin, 130, 80, 110, 21, 500))
    v = "OK" if (kept > 0 and worst == 0) else ("MISS" if kept == 0 else "FALSE POS")
    print("     h_min=%-3d s>=80 v>=110%-8s %-12d %-16d %s"
          % (hmin, "", kept, worst, v))
print()
print("     For reference, with the CURRENT v_min=130:")
for hmin in (90, 92, 94, 95):
    kept = biggest(hand, hmin, 130, 80, 130, 21, 500)
    worst = 0
    for c in CLEAN:
        worst = max(worst, biggest(c, hmin, 130, 80, 130, 21, 500))
    v = "OK" if (kept > 0 and worst == 0) else ("MISS" if kept == 0 else "FALSE POS")
    print("     h_min=%-3d s>=80 v>=130%-8s %-12d %-16d %s"
          % (hmin, "", kept, worst, v))
