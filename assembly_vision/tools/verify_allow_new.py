# -*- coding: utf-8 -*-
r"""Verify the two `allow_new` gates actually reject edge noise and keep real parts.

Reproduces the demo's gate logic offline on real frames, and reports for every
part-shaped blob whether it would be allowed to spawn a new track.

ASCII-only on lines containing quotes.
"""
import os
import sys

import numpy as np

ROOT = r"D:\university resource\大三上\视觉\assembly_vision"
sys.path.insert(0, ROOT)

import cv2                                                        # noqa: E402
from assemvision.config import load_config                        # noqa: E402
from assemvision.core.checks import Qualifier                     # noqa: E402
from assemvision.core.part_id import default_templates            # noqa: E402
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402

wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
cal = load_config(os.path.join(ROOT, "config", "calibration.yaml"))
tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
               ref_scale=float(cal.get("tray_map.scale", 1.0)))
q = Qualifier(wp, tray, cal)
seg = Segmenter(wp)
clf = q._classifier()
mmpp = tray.mm_per_px(1.0)

MIN_A, MAX_A = 400.0, 8000.0
INSET = 6
MIN_SCORE = 0.60


def classify(feats):
    f = dict(feats)
    f["area_mm2"] = float(f.get("area_px", 0.0)) * mmpp * mmpp
    rw, rh = float(f.get("rect_w", 0)), float(f.get("rect_h", 0))
    f["length_mm"] = max(rw, rh) * mmpp
    f["short_side_mm"] = min(rw, rh) * mmpp
    pid, sc, _ = clf.classify(f)
    return pid, sc


for name in ("assembly_choice.png", "glove_miss.png", "now_raw.png"):
    p = os.path.join(ROOT, "data", "refs", name)
    img = imread_safe(p)
    if img is None:
        continue
    if img.shape[1] != 640:
        img = cv2.resize(img, (640, int(round(img.shape[0] * 640.0 / img.shape[1]))),
                         interpolation=cv2.INTER_AREA)
    m = q.work_roi_mask(img.shape, 1.0, margin_mm=25.0)
    e = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (INSET * 2 + 1,) * 2))
    r = seg.segment(img, scale=1.0)          # whole frame, like the full search
    print()
    print("  == %s ==" % name)
    print("     %-9s %-13s %-9s %-14s %-9s %s"
          % ("area", "centroid", "inROI", "classified", "score", "allow_new"))
    n_ok = 0
    for b in sorted(r.brass_blobs, key=lambda z: -z.area_px):
        if not (MIN_A <= b.area_px <= MAX_A):
            continue
        cx, cy = int(b.centroid[0]), int(b.centroid[1])
        inside = (0 <= cy < e.shape[0] and 0 <= cx < e.shape[1] and e[cy, cx] > 0)
        pid, sc = classify(dict(b.features or {}))
        ok = bool(inside and pid and float(sc) >= MIN_SCORE)
        n_ok += 1 if ok else 0
        print("     %-9.0f (%3d,%3d)%-3s %-9s %-14s %-9s %s"
              % (b.area_px, cx, cy, "", "yes" if inside else "NO",
                 pid or "-", "%.2f" % sc, "ALLOW" if ok else "reject"))
    print("     -> %d 个块会被允许建新轨迹" % n_ok)
