# -*- coding: utf-8 -*-
"""Offline check of the brass size gate (segment.max_diam_mm) on real frames.

Runs the real Segmenter twice on each stored frame -- with and without
mm_per_px -- and reports what the gate would remove and whether any
part-sized blob gets dropped.  No camera needed.

Why this exists: the gate is SILENT.  When it removes the wooden desk the
UI looks exactly the same as when it removes a real part, so "it looks fine"
proves nothing.  This tool prints the numbers instead.

ASCII-only in printed lines (GBK console safety).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2                                                        # noqa: E402
from assemvision.config import load_config                        # noqa: E402
from assemvision.core.checks import Qualifier                     # noqa: E402
from assemvision.core.roi import TrayMap                          # noqa: E402
from assemvision.core.segment import Segmenter                    # noqa: E402
from assemvision.imgio import imread_safe                         # noqa: E402

# Frames kept in the repo (real HBV-era captures). now_raw.png is the one from
# the 2.12 report: whole wooden desk turned into "brass".
FRAMES = ["now_raw.png", "asm_raw.png", "assembly_choice.png", "glove_miss.png",
          "tray_empty.png"]
# A blob this small is not a part; the gate is not supposed to care about it.
PART_MIN_PX = 400.0


def _report(tag, r_no, r_on, mmpp, maxd):
    """Print one segmentation pair and return the number of dropped part blobs.

    ★ How to tell "this blob was removed by the gate": compare the MASK, not
      the centroid pixel.  A wheel's centroid sits in its hub HOLE, so
      "centroid pixel is background" happens on frames where nothing was
      dropped at all (measured on glove_miss.png, dropped_px=0).
    """
    det = r_on.detail.get("size_gate", {})
    a_no = int((r_no.brass_mask > 0).sum())
    a_on = int((r_on.brass_mask > 0).sum())
    print("     [%s] gate on=%s  dropped_px=%d  brass px: %d -> %d (%.1f%% kept)"
          % (tag, det.get("on"), int(det.get("dropped_px", 0)), a_no, a_on,
             100.0 * a_on / max(1, a_no)))
    print("     %-10s %-9s %-8s %-11s %s"
          % ("area_px", "diam_mm", "verdict", "centroid", "note"))
    dropped = []
    for b in sorted(r_no.brass_blobs, key=lambda z: -z.area_px):
        d = 2.0 * float(np.sqrt(b.area_px / np.pi)) * mmpp
        over = d > maxd
        cx, cy = int(round(b.centroid[0])), int(round(b.centroid[1]))
        # does any pixel of this blob survive the gate?
        bx, by, bw, bh = b.bbox
        sub_no = r_no.brass_mask[by:by + bh, bx:bx + bw] > 0
        sub_on = r_on.brass_mask[by:by + bh, bx:bx + bw] > 0
        n_before = int(sub_no.sum())
        n_after = int((sub_no & sub_on).sum())
        note = ""
        if over:
            note = "REMOVED" if n_after == 0 else "!! %d/%d px left" % (n_after, n_before)
        elif b.area_px >= PART_MIN_PX:
            note = "kept" if n_after >= 0.9 * n_before else "!! PART TRIMMED %d/%d" % (
                n_after, n_before)
            if n_after < 0.9 * n_before:
                dropped.append(b)
        print("     %-10.0f %-9.1f %-8s (%3d,%3d)  %s"
              % (b.area_px, d, "over" if over else "ok", cx, cy, note))
    return dropped, det


def main() -> int:
    wp = load_config(os.path.join(ROOT, "config", "workpiece.yaml"))
    cal = load_config(os.path.join(ROOT, "config", "calibration.yaml"))
    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"), float),
                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
    q = Qualifier(wp, tray, cal)
    seg = Segmenter(wp)
    mmpp = tray.mm_per_px(1.0)
    maxd = float(seg.max_diam_mm)

    print("=== size gate check: max_diam_mm=%.0fmm, mm_per_px=%.4f (1 px = %.3f mm)"
          % (maxd, mmpp, mmpp))
    print("    config work_roi_mode = %s; also runs a forced 'full' pass"
          % seg.work_roi_mode)
    print("    judgement: desk measured 201mm, biggest real part 64.5mm (2.7x apart)")
    print("    frames: %s" % ", ".join(FRAMES))

    # ★ 对账锚点：docs 记的"木桌等效直径 201mm"应当能在这几帧上复现出来。
    #   复现不出来就说明阈值/尺度/ROI 有一处变了 —— 那正是这条要抓的。
    seen_over = []
    seen_parts = []
    rc = 0
    for name in FRAMES:
        p = os.path.join(ROOT, "data", "refs", name)
        img = imread_safe(p)
        if img is None:
            print("\n  == %s ==  (missing, skipped)" % name)
            continue
        if img.shape[1] != 640:
            img = cv2.resize(img, (640, int(round(img.shape[0] * 640.0 / img.shape[1]))),
                             interpolation=cv2.INTER_AREA)
        print("\n  == %s ==  %dx%d" % (name, img.shape[1], img.shape[0]))
        # (a) the real runtime path: work ROI clamps brass first
        roi = q.work_roi_mask(img.shape, 1.0, margin_mm=25.0)
        dropped, det = _report("roi ",
                               seg.segment(img, scale=1.0, work_roi=roi),
                               seg.segment(img, scale=1.0, work_roi=roi, mm_per_px=mmpp),
                               mmpp, maxd)
        # (b) full-frame path: this is where the wooden desk actually shows up
        #     (measured: brass 86048px = 28.0% of the frame).  work_roi_mode is
        #     forced to 'full' so the ROI cannot do the blocking.
        keep_mode = seg.work_roi_mode
        seg.work_roi_mode = "full"
        try:
            r_full_no = seg.segment(img, scale=1.0)
            dropped_f, det_f = _report("full", r_full_no,
                                       seg.segment(img, scale=1.0, mm_per_px=mmpp),
                                       mmpp, maxd)
        finally:
            seg.work_roi_mode = keep_mode
        for b in r_full_no.brass_blobs:
            d = 2.0 * float(np.sqrt(b.area_px / np.pi)) * mmpp
            if d > maxd:
                seen_over.append((name, d))
            elif b.area_px >= PART_MIN_PX:
                seen_parts.append((name, d))
        if dropped or dropped_f:
            print("     FAIL: %d (roi) + %d (full) part-sized blob(s) dropped"
                  % (len(dropped), len(dropped_f)))
            rc = 1
        if not (det.get("on") and det_f.get("on")):
            print("     FAIL: gate did not report itself as ON")
            rc = 1

    print("\n  --- cross-check against the documented numbers ---")
    if seen_over:
        ds = [d for _, d in seen_over]
        print("     over-size blobs seen: %s"
              % ", ".join("%s %.1fmm" % (n, d) for n, d in seen_over))
        print("     range %.1f~%.1f mm (docs say the desk is 201mm)" % (min(ds), max(ds)))
        bad = [d for d in ds if not (170.0 <= d <= 240.0)]
        if bad:
            print("     FAIL: %s outside the documented 170~240mm band" % bad)
            rc = 1
    else:
        print("     note: no over-size blob on these frames (nothing to compare)")
    if seen_parts:
        dmax_part = max(d for _, d in seen_parts)
        print("     biggest real part: %.1fmm (docs say 64.5mm) over %d blob(s)"
              % (dmax_part, len(seen_parts)))
        if dmax_part > 100.0:
            print("     FAIL: a 'part' blob is %.1fmm -- parts must stay well under %.0fmm"
                  % (dmax_part, maxd))
            rc = 1
    print("\n%s" % ("OK: no part-sized blob was dropped, gate reported ON everywhere."
                    if rc == 0 else "FAILED: see lines above."))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
