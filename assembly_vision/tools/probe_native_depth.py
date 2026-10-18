# -*- coding: utf-8 -*-
"""B 模块有界取流诊断：只核验对齐/单位/点云，不生成高度基线。"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assemvision.config import Config, load_config
from assemvision.imgio import imwrite_safe
from assemvision.sources.d435 import RealSenseSource


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--camera", default=str(ROOT / "config/camera.yaml"))
    ap.add_argument("--out", required=True, help="新证据目录，拒绝覆盖")
    ap.add_argument("--frames", type=int, default=5)
    args = ap.parse_args()
    if args.frames < 1:
        ap.error("--frames 必须为正整数")
    folder = Path(args.out).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    cam = Config(copy.deepcopy(load_config(args.camera).data))
    cam.set("depth.enabled", True)
    cam.set("depth.backend", "realsense")
    cam.set("realsense.open_retries", 1)
    src = RealSenseSource(cam)
    report = dict(status="incomplete", frames=[], height_baseline=None)
    rc = 1
    try:
        src.open()
        src.apply_locked_controls()
        for _ in range(int(cam.get("capture.warmup_frames", 30))):
            src.read(want_full=False)
        for index in range(args.frames):
            f = src.read(want_full=True)
            if f is None or f.depth_mm is None or not np.isfinite(f.depth_mm).any():
                raise RuntimeError("没有有效的原生深度")
            valid = np.isfinite(f.depth_mm)
            item = dict(color_shape=list(f.left_full.shape), depth_shape=list(f.depth_mm.shape),
                        xyz_shape=list(f.depth_points_mm.shape), valid_ratio=float(valid.mean()),
                        median_depth_mm=float(np.nanmedian(f.depth_mm)),
                        xyz_z_max_difference_mm=float(np.nanmax(np.abs(f.depth_points_mm[..., 2] - f.depth_mm))))
            report["frames"].append(item)
            report["profile"] = f.depth_profile
            if index in (0, args.frames - 1):
                np.savez_compressed(folder / ("frame_%02d.npz" % index),
                                    depth_mm=f.depth_mm, points_mm=f.depth_points_mm)
                if not imwrite_safe(str(folder / ("frame_%02d.png" % index)), f.left_full):
                    raise IOError("彩色证据保存失败")
        report["status"] = "native_stream_ok_no_height_calibration"
        rc = 0
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        src.close()
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                        allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
