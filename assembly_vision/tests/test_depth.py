# -*- coding: utf-8 -*-
"""B 模块：真正的三维倾斜平面、空洞质量门禁、SDK 原生接口及过期数据。"""
from __future__ import annotations

import copy
import os
import sys
import time
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import check
from assemvision.config import Config, load_config
from assemvision.core.depth import (NativeHeightEstimator, native_depth_mm,
                                   rectangle_mask, compare_native_height)
from assemvision.types import Frame, SegResult
from test_d435_source import _FakeSDK, _cam_cfg, _patched_sdk, _FakeFrameset
from assemvision.sources.d435 import RealSenseSource

ROOT = Path(__file__).resolve().parents[1]


def scene():
    cfg = load_config(str(ROOT / "config/depth_d435.yaml"))
    v, u = np.mgrid[:80, :100]
    # 透视相机射线与三维斜桌面的交点；不是在像素域直接造线性 Z。
    rx, ry = (u - 50) / 110.0, (v - 40) / 110.0
    a, b, c = .2, -.15, 440.0
    norm = np.sqrt(1 + a*a + b*b)
    target = rectangle_mask(u.shape, [65, 30, 20, 20])
    reference = rectangle_mask(u.shape, [5, 5, 35, 65])
    z = (c - target * 35.0 * norm) / (1 - a*rx - b*ry)
    xyz = np.stack((rx*z, ry*z, z), axis=-1).astype(np.float32)
    return cfg, xyz, reference, target


def main():
    raw = np.array([[0, 1000], [500, 2000]], np.uint16)
    depth = native_depth_mm(raw, .0005)
    check(np.isnan(depth[0, 0]) and depth[0, 1] == 500 and depth[1, 0] == 250,
          "设备非默认 depth_scale 正确换毫米，0 为未知")
    for scale in (0, -1, float("nan")):
        try:
            native_depth_mm(raw, scale)
            raise AssertionError("无效单位未拒绝")
        except ValueError:
            pass
    cfg, xyz, ref, target = scene()
    estimator = NativeHeightEstimator(cfg)
    result = estimator.measure(xyz, ref, target)
    check(result["ok"] and abs(result["height_mm"] - 35) < .001,
          "倾斜桌面法向高度35mm准确，目标不参与平面拟合")
    # 正常工件本身也可能倾斜，不能把所有目标高度梯度都当成错误。
    tilted = xyz.copy()
    small = rectangle_mask(target.shape, [65, 30, 8, 8])
    vv, uu = np.mgrid[:80, :100]
    rx, ry = (uu-50)/110., (vv-40)/110.
    normal = np.array([-.2, .15, 1.])
    norm = np.linalg.norm(normal)
    normal /= norm
    z = (440/norm-35)/(normal[0]*rx + normal[1]*ry + normal[2] + .2*rx)
    tilted[small] = np.stack((rx*z, ry*z, z), axis=-1)[small]
    answer = 35 + .2*tilted[..., 0][small]
    result = estimator.measure(tilted, ref, small)
    check(result["ok"] and abs(result["height_mm"] - float(np.median(answer))) < .001,
          "倾斜工件的小测区输出局部代表高度，不要求目标水平")
    noisy = xyz.copy()
    ids = np.argwhere(ref)[::15]
    noisy[ids[:, 0], ids[:, 1], 2] += 50
    result = estimator.measure(noisy, ref, target)
    check(result["ok"] and abs(result["height_mm"] - 35) < .01,
          "少量桌面离群点不能拖动高度基线")
    holes = xyz.copy()
    ids = np.argwhere(target)
    holes[ids[:250, 0], ids[:250, 1]] = np.nan
    result = estimator.measure(holes, ref, target)
    check(not result["ok"] and result["height_mm"] is None, "目标大面积空洞返回未知")
    check(not estimator.measure(xyz, ref, ref)["ok"], "参考区与目标重叠必须拒绝")
    thin = np.zeros_like(ref)
    thin[10, :80] = True
    check(not estimator.measure(xyz, thin, target)["ok"], "共线参考点不能拟合桌面")
    noisy = xyz.copy()
    ids = np.argwhere(target)[::2]
    noisy[ids[:, 0], ids[:, 1], 2] -= 25
    check(not estimator.measure(noisy, ref, target)["ok"], "目标混入不同高度时拒绝")
    rough = xyz.copy()
    rough[..., 2] += np.where(np.indices(ref.shape).sum(axis=0) % 2, 7, -7)
    check(not estimator.measure(rough, ref, target)["ok"], "非平面桌面残差超限拒绝")
    check(compare_native_height(34, 35, 15) is True
          and compare_native_height(5, 35, 15) is False
          and compare_native_height(None, 35, 15) is None,
          "高度正常/偏差/未知三种状态分离")
    check(compare_native_height(float("nan"), 35, 15) is None, "非有限高度不输出合格")
    for rect in ([99, 0, 10, 10], [0, 0, 0, 5], [.5, 0, 4, 5]):
        try:
            rectangle_mask(ref.shape, rect)
            raise AssertionError("错误ROI未拒绝")
        except ValueError:
            pass
    cfg.set("native_depth.reference_rect", [5, 5, 35, 65])
    cfg.set("native_depth.target_rect", [65, 30, 20, 20])
    cfg.set("native_depth.profile", {"serial": "unit-A"})
    f = Frame(1, 0, depth_points_mm=xyz, depth_profile={"serial": "unit-B"})
    check(not estimator.measure_frame(f)["ok"], "不同相机不能复用本机基线")
    f.depth_profile = {"serial": "unit-A"}
    check(estimator.measure_frame(f)["ok"], "相同来源允许使用显式ROI测量")
    test_native_source()
    test_pipeline(cfg, xyz)
    test_calibration_workflow()
    print("B 模块深度离线测试通过；真实反光金属精度仍须实测。")


def test_native_source():
    # 假 SDK 验证同帧对齐/设备单位/反投影；默认彩色路径由旧测试独立覆盖。
    sdk = _FakeSDK(frames=[np.ones((480, 640, 3), np.uint8)] * 3)
    sdk.stream.depth = "DEPTH"
    sdk.format.z16 = "Z16"
    sdk.devices[0].first_depth_sensor = lambda: SimpleNamespace(get_depth_scale=lambda: .0005)
    raw = np.full((480, 640), 880, np.uint16)
    raw[0, 0] = 0
    intr = SimpleNamespace(width=640, height=480, fx=603., fy=602., ppx=325., ppy=255.,
                           model="none", coeffs=[0.] * 5)
    df = SimpleNamespace(get_data=lambda: raw,
                         profile=SimpleNamespace(as_video_stream_profile=lambda:
                                                 SimpleNamespace(get_intrinsics=lambda: intr)))
    calls = []
    def align_process(frames):
        calls.append("align")
        return SimpleNamespace(get_color_frame=frames.get_color_frame, get_depth_frame=lambda: df)
    sdk.align = lambda stream: SimpleNamespace(process=align_process)
    points = np.zeros((480, 640, 3), np.float32)
    points[..., 2] = .44
    def calculate(frame):
        assert frame is df
        calls.append("xyz")
        return SimpleNamespace(get_vertices=lambda: points.reshape(-1))
    sdk.pointcloud = lambda: SimpleNamespace(calculate=calculate)
    cam = _cam_cfg(**{"depth.enabled": True, "depth.backend": "realsense"})
    with _patched_sdk(sdk):
        src = RealSenseSource(cam)
        src.open()
        check(len(sdk.last_config.streams) == 2, "显式原生深度模式同时开彩色和Z16")
        f = src.read(want_full=False)
        check(f.depth_mm is None and not calls, "普通实时帧不生成低频点云")
        f = src.read(want_full=True)
        check(calls == ["align", "xyz"] and abs(float(f.depth_mm[10, 10]) - 440) < .001,
              "全分辨率请求先对齐，再生成毫米深度")
        check(np.isnan(f.depth_points_mm[0, 0]).all() and f.right is None and f.right_full is None,
              "深度空洞清空XYZ，右目字段保持为空")
        check(abs(float(f.depth_points_mm[10, 10, 2]) - 440) < .001
              and f.depth_profile["serial"] == src.dev_serial, "SDK三维点米转毫米并绑定设备")
        src.close()
        check(src.align is None and src.pointcloud is None, "关闭释放原生资源")


def test_pipeline(cfg, xyz):
    from assemvision.pipeline import Pipeline
    cam = load_config(str(ROOT / "config/camera.yaml"))
    cam = Config(copy.deepcopy(cam.data))
    cam.set("depth.enabled", True)
    cam.set("depth.backend", "realsense")
    cam.set("native_depth", copy.deepcopy(cfg.data["native_depth"]))
    cam.set("native_depth.baseline_height_mm", 35.)
    cam.set("native_depth.baseline_accepted", True)
    wp = load_config(str(ROOT / "config/workpiece.yaml"))
    calib = load_config(str(ROOT / "config/calibration.synthetic.yaml"))
    old = copy.deepcopy(calib.data)
    pipe = Pipeline(cam, wp, calib, source_override="images", path_override=str(ROOT / "data/synthetic_seq"))
    check(pipe.depth is None and pipe.native_depth is not None, "原生模式不初始化旧双目匹配器")
    check(pipe.qualifier.check_assembly(SegResult()).height_baseline_mm == 35
          and calib.data == old and wp.get("assembly.baseline.wheel_height_mm") is None,
          "实际终检读取原生基线且不改原工件/标定对象")
    pipe.sh.depth_frame = Frame(1, 0, depth_points_mm=xyz, depth_profile={"serial": "unit-A"})
    pipe.sh.depth_captured_at = time.monotonic()
    pipe._do_depth()
    check(abs(pipe._current_height() - 35) < .001, "pipeline消费原生高度")
    pipe.sh.depth_updated_at -= 10
    check(pipe._current_height() is None, "旧高度过期后返回未知")
    pipe.sh.depth_frame = Frame(2, 0)
    pipe._do_depth()
    check(pipe.sh.wheel_height_mm is None, "新帧缺深度立即清掉历史高度")
    cam.set("native_depth.baseline_accepted", False)
    pending = Pipeline(cam, wp, calib, source_override="images", path_override=str(ROOT / "data/synthetic_seq"))
    check(pending.qualifier.check_assembly(SegResult()).height_baseline_mm is None,
          "尚未人工复核及独立验证的候选基线不进入终检")


def test_calibration_workflow():
    """实跑 c5 新入口：证据落盘、基线加载及独立缺立柱高度对照。"""
    sys.path.insert(0, str(ROOT / "calib"))
    import c5_assembly_baseline as c5
    cfg, xyz, ref, target = scene()
    cfg.set("native_depth.warmup_frames", 1)
    cfg.set("native_depth.max_sample_attempts", 6)
    current = [xyz]
    class Source:
        closed = 0
        def __init__(self, cam):
            self.index = 0
        def open(self):
            pass
        def apply_locked_controls(self):
            pass
        def close(self):
            Source.closed += 1
        def read(self, want_full=False):
            self.index += 1
            return Frame(self.index, 0, left_full=np.zeros((80, 100, 3), np.uint8),
                         depth_mm=current[0][..., 2], depth_points_mm=current[0],
                         depth_profile={"serial": "fixture"})
    with tempfile.TemporaryDirectory(prefix="native_depth_test_") as tmp:
        folder = Path(tmp)
        policy = folder / "policy.json"
        cfg.save(str(policy))
        args = SimpleNamespace(native_depth="calibrate", out=str(folder / "full"),
                               camera=str(ROOT / "config/camera.yaml"), depth_policy=str(policy),
                               frames=3, no_preview=False, native_label="full", native_reference="")
        args.measurement_surface = "cap_region"
        with patch("assemvision.sources.d435.RealSenseSource", Source), \
             patch("cv2.selectROI", side_effect=[(5, 5, 35, 65), (65, 30, 20, 20)]), \
             patch("cv2.destroyAllWindows"), patch("cv2.imshow"), patch("cv2.waitKey", return_value=-1):
            check(c5.run_native_depth(args) == 0, "c5原生标定完整执行并保存证据")
            runtime = folder / "full/camera.native-depth.json"
            saved = load_config(str(runtime))
            check(abs(saved.get("native_depth.baseline_height_mm") - 35) < .001,
                  "运行配置保存新三维高度基线")
            current[0] = xyz.copy()
            # 缺立柱样例目标从35mm降到5mm，其余参考面保持原样。
            normal = np.array([-.2, .15, 1.])
            normal /= np.linalg.norm(normal)
            current[0][target] += 30 * normal
            args.native_depth, args.native_reference = "check", str(runtime)
            args.out, args.no_preview, args.native_label = str(folder / "missing"), False, "missing-column"
            args.reselect_target = True
            with patch("cv2.selectROI", return_value=(65, 30, 20, 20)):
                check(c5.run_native_depth(args) == 0, "c5独立检查允许手动重定位相同物理表面")
            report = json.loads((folder / "missing/report.json").read_text(encoding="utf-8"))
            check(report["height_matches_baseline"] is False and report["label"] == "missing-column",
                  "缺立柱高度偏差记为不符合，场景标签单独保留")
            check(report["target_selection"] == "manual_reselection"
                  and report["measurement_surface"] == "cap_region"
                  and load_config(str(runtime)).get("native_depth.baseline_height_mm") == saved.get("native_depth.baseline_height_mm"),
                  "人工重选记录来源，不改原基线")
            args.out = str(folder / "failed")
            args.reselect_target, args.no_preview = False, True
            current[0] = np.full_like(xyz, np.nan)
            check(c5.run_native_depth(args) == 1 and not (folder / "failed/camera.native-depth.json").exists(),
                  "坏深度不生成基线且退出失败")
    check(Source.closed == 3, "标定、检查和失败路径均释放相机")


if __name__ == "__main__":
    main()
