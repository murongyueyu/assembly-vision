# -*- coding: utf-8 -*-
"""阶段 1 引导采集。先用 --check 检查一张原图；s 保存，q/ESC 退出。"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import sys
import time
from pathlib import Path

import cv2

from part_dataset import (ROOT, DEFAULT_DATA, POLICY, PARTS, capture_tasks,
                          file_hash, load_config, read_json, remaining_tasks, selected_tasks, save_image, write_json)
from assemvision.sources.factory import make_source


def actual_controls(src):
    """记录 RGB SDK 实际回读，读取失败保留错误；不把请求值伪装成实测。"""
    out = {}
    if not hasattr(src, "_target_sensors"):
        return {"unavailable": "非 RealSense 源，无 SDK 控件回读"}
    for sensor in src._target_sensors():
        name = src._sensor_name(sensor)
        opts = {}
        try:
            options = sensor.get_supported_options()
            for opt in options:
                try:
                    opts[str(opt)] = float(sensor.get_option(opt))
                except Exception as exc:
                    opts[str(opt)] = {"error": str(exc)}
        except Exception as exc:
            opts["error"] = str(exc)
        out[name] = opts
    return out


def instruction(task):
    part = PARTS.get(task["part"], "无零件")
    backgrounds = {"tray": "黑料盘", "green": "绿幕", "empty_tray": "空料盘",
                   "wood": "保留木桌入画", "shadow": "有阴影但无零件",
                   "glove": "手套入画但无零件", "empty_workarea": "作业区域内无零件"}
    positions = {"left": "画面左侧", "center": "画面中部", "right": "画面右侧"}
    print("\n当前 %s：%s；背景：%s；位置：%s；参考转角：%s°" % (
        task["id"], part, backgrounds.get(task["background"], task["background"]),
        positions[task["position"]], task["angle_deg"]))
    print("请只摆当前这一件，完整入画，手移开后按 s；位置/角度仅作多样性提示，无须精确量角。")
    if task.get("appearance"):
        print("顶盖朝上面：%s；在本会话保持这一面朝上。" % {
            "hole_up": "有孔面", "smooth_up": "光滑面"}[task["appearance"]])
    if task["part"] is None:
        print("本张应没有四类零件；手套场景请戴约定的双层手套。")


def windows_mf_exposure(requested):
    """对应 SDK v2.59.1 Windows MF 的 from_100micros/to_100micros。

    100µs 是 SDK 单位，Windows 用 2 的整次幂秒；先 roundl 再截断回整数。
    这是单位转换，不是经验容差。来源：librealsense/src/mf/mf-uvc.cpp。
    """
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError("曝光请求必须是有限正数")
    exponent = math.log2(int(requested) * 0.0001)
    rounded = math.floor(exponent + .5) if exponent >= 0 else math.ceil(exponent - .5)
    return float(int(2.0 ** rounded * 10000))


def check_locked(src, controls, cam, exposure_mode="strict"):
    """精确核验实际控件；仅显式启用的 Windows 曝光允许 SDK 已知的离散换算。"""
    if not hasattr(src, "rs") or src.rs is None:
        return []
    if exposure_mode not in ("strict", "windows_mf"):
        raise ValueError("未知曝光回读策略：%s" % exposure_mode)
    validation = []
    expected = cam.get("realsense.controls", {}) or {}
    from assemvision.sources.d435 import _opt_name
    for key, target in expected.items():
        name = str(getattr(src.rs.option, _opt_name(key), _opt_name(key)))
        observed = [opts[name] for opts in controls.values() if isinstance(opts, dict) and name in opts]
        numeric = [v for v in observed if isinstance(v, (int, float))]
        if not numeric:
            # 不支持的非关键画质控件可留证，自动开关缺失则无法确认采集条件。
            if key in ("enable_auto_exposure", "enable_auto_white_balance"):
                raise RuntimeError("无法回读 %s，不能确认自动策略已经关闭" % key)
            continue
        status = "exact"
        if any(not math.isfinite(v) or abs(v - float(target)) > 1e-6 for v in numeric):
            quantized = (windows_mf_exposure(float(target))
                         if key == "exposure" and exposure_mode == "windows_mf" and sys.platform == "win32"
                         else None)
            if quantized is None or any(not math.isfinite(v) or v != quantized for v in numeric):
                raise RuntimeError("控件实际值与配置不符：%s 请求 %s，回读 %s" % (key, target, numeric))
            status = "windows_mf_quantized"
        validation.append(dict(control=key, requested=float(target), observed=numeric, status=status))
    return validation


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--session", required=True, help="会话名，只用英文字母、数字、横线或下划线")
    ap.add_argument("--split", choices=["dev", "test", "check"], required=True)
    ap.add_argument("--check", action="store_true", help="先检查相机，按 s 保存一张后退出")
    ap.add_argument("--clip", default="", help="录制无损原图序列，例如 glove 或 pick_place")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--camera", type=Path, default=ROOT / "config" / "camera.yaml")
    ap.add_argument("--policy", type=Path, default=POLICY)
    ap.add_argument("--source", default=None, help="离线自检可指定 images")
    ap.add_argument("--path", default=None, help="离线图片目录")
    ap.add_argument("--headless", action="store_true", help="仅自检用：自动保存第一张并退出")
    ap.add_argument("--max-frames", type=int, default=0, help="仅自检用：限制读取帧数")
    ap.add_argument("--limit", type=int, default=0, help="本次最多保存几张后自动退出；0 表示不限制")
    ap.add_argument("--part", choices=list(PARTS), default=None, help="新会话可只补采指定零件")
    ap.add_argument("--appearance", choices=["hole_up", "smooth_up"], default=None, help="顶盖朝上面提示，需 --part P4_cap")
    ap.add_argument("--background", choices=["tray", "green"], default=None, help="指定零件补采的背景")
    args = ap.parse_args(argv)
    if args.limit < 0:
        ap.error("limit 不能为负数")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.session):
        ap.error("session 必须为安全的单级目录名")
    if args.check != (args.split == "check"):
        ap.error("--check 必须与 --split check 一起使用")
    if args.headless and not args.check:
        ap.error("headless 只能保存检查图，不允许自动确认正式采集任务")
    if args.clip and (args.check or args.seconds <= 0):
        ap.error("视频会话不能与 check 混用，seconds 必须大于零")
    policy, cam = load_config(str(args.policy)), load_config(str(args.camera))
    exposure_mode = policy.get("capture.exposure_readback", "strict")
    folder = args.dataset.resolve() / args.session
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "session.json"
    tasks = selected_tasks(policy, args.part, args.appearance, args.background)
    selection = dict(part=args.part, appearance=args.appearance, background=args.background)
    if (args.part or args.appearance or args.background) and (args.check or args.clip):
        ap.error("零件补采筛选只用于正式静态照片会话")
    mode = "check" if args.check else ("clip" if args.clip else "guided")
    fingerprints = {"camera": file_hash(args.camera), "policy": file_hash(args.policy),
                    "workpiece": file_hash(ROOT / "config" / "workpiece.yaml")}
    meta = read_json(manifest) if manifest.exists() else None
    if meta:
        if meta.get("capture_selection", dict(part=None, appearance=None, background=None)) != selection:
            raise ValueError("会话的零件/朝上面/背景筛选已变化，请保持原筛选或使用新会话")
        if (meta["split"], meta["mode"], meta["fingerprints"], meta.get("clip", "")) != (
                args.split, mode, fingerprints, args.clip):
            raise ValueError("会话配置/用途已变化，请使用新的 session 名，不混入旧批次")
        if args.check or args.clip:
            raise ValueError("检查或录像会话已存在，请换 session 名，保留已有原图")
        if not remaining_tasks(tasks, meta["samples"], meta.get("skipped_tasks", [])):
            print("本批次任务已处理：保存 %d 张，跳过 %d 项；%s" % (
                len(meta["samples"]), len(meta.get("skipped_tasks", [])), folder))
            return 0
    src = make_source(cam, args.source, args.path)
    last_task, reads, failures = None, 0, 0
    started = None
    saved_this_run = 0
    print("原图输出：%s" % folder)
    print("窗口内 s=保存当前原图，q/ESC=退出；保存不包含屏幕上的文字。")
    if args.check:
        print("检查图：先取下全部零件，让空料盘和绿幕完整入画；按 s 保存一张后自动退出。")
    try:
        src.open()
        # ★ open() 只开流，不会应用配置。必须在开流后显式锁控件，再预热和回读。
        # 用户真机复现：漏掉此调用时请求 exposure=126，实际仍是驱动保留的 156。
        if hasattr(src, "apply_locked_controls"):
            src.apply_locked_controls()
        warmup = int(policy.get("capture.warmup_frames"))
        while True:
            f = src.read(want_full=True)
            if f is None:
                failures += 1
                if failures >= int(policy.get("capture.failed_reads_limit")):
                    raise RuntimeError("连续取帧失败，已停止；已有采集清单保留")
                continue
            failures = 0
            reads += 1
            if reads <= warmup:
                continue
            img = f.left_full if f.left_full is not None else f.left
            if meta is None:
                controls = actual_controls(src)
                validation = check_locked(src, controls, cam, exposure_mode)
                for row in validation:
                    if row["status"] == "windows_mf_quantized":
                        print("[采集] Windows MF 曝光离散换算：请求 %g，实际 %s；按实际值留档。" % (
                            row["requested"], row["observed"]))
                meta = dict(schema=1, session=args.session, split=args.split, mode=mode,
                            capture_selection=selection,
                            clip=args.clip, created_at=dt.datetime.now().astimezone().isoformat(),
                            source=src.name, fingerprints=fingerprints, camera_config=cam.data,
                            policy=policy.data, workpiece_config=load_config(str(ROOT / "config" / "workpiece.yaml")).data,
                            actual_controls=controls, control_validation=validation,
                            intrinsics=src.color_intrinsics() if hasattr(src, "color_intrinsics") else None,
                            distance_mm=policy.get("capture.distance_mm"),
                            scale_note="distance/fx 是桌面近似尺度，不是旧单应或精确零件尺寸",
                            samples=[])
                write_json(manifest, meta)
            if meta.get("image_shape") and meta["image_shape"] != list(img.shape):
                raise RuntimeError("帧尺寸变化，不能继续混入本批次")
            meta["image_shape"] = list(img.shape)
            index = len(meta["samples"])
            task = remaining_tasks(tasks, meta["samples"], meta.get("skipped_tasks", []))[0] if mode == "guided" else None
            if task and task["id"] != last_task:
                instruction(task)
                last_task = task["id"]
            if started is None:
                started = time.perf_counter()
                if mode == "clip":
                    print("开始录制 %s，约 %.0f 秒；每帧 PNG 和时间戳保留，按 q 可提前结束。" % (args.clip, args.seconds))
            if not args.headless:
                view = img.copy()
                line = ("CHECK: empty tray / S save and exit" if args.check else
                        ("CLIP %s / Q stop" % args.clip if args.clip else
                         "%s/%d %s %s %sdeg / S save Q quit" % (
                             task["id"], len(tasks), task["part"] or "NEGATIVE", task["background"], task["angle_deg"])))
                cv2.putText(view, line, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .44, (0, 255, 255), 1)
                cv2.imshow("Stage1 raw capture", view)
                key = cv2.waitKey(1) & 0xFF
                if cv2.getWindowProperty("Stage1 raw capture", cv2.WND_PROP_VISIBLE) < 1:
                    break
            else:
                key = ord("s")
            if key in (27, ord("q")):
                break
            if key == ord("s") or mode == "clip":
                # ★ 录像只在首尾回读，避免逐帧 SDK 查询降低真实采集帧率。
                controls = meta["actual_controls"] if mode == "clip" else actual_controls(src)
                if mode != "clip":
                    current = check_locked(src, controls, cam, exposure_mode)
                    if current != meta["control_validation"]:
                        raise RuntimeError("实际控件相对本会话起始值发生变化，停止混入素材")
                rel = "raw/%06d.png" % (index + 1)
                path = folder / rel
                if path.exists():
                    raise FileExistsError("发现未登记原图，停止以避免覆盖：%s" % path)
                save_image(path, img)
                rec = dict(image=rel, sha256=file_hash(path), kind="clip_frame" if args.clip else mode,
                           captured_at=dt.datetime.now().astimezone().isoformat(), source_ts=float(f.ts),
                           elapsed_s=time.perf_counter() - started, task=task)
                if mode != "clip":
                    rec["actual_controls"] = controls
                if mode == "guided":
                    rec["kind"] = "image"
                meta["samples"].append(rec)
                saved_this_run += 1
                write_json(manifest, meta)
                if mode != "clip":
                    print("保存成功：%s" % path)
                if (args.check or (args.limit and saved_this_run >= args.limit)
                        or (mode == "guided" and not remaining_tasks(tasks, meta["samples"], meta.get("skipped_tasks", [])))):
                    break
            if args.clip and time.perf_counter() - started >= args.seconds:
                break
            if args.max_frames and reads - warmup >= args.max_frames:
                break
        if meta and mode == "clip":
            meta["controls_after_clip"] = actual_controls(src)
            final_controls = check_locked(src, meta["controls_after_clip"], cam, exposure_mode)
            if final_controls != meta["control_validation"]:
                raise RuntimeError("录像结束时实际控件发生变化，本会话不能作为固定参数素材")
            meta["recording_summary"] = dict(
                frames=len(meta["samples"]), requested_seconds=args.seconds,
                last_saved_elapsed_s=meta["samples"][-1]["elapsed_s"] if meta["samples"] else 0,
                controls_end_verified=True, stopped_normally=True)
            write_json(manifest, meta)
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as exc:
        if meta and mode == "clip":
            # 原图先落盘、记录已在内存中；异常退出前尽量补齐清单和首尾控件证据。
            meta["recording_summary"] = dict(
                frames=len(meta["samples"]), requested_seconds=args.seconds,
                stopped_normally=False, controls_end_verified=False,
                error="%s: %s" % (type(exc).__name__, exc))
            try:
                meta["controls_after_clip"] = actual_controls(src)
                checked = check_locked(src, meta["controls_after_clip"], cam, exposure_mode)
                meta["recording_summary"]["controls_end_verified"] = checked == meta["control_validation"]
            except Exception as end_error:
                meta["recording_summary"]["controls_error"] = str(end_error)
            try:
                write_json(manifest, meta)
            except OSError as save_error:
                print("异常收尾仍无法写入清单，保留临时清单供恢复：%s" % save_error)
        raise
    finally:
        src.close()
        if not args.headless:
            cv2.destroyAllWindows()
    print("本会话已保存 %d 张；清单：%s" % (len(meta["samples"]) if meta else 0, manifest))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print("采集停止：%s" % exc)
        raise SystemExit(1)
