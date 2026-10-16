# -*- coding: utf-8 -*-
"""原图上的人工真值。1~4 选零件，g 手套，拖框；n 确认空图，s 保存。"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from part_dataset import (DEFAULT_DATA, PARTS, file_hash, read_json, save_image,
                          validate_annotation, write_json, imread_safe)

WINDOW = "Stage1 manual labels"
MASK_TOOLBAR = 64


def mask_view_geometry(shape, bbox):
    """自动裁出零件及邻近背景，返回原图裁切坐标和整数显示倍率。"""
    h, w = shape[:2]
    x, y, bw, bh = bbox
    pad = max(8, int(round(max(bw, bh) * .12)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    cw, ch = x1 - x0, y1 - y0
    scale = max(1, min(8, 960 // cw, 600 // ch))
    return x0, y0, cw, ch, scale


def mask_view_point(x, y, geometry):
    """鼠标显示坐标还原到原图像素；工具栏/画面外点击无效。"""
    x0, y0, cw, ch, scale = geometry
    y -= MASK_TOOLBAR
    if not (0 <= x < cw * scale and 0 <= y < ch * scale):
        return None
    return x0 + x // scale, y0 + y // scale


def paint_mask(image, obj, smoke=False):
    """手点多边形；前景和孔洞分开提交，不用算法结果自动当真值。"""
    mask = np.zeros(image.shape[:2], np.uint8)
    points = []
    geometry = list(mask_view_geometry(image.shape, obj["bbox"]))
    show_overlay = True
    print("精细掩膜 %s：左键沿可见边界点多边形，f 填前景，h 扣孔洞，右键撤点。" % obj["part"])
    print("可多次画前景/孔洞；s 确认，r 清空，q 取消。边界不清楚请取消并换样本。")
    print("已自动放大零件局部；+/- 调整倍率，v 隐藏/显示填充。保存仍为原图像素。")

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point = mask_view_point(x, y, geometry)
            if point is not None:
                points.append(point)
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.setMouseCallback(WINDOW, mouse)
    while True:
        view = image.copy()
        selected = mask > 0
        if show_overlay:
            view[selected] = (view[selected] * .7 + np.array([0, 76, 0])).astype(np.uint8)
        x0, y0, cw, ch, scale = geometry
        crop = cv2.resize(view[y0:y0 + ch, x0:x0 + cw], (cw * scale, ch * scale),
                          interpolation=cv2.INTER_NEAREST)
        view = np.zeros((ch * scale + MASK_TOOLBAR, max(640, cw * scale), 3), np.uint8)
        view[MASK_TOOLBAR:, :cw * scale] = crop

        def display_point(px, py):
            return int((px - x0) * scale + scale // 2), int((py - y0) * scale + scale // 2 + MASK_TOOLBAR)

        x, y, w, h = obj["bbox"]
        cv2.rectangle(view, display_point(x, y), display_point(x + w - 1, y + h - 1), (255, 255, 0), 1)
        if points:
            displayed = [display_point(*point) for point in points]
            cv2.polylines(view, [np.asarray(displayed, np.int32)], False, (0, 0, 255), 1)
            for point in displayed:
                cv2.circle(view, point, 2, (0, 0, 255), -1)
        cv2.putText(view, "F fill / H hole / S save / R reset / Q cancel", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
        cv2.putText(view, "Zoom %dx   +/- zoom   V overlay   Right click undo point" % scale, (6, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(20) & 0xFF
        if smoke or key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return None
        if key in (ord("+"), ord("=")):
            geometry[-1] = min(scale + 1, max(1, min(16, 1280 // cw, 800 // ch)))
        elif key in (ord("-"), ord("_")):
            geometry[-1] = max(1, scale - 1)
        elif key == ord("v"):
            show_overlay = not show_overlay
        if key in (ord("f"), ord("h")) and len(points) >= 3:
            cv2.fillPoly(mask, [np.asarray(points, np.int32)], 255 if key == ord("f") else 0)
            points.clear()
        elif key == ord("r"):
            mask[:] = 0
            points.clear()
        elif key == ord("s"):
            outside = mask.copy()
            outside[y:y + h, x:x + w] = 0
            if points:
                print("还有未提交多边形，先按 f 或 h。")
            elif not np.any(mask):
                print("掩膜为空，不能保存。")
            elif np.any(outside):
                print("掩膜越出人工位置框；请先修正位置框或重画掩膜。")
            else:
                return mask


def annotate(image, previous, smoke=False):
    data = {"objects": [], "work_roi": None, "reviewed": False}
    if previous:
        data.update(previous)
    state = dict(label=None, start=None, cursor=None, area=False, empty=False)
    h, w = image.shape[:2]

    def mouse(event, x, y, flags, param):
        x, y = max(0, min(w - 1, x)), max(0, min(h - 1, y))
        state["cursor"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and (state["label"] or state["area"]):
            state["start"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["start"]:
            x0, y0 = state["start"]
            box = [min(x0, x), min(y0, y), abs(x - x0) + 1, abs(y - y0) + 1]
            if state["area"]:
                data["work_roi"] = box
                state["area"] = False
            else:
                data["objects"].append(dict(part=state["label"], bbox=box, occluded=False))
                state["empty"] = False
            state["start"] = None

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)  # ★ 原像素坐标，不做拉伸显示。
    cv2.setMouseCallback(WINDOW, mouse)
    print("按 1 底座 / 2 立柱 / 3 轮盘 / 4 顶盖 / g 手套，再左键拖框。")
    print("a 拖作业区域（可选），d 撤最后一框，o 切换最后一件遮挡状态。")
    print("s 确认保存，n 明确确认空图，q 退出；任务提示不自动变成标注。")
    while True:
        view = image.copy()
        for i, obj in enumerate(data["objects"]):
            x, y, bw, bh = obj["bbox"]
            cv2.rectangle(view, (x, y), (x + bw - 1, y + bh - 1), (0, 255, 0), 1)
            cv2.putText(view, "%d %s%s" % (i, obj["part"], " OCC" if obj.get("occluded") else ""),
                        (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
        if data.get("work_roi"):
            x, y, bw, bh = data["work_roi"]
            cv2.rectangle(view, (x, y), (x + bw - 1, y + bh - 1), (255, 200, 0), 1)
        if state["start"] and state["cursor"]:
            cv2.rectangle(view, state["start"], state["cursor"], (0, 255, 255), 1)
        cv2.putText(view, "label=%s / 1-4 G label A area D undo N empty S save Q quit" % state["label"],
                    (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 255, 255), 1)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(100 if smoke else 20) & 0xFF
        if smoke or key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return None
        if key in [ord(str(i)) for i in range(1, 5)]:
            state["label"] = list(PARTS)[key - ord("1")]
        elif key == ord("g"):
            state["label"] = "glove"
        elif key == ord("a"):
            state["area"] = True
        elif key == ord("d") and data["objects"]:
            data["objects"].pop()
        elif key == ord("o") and data["objects"]:
            obj = data["objects"][-1]
            obj["occluded"] = not obj.get("occluded", False)
        elif key == ord("n") and not data["objects"]:
            state["empty"] = True
            print("已人工确认：这张图没有零件或手套。按 s 保存。")
        elif key == ord("s"):
            if not data["objects"] and not state["empty"]:
                print("空图需要先按 n 确认；有手套的图请用 g 标出手套。")
                continue
            data["reviewed"] = True
            validate_annotation(data, image.shape)
            return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--session", required=True)
    ap.add_argument("--masks", action="store_true", help="为已确认的框逐件人工画掩膜")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--limit", type=int, help="本次最多标注的图片数，便于分批或单张重画")
    ap.add_argument("--clips", action="store_true", help="标注视频序列，每秒最多抽一张，按时间戳抽样")
    ap.add_argument("--smoke", action="store_true", help="仅检查首张图窗口，不保存标注")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit 必须为正整数")
    if Path(args.session).name != args.session or args.session in (".", ".."):
        ap.error("session 只能是单级目录名")
    folder = args.dataset / args.session
    meta = read_json(folder / "session.json")
    interval = float(meta.get("policy", {}).get("annotation", {}).get("clip_interval_s", 1.0))
    last_time = -float("inf")
    completed = 0
    try:
        for rec in meta["samples"]:
            if rec["kind"] == "clip_frame":
                if not args.clips or rec["elapsed_s"] - last_time < interval:
                    continue
                last_time = rec["elapsed_s"]
            path = folder / rec["image"]
            ann_path = path.with_suffix(".json")
            if ann_path.exists() and not args.redo and not args.masks and not args.smoke:
                continue
            image = imread_safe(str(path))
            if image is None or file_hash(path) != rec["sha256"]:
                raise ValueError("原图缺失或发生变化：%s" % path)
            previous = read_json(ann_path) if ann_path.exists() else None
            if previous and previous.get("image_sha256") != rec["sha256"]:
                raise ValueError("旧标注与原图 hash 不一致：%s" % ann_path)
            print("\n原图：%s\n采集提示（请人工核对）：%s" % (path, rec.get("task")))
            if args.masks:
                if not previous:
                    raise ValueError("请先完成位置框标注：%s" % path)
                validate_annotation(previous, image.shape)
                if not args.redo and not args.smoke and all(o.get("mask") for o in previous["objects"]):
                    continue
                cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
                for i, obj in enumerate(previous["objects"]):
                    if obj.get("mask") and not args.redo:
                        continue
                    mask = paint_mask(image, obj, smoke=args.smoke)
                    if mask is None:
                        return 0
                    rel = "%s_mask_%d.png" % (path.stem, i)
                    save_image(path.parent / rel, mask)
                    obj["mask"] = rel
                    obj["mask_sha256"] = file_hash(path.parent / rel)
                    write_json(ann_path, previous)
            else:
                data = annotate(image, previous, args.smoke)
                if data is None:
                    return 0
                # 重新画框后旧掩膜不能自动沿用。
                data["image_sha256"] = rec["sha256"]
                write_json(ann_path, data)
                print("标注保存成功：%s" % ann_path)
            completed += 1
            if args.limit is not None and completed >= args.limit:
                break
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
