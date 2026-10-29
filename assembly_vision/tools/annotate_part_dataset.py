# -*- coding: utf-8 -*-
"""原图上的人工真值。1~4 选零件，g 手套，拖框；n 确认空图，s 保存。"""
from __future__ import annotations

import argparse
import base64
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


class MaskEditor:
    """掩膜和顶点作为同一步历史保存；所有修改可撤销，历史间共享未修改掩膜。"""

    def __init__(self, shape, initial=None):
        self.mask = np.zeros(shape[:2], np.uint8) if initial is None else initial.copy()
        self.points = []
        self.past, self.future = [], []

    def remember(self):
        self.past.append((self.mask, self.points.copy()))
        self.future.clear()

    def point(self, point):
        self.remember()
        self.points.append(tuple(point))

    def fill(self, value):
        if len(self.points) < 3:
            return False
        self.remember()
        self.mask = self.mask.copy()
        cv2.fillPoly(self.mask, [np.asarray(self.points, np.int32)], value)
        self.points.clear()
        return True

    def cancel_polygon(self):
        if self.points:
            self.remember()
            self.points.clear()

    def clear(self):
        self.remember()
        self.mask = np.zeros_like(self.mask)
        self.points.clear()

    def undo(self):
        if self.past:
            self.future.append((self.mask, self.points.copy()))
            self.mask, self.points = self.past.pop()

    def redo(self):
        if self.future:
            self.past.append((self.mask, self.points.copy()))
            self.mask, self.points = self.future.pop()


def save_mask_draft(path, editor, image_hash, part):
    """每次编辑保留草稿；不将未确认草稿作为评测真值。"""
    ok, encoded = cv2.imencode(".png", editor.mask)
    if not ok:
        raise OSError("掩膜草稿编码失败")
    write_json(path, dict(image_sha256=image_hash, part=part, points=editor.points,
                         mask_png=base64.b64encode(encoded.tobytes()).decode("ascii")))


def load_mask_draft(path, shape, image_hash, part):
    data = read_json(path)
    if data.get("image_sha256") != image_hash or data.get("part") != part:
        raise ValueError("草稿与原图或类别不一致：%s" % path)
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(data["mask_png"]), np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != tuple(shape[:2]):
        raise ValueError("草稿掩膜尺寸不正确：%s" % path)
    editor = MaskEditor(shape, mask)
    editor.points = [tuple(p) for p in data["points"]]
    if any(len(p) != 2 or not (0 <= p[0] < shape[1] and 0 <= p[1] < shape[0]) for p in editor.points):
        raise ValueError("草稿顶点超出原图：%s" % path)
    return editor


def paint_mask(image, obj, smoke=False, initial=None, draft_path=None, image_hash=None):
    """手点多边形；前景和孔洞分开提交，不用算法结果自动当真值。"""
    editor = MaskEditor(image.shape, initial)
    if draft_path is not None and draft_path.exists():
        editor = load_mask_draft(draft_path, image.shape, image_hash, obj["part"])
        print("已恢复上次未保存的掩膜草稿与顶点。")
    geometry = list(mask_view_geometry(image.shape, obj["bbox"]))
    show_overlay = True
    print("精细掩膜 %s：左键沿可见边界点多边形，f 填前景，h 扣孔洞，右键撤点。" % obj["part"])
    print("z/右键撤销一步，y 恢复；r 只取消当前多边形，x 清空整张（可撤销）。")
    print("s 确认，q 退出保留草稿；矩形框仅作参考，不限制掩膜边界。")
    print("已自动放大零件局部；+/- 调整倍率，v 隐藏/显示填充。保存仍为原图像素。")
    if obj["part"] == "P4_cap":
        print("顶盖统一按外轮廓标注，不要求挖出看不清的小孔；不以孔数区分两面。")

    def checkpoint():
        if draft_path is not None and not smoke:
            save_mask_draft(draft_path, editor, image_hash, obj["part"])

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point = mask_view_point(x, y, geometry)
            if point is not None:
                editor.point(point)
                checkpoint()
        elif event == cv2.EVENT_RBUTTONDOWN:
            editor.undo()
            checkpoint()

    cv2.setMouseCallback(WINDOW, mouse)
    while True:
        mask, points = editor.mask, editor.points
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
        cv2.putText(view, "F fill H hole S save Z undo Y redo R cancel X clear Q exit", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
        cv2.putText(view, "Zoom %dx   +/- zoom   0 full/local   V overlay   Right click undo" % scale, (6, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(20) & 0xFF
        if smoke or key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return None
        if key == ord("0"):
            if geometry[:4] == [0, 0, image.shape[1], image.shape[0]]:
                geometry = list(mask_view_geometry(image.shape, obj["bbox"]))
            else:
                geometry = [0, 0, image.shape[1], image.shape[0], 1]
        elif key in (ord("+"), ord("=")):
            geometry[-1] = min(scale + 1, max(1, min(16, 1280 // cw, 800 // ch)))
        elif key in (ord("-"), ord("_")):
            geometry[-1] = max(1, scale - 1)
        elif key == ord("v"):
            show_overlay = not show_overlay
        if key in (ord("f"), ord("h")):
            editor.fill(255 if key == ord("f") else 0)
            checkpoint()
        elif key == ord("r"):
            editor.cancel_polygon()
            checkpoint()
        elif key in (ord("z"), 26):
            editor.undo()
            checkpoint()
        elif key in (ord("y"), 25):
            editor.redo()
            checkpoint()
        elif key == ord("x"):
            editor.clear()
            checkpoint()
        elif key == ord("s"):
            if editor.points:
                print("还有未提交多边形，先按 f 或 h。")
            elif not np.any(editor.mask):
                print("掩膜为空，不能保存。")
            else:
                return editor.mask


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
    ap.add_argument("--part", choices=list(PARTS), help="掩膜模式只处理指定类别，供多人分工")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--limit", type=int, help="本次最多标注的图片数，便于分批或单张重画")
    ap.add_argument("--clips", action="store_true", help="标注视频序列，每秒最多抽一张，按时间戳抽样")
    ap.add_argument("--smoke", action="store_true", help="仅检查首张图窗口，不保存标注")
    args = ap.parse_args(argv)
    if args.part and not args.masks:
        ap.error("--part 仅用于已有框标注的 --masks 模式")
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
                selected = [o for o in previous["objects"] if not args.part or o["part"] == args.part]
                if not selected:
                    continue
                if not args.redo and not args.smoke and all(o.get("mask") for o in selected):
                    continue
                cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
                for i, obj in enumerate(previous["objects"]):
                    if args.part and obj["part"] != args.part:
                        continue
                    if obj.get("mask") and not args.redo:
                        continue
                    draft_path = path.parent / ("%s_mask_%d.draft.json" % (path.stem, i))
                    initial = None
                    if obj.get("mask"):
                        mask_path = path.parent / obj["mask"]
                        initial = imread_safe(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if initial is None or initial.shape != image.shape[:2] or file_hash(mask_path) != obj.get("mask_sha256"):
                            raise ValueError("已有掩膜缺失、尺寸或hash不匹配：%s" % mask_path)
                    mask = paint_mask(image, obj, smoke=args.smoke, initial=initial,
                                      draft_path=draft_path, image_hash=rec["sha256"])
                    if mask is None:
                        return 0
                    rel = "%s_mask_%d.png" % (path.stem, i)
                    save_image(path.parent / rel, mask)
                    obj["mask"] = rel
                    obj["mask_sha256"] = file_hash(path.parent / rel)
                    if obj["part"] == "P4_cap":
                        obj["mask_scope"] = "outer_silhouette"
                        obj.setdefault("holes_reviewed", False)
                    write_json(ann_path, previous)
                    draft_path.unlink(missing_ok=True)
                    print("掩膜保存成功：%s" % (path.parent / rel))
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
