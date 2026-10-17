# -*- coding: utf-8 -*-
"""阶段 1 的数据契约、独立评测与实验分割；不依赖槽位或旧单应。"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assemvision.config import Config, load_config
from assemvision.core.segment import Segmenter
from assemvision.core.features import contour_features
from assemvision.core.part_id import PartClassifier, PartTemplate, classify_features
from assemvision.imgio import imread_safe, imwrite_safe

PARTS = {"P1_base": "底座圆盘", "P2_column": "立柱",
         "P3_wheel": "轮盘组件", "P4_cap": "顶盖铭牌"}
DEFAULT_DATA = ROOT.parent / "ai_测试工具" / "阶段1数据"
POLICY = ROOT / "config" / "part_dataset.yaml"


def write_json(path, data):
    """先写临时文件再原子替换；中断不会留下半份清单。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    # Windows读者/索引服务可能短暂阻止替换；保留旧清单，有限重试后仍明确报错。
    for attempt in range(8):
        try:
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.01 * 2 ** attempt, .2))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_image(path, img):
    if not imwrite_safe(str(path), img):
        raise OSError("图片保存失败：%s" % path)


def capture_tasks(policy):
    """人工摆放任务只是采集提示，不能自动变成评测真值。"""
    tasks = []
    angles = policy.get("capture.poses_deg")
    positions = policy.get("capture.positions")
    if len(angles) != len(positions):
        raise ValueError("角度和位置提示必须逐项对应")
    for part in PARTS:
        for bg in policy.get("capture.backgrounds"):
            for angle, pos in zip(angles, positions):
                tasks.append(dict(part=part, background=bg, angle_deg=angle, position=pos))
    for scene in policy.get("capture.negative_scenes"):
        for repeat in range(int(policy.get("capture.negative_repeats"))):
            tasks.append(dict(part=None, background=scene, repeat=repeat,
                              angle_deg=None, position="center"))
    for index, task in enumerate(tasks):
        task["id"] = "%03d" % (index + 1)
    return tasks


def remaining_tasks(tasks, samples, skipped=()):
    """按已完成任务续采，而非原图张数；允许复核后把错背景照片计入另一任务。

    fulfilled_task_id 只表示占用哪个采集名额；实际场景仍以 capture_review 为准，
    原始 task 提示和原图均保留，不把提示的角度/位置当成实测。
    """
    known = {t["id"] for t in tasks}
    completed = set()
    for sample in samples:
        if sample.get("kind") != "image":
            continue
        original = sample.get("task") or {}
        task_id = sample.get("fulfilled_task_id", original.get("id"))
        if task_id is None:
            continue
        if task_id not in known or task_id in completed:
            raise ValueError("采集任务编号未知或重复，请先复核会话清单：%s" % task_id)
        completed.add(task_id)
    excluded = set()
    for record in skipped:
        task_id = record.get("id")
        if (task_id not in known or task_id in completed or task_id in excluded
                or not str(record.get("reason", "")).strip()):
            raise ValueError("跳过任务必须编号有效、未采集、不重复并注明原因：%s" % task_id)
        excluded.add(task_id)
    return [t for t in tasks if t["id"] not in completed and t["id"] not in excluded]


def selected_tasks(policy, part=None, appearance=None, background=None):
    """补采只选指定零件，原任务编号仍可追溯；朝上面是提示而非自动真值。"""
    if part is not None and part not in PARTS:
        raise ValueError("未知零件")
    if appearance and (part != "P4_cap" or appearance not in ("hole_up", "smooth_up")):
        raise ValueError("朝上面只适用于明确选择的顶盖")
    if background is not None and (part is None or background not in ("tray", "green")):
        raise ValueError("背景筛选需要先指定零件，且必须是 tray/green")
    tasks = [dict(t) for t in capture_tasks(policy)
             if (part is None or t["part"] == part) and (background is None or t["background"] == background)]
    if appearance:
        for task in tasks:
            task["appearance"] = appearance
    return tasks


def bbox_iou(a, b):
    """位置框采用 x,y,w,h，类别不参与空间匹配。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def match_boxes(truth, predictions, threshold):
    """全局一对一匹配：优先最多匹配，再最大 IoU；避免贪心抢走唯一候选。"""
    if len(truth) > 12:
        raise ValueError("本工位只支持少量物体，单帧真值超过 12 个请检查标注")
    states = {0: (0.0, [])}
    for pi, pred in enumerate(predictions):
        nxt = dict(states)
        for mask, (score, pairs) in states.items():
            for ti, obj in enumerate(truth):
                if mask & (1 << ti):
                    continue
                ov = bbox_iou(obj["bbox"], pred["bbox"])
                if ov < threshold:
                    continue
                key = mask | (1 << ti)
                candidate = (score + ov, pairs + [(ti, pi, ov)])
                if key not in nxt or candidate[0] > nxt[key][0]:
                    nxt[key] = candidate
        states = nxt
    return max(states.items(), key=lambda item: (item[0].bit_count(), item[1][0]))[1][1]


def score_frame(truth, predictions, threshold):
    """错认同时计入真类 FN 和预测类 FP；没有预测的真件也必须计漏检。"""
    pairs = match_boxes(truth, predictions, threshold)
    by_truth = {ti: pi for ti, pi, _ in pairs}
    matched_pred = {pi for _, pi, _ in pairs}
    counts = {part: dict(tp=0, fp=0, fn=0) for part in PARTS}
    confusion = {part: {p: 0 for p in [*PARTS, "unknown"]} for part in PARTS}
    for ti, obj in enumerate(truth):
        actual = obj["part"]
        got = predictions[by_truth[ti]].get("part") if ti in by_truth else None
        if got not in PARTS:
            got = None
        confusion[actual][got or "unknown"] += 1
        if got == actual:
            counts[actual]["tp"] += 1
        else:
            counts[actual]["fn"] += 1
            if got:
                counts[got]["fp"] += 1
    for pi, pred in enumerate(predictions):
        if pi not in matched_pred and pred.get("part") in PARTS:
            counts[pred["part"]]["fp"] += 1
    return counts, confusion, pairs


def metrics(counts):
    out = copy.deepcopy(counts)
    for row in out.values():
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        row["precision"] = tp / (tp + fp) if tp + fp else None
        row["recall"] = tp / (tp + fn) if tp + fn else None
    return out


def validate_annotation(data, shape):
    if data.get("reviewed") is not True:
        raise ValueError("标注尚未人工确认")
    h, w = shape[:2]
    for obj in data.get("objects", []):
        if obj.get("part") not in {*PARTS, "glove"}:
            raise ValueError("未知类别")
        x, y, bw, bh = obj["bbox"]
        if not (0 <= x < w and 0 <= y < h and bw > 0 and bh > 0 and x + bw <= w and y + bh <= h):
            raise ValueError("位置框越出原图或面积为零")
    if "work_roi" in data and data["work_roi"] is not None:
        x, y, bw, bh = data["work_roi"]
        if not (x >= 0 and y >= 0 and bw > 0 and bh > 0 and x + bw <= w and y + bh <= h):
            raise ValueError("作业区域越出原图")


def load_samples(dataset, split, include_clips=False):
    """按会话划分开发/验收，检查原图 hash 和人工确认，拒绝静默跳过。"""
    samples = []
    for meta_path in sorted(Path(dataset).glob("*/session.json")):
        meta = read_json(meta_path)
        if meta["split"] != split:
            continue
        for rec in meta["samples"]:
            if rec.get("kind") != "image" and not (include_clips and rec.get("kind") == "clip_frame"):
                continue
            image_path = meta_path.parent / rec["image"]
            ann_path = image_path.with_suffix(".json")
            if not ann_path.exists():
                if rec.get("kind") == "clip_frame":
                    continue  # 视频仅评测按时间间隔人工抽样确认过的帧。
                raise ValueError("未标注：%s" % image_path)
            image = imread_safe(str(image_path))
            if image is None:
                raise OSError("读图失败：%s" % image_path)
            ann = read_json(ann_path)
            validate_annotation(ann, image.shape)
            digest = file_hash(image_path)
            if digest != rec["sha256"] or ann.get("image_sha256") != digest:
                raise ValueError("原图已变化，拒绝使用失配标注：%s" % image_path)
            samples.append(dict(path=image_path, image=image, annotation=ann,
                                record=rec, session=meta_path.parent.name, meta=meta))
    if not samples:
        raise ValueError("没有已采集且完成标注的 %s 静态素材" % split)
    return samples


def segment_variant(image, wp, policy, variant):
    """实验分割直接处理原图；不给旧单应或物理尺寸闸门，木桌也保留供评测。"""
    if variant not in policy.get("experiment.variants"):
        raise ValueError("未知掩膜方案：%s" % variant)
    # 独立实验只在内存副本上覆盖分割参数，保留正式配置及采集快照用于对账。
    overrides = policy.get("experiment.segmentation_overrides", {})
    if overrides:
        wp = type(wp)(copy.deepcopy(wp.data))
        for key, value in overrides.items():
            if not key.startswith("segment."):
                raise ValueError("实验覆盖只允许segment分割参数：%s" % key)
            wp.set(key, value)
    seg = Segmenter(wp)
    result = seg.segment(image, mm_per_px=None, work_roi=None)
    mask = result.brass_mask.copy()
    if variant != "brass":
        # ★ metal 依赖 brass 种子；这是候选补全实验，不是独立检测能力。
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        metal = seg._metal_mask(hsv)
        radius = int(policy.get("experiment.seed_margin_px"))
        seed = cv2.dilate(mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(metal, seed))
        if variant == "seeded_color":
            # 暗边允许更高饱和度和更低亮度，但只能补在原黄铜种子邻域。
            # ★ 不能全图放宽色域：真实开发图会把背景连进底座并产生误认。
            spec = policy.get("experiment.seeded_color")
            if not isinstance(spec, dict):
                raise ValueError("seeded_color缺少弱色域配置")
            weak = seg._mask(hsv, spec, morph=False)
            # 早期候选未单列暗边半径时沿用其已留档的种子半径，保留实验可重放性。
            dark_radius = int(spec.get("margin_px", radius))
            if dark_radius < 0:
                raise ValueError("弱色域补全半径不能为负")
            dark_seed = cv2.dilate(result.brass_mask, np.ones(
                (2 * dark_radius + 1, 2 * dark_radius + 1), np.uint8))
            mask = cv2.bitwise_or(mask, cv2.bitwise_and(weak, dark_seed))
        if variant in ("seeded_metal_close", "seeded_color"):
            k = int(policy.get("experiment.close_kernel_px"))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    blobs = seg._blobs(mask, min_area_px=float(seg.brass.get("min_area_px", 500)))
    for blob in blobs:
        blob.features = contour_features(blob.contour, mask)
    return mask, blobs, result


def sample_scale(sample):
    intr = sample["meta"].get("intrinsics") or {}
    distance = sample["meta"].get("distance_mm")
    fx = intr.get("fx", 0)
    return float(distance) / float(fx) if distance and fx > 0 else None


def extract_features(blob, image, mask, mmpp=None):
    return classify_features(blob, mm_per_px=mmpp, bgr=image, mask=mask)


def build_templates(rows, policy, feature_set):
    keys = policy.get("experiment.feature_sets.%s" % feature_set)
    out = {}
    for part in PARTS:
        feats = [row["features"] for row in rows if row["part"] == part]
        if len(feats) < int(policy.get("experiment.min_samples_per_class")):
            raise ValueError("%s 只有 %d 个可匹配开发样本，先检查漏检/标注" % (part, len(feats)))
        bands = {}
        for key in keys:
            if key in policy.get("experiment.ignored_features_by_part", {}).get(part, []):
                continue
            vals = [f[key] for f in feats if key in f and np.isfinite(f[key])]
            if len(vals) != len(feats):
                raise ValueError("开发样本缺少特征 %s；不允许静默降低维度" % key)
            lo, hi = np.percentile(vals, [policy.get("experiment.lower_percentile"),
                                          policy.get("experiment.upper_percentile")])
            pad = max(float(policy.get("experiment.minimum_padding")),
                      float(np.median(np.abs(vals))) * float(policy.get("experiment.relative_padding")))
            bands[key] = [float(lo - pad), float(hi + pad)]
        out[part] = dict(name=PARTS[part], ranges=bands, samples=len(feats))
    return out


def classifier_from_templates(data, policy, allowed=None):
    templates = []
    for part, item in data.items():
        # 旧标定把说明文字存为 _ranges_note，不能当作零件模板解析。
        if part.startswith("_"):
            continue
        if part not in PARTS or not isinstance(item, dict):
            raise ValueError("无效的零件模板：%s" % part)
        bands = item.get("ranges", item.get("features", {}))
        ranges = {k: tuple(v) for k, v in bands.items()
                  if isinstance(v, (list, tuple)) and len(v) == 2 and (allowed is None or k in allowed)}
        if not ranges:
            raise ValueError("%s 没有可用特征区间" % part)
        required = policy.get("experiment.required_features", [])
        if any(key not in ranges for key in required):
            raise ValueError("%s 缺少配置要求的必检特征区间" % part)
        weights = dict(item.get("weights", {}))
        tol = {"holes": float(weights.pop("holes_tol"))} if "holes_tol" in weights else {}
        templates.append(PartTemplate(part=part, ranges=ranges, weights=weights, discrete_tol=tol))
    return PartClassifier(templates, min_score=policy.get("experiment.min_score"),
                          holes_tol=policy.get("experiment.holes_tol"))


def predict(clf, features, policy):
    part, score, _ = clf.classify(features)
    ranked = clf.rank(features)
    margin = float(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 1.0
    if margin < float(policy.get("experiment.min_margin")):
        part = None
    required = policy.get("experiment.required_features", [])
    if part and required:
        template = next(t for t in clf.templates if t.part == part)
        # 单个尺寸严重越界不能被其他形状特征的平均分掩盖；拒识后不猜第二名。
        if not required_features_match(features, template.ranges, required):
            part = None
    return part, float(score), margin


def required_features_match(features, ranges, required):
    """显式配置的必检特征必须存在、有限且落在模板区间；不另加硬编码容差。"""
    for key in required:
        value = features.get(key)
        if key not in ranges or value is None or not np.isfinite(value):
            return False
        lo, hi = ranges[key]
        if not lo <= value <= hi:
            return False
    return True
