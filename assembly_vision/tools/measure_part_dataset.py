# -*- coding: utf-8 -*-
"""阶段 1：fit 只读开发集，evaluate 独立验收；不读旧槽位坐标。"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import time
from pathlib import Path

import cv2
import numpy as np

from part_dataset import (ROOT, DEFAULT_DATA, POLICY, PARTS, build_templates,
                          classifier_from_templates, extract_features, file_hash,
                          load_config, load_samples, match_boxes, metrics, predict,
                          read_json, save_image, score_frame, segment_variant,
                          sample_scale, write_json, imread_safe)


def provenance(samples):
    return [dict(session=s["session"], image=str(s["path"]), sha256=s["record"]["sha256"],
                 annotation_sha256=file_hash(s["path"].with_suffix(".json"))) for s in samples]


def verify_capture(samples, wp_path):
    # ★ 同一个实验固定相机配置；请求值相同之外，逐帧 SDK 回读也在采集时核验。
    cameras = {s["meta"]["fingerprints"]["camera"] for s in samples}
    if len(cameras) != 1:
        raise ValueError("本次素材包含不同相机配置，请分批实验")
    if any(s["meta"]["fingerprints"]["workpiece"] != file_hash(wp_path) for s in samples):
        raise ValueError("当前正式分割配置与采集快照不一致；请用采集时的配置副本做独立实验")


def fit(args):
    samples = load_samples(args.dataset, "dev")
    verify_capture(samples, args.workpiece)
    policy, wp = load_config(str(args.policy)), load_config(str(args.workpiece))
    if args.output.exists():
        raise FileExistsError("模板已存在，请使用新文件名以保留实验历史")
    rows, misses = [], []
    for sample in samples:
        image = sample["image"]
        ann = sample["annotation"]
        truth = [o for o in ann["objects"] if o["part"] in PARTS]
        if any(o.get("occluded") for o in truth):
            raise ValueError("建模板只使用未遮挡静态件：%s" % sample["path"])
        mask, blobs, _ = segment_variant(image, wp, policy, args.variant)
        pairs = match_boxes(truth, [dict(bbox=list(b.bbox)) for b in blobs], policy.get("evaluation.bbox_iou"))
        found = set()
        mmpp = sample_scale(sample) if args.features == "shape_scale" else None
        if args.features == "shape_scale" and mmpp is None:
            raise ValueError("缺当前 SDK 内参，不能生成近似物理尺度特征")
        for ti, pi, _ in pairs:
            found.add(ti)
            rows.append(dict(part=truth[ti]["part"], image=str(sample["path"]),
                             features=extract_features(blobs[pi], image, mask, mmpp)))
        for ti, obj in enumerate(truth):
            if ti not in found:
                misses.append(dict(image=str(sample["path"]), part=obj["part"]))
    data = dict(schema=1, created_at=dt.datetime.now().astimezone().isoformat(),
                variant=args.variant, feature_set=args.features,
                scale_note="shape_scale 使用本次采集 distance/fx 的近似桌面尺度，不读取旧标定",
                policy=policy.data, policy_sha256=file_hash(args.policy),
                workpiece_sha256=file_hash(args.workpiece),
                camera_sha256=samples[0]["meta"]["fingerprints"]["camera"],
                source=provenance(samples), development_detection_misses=misses,
                templates=build_templates(rows, policy, args.features))
    write_json(args.output, data)
    print("模板保存成功：%s\n开发素材 %d 张，可匹配零件 %d 个，未匹配 %d 个（已登记，未伪装成成功）。" % (
        args.output, len(samples), len(rows), len(misses)))


def contains(box, roi):
    x, y, w, h = box
    rx, ry, rw, rh = roi
    return rx <= x + w / 2 < rx + rw and ry <= y + h / 2 < ry + rh


def mask_iou_score(truth, predicted, scope):
    """外轮廓标注仅比较填孔后的轮廓；与完整金属掩膜指标分开。"""
    if scope not in ("full_mask", "outer_silhouette"):
        raise ValueError("未知掩膜标注范围：%s" % scope)
    masks = [np.uint8(truth > 0) * 255, np.uint8(predicted > 0) * 255]
    if scope == "outer_silhouette":
        for mask in masks:
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, contours, -1, 255, -1)
    union = np.count_nonzero((masks[0] > 0) | (masks[1] > 0))
    overlap = np.count_nonzero((masks[0] > 0) & (masks[1] > 0))
    return float(overlap / union) if union else 0.0


def evaluate(args):
    policy, wp = load_config(str(args.policy)), load_config(str(args.workpiece))
    samples = load_samples(args.dataset, args.split, include_clips=args.clips)
    verify_capture(samples, args.workpiece)
    if args.output.exists():
        raise FileExistsError("结果目录已存在，请换新目录；不覆盖历史验收结果")
    artifact = read_json(args.templates) if args.templates else None
    if artifact:
        if artifact["policy_sha256"] != file_hash(args.policy) or artifact["workpiece_sha256"] != file_hash(args.workpiece):
            raise ValueError("模板生成后配置发生变化，必须另建实验")
        if artifact["camera_sha256"] != samples[0]["meta"]["fingerprints"]["camera"]:
            raise ValueError("模板与验收相机配置不一致")
        if args.split == "test":
            trained_hashes = {r["sha256"] for r in artifact["source"]}
            trained_sessions = {r["session"] for r in artifact["source"]}
            if any(s["record"]["sha256"] in trained_hashes or s["session"] in trained_sessions for s in samples):
                raise ValueError("验收集与开发集存在原图或会话重叠，拒绝评测")
        variant, feature_set = artifact["variant"], artifact["feature_set"]
        clf = classifier_from_templates(artifact["templates"], policy)
        model_label = "新模板（开发集拟合）"
    else:
        if not args.baseline:
            raise ValueError("请选择 --templates 或 --baseline")
        # 旧模板独立重放：保留正式特征入口，但不用旧单应/槽位身份当证据。
        calibration = load_config(str(args.calibration))
        clf = classifier_from_templates(calibration.get("part_templates"), policy)
        clf.min_score = float(wp.get("part_id.min_score", .55))
        variant, feature_set = "brass", "legacy"
        model_label = "旧模板独立重放（当前近似尺度；非旧整机逐字复现）"
    args.output.mkdir(parents=True)
    if args.split == "test":
        # ★ 记录每次查看验收集，调参后应另采新验收批次。
        log = args.dataset / "test_access.jsonl"
        with log.open("a", encoding="utf-8") as stream:
            import json
            stream.write(json.dumps(dict(at=dt.datetime.now().astimezone().isoformat(),
                                         output=str(args.output.resolve()), source=provenance(samples)), ensure_ascii=False) + "\n")
    counts = {p: dict(tp=0, fp=0, fn=0) for p in PARTS}
    confusion = {p: {q: 0 for q in [*PARTS, "unknown"]} for p in PARTS}
    static_counts = {p: dict(tp=0, fp=0, fn=0) for p in PARTS}
    static_images = 0
    mask_scores, timings, frames = [], [], []
    negative_total = negative_fp = 0
    glove_stats = dict(tp=0, fn=0, fp=0, tn=0, matched_boxes=0, truth_boxes=0)
    for index, sample in enumerate(samples):
        image, ann = sample["image"], sample["annotation"]
        roi = ann.get("work_roi") if args.scope == "work" else None
        if args.scope == "work" and roi is None:
            raise ValueError("work 评测要求每张图人工标注作业区域：%s" % sample["path"])
        truth = [o for o in ann["objects"] if o["part"] in PARTS and (roi is None or contains(o["bbox"], roi))]
        start = time.perf_counter()
        mask, blobs, result = segment_variant(image, wp, policy, variant)
        after_seg = time.perf_counter()
        mmpp = sample_scale(sample) if feature_set in ("legacy", "shape_scale") else None
        if feature_set in ("legacy", "shape_scale") and mmpp is None:
            raise ValueError("无当前相机内参，不能以旧尺度或零值冒充近似尺度")
        predictions = []
        for blob in blobs:
            feats = extract_features(blob, image, mask, mmpp)
            if artifact:
                got, score, margin = predict(clf, feats, policy)
            else:
                got, score, _ = clf.classify(feats)
                ranks = clf.rank(feats)
                margin = ranks[0][1] - ranks[1][1] if len(ranks) > 1 else 1.0
            predictions.append(dict(part=got, bbox=list(blob.bbox), score=float(score), margin=float(margin), features=feats))
        after_class = time.perf_counter()
        # 报告范围仅用于指标归属；原图检测始终不读取人工框/作业区域。
        selected = [p for p in predictions if roi is None or contains(p["bbox"], roi)]
        current, matrix, pairs = score_frame(truth, selected, policy.get("evaluation.bbox_iou"))
        static = sample["record"]["kind"] == "image" and not any(o.get("occluded") for o in truth)
        if static:
            static_images += 1
        for p in PARTS:
            for key in counts[p]:
                counts[p][key] += current[p][key]
                if static:
                    static_counts[p][key] += current[p][key]
            for q in confusion[p]:
                confusion[p][q] += matrix[p][q]
        glove_truth = [o for o in ann["objects"] if o["part"] == "glove" and (roi is None or contains(o["bbox"], roi))]
        glove_pred = [dict(bbox=list(result.glove_bbox))] if result.glove_bbox is not None else []
        glove_pred = [p for p in glove_pred if roi is None or contains(p["bbox"], roi)]
        gp = match_boxes(glove_truth, glove_pred, policy.get("evaluation.bbox_iou"))
        glove_stats["truth_boxes"] += len(glove_truth)
        glove_stats["matched_boxes"] += len(gp)
        glove_stats["tp" if glove_truth and glove_pred else "fn" if glove_truth else "fp" if glove_pred else "tn"] += 1
        if not truth:
            negative_total += 1
            negative_fp += int(any(p["part"] for p in selected))
        # 每个手工掩膜与一个完整预测连通块比较，不把预测裁到真值框内提高 IoU。
        localization = match_boxes(truth, predictions, policy.get("evaluation.bbox_iou"))
        by_truth = {ti: pi for ti, pi, _ in localization}
        for ti, obj in enumerate(truth):
            if not obj.get("mask"):
                continue
            gt = imread_safe(str(sample["path"].parent / obj["mask"]), cv2.IMREAD_GRAYSCALE)
            if gt is None or gt.shape != mask.shape or not np.any(gt):
                raise ValueError("手工掩膜缺失、空白或尺寸不符")
            if obj.get("mask_sha256") != file_hash(sample["path"].parent / obj["mask"]):
                raise ValueError("手工掩膜 hash 不符，请重新确认标注")
            proposed = np.zeros_like(mask)
            if ti in by_truth:
                cv2.drawContours(proposed, [blobs[by_truth[ti]].contour], -1, 255, -1)
                proposed = cv2.bitwise_and(proposed, mask)
            mask_scope = obj.get("mask_scope", "full_mask")
            mask_scores.append(dict(image=str(sample["path"]), part=obj["part"], static=static,
                                    mask_scope=mask_scope, iou=mask_iou_score(gt, proposed, mask_scope)))
        timing = dict(seg_ms=(after_seg - start) * 1000,
                      features_classify_ms=(after_class - after_seg) * 1000,
                      total_ms=(after_class - start) * 1000)
        timings.append(timing)
        row = dict(image=str(sample["path"]), session=sample["session"], truth=truth,
                   predictions=predictions, counts=current, timing=timing,
                   capture_context=sample["record"].get("capture_review", {}),
                   capture_prompt=sample["record"].get("task"))
        frames.append(row)
        vis = image.copy()
        for obj in ann["objects"]:
            x, y, w, h = obj["bbox"]
            cv2.rectangle(vis, (x, y), (x + w - 1, y + h - 1), (255, 180, 0), 1)
        for pred in predictions:
            x, y, w, h = pred["bbox"]
            cv2.rectangle(vis, (x, y), (x + w - 1, y + h - 1), (0, 255, 0) if pred["part"] else (0, 180, 255), 1)
            cv2.putText(vis, "%s %.2f" % (pred["part"] or "unknown", pred["score"]),
                        (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 255, 0), 1)
        save_image(args.output / "visuals" / ("%04d.png" % index), vis)
        save_image(args.output / "masks" / ("%04d.png" % index), mask)
    class_metrics = metrics(static_counts)
    coverage_gaps = {sample["session"]: sample["meta"].get("skipped_tasks", []) for sample in samples
                     if sample["meta"].get("skipped_tasks")}
    latency = {k: dict(median=float(np.median([t[k] for t in timings])),
                       p95=float(np.percentile([t[k] for t in timings], 95))) for k in timings[0]}
    passed_classes = {p: (r["precision"] is not None and r["recall"] is not None
                         and r["precision"] >= policy.get("evaluation.precision_target")
                         and r["recall"] >= policy.get("evaluation.recall_target")) for p, r in class_metrics.items()}
    report = dict(model=model_label, split=args.split, scope=args.scope, variant=variant,
                  feature_set=feature_set, image_count=len(samples), source=provenance(samples),
                  template_sha256=file_hash(args.templates) if artifact else file_hash(args.calibration),
                  policy_sha256=file_hash(args.policy), workpiece_sha256=file_hash(args.workpiece),
                  static_image_count=static_images, class_metrics=class_metrics,
                  diagnostic_metrics=metrics(counts), confusion_all_frames=confusion, class_targets_met=passed_classes,
                  negatives=dict(frames=negative_total, frames_with_identified_part=negative_fp),
                  glove=glove_stats, latency=latency, mask_scores=mask_scores, frames=frames,
                  coverage_gaps=coverage_gaps,
                  note="精度仅指本批素材；跳过的场景没有完成验证。视频为间隔抽样，遮挡样本单列。")
    write_json(args.output / "report.json", report)
    with (args.output / "classes.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["part", "tp", "fp", "fn", "precision", "recall"])
        writer.writeheader()
        writer.writerows(dict(part=p, **r) for p, r in class_metrics.items())
    lines = ["# 阶段 1 真实素材评测", "", model_label, "",
             "范围：%s；素材：%d 张，其中未遮挡静态图 %d 张；方案：%s / %s。" % (args.scope, len(samples), static_images, variant, feature_set), "",
             "| 类别 | TP | FP | FN | 精确率 | 召回率 |", "|---|---|---|---|---|---|"]
    for p, row in class_metrics.items():
        percent = lambda x: "无分母" if x is None else "%.1f%%" % (100 * x)
        lines.append("| %s | %d | %d | %d | %s | %s |" % (
            PARTS[p], row["tp"], row["fp"], row["fn"], percent(row["precision"]), percent(row["recall"])))
    lines += ["", "负样本误报帧：%d/%d。" % (negative_fp, negative_total), "",
              "算法总耗时中位 %.2fms，P95 %.2fms；不含图像读取、人工标注、显示或结果落盘。" % (
                  latency["total_ms"]["median"], latency["total_ms"]["p95"]), "",
              "原始预测、混淆矩阵、人工掩膜 IoU 与来源记录见 report.json；全部原图叠加结果见 visuals/。", "",
              "当前实验不会修改正式配置。静态验收必须另查遮挡比例、每类样本量、掩膜数量与手套误报。"]
    for mask_scope, label in [("full_mask", "完整金属掩膜（含孔洞）"), ("outer_silhouette", "外轮廓（内部孔洞不评测）")]:
        values = [r["iou"] for r in mask_scores if r["static"] and r["mask_scope"] == mask_scope]
        lines.append("%s：%d份%s。两种口径不合并验收。" % (
            label, len(values), "，IoU中位%.3f" % float(np.median(values)) if values else "，未覆盖"))
    if coverage_gaps:
        lines += ["", "## 未覆盖的采集场景", ""]
        for session, skipped in coverage_gaps.items():
            lines.extend("- %s / %s：%s（未采集，不能据此宣称验证通过）" % (session, row["id"], row["reason"])
                         for row in skipped)
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("评测保存成功：%s\n每类是否达到精确率/召回率门槛：%s" % (args.output, passed_classes))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--policy", type=Path, default=POLICY)
    ap.add_argument("--workpiece", type=Path, default=ROOT / "config" / "workpiece.yaml")
    sub = ap.add_subparsers(dest="command", required=True)
    train = sub.add_parser("fit", help="只从 dev 素材拟合，不读取 test")
    train.add_argument("--variant", choices=["brass", "seeded_metal", "seeded_metal_close", "seeded_color"], required=True)
    train.add_argument("--features", choices=["shape", "shape_scale", "shape_color"], required=True)
    train.add_argument("--output", type=Path, required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--split", choices=["dev", "test"], required=True)
    model = ev.add_mutually_exclusive_group(required=True)
    model.add_argument("--templates", type=Path)
    model.add_argument("--baseline", action="store_true")
    ev.add_argument("--calibration", type=Path, default=ROOT / "config" / "calibration.yaml")
    ev.add_argument("--scope", choices=["full", "work"], default="full")
    ev.add_argument("--clips", action="store_true", help="同时评测手工间隔标注的视频帧")
    ev.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    (fit if args.command == "fit" else evaluate)(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print("阶段 1 评测停止：%s" % exc)
        raise SystemExit(1)
