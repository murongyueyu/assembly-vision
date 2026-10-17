# -*- coding: utf-8 -*-
"""阶段 1 独立真值评测：覆盖错认双计数、全局匹配、数据完整性和完整工具链。"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from _common import check
from part_dataset import (ROOT, POLICY, PARTS, bbox_iou, capture_tasks, file_hash,
                          load_config, load_samples, match_boxes, read_json, save_image,
                          score_frame, validate_annotation, write_json, imread_safe)
from part_dataset import remaining_tasks, selected_tasks, classifier_from_templates
from part_dataset import required_features_match, predict
from measure_part_dataset import fit, evaluate, mask_iou_score
from part_track_demo import run_dataset_model
from capture_part_dataset import main as capture_main, check_locked, windows_mf_exposure
from annotate_part_dataset import mask_view_geometry, mask_view_point, MASK_TOOLBAR
from annotate_part_dataset import main as annotate_main
from annotate_part_dataset import MaskEditor, paint_mask, save_mask_draft, load_mask_draft


def raises(call, message):
    try:
        call()
    except (ValueError, OSError, RuntimeError):
        check(True, message)
    else:
        raise AssertionError(message)


def make_session(folder, split, wp_path, policy_path, count):
    """只用于验证工具链，明确标记为合成，不冒充真实采集素材。"""
    camera_hash = "OFFLINE_FIXTURE_NOT_CAMERA"
    meta = dict(schema=1, split=split, mode="guided", intrinsics=dict(fx=603.82),
                distance_mm=440, fingerprints=dict(camera=camera_hash, workpiece=file_hash(wp_path),
                                                  policy=file_hash(policy_path)), samples=[])
    color = cv2.cvtColor(np.array([[[25, 110, 180]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    index = 0
    for part in PARTS:
        for repeat in range(count):
            image = np.zeros((180, 200, 3), np.uint8)
            cx, cy = 95 + repeat * 3 + (1 if split == "test" else 0), 90
            if part == "P1_base":
                cv2.circle(image, (cx, cy), 30, color, -1)
            elif part == "P2_column":
                cv2.rectangle(image, (cx - 40, cy - 8), (cx + 40, cy + 8), color, -1)
            elif part == "P3_wheel":
                points = []
                for i in range(16):
                    angle = i * np.pi / 8
                    r = 35 if i % 2 == 0 else 19
                    points.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
                cv2.fillPoly(image, [np.asarray(points, np.int32)], color)
            else:
                cv2.circle(image, (cx, cy), 17, color, -1)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            box = list(cv2.boundingRect(cv2.findNonZero(gray)))
            path = folder / "raw" / ("%03d.png" % index)
            save_image(path, image)
            digest = file_hash(path)
            write_json(path.with_suffix(".json"), dict(reviewed=True, image_sha256=digest,
                                                        objects=[dict(part=part, bbox=box, occluded=False)]))
            meta["samples"].append(dict(image="raw/" + path.name, sha256=digest, kind="image"))
            index += 1
    write_json(folder / "session.json", meta)


def main():
    # 小件面积实验必须只改内存副本，不能污染工位配置或采集快照。
    from assemvision.config import Config
    from part_dataset import segment_variant
    original_wp = load_config(str(ROOT / "config" / "workpiece.yaml"))
    original_data = copy.deepcopy(original_wp.data)
    experiment = copy.deepcopy(load_config(str(POLICY)).data)
    image = np.zeros((80, 120, 3), np.uint8)
    image[:] = cv2.cvtColor(np.array([[[82, 255, 130]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    image[20:30, 40:60] = cv2.cvtColor(np.array([[[25, 110, 180]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    _, before, _ = segment_variant(image, original_wp, Config(experiment), "brass")
    experiment["experiment"]["segmentation_overrides"] = {"segment.brass.min_area_px": 100}
    _, after, _ = segment_variant(image, original_wp, Config(experiment), "brass")
    check(not before and len(after) == 1 and original_wp.data == original_data,
          "独立面积实验可保留小件但不修改正式配置对象")
    experiment["experiment"]["max_candidate_extent_px"] = 15
    visible_mask, rejected, _ = segment_variant(image, original_wp, Config(experiment), "brass")
    check(not rejected and np.any(visible_mask),
          "像素跨度闸门拒绝过大候选但保留原始分割掩膜供诊断")
    experiment["experiment"]["max_candidate_extent_px"] = 0
    raises(lambda: segment_variant(image, original_wp, Config(experiment), "brass"),
           "候选跨度配置不得为零或负值")
    experiment["experiment"]["max_candidate_extent_px"] = None
    border_image = np.zeros((150, 180, 3), np.uint8)
    color = image[25, 45].tolist()
    for x, y in [(70, 0), (70, 130), (0, 60), (160, 60), (70, 60)]:
        border_image[y:y+20, x:x+20] = color
    mask_before, border_before, _ = segment_variant(border_image, original_wp, Config(experiment), "brass")
    experiment["experiment"]["reject_border_candidates"] = True
    mask_after, border_after, border_result = segment_variant(border_image, original_wp, Config(experiment), "brass")
    check(len(border_before) == 5 and len(border_after) == 1
          and 0 < border_after[0].bbox[0] < 100
          and len(border_result.detail["rejected_candidates"]) == 4
          and np.array_equal(mask_before, mask_after) and original_wp.data == original_data,
          "贴边开关拒识四边截断候选、保留内部目标与完整掩膜，并逐一登记原因；默认不变")
    experiment["experiment"]["reject_border_candidates"] = "false"
    raises(lambda: segment_variant(border_image, original_wp, Config(experiment), "brass"),
           "贴边开关拒绝易误解的字符串配置")
    experiment["experiment"]["reject_border_candidates"] = False
    experiment["experiment"]["segmentation_overrides"] = {"capture.exposure": 10}
    raises(lambda: segment_variant(image, original_wp, Config(experiment), "brass"),
           "分割实验禁止覆盖相机采集参数")
    weak_policy = copy.deepcopy(load_config(str(POLICY)).data)
    weak_policy["experiment"]["segmentation_overrides"] = {"segment.brass.min_area_px": 100}
    weak_image = image.copy()
    dark = cv2.cvtColor(np.array([[[25, 230, 60]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    weak_image[22:28, 60:68] = dark
    weak_image[22:28, 90:100] = dark
    original_mask, _, _ = segment_variant(weak_image, original_wp, Config(weak_policy), "brass")
    weak_mask, _, _ = segment_variant(weak_image, original_wp, Config(weak_policy), "seeded_color")
    check(not original_mask[25, 61] and weak_mask[25, 61] and not weak_mask[25, 65]
          and not np.any(weak_mask[22:28, 90:100]) and original_wp.data == original_data,
          "弱色域只能补种子近邻暗边，不能独立发现远处暗块或污染正式配置")
    weak_policy["experiment"]["seeded_color"]["margin_px"] = -1
    raises(lambda: segment_variant(weak_image, original_wp, Config(weak_policy), "seeded_color"),
           "拒绝负的暗边补全半径")
    bands = {"length_mm": (50, 60)}
    check(required_features_match({"length_mm": 50}, bands, ["length_mm"])
          and required_features_match({"length_mm": 60}, bands, ["length_mm"])
          and not required_features_match({"length_mm": 61}, bands, ["length_mm"])
          and not required_features_match({}, bands, ["length_mm"])
          and not required_features_match({"length_mm": float("nan")}, bands, ["length_mm"]),
          "必检特征边界包含端点，越界、缺失和非有限值必须拒识")
    required_policy = Config(copy.deepcopy(load_config(str(POLICY)).data))
    example_templates = {"P2_column": dict(ranges=dict(length_mm=[50, 60],
        circularity=[.5, 1], solidity=[.8, 1]), weights=dict(length_mm=1, circularity=3, solidity=3))}
    required_clf = classifier_from_templates(example_templates, required_policy)
    feats = dict(length_mm=500, circularity=.8, solidity=.9)
    check(predict(required_clf, feats, required_policy)[0] == "P2_column",
          "未启用必检特征时保持原有软评分")
    required_policy.set("experiment.required_features", ["length_mm"])
    check(predict(required_clf, feats, required_policy)[0] is None
          and predict(required_clf, dict(feats, length_mm=55), required_policy)[0] == "P2_column",
          "尺寸严重越界不能被其他形状高分掩盖，合法尺寸仍可识别")
    required_policy.set("experiment.required_features", ["short_side_mm"])
    raises(lambda: classifier_from_templates(example_templates, required_policy),
           "模板缺少必检区间时提前报错，不静默放行")
    compatibility_policy = Config(copy.deepcopy(load_config(str(POLICY)).data))
    tie_clf = classifier_from_templates({
        "P1_base": dict(ranges={"length_mm": [50, 60]}),
        "P2_column": dict(ranges={"length_mm": [50, 60]})}, compatibility_policy)
    check(predict(tie_clf, {"length_mm": 55}, compatibility_policy) == (None, 1.0, 0.0),
          "复用模板得分后并列高分仍按余量拒识，不由模板顺序决定身份")
    low_clf = classifier_from_templates({"P1_base": dict(ranges={"length_mm": [50, 60]})},
                                       compatibility_policy)
    check(predict(low_clf, {"length_mm": 500}, compatibility_policy)[0] is None,
          "复用排序不能绕过分类最低分门槛")
    silhouette = np.zeros((30, 30), np.uint8)
    cv2.circle(silhouette, (15, 15), 10, 255, -1)
    ring = silhouette.copy()
    cv2.circle(ring, (15, 15), 4, 0, -1)
    check(mask_iou_score(silhouette, ring, "outer_silhouette") == 1 and
          mask_iou_score(silhouette, ring, "full_mask") < 1,
          "孔洞未标时只比较外轮廓，与完整金属掩膜指标区分")
    ring[0:3, 0:3] = 255
    check(mask_iou_score(silhouette, ring, "outer_silhouette") < 1,
          "外轮廓口径仍保留额外背景误检惩罚，不裁切预测")
    raises(lambda: mask_iou_score(silhouette, ring, "wrong"), "未知标注口径必须报错")
    editor = MaskEditor((40, 40))
    for p in [(5, 5), (30, 5), (30, 30), (5, 30)]:
        editor.point(p)
    editor.fill(255)
    outer = editor.mask.copy()
    for p in [(12, 12), (20, 12), (20, 20), (12, 20)]:
        editor.point(p)
    editor.fill(0)
    hole = editor.mask.copy()
    check(hole[15, 15] == 0 and outer[15, 15] == 255, "孔洞操作只扣指定区域")
    editor.undo()
    check(np.array_equal(editor.mask, outer) and len(editor.points) == 4,
          "撤销挖孔恢复之前的金属和孔洞顶点，不清空整张")
    editor.redo()
    check(np.array_equal(editor.mask, hole) and not editor.points, "恢复操作可重新应用孔洞")
    editor.point((10, 10))
    editor.cancel_polygon()
    check(np.array_equal(editor.mask, hole) and not editor.points, "取消当前多边形保留全部已画掩膜")
    editor.undo()
    check(editor.points == [(10, 10)], "取消多边形也能撤销")
    editor.clear()
    editor.undo()
    check(np.array_equal(editor.mask, hole), "误清空整张后可撤销恢复成果")
    editor.redo()
    editor.point((1, 1))
    check(not editor.future, "撤销后新编辑废弃旧恢复分支，避免恢复错误版本")
    with TemporaryDirectory(prefix="mask_draft_") as tmp:
        draft = Path(tmp) / "draft.json"
        editor = MaskEditor((40, 40), hole)
        editor.point((8, 9))
        save_mask_draft(draft, editor, "raw_hash", "P1_base")
        restored = load_mask_draft(draft, (40, 40), "raw_hash", "P1_base")
        check(np.array_equal(restored.mask, hole) and restored.points == [(8, 9)],
              "退出后草稿可恢复已填掩膜和未提交顶点")
        raises(lambda: load_mask_draft(draft, (40, 40), "other_raw", "P1_base"),
               "其他原图的草稿不得误恢复")
    with patch("annotate_part_dataset.cv2.setMouseCallback"), patch("annotate_part_dataset.cv2.imshow"), \
            patch("annotate_part_dataset.cv2.waitKey", return_value=ord("s")), \
            patch("annotate_part_dataset.cv2.getWindowProperty", return_value=1):
        result = paint_mask(np.zeros((40, 40, 3), np.uint8),
                            {"part": "P1_base", "bbox": [12, 12, 5, 5]}, initial=outer)
        check(np.array_equal(result, outer), "人工掩膜允许越出粗定位框，保存不裁切金属")
    geometry = mask_view_geometry((480, 640, 3), [366, 225, 69, 67])
    x0, y0, cw, ch, scale = geometry
    check(scale > 1 and x0 <= 366 and y0 <= 225 and x0 + cw >= 435 and y0 + ch >= 292,
          "精细掩膜自动放大完整零件并保留周围背景")
    for factor in (1, scale, 12):
        g = (x0, y0, cw, ch, factor)
        check(mask_view_point((400 - x0) * factor + factor - 1,
                              (258 - y0) * factor + MASK_TOOLBAR + factor - 1, g) == (400, 258),
              "放大后的鼠标坐标精确映射回原图像素，倍率%d" % factor)
        check(mask_view_point(0, MASK_TOOLBAR - 1, g) is None and
              mask_view_point(cw * factor, MASK_TOOLBAR, g) is None,
              "工具栏和图像外点击不写入多边形")
    edge = mask_view_geometry((480, 640, 3), [0, 0, 30, 30])
    check(edge[0] == edge[1] == 0 and mask_view_point(0, MASK_TOOLBAR, edge) == (0, 0),
          "靠图像边缘的零件裁切与坐标映射不发生负偏移")
    policy = load_config(str(POLICY))
    # 本机标定不入库；复现其四类模板加说明字段的结构，避免克隆后依赖私有标定。
    legacy = {part: {"ranges": {"circularity": [.1, .9]}} for part in PARTS}
    legacy["_ranges_note"] = "旧标定中的区间说明，不是零件模板"
    classifier = classifier_from_templates(legacy, policy)
    check({t.part for t in classifier.templates} == set(PARTS),
          "正式旧标定的说明字段不应被解析为零件，四类模板均保留")
    raises(lambda: classifier_from_templates({"P1_base": "损坏模板"}, policy),
           "非说明字段的损坏模板必须明确报错")
    print("== 人工真值与独立计分 ==")
    policy = load_config(str(POLICY))
    tasks = capture_tasks(policy)
    check(len(tasks) == 60 and sum(t["part"] is None for t in tasks) == 12,
          "每批 48 个零件任务 + 12 个负样本，任务不自动成为标注")
    repaired = [dict(kind="image", task=tasks[0], fulfilled_task_id="007"),
                dict(kind="image", task=tasks[1]), dict(kind="image", task=tasks[2])]
    pending = remaining_tasks(tasks, repaired)
    check(pending[0]["id"] == "001" and all(t["id"] != "007" for t in pending),
          "错背景样本复用为绿幕名额后，补最初黑盘任务且不重复绿幕任务")
    raises(lambda: remaining_tasks(tasks, repaired + [repaired[0]]),
           "同一采集名额不能被两张原图重复占用")
    completed_54 = [dict(kind="image", task=t) for t in tasks[:54]]
    omitted = [dict(id="055", reason="现场不能形成明显阴影"), dict(id="056", reason="现场不能形成明显阴影")]
    check(remaining_tasks(tasks, completed_54, omitted)[0]["id"] == "057" and len(completed_54) == 54,
          "现场无法采阴影时直接进入手套任务，不生成虚假已采样本")
    raises(lambda: remaining_tasks(tasks, completed_54, [dict(id="001", reason="已有样本")]),
           "不能用跳过任务掩盖已采集的失败图")
    raises(lambda: remaining_tasks(tasks, [], [dict(id="055", reason="")]), "跳过必须留原因")
    supplement = selected_tasks(policy, "P4_cap", "smooth_up", "green")
    check(len(supplement) == 6 and supplement[0]["id"] == "043"
          and all(t["appearance"] == "smooth_up" and t["part"] == "P4_cap" for t in supplement),
          "顶盖光滑面补采可独立选6张绿幕，不重复其他类别或有孔面")
    raises(lambda: selected_tasks(policy, "P1_base", "smooth_up"), "不能给底座冒记顶盖朝上面")
    check(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1, "相同位置框 IoU=1")
    truth = [dict(part="P1_base", bbox=[0, 0, 10, 10])]
    wrong = [dict(part="P3_wheel", bbox=[0, 0, 10, 10])]
    counts, matrix, _ = score_frame(truth, wrong, .5)
    check(counts["P1_base"]["fn"] == 1 and counts["P3_wheel"]["fp"] == 1,
          "错认同时计真类漏检和预测类误检，不虚增精确率")
    counts, _, _ = score_frame(truth, [], .5)
    check(counts["P1_base"]["fn"] == 1, "没有任何候选块也必须计漏检")
    counts, _, pairs = score_frame(truth, [dict(part="P1_base", bbox=[0, 0, 10, 10])] * 2, .5)
    check(counts["P1_base"]["tp"] == 1 and counts["P1_base"]["fp"] == 1,
          "重复检出只能一个 TP，另一个 FP")
    pairs = match_boxes([dict(bbox=[0, 0, 10, 10]), dict(bbox=[7, 0, 10, 10])],
                        [dict(bbox=[2, 0, 10, 10]), dict(bbox=[-3, 0, 10, 10])], .3)
    check(len(pairs) == 2 and {(a, b) for a, b, _ in pairs} == {(0, 1), (1, 0)},
          "全局匹配保住第二件唯一候选，避免贪心丢真件")
    raises(lambda: validate_annotation(dict(reviewed=False, objects=[]), (100, 100, 3)),
           "拒绝尚未确认的空标注")
    raises(lambda: validate_annotation(dict(reviewed=True, objects=[dict(part="P1_base", bbox=[95, 0, 10, 10])]),
                                       (100, 100, 3)), "拒绝越界真值")
    print("== 采集契约、拟合和评测端到端（合成工具自检） ==")
    with TemporaryDirectory(prefix="part_dataset_test_") as tmp:
        base = Path(tmp)
        stable = base / "atomic.json"
        write_json(stable, {"version": 1})
        replace_real = os.replace
        calls = []

        def transient_replace(source, target):
            calls.append(1)
            if len(calls) <= 2:
                check(read_json(stable) == {"version": 1}, "替换被拒期间原有效清单保持完整")
                raise PermissionError("模拟Windows短暂占用")
            return replace_real(source, target)

        with patch("part_dataset.os.replace", side_effect=transient_replace), patch("part_dataset.time.sleep"):
            write_json(stable, {"version": 2})
        check(len(calls) == 3 and read_json(stable) == {"version": 2},
              "临时拒绝替换可自动重试，不导致录像立即中断")
        with patch("part_dataset.os.replace", side_effect=PermissionError("持续拒绝")) as replacer, \
                patch("part_dataset.time.sleep"):
            raises(lambda: write_json(stable, {"version": 3}), "持续权限问题有限重试后明确报错")
        check(replacer.call_count == 8 and read_json(stable) == {"version": 2}
              and read_json(stable.with_suffix(".json.tmp")) == {"version": 3},
              "永久失败保留原有效清单及完整临时清单供恢复")
        print("== 回归真机曝光未应用：必须先开流，再锁参数，再预热/读图 ==")

        class FakeSensor:
            values = {"option.exposure": 156.0, "option.enable_auto_exposure": 1.0,
                      "option.enable_auto_white_balance": 1.0}

            def get_supported_options(self):
                return list(self.values)

            def get_option(self, key):
                return self.values[key]

        class FakeSource:
            rs = SimpleNamespace(option=SimpleNamespace(exposure="option.exposure", gain="option.gain",
                enable_auto_exposure="option.enable_auto_exposure",
                enable_auto_white_balance="option.enable_auto_white_balance"))
            name = "FAKE_SDK_CAPTURE_CONTRACT"
            opened = applied = closed = False
            effective_exposure = 126.0

            def __init__(self):
                self.sensor = FakeSensor()

            def open(self):
                self.opened = True

            def apply_locked_controls(self):
                assert self.opened, "锁控件必须在 open 后"
                self.sensor.values = {"option.exposure": self.effective_exposure, "option.enable_auto_exposure": 0.0,
                                      "option.enable_auto_white_balance": 0.0}
                self.applied = True

            def _target_sensors(self):
                return [self.sensor]

            def _sensor_name(self, sensor):
                return "RGB Camera"

            def color_intrinsics(self):
                return dict(fx=603.82)

            def read(self, want_full=False):
                assert self.opened and self.applied, "任何预热或保存前必须先应用控件"
                img = np.zeros((48, 64, 3), np.uint8)
                return SimpleNamespace(left_full=img, left=img, ts=1.0)

            def close(self):
                self.closed = True

        source = FakeSource()
        failed_clip_source = FakeSource()
        clip_writes = []

        def fail_frame_manifest(path, data):
            clip_writes.append(1)
            if len(clip_writes) == 3:
                raise PermissionError("模拟第2帧已写盘但清单替换失败")
            return write_json(path, data)

        with patch("capture_part_dataset.make_source", return_value=failed_clip_source), \
                patch("capture_part_dataset.write_json", side_effect=fail_frame_manifest), \
                patch("capture_part_dataset.cv2.imshow"), \
                patch("capture_part_dataset.cv2.waitKey", return_value=255), \
                patch("capture_part_dataset.cv2.getWindowProperty", return_value=1), \
                patch("capture_part_dataset.cv2.destroyAllWindows"):
            raises(lambda: capture_main(["--dataset", str(base / "capture"), "--session", "clip_failed",
                                         "--split", "dev", "--clip", "pick_place", "--seconds", "30"]),
                   "异常录像不能伪装为正常完成")
        failed = read_json(base / "capture" / "clip_failed" / "session.json")
        check(len(failed["samples"]) == 2 and not failed["recording_summary"]["stopped_normally"]
              and failed["recording_summary"]["controls_end_verified"] and failed_clip_source.closed,
              "异常收尾补齐已写原图记录、回读控件并释放设备")
        with patch("capture_part_dataset.make_source", return_value=source):
            capture_main(["--dataset", str(base / "capture"), "--session", "check_fixed",
                          "--split", "check", "--check", "--headless"])
        captured = read_json(base / "capture" / "check_fixed" / "session.json")
        check(source.closed and captured["actual_controls"]["RGB Camera"]["option.exposure"] == 126,
              "回读 156 的设备先应用 126，再保存原图和实际控件；退出时释放相机")
        guided_source = FakeSource()
        with patch("capture_part_dataset.make_source", return_value=guided_source), \
                patch("capture_part_dataset.cv2.imshow"), \
                patch("capture_part_dataset.cv2.waitKey", return_value=ord("s")), \
                patch("capture_part_dataset.cv2.getWindowProperty", return_value=1), \
                patch("capture_part_dataset.cv2.destroyAllWindows"):
            capture_main(["--dataset", str(base / "capture"), "--session", "guided_one",
                          "--split", "dev", "--limit", "1"])
        guided_meta = read_json(base / "capture" / "guided_one" / "session.json")
        check(len(guided_meta["samples"]) == 1 and guided_source.closed,
              "逐步采集 limit=1 在保存一次后自动退出并释放相机")
        guided_meta["skipped_tasks"] = [dict(id="002", reason="现场无法复现该采集条件")]
        write_json(base / "capture" / "guided_one" / "session.json", guided_meta)
        with patch("capture_part_dataset.make_source", return_value=FakeSource()), \
                patch("capture_part_dataset.cv2.imshow"), \
                patch("capture_part_dataset.cv2.waitKey", return_value=ord("s")), \
                patch("capture_part_dataset.cv2.getWindowProperty", return_value=1), \
                patch("capture_part_dataset.cv2.destroyAllWindows"):
            capture_main(["--dataset", str(base / "capture"), "--session", "guided_one",
                          "--split", "dev", "--limit", "1"])
        continued = read_json(base / "capture" / "guided_one" / "session.json")
        check(len(continued["samples"]) == 2 and continued["samples"][-1]["task"]["id"] == "003",
              "真实采集入口断点续采会跳过有原因的任务，原图编号保持连续")
        with patch("capture_part_dataset.make_source", return_value=FakeSource()), \
                patch("capture_part_dataset.cv2.imshow"), \
                patch("capture_part_dataset.cv2.waitKey", return_value=ord("s")), \
                patch("capture_part_dataset.cv2.getWindowProperty", return_value=1), \
                patch("capture_part_dataset.cv2.destroyAllWindows"):
            capture_main(["--dataset", str(base / "capture"), "--session", "cap_smooth",
                          "--split", "dev", "--part", "P4_cap", "--appearance", "smooth_up",
                          "--background", "green", "--limit", "1"])
        supplementary = read_json(base / "capture" / "cap_smooth" / "session.json")
        check(supplementary["samples"][0]["task"]["id"] == "043"
              and supplementary["samples"][0]["task"]["appearance"] == "smooth_up",
              "补采实际入口保存独立会话和朝上面提示，原图从000001开始")
        check(windows_mf_exposure(126) == 156 and windows_mf_exposure(78) == 78
              and windows_mf_exposure(600) == 625,
              "Windows MF 的已知曝光换算：126→156、78→78、600→625")
        exposure_cam = load_config(str(ROOT / "config" / "camera.yaml"))
        observed = {"RGB Camera": {"option.exposure": 156.0, "option.enable_auto_exposure": 0.0,
                                    "option.enable_auto_white_balance": 0.0}}
        with patch("capture_part_dataset.sys.platform", "win32"):
            result = check_locked(source, observed, exposure_cam, "windows_mf")
            check(next(r for r in result if r["control"] == "exposure")["status"] == "windows_mf_quantized",
                  "仅已知的 Windows 曝光档位换算可以通过，元数据保留换算标记")
            raises(lambda: check_locked(source, observed, exposure_cam, "strict"),
                   "严格策略仍拒绝不同曝光，不直接放宽容差")
            broken = copy.deepcopy(observed)
            broken["RGB Camera"]["option.exposure"] = 312.0
            raises(lambda: check_locked(source, broken, exposure_cam, "windows_mf"),
                   "真正错误的曝光 312 不能冒充 126 的有效档位")
            broken = copy.deepcopy(observed)
            broken["RGB Camera"]["option.enable_auto_exposure"] = 1.0
            raises(lambda: check_locked(source, broken, exposure_cam, "windows_mf"),
                   "曝光换算不放行自动曝光开启")
            broken = copy.deepcopy(observed)
            broken["RGB Camera"]["option.gain"] = 64.0
            raises(lambda: check_locked(source, broken, exposure_cam, "windows_mf"),
                   "曝光换算不放行其他控件漂移")
            quantized_source = FakeSource()
            quantized_source.effective_exposure = 156.0
            with patch("capture_part_dataset.make_source", return_value=quantized_source):
                capture_main(["--dataset", str(base / "capture"), "--session", "check_quantized",
                              "--split", "check", "--check", "--headless"])
            quantized_meta = read_json(base / "capture" / "check_quantized" / "session.json")
            exposure_row = next(r for r in quantized_meta["control_validation"] if r["control"] == "exposure")
            check(exposure_row["requested"] == 126 and exposure_row["observed"] == [156]
                  and len(quantized_meta["samples"]) == 1 and quantized_source.closed,
                  "真机日志的请求126/回读156路径可完整保存一张原图并释放设备")
        with patch("capture_part_dataset.sys.platform", "linux"):
            raises(lambda: check_locked(source, observed, exposure_cam, "windows_mf"),
                   "Linux 不套用 Windows 后端曝光换算")
        cfg = copy.deepcopy(policy.data)
        cfg["experiment"]["min_samples_per_class"] = 2
        cfg_path = base / "policy.json"
        write_json(cfg_path, cfg)
        wp = ROOT / "config" / "workpiece.yaml"
        make_session(base / "dev_a", "dev", wp, cfg_path, 2)
        make_session(base / "test_b", "test", wp, cfg_path, 1)
        samples = load_samples(base, "dev")
        check(len(samples) == 8, "只读取指定开发会话，保留原图与标注 hash")
        model = base / "model.json"
        fit(SimpleNamespace(dataset=base, policy=cfg_path, workpiece=wp,
                            output=model, variant="brass", features="shape_scale"))
        check(len(read_json(model)["templates"]) == 4, "四类模板由开发集建立")
        check("holes" not in read_json(model)["templates"]["P4_cap"]["ranges"] and
              "holes" in read_json(model)["templates"]["P1_base"]["ranges"],
              "新顶盖模板不使用孔数区分身份，其他类别特征保持配置")
        check(all(r["session"] == "dev_a" for r in read_json(model)["source"]), "模板来源不包含验收会话")
        demo_artifact = read_json(model)
        demo_artifact["policy"]["experiment"]["tracking_overrides"] = dict(backend="detection", detect_every_n=1)
        demo_model = base / "demo_model.json"
        write_json(demo_model, demo_artifact)
        replay_meta = read_json(base / "dev_a" / "session.json")
        for index, record in enumerate(replay_meta["samples"]):
            record["elapsed_s"] = index * .1  # 合成测试时钟，不是真机帧率证据。
        replay_path = base / "dev_a" / "demo_session.json"
        write_json(replay_path, replay_meta)
        demo_args = SimpleNamespace(dataset_model=demo_model, workpiece=wp,
            camera=ROOT / "config/camera.yaml", session=replay_path,
            report=base / "demo_report.json", bench=True, no_preview=False,
            max_frames=2, source=None, path=None)
        demo_report = run_dataset_model(demo_args)
        check(demo_report["frames"] == 2 and demo_report["tracker_config"]["backend"] == "detection"
              and (base / "demo_report.json").exists(),
              "独立模型预览实际入口可按清单时钟回放并生成报告，不读取旧标定")
        demo_args.report = ""
        demo_args.bench, demo_args.max_frames = False, 4
        demo_args.snapshot_dir = base / "中文诊断快照"
        with patch("part_track_demo.cv2.imshow"), \
                patch("part_track_demo.cv2.waitKey", side_effect=[ord("s"), ord("s"), ord("S"), ord("q")]), \
                patch("part_track_demo.cv2.getWindowProperty", return_value=1), \
                patch("part_track_demo.cv2.destroyAllWindows"):
            saved_demo = run_dataset_model(demo_args)
        paths = [Path(p) for p in saved_demo["snapshots"]]
        check(len(paths) == 3 and len(set(paths)) == 3 and saved_demo["frames"] == 4,
              "窗口内连续按s/s/S分别保存三次，q正常退出，快照互不覆盖")
        first_evidence = read_json(paths[0])
        first_original = imread_safe(str(base / "dev_a" / replay_meta["samples"][0]["image"]))
        check(np.array_equal(imread_safe(str(paths[0].parent / "raw.png")), first_original)
              and set(first_evidence["files"]) == {"raw.png", "overlay.png", "mask.png"}
              and first_evidence["frame"] == 1 and first_evidence["elapsed_s"] == 0
              and "predictions" in first_evidence and "stages_ms" in first_evidence
              and all(file_hash(paths[0].parent / name) == digest for name,digest in first_evidence["files"].items()),
              "中文路径快照保留无文字原图、叠加图和掩膜，关联当前帧预测/时钟/耗时及hash")
        with patch("part_track_demo.cv2.imshow"), \
                patch("part_track_demo.cv2.waitKey", side_effect=[ord("s"), ord("q")]), \
                patch("part_track_demo.cv2.getWindowProperty", return_value=1), \
                patch("part_track_demo.cv2.destroyAllWindows"):
            repeated_snapshot = run_dataset_model(demo_args)
        check(Path(repeated_snapshot["snapshots"][0]) not in paths
              and all(p.exists() for p in paths), "重启后再保存第1帧也不会覆盖上次证据")
        with patch("part_track_demo.cv2.imshow"), \
                patch("part_track_demo.cv2.waitKey", side_effect=[ord("s"), ord("q")]), \
                patch("part_track_demo.cv2.getWindowProperty", return_value=1), \
                patch("part_track_demo.cv2.destroyAllWindows"), \
                patch("part_track_demo.save_preview_snapshot", side_effect=OSError("模拟写入被拒")):
            failed_snapshot = run_dataset_model(demo_args)
        check(failed_snapshot["snapshot_save_errors"] == 1 and not failed_snapshot["snapshots"]
              and all(p.exists() for p in paths), "快照失败明确记录且预览可退出，先前快照保持完整")
        # 自动诊断使用真实分割，再制造同一帧重复候选，验证原始错误不会被轨迹去重吞掉。
        from part_dataset import segment_variant as original_segment
        def duplicate_candidates(*args):
            mask, blobs, result = original_segment(*args)
            return mask, blobs + blobs, result
        demo_args.bench, demo_args.max_frames = True, 3
        demo_args.diagnostics = True
        demo_args.diagnostic_slow_ms = 1e-9
        demo_args.diagnostic_limit = 1
        demo_args.report = base / "自动诊断报告.json"
        with patch("part_dataset.segment_variant", side_effect=duplicate_candidates), \
                patch("part_track_demo.cv2.imshow") as show:
            automatic = run_dataset_model(demo_args)
        diagnostic = automatic["diagnostics"]
        trace = diagnostic["frame_trace"]
        check(len(trace) == 3 and automatic["raw_duplicate_identity_frames"] == 3
              and all(t["reasons"] == ["slow", "duplicate_identity"] for t in trace)
              and [t["elapsed_s"] for t in trace] == [0, .1, .2]
              and all(t["process_cpu_ms"] >= 0 and t["tracks"] for t in trace),
              "逐帧保留真实清单时间、分段/CPU耗时、原始重复身份与轨迹，限量后仍记录异常")
        check(diagnostic["attempts"] == {"slow": 1, "duplicate_identity": 1}
              and len(diagnostic["snapshots"]) == 1 and not show.called
              and not automatic["snapshots"],
              "无窗口也自动保留异常，同帧两原因共用一份快照，各类独立限量且不混入人工计数")
        evidence_path = Path(diagnostic["snapshots"][0])
        evidence = read_json(evidence_path)
        check(evidence["triggers"] == ["slow", "duplicate_identity"]
              and evidence["frame"] == trace[0]["frame"]
              and evidence["tracks"] == trace[0]["tracks"]
              and evidence["stages_ms"] == trace[0]["stages_ms"]
              and np.array_equal(imread_safe(str(evidence_path.parent / "raw.png")), first_original)
              and all(file_hash(evidence_path.parent / n) == h for n,h in evidence["files"].items()),
              "自动快照和触发帧一致，保留原像素、完整预测、稳定key及hash")
        demo_args.report = base / "自动诊断失败.json"
        with patch("part_track_demo.save_preview_snapshot", side_effect=OSError("拒绝访问")) as saving:
            automatic_failure = run_dataset_model(demo_args)
        check(saving.call_count == 1 and automatic_failure["diagnostics"]["save_errors"] == 1
              and len(automatic_failure["diagnostics"]["frame_trace"]) == 3
              and evidence_path.exists(), "自动保存失败也消耗额度，后续帧继续记账且旧证据保留")
        confirmed_artifact = copy.deepcopy(demo_artifact)
        confirmed_artifact["policy"]["experiment"]["confirmation"] = dict(frames=7, min_score=0,
            max_gap_seconds=.25, dist_match_px=60, iou_match=.15, area_ratio=[.8, 1.25])
        confirmed_model = base / "confirmed_model.json"
        write_json(confirmed_model, confirmed_artifact)
        confirmed_meta = copy.deepcopy(replay_meta)
        confirmed_meta["samples"] = [dict(replay_meta["samples"][0], elapsed_s=i/30) for i in range(9)]
        confirmed_session = base / "dev_a" / "confirmed_session.json"
        write_json(confirmed_session, confirmed_meta)
        confirmed_args = copy.copy(demo_args)
        confirmed_args.dataset_model, confirmed_args.session = confirmed_model, confirmed_session
        confirmed_args.max_frames, confirmed_args.report = 0, base / "confirmed_report.json"
        confirmed_args.diagnostic_slow_ms = 10000
        confirmed_report = run_dataset_model(confirmed_args)
        confirmed_trace = confirmed_report["diagnostics"]["frame_trace"]
        check([len(row["tracks"]) for row in confirmed_trace] == [0]*6+[1]*3
              and [row["confirmation"][0]["frames"] for row in confirmed_trace] == [1,2,3,4,5,6,7,7,7],
              "实际演示入口前六帧不建身份，第七帧建轨迹，后续继续同一key且保留确认计数")
        confirmed_args.report = base / "confirmed_ambiguous_report.json"
        with patch("part_dataset.segment_variant", side_effect=duplicate_candidates):
            ambiguous_report = run_dataset_model(confirmed_args)
        check(ambiguous_report["raw_duplicate_identity_frames"] == 9
              and ambiguous_report["track_count_range"] == [0,0],
              "多帧闸门拒绝同类歧义后仍完整报告原始九帧重复身份，不能伪装成原始检测正确")
        demo_args.report = ""
        raises(lambda: run_dataset_model(demo_args), "自动诊断必须指定可保存的报告")
        demo_args.report = base / "invalid_diagnostic.json"
        demo_args.diagnostic_slow_ms = float("nan")
        raises(lambda: run_dataset_model(demo_args), "拒绝非有限慢帧阈值")
        demo_args.diagnostic_slow_ms = 33
        demo_args.diagnostic_limit = 0
        raises(lambda: run_dataset_model(demo_args), "拒绝无效自动保存上限")
        demo_args.diagnostics = False
        demo_args.report = base / "demo_report.json"
        demo_args.bench, demo_args.max_frames = True, 2
        raises(lambda: run_dataset_model(demo_args), "模型预览报告禁止覆盖")
        demo_args.report = ""
        replay_meta["samples"][0]["sha256"] = "broken"
        write_json(replay_path, replay_meta)
        raises(lambda: run_dataset_model(demo_args), "预览也拒绝原图hash失配")
        demo_artifact["camera_sha256"] = file_hash(demo_args.camera)
        write_json(demo_model, demo_artifact)
        demo_args.session, demo_args.max_frames = "", 1
        live_source = FakeSource()
        with patch("assemvision.sources.factory.make_source", return_value=live_source):
            live_report = run_dataset_model(demo_args)
        check(live_report["frames"] == 1 and live_source.applied and live_source.closed
              and live_report["controls_before"] == live_report["controls_after"],
              "模型真机入口契约：锁参数、预热、首尾校验并关闭源；模拟SDK不代表真机已验收")
        class DriftingSource(FakeSource):
            reads = 0
            def read(self, want_full=False):
                frame = super().read(want_full)
                self.reads += 1
                if self.reads > int(demo_artifact["policy"]["capture"]["warmup_frames"]):
                    self.sensor.values["option.enable_auto_exposure"] = 1.0
                return frame
        drifting_source = DriftingSource()
        with patch("assemvision.sources.factory.make_source", return_value=drifting_source):
            raises(lambda: run_dataset_model(demo_args), "预览结束发现相机参数漂移必须报错")
        check(drifting_source.closed, "预览异常退出也关闭相机源")
        args = SimpleNamespace(dataset=base, policy=cfg_path, workpiece=wp, templates=model,
                               split="test", baseline=False, scope="full", clips=False, output=base / "report")
        evaluate(args)
        report = read_json(args.output / "report.json")
        check(report["image_count"] == 4 and report["static_image_count"] == 4,
              "完整评测原图并输出未遮挡静态统计")
        check(sum(r["tp"] + r["fn"] for r in report["class_metrics"].values()) == 4,
              "所有真件都进入分母，不能只统计认出的块")
        check((args.output / "report.md").exists() and (base / "test_access.jsonl").exists(),
              "保留可读报告与验收集读取日志")
        raises(lambda: evaluate(args), "拒绝覆盖历史验收报告")
        data = read_json(model)
        data["source"][0]["session"] = "test_b"
        write_json(model, data)
        args.output = base / "leak"
        raises(lambda: evaluate(args), "验收会话与建模板来源重叠时拒绝执行")
        with patch("annotate_part_dataset.cv2.namedWindow"), patch("annotate_part_dataset.cv2.destroyAllWindows"), \
                patch("annotate_part_dataset.paint_mask", return_value=np.ones((180, 200), np.uint8)) as painter:
            annotate_main(["--dataset", str(base), "--session", "dev_a", "--masks",
                           "--part", "P4_cap", "--limit", "1"])
            check(painter.call_count == 1 and painter.call_args.args[1]["part"] == "P4_cap",
                  "按类别筛选掩膜时跳过其他零件，limit仅计实际处理图片")
            saved = [read_json(p) for p in (base / "dev_a" / "raw").glob("*.json")]
            masked = [o for a in saved for o in a["objects"] if o.get("mask")]
            check(len(masked) == 1 and masked[0]["part"] == "P4_cap",
                  "分工标注只保存指定类别的一张，不写其他同学的零件")
        raw = base / "dev_a" / "raw" / "000.png"
        raw.write_bytes(b"changed")
        raises(lambda: load_samples(base, "dev"), "原图被覆盖或无法解码必须报错，不静默跳过")
    print("阶段 1 数据工具测试全部通过；这些合成结果不代表真机识别精度。")


if __name__ == "__main__":
    main()
