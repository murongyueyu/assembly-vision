# -*- coding: utf-8 -*-
"""阶段 1 独立真值评测：覆盖错认双计数、全局匹配、数据完整性和完整工具链。"""
from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from _common import check
from part_dataset import (ROOT, POLICY, PARTS, bbox_iou, capture_tasks, file_hash,
                          load_config, load_samples, match_boxes, read_json, save_image,
                          score_frame, validate_annotation, write_json)
from part_dataset import remaining_tasks, selected_tasks, classifier_from_templates
from measure_part_dataset import fit, evaluate
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
        check(all(r["session"] == "dev_a" for r in read_json(model)["source"]), "模板来源不包含验收会话")
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
