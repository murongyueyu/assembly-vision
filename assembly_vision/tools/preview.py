# -*- coding: utf-8 -*-
"""带判据叠加的实时预览 —— 标定/调试时"看得见画面"的统一入口。

用法（在 assembly_vision 目录下）：
    python tools/preview.py                 # 看全部：ROI + 分割 + 覆盖率 + 过曝
    python tools/preview.py --guide assembly   # 绿圈高亮装配位（摆装配体时用）
    python tools/preview.py --guide slot_wheel # 绿圈高亮轮盘槽
    python tools/preview.py --plain            # 只看原图（不叠加任何东西）
    python tools/preview.py --save out.png     # 退出时存一张

窗口里能看到什么：
    红色多边形   各槽位 ROI + 装配位 ROI
    绿色粗圈     --guide 指定的那个位置（"东西该放这里"）
    黄色         被切成"黄铜"的像素
    品红         被切成"金属"的像素（不靠色相那条通路，给装配体数辐条用）
    左上角       分辨率 / 帧率 / 过曝量 / V 均值 / 当前曝光
    左下角       各槽覆盖率 + 识别结果 + 判定
    右下角       操作提示
    ★ 过曝 > 0.5% 时整屏描红边 + 大字报警

★★ 为什么要有这个工具（实测踩过，代价很大）：
    标定脚本原来是"盲跑" —— 在终端按回车，脚本自己去抓帧判断，中途看不到画面。
    一次严重过曝（曝光 -5.0 在当时的工位光照下近全白像素 8%，
    轮盘的黄铜面积从 3200px 塌到 626px）把整条判据链弄废，
    而终端只报一句"扫优失败"，完全无从下手。
    根因是**判据失效时画面看上去只是"亮一点"**，不叠加就看不出来。
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from assemvision.config import load_config                       # noqa: E402
from assemvision.core.checks import Qualifier                    # noqa: E402
from assemvision.core.roi import TrayMap                         # noqa: E402
from assemvision.core.segment import Segmenter                   # noqa: E402
from assemvision.imgio import imwrite_safe                       # noqa: E402
from assemvision.sources.factory import make_source              # noqa: E402
from calib._util import open_source                             # noqa: E402

WIN = "预览  [q/ESC=退出  s=存图  r=重载标定]"


def _poly_draw(vis, targets, guide):
    for key, poly in targets.items():
        if poly is None or len(poly) < 3:
            continue
        pts = np.round(np.asarray(poly)).astype(np.int32).reshape(-1, 1, 2)
        if guide and key == guide:
            cv2.polylines(vis, [pts], True, (0, 255, 0), 5)      # 绿圈 = 该放这里
        else:
            cv2.polylines(vis, [pts], True, (0, 0, 255), 2)


def _txt(vis, s, x, y, scale=0.5):
    cv2.putText(vis, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(vis, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="uvc | video | images")
    ap.add_argument("--path", default=None, help="--source video/images 时的路径")
    ap.add_argument("--camera", default="")
    ap.add_argument("--workpiece", default="")
    ap.add_argument("--calibration", default="")
    ap.add_argument("--guide", default="",
                    help="绿圈高亮：assembly | slot_wheel | slot_column | slot_cap | slot_base")
    ap.add_argument("--plain", action="store_true", help="只看原图，不做任何叠加")
    ap.add_argument("--save", default="data/refs/preview.png", help="按 s 或退出时存图")
    ap.add_argument("--exposure", type=float, default=None,
                    help="临时改曝光（不改配置文件），用来找不过曝的值")
    args = ap.parse_args()

    C = os.path.join(ROOT, "config")
    cam = load_config(args.camera or os.path.join(C, "camera.yaml"))
    wp = load_config(args.workpiece or os.path.join(C, "workpiece.yaml"))
    cal = load_config(args.calibration or os.path.join(C, "calibration.yaml"),
                      required=False)

    seg = Segmenter(wp)
    q = None
    if cal.data and cal.get("tray_map.homography", None) is not None:
        try:
            tray = TrayMap(wp,
                           homography=np.asarray(cal.get("tray_map.homography"), np.float64),
                           ref_scale=float(cal.get("tray_map.scale", 1.0)))
            q = Qualifier(wp, tray, cal)
        except Exception as e:                                   # noqa: BLE001
            print("（标定数据用不了，只显示分割：%s）" % e)
    else:
        print("（没有 calibration.yaml 的 tray_map，只显示分割、不画 ROI）")

    src = open_source(cam, args)
    expo = None
    if args.exposure is not None:
        cap = getattr(src, "cap", None) or getattr(src, "_cap", None)
        if cap is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, float(args.exposure))
            expo = float(args.exposure)
            print("已临时把曝光设为 %.1f（不改配置文件）" % expo)
    print("窗口按键：q/ESC 退出   s 存图   r 重载标定")
    print("颜色：红=ROI  绿=--guide 指定位置  黄=黄铜  品红=金属掩膜")

    try:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    except Exception:                                            # noqa: BLE001
        pass

    last = None
    fps = 0.0
    import time
    t0 = time.time()
    while True:
        f = src.read(want_full=False)
        if f is None:
            time.sleep(0.02)
            continue
        bgr = f.left if getattr(f, "left", None) is not None else f.left_full
        if bgr is None:
            continue
        last = bgr
        vis = bgr.copy()
        info, tail = [], []
        blown = 0.0
        if not args.plain:
            try:
                res = seg.segment(bgr, scale=f.scale)
                ov = bgr.copy()
                # ★ 黄铜半透明填充，金属掩膜只画轮廓 ——
                #   金属掩膜在真机帧上占 28%，填充会把画面糊住看不见零件。
                if res.brass_mask is not None:
                    ov[res.brass_mask > 0] = (0, 255, 255)
                vis = cv2.addWeighted(vis, 0.6, ov, 0.4, 0)
                mm_ = getattr(res, "metal_mask", None)
                if mm_ is not None and int(np.count_nonzero(mm_)) > 200:
                    cnts, _ = cv2.findContours(mm_, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    big = [c for c in cnts if cv2.contourArea(c) > 40]
                    cv2.drawContours(vis, big, -1, (255, 0, 255), 2)
                blown = float(res.detail.get("blown_ratio", 0.0))
                info.append("blown %.2f%%   V %.0f" % (100 * blown,
                                                       res.detail.get("v_mean", 0.0)))
                if q is not None:
                    for s in q.check_slots(res, f.scale, bgr=bgr):
                        idp = "?"
                        try:
                            idp = s.features.get("id_part") or "?"
                        except Exception:                        # noqa: BLE001
                            pass
                        tail.append("%-7s cov %.2f  %-10s %s"
                                    % (s.slot_id.replace("slot_", ""), s.coverage,
                                       idp, s.verdict))
            except Exception as e:                               # noqa: BLE001
                info.append("分段失败: %s" % e)
            if q is not None:
                try:
                    tgt = dict(q.tray.slot_polygons_px(f.scale))
                    ap_ = q.tray.assembly_polygon_px(f.scale)
                    if ap_ is not None:
                        tgt["assembly"] = ap_
                    _poly_draw(vis, tgt, args.guide)
                except Exception:                                # noqa: BLE001
                    pass

        now = time.time()
        dt = now - t0
        t0 = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        _txt(vis, "%dx%d  %.1f fps" % (bgr.shape[1], bgr.shape[0], fps), 8, 22, 0.55)
        for i, s in enumerate(info):
            _txt(vis, s, 8, 44 + 20 * i, 0.55)
        if expo is not None:
            _txt(vis, "曝光 %.1f (临时)" % expo, 8, 44 + 20 * len(info), 0.55)
        for i, s in enumerate(reversed(tail)):
            _txt(vis, s, 8, vis.shape[0] - 34 - 18 * i, 0.45)
        _txt(vis, "q/ESC 退出   s 存图   r 重载标定", 8, vis.shape[0] - 12, 0.45)
        if blown > 0.005:
            cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 0, 255), 8)
            msg = ("!! 过曝 %.1f%% -> 黄铜会被判成绿垫，判据会失效 !! 调低曝光"
                   % (100 * blown))
            _txt(vis, msg, 10, 44 + 20 * (len(info) + 2), 0.6)
            cv2.putText(vis, msg, (10, 44 + 20 * (len(info) + 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(WIN, vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s") and last is not None:
            out = args.save if os.path.isabs(args.save) else os.path.join(ROOT, args.save)
            imwrite_safe(out, vis)
            print("已存图:", out)
        if k == ord("r"):
            cal = load_config(args.calibration or os.path.join(C, "calibration.yaml"),
                              required=False)
            if cal.data and cal.get("tray_map.homography", None) is not None:
                try:
                    tray = TrayMap(wp, homography=np.asarray(cal.get("tray_map.homography"),
                                                             np.float64),
                                   ref_scale=float(cal.get("tray_map.scale", 1.0)))
                    q = Qualifier(wp, tray, cal)
                    print("已重载标定")
                except Exception as e:                           # noqa: BLE001
                    print("重载失败:", e)
        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:                                        # noqa: BLE001
            pass

    if last is not None:
        out = args.save if os.path.isabs(args.save) else os.path.join(ROOT, args.save)
        imwrite_safe(out, last)
        print("退出时已存原图:", out)
    src.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
