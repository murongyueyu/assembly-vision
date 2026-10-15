# -*- coding: utf-8 -*-
"""D435 工位**实时交互**调试台 —— 边看画面边调控件、边看分割结果。

★ 为什么要有它：`capture_site_frame.py` / `d435_intrinsics_check.py` /
  `sweep_camera_controls.py` 都是"抓完再打印"，**看不到实时画面** ——
  调白平衡/曝光、看掩膜切没切到零件，都要**盯着画面**才有手感。
  本工具把这些合成一个窗口：实时画面 + 分割叠加 + HUD 数字 + 键盘调参。

★ 窗口里能看到什么（可各自开关）：
  · 实时彩色画面（就是工程 `make_source` 取的那一路图）
  · **零件掩膜**（`segment.brass`）叠加 —— 绿色
  · **金属掩膜**（不靠色相那条通路）叠加 —— 青色
  · **手套掩膜** 叠加 —— 品红
  · 活动轮廓（`findContours` 描出来的真实形状，不是方框）
  · HUD：帧率 / V 均值 / **过曝%** / 绿幕 H / 零件暖色 H / brass 面积与连通域数 /
    当前曝光与白平衡

★ 键盘（**这是重点：调完立刻能看到画面变化**）：
    q / ESC   退出
    s         存一张原图 + 一张叠加图到 `ai_测试工具/工位帧/`
    空格      暂停 / 继续（暂停后仍可调参、仍会重算）
    ── 画面 ──
    a         掩膜叠加 开/关
    c         活动轮廓 开/关
    g         金属掩膜 开/关
    h         手套掩膜 开/关
    d         调试详情 开/关
    ── 相机控件（与 camera.yaml 一致的那些；调完按 w 才写配置）──
    [ / ]     曝光 -/+ 10（自动关掉后才有用）
    - / =     增益 -/+ 4
    , / .     白平衡 -/+ 200（会自动把 AWB 关掉）
    ; / '     亮度 -/+ 5
    9 / 0     对比度 -/+ 5
    o / p     伽马 -/+ 20
    k / l     饱和度 -/+ 5
    n / m     锐度 -/+ 10
    b         AWB 自动 开/关
    f         ★ 精细调节 开/关（步长缩到 1/5；也可以用 --step-scale 0.2 启动即精细）
    w         ★ 把当前控件值**写回 `config/camera.yaml`**（会打印改了什么）
    r         把控件恢复成配置文件里的值

★ 本工具一次把三件事都带上（不用换脚本）：
    ① **掩膜** —— a 零件口径 / g 金属 / h 手套
    ② **轮廓** —— c 描真实形状（默认用 `brass ∪ metal`，形状比单用 brass 完整）
    ③ **调参** —— 亮度/对比度/伽马/饱和度/锐度/曝光/增益/白平衡，全部键盘直调

★ 用法（★ 在**不受限**的进程里跑；VSCode 里打开本文件按 F5）：
    & D:\\miniconda\\envs\\yolov11\\python.exe "ai_测试工具\\live_site_debug.py"
    ... --step-scale 0.2     启动就用小步长（等价于启动后按 f）
    ... --parts-mask brass   轮廓用工程现状口径（默认 union）
    ... --s-min 25 --h-min 5 临时改 brass 阈值（不写配置）
    ... --no-controls        只看画面，不动相机控件
    ... --source images --path ..\\assembly_vision\\data\\synthetic_seq   # 离线喂图
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AV = os.path.join(ROOT, "assembly_vision")
sys.path.insert(0, AV)

OUT_DIR = os.path.join(ROOT, "ai_测试工具", "工位帧")
WIN = "D435 工位实时调试"

# 键盘 -> (控件名, 步长)。负号键在 OpenCV 里是 `-`，加号是 `=`（不用按 Shift）
KEY_STEPS = {
    "[": ("exposure", -10), "]": ("exposure", +10),
    "-": ("gain", -4), "=": ("gain", +4),
    ",": ("white_balance", -200), ".": ("white_balance", +200),
    ";": ("brightness", -5), "'": ("brightness", +5),
    "9": ("contrast", -5), "0": ("contrast", +5),
    "o": ("gamma", -20), "p": ("gamma", +20),
    "k": ("saturation", -5), "l": ("saturation", +5),
    "n": ("sharpness", -10), "m": ("sharpness", +10),
}
STEP_UNIT = {
    "exposure": "", "gain": "", "white_balance": "K", "brightness": "",
    "contrast": "", "gamma": "", "saturation": "", "sharpness": "",
}

# ★★ 分割阈值也能现场调（**不是**相机控件，是 `workpiece.yaml` 里的 segment.brass）。
#   为什么必须现场调：从**截图**量阈值不可靠（截图带窗框、还可能被缩放 ——
#   实测同一帧我从截图量到 brass 2369px、而 HUD 上是 19488px，差 8 倍）。
#   而这些值的取舍（亮一点还是暗一点、边缘闪不闪）**只能用眼睛对着实时画面定**。
#
#   ★ 键位冲突的解法：**不新增键**，改成"方向键随模式切换" ——
#     按 f 在「相机模式」与「分割模式」之间切（f 原来管精细调节，并入相机模式）。
#     两个模式共用同一批方向键，屏幕提示会跟着变，不会记混。
#       相机模式：[/] 曝光 · -/= 增益 · ,/. 白平衡 · ;/' 亮度 · 9/0 对比
#                 o/p 伽马 · k/l 饱和 · n/m 锐度 · b AWB · 1 精细调节
#       分割模式：[/] v_min · -/= s_min · ,/. morph_close · ;/' h_min · 9/0 h_max
#                 ★ 边缘闪烁多半是 v_min 卡边界 —— 分割模式下按 [ 降它
SEG_STEPS = {
    "[": ("v_min", -5), "]": ("v_min", +5),
    "-": ("s_min", -5), "=": ("s_min", +5),
    ",": ("morph_close", -2), ".": ("morph_close", +2),
    ";": ("h_min", -2), "'": ("h_min", +2),
    "9": ("h_max", -2), "0": ("h_max", +2),
}
SEG_LIMITS = {
    "h_min": (0, 179), "h_max": (0, 179),
    "s_min": (0, 255), "v_min": (0, 255), "morph_close": (1, 31),
}


def imwrite_safe(p, img):
    """★ 中文路径必须走 imencode+tofile（裸 cv2.imwrite 会静默失败，实录 §2.36）。"""
    try:
        ok, buf = cv2.imencode(os.path.splitext(p)[1] or ".png", img)
        if not ok:
            return False
        buf.tofile(p)
        return os.path.exists(p) and os.path.getsize(p) > 0
    except Exception as e:                                              # noqa: BLE001
        print("   写盘异常：%s" % e)
        return False


def find_rgb_sensor(rs, profile):
    """挑 RGB 传感器（exposure/gain 在深度传感器上也有，写错地方没用）。"""
    try:
        sensors = list(profile.get_device().sensors)
    except Exception:                                                   # noqa: BLE001
        return None
    for s in sensors:
        try:
            if "rgb" in str(s.get_info(rs.camera_info.name)).lower():
                return s
        except Exception:                                               # noqa: BLE001
            continue
    return sensors[0] if sensors else None


class Ctrl:
    """相机控件读写（全部走 sensor.set_option / get_option）。"""

    def __init__(self, rs, sensor, cfg_controls: dict) -> None:
        self.rs = rs
        self.s = sensor
        self.cfg = dict(cfg_controls or {})       # 配置文件里的值（用于 r 恢复）
        self.cur: dict = {}
        if self.s is not None:
            self.read_all()
        # ★★ 把**配置文件里已有、但传感器没读到**的键补进 cur。
        #   为什么必须补（2026-10 实测踩到）：
        #     `save_to_config` 写的是 `self.cur`，而 `cur` 只含 `read_all` 读到的键。
        #     于是按一次 `w` 就会把配置里**没被读到的键**（`hue` /
        #     `backlight_compensation` / `power_line_frequency` 三项）**从文件里挤掉** ——
        #     配置在 13 项 ↔ 10 项之间来回变。
        #   ⇒ 语义定成：**cur 是"这次要写回去的整套控件"**，
        #     没读到的键沿用配置里的值（而不是消失）。
        for k, v in self.cfg.items():
            if k not in self.cur:
                try:
                    self.cur[k] = float(v)
                except (TypeError, ValueError):
                    continue

    def _opt(self, name):
        return getattr(self.rs.option, name, None)

    def supports(self, name) -> bool:
        o = self._opt(name)
        if o is None or self.s is None:
            return False
        try:
            return bool(self.s.supports(o))
        except Exception:                                               # noqa: BLE001
            return False

    def read_all(self) -> None:
        for n in ("enable_auto_exposure", "exposure", "gain",
                  "enable_auto_white_balance", "white_balance", "brightness",
                  "contrast", "gamma", "saturation", "sharpness",
                  # ★ 这三项以前没读 —— 结果按 w 时它们被挤出了配置文件（见 __init__ 的说明）
                  "hue", "backlight_compensation", "power_line_frequency"):
            if not self.supports(n):
                continue
            try:
                self.cur[n] = float(self.s.get_option(self._opt(n)))
            except Exception:                                           # noqa: BLE001
                pass

    def set(self, name, val) -> bool:
        if not self.supports(name):
            return False
        try:
            self.s.set_option(self._opt(name), float(val))
            self.cur[name] = float(val)
            return True
        except Exception:                                               # noqa: BLE001
            return False

    def bump(self, name, delta) -> str:
        """按步长改一个值。★ 改 exposure/gain 时自动关 AE；改 white_balance 时自动关 AWB。"""
        note = ""
        if name in ("exposure", "gain") and self.cur.get("enable_auto_exposure", 0) != 0:
            self.set("enable_auto_exposure", 0)
            note = "（已自动关掉 AE）"
        if name == "white_balance" and self.cur.get("enable_auto_white_balance", 0) != 0:
            self.set("enable_auto_white_balance", 0)
            note += "（已自动关掉 AWB）"
        if name not in self.cur:
            try:
                self.cur[name] = float(self.s.get_option(self._opt(name)))
            except Exception:                                           # noqa: BLE001
                return "★ 读不到 %s" % name
        new = self.cur[name] + float(delta)
        if name == "exposure":
            new = float(np.clip(new, 1, 10000))
        elif name == "gain":
            new = float(np.clip(new, 0, 128))
        elif name == "white_balance":
            new = float(np.clip(new, 2800, 6500))
        elif name == "gamma":
            new = float(np.clip(new, 100, 500))
        elif name == "sharpness":
            new = float(np.clip(new, 0, 100))
        elif name == "saturation":
            new = float(np.clip(new, 0, 100))
        if not self.set(name, new):
            return "★ %s 写失败" % name
        return "%s -> %.0f%s" % (name, new, note)

    def restore(self) -> str:
        """恢复成配置文件里的值。"""
        n = 0
        items = sorted(self.cfg.items(),
                       key=lambda kv: 0 if str(kv[0]).startswith("enable_") else 1)
        for k, v in items:
            if self.set(str(k), v):
                n += 1
        return "已恢复 %d 项（配置文件的值）" % n

    def save_to_config(self, cam_path: str, wp_cfg=None) -> str:
        """把当前控件值写回 `camera.yaml`，**只替换 `controls:` 那一段**。

        ★★ 为什么不能直接 `yaml.safe_load` + `safe_dump`（第一版就是这么写的）：
          `safe_load` 会**把所有注释丢掉**，`safe_dump` 再一写，
          整个 `camera.yaml` 里的 `★` 依据、实测数字、历史说明**全部消失** ——
          而本项目 `AGENTS.md §3` 明确要求「改配置时保留 ★ 注释并更新依据」。
          所以这里改成**按行做外科手术**：只把 `realsense.controls:` 下面那段键值
          换成新值，其余每一行（含注释）**原样不动**。

        ★ 另外：会先备份成 `camera.yaml.bak`，并打印每一项的旧值→新值。
        """
        try:
            with open(cam_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines(keepends=True)
        except Exception as e:                                          # noqa: BLE001
            return "★ 读 %s 失败：%s" % (cam_path, e)

        # ── 定位 `controls:`（必须在 `realsense:` 段下）──
        i_ctrl = -1
        for i, ln in enumerate(lines):
            s = ln.rstrip("\n")
            if s.startswith("realsense:"):
                continue
            if s.strip().startswith("controls:"):
                i_ctrl = i
                break
        if i_ctrl < 0:
            return "★ 在 %s 里找不到 `controls:`，未改动" % os.path.basename(cam_path)

        # controls 的缩进 = 它自己的缩进 + 2
        base_indent = len(lines[i_ctrl]) - len(lines[i_ctrl].lstrip())
        key_indent = " " * (base_indent + 2)
        # 段落范围：从 controls: 的下一行起，到"缩进 <= controls 缩进"的非空行为止
        j = i_ctrl + 1
        while j < len(lines):
            s = lines[j]
            if s.strip() == "":
                j += 1
                continue
            ind = len(s) - len(s.lstrip())
            if ind <= base_indent:
                break
            j += 1
        end = j

        # ── 旧值 + **旧的行尾注释**（注释必须原样保留：AGENTS §3 要求保留 ★ 依据）──
        def parse_val(t):
            t = t.strip()
            try:
                return int(t)
            except ValueError:
                try:
                    return float(t)
                except ValueError:
                    return t
        old = {}
        old_comment = {}
        for ln in lines[i_ctrl + 1:end]:
            s = ln.strip()
            if not s or s.startswith("#") or ":" not in s:
                continue
            k, _, rest = s.partition(":")
            k = k.strip()
            pre, sep, cmt = rest.partition("#")
            old[k] = parse_val(pre.strip())
            # ★ 连 `#` 一起留着，但**自己补空格** ——
            #   原来写的是 `exposure: 156                  # 注释`（对齐过的多空格），
            #   直接照抄会变成 `exposure: 61# 注释`（挤在一起，不好看也不好读）。
            #   统一成"键值 + 2 空格 + # 注释"。
            old_comment[k] = ("  #" + cmt) if sep else ""

        # ── 新值：按 d435.py 的写盘顺序（enable_* 在前）──
        def fmt(v):
            return str(int(v)) if abs(v - round(v)) < 1e-6 else str(round(v, 4))

        def default_note(k):
            # 只有"原来没有注释"的键才用这两句兜底说明
            if k == "enable_auto_exposure":
                return "  # ★★ 0=锁定（漂的是它）"
            if k == "enable_auto_white_balance":
                return "  # ★★ 1=保持自动（扫描结论：固定值更差）"
            return ""

        order = sorted(self.cur.items(),
                       key=lambda kv: 0 if str(kv[0]).startswith("enable_") else 1)
        # 保留段内原有的键顺序（新值里有、旧段没有的键，追加到末尾）
        seq = [k for k in old if k in self.cur] + [k for k, _ in order if k not in old]
        new_lines = []
        for k in seq:
            if k not in self.cur:
                continue
            cmt = old_comment.get(k) or default_note(k)
            new_lines.append("%s%s: %s%s\n" % (key_indent, k, fmt(self.cur[k]), cmt))
        if not new_lines:
            return "★ 没有可写的控件值，未改动"

        # ── 备份 + 替换 ──
        bak = cam_path + ".bak"
        try:
            with open(bak, "w", encoding="utf-8") as f:
                f.writelines(lines)
            out = lines[:i_ctrl + 1] + new_lines + lines[end:]
            with open(cam_path, "w", encoding="utf-8") as f:
                f.writelines(out)
        except Exception as e:                                          # noqa: BLE001
            return "★ 写 %s 失败：%s" % (cam_path, e)

        diff = [(k, old.get(k), self.cur[k]) for k in seq
                if k in self.cur and old.get(k) != self.cur[k]]
        msg = "已写回 %s（第 %d~%d 行替换；备份 %s）；变化 %d 项" % (
            os.path.basename(cam_path), i_ctrl + 2, end, os.path.basename(bak), len(diff))
        if diff:
            msg += "：\n     " + "\n     ".join(
                "%s: %s -> %s" % (k, a, fmt(b)) for k, a, b in diff)
        else:
            msg += "（与配置里已有的值相同）"
        return msg


def main() -> int:
    ap = argparse.ArgumentParser("live_site_debug")
    ap.add_argument("--camera", default=os.path.join(AV, "config", "camera.yaml"))
    ap.add_argument("--workpiece", default=os.path.join(AV, "config", "workpiece.yaml"))
    ap.add_argument("--source", default=None, help="留空=用 camera.yaml；images/video 可离线喂")
    ap.add_argument("--path", default=None)
    ap.add_argument("--no-controls", action="store_true",
                    help="只读模式：不写任何相机控件（离线源必须加这个）")
    ap.add_argument("--mask-every", type=int, default=1,
                    help="每 N 帧算一次掩膜（省算力；HUD 仍每帧更新）")
    ap.add_argument("--scale", type=float, default=1.0, help="窗口显示放大倍数")
    # ★ 手调步长：默认值是"扫一遍找范围"用的（曝光 ±10、白平衡 ±200）。
    #   找到范围之后要**细调**，用这个整体缩小步长（例如 0.2 ⇒ 曝光 ±2、白平衡 ±40）。
    ap.add_argument("--step-scale", type=float, default=1.0,
                    help="键调步长的整体倍率（默认 1.0；细调用 0.2）")
    # ★★ 轮廓口径（2026-10 用户实测："顶盖/立柱没轮廓、底座不是圆"）：
    #   根因是**镜面高光把同一个零件切成碎片** ——
    #     底座：brass 只抓到亮顶面、metal 只抓到暗边，两块不重合
    #           ⇒ brass 单独看是月牙（圆度 0.26），union 才是实心圆（0.88）
    #   实测 union **不引入新假阳性**（多出的像素 99.7% 在料盘区内）。
    #   默认 union（这是**给你看画面**的工具，形状完整更好判断）；
    #   要看工程现状（brass）就加 --parts-mask brass。
    ap.add_argument("--parts-mask", choices=["brass", "union"], default="union",
                    help="零件轮廓用哪个掩膜：union=brass∪metal（默认，形状完整）| brass（工程现状）")
    # ★ 可调的两个 brass 阈值（回车立即生效，不用改配置）——用来现场试"降 s_min / 降 h_min"
    ap.add_argument("--s-min", type=float, default=None,
                    help="临时覆盖 segment.brass.s_min（现场试阈值用）")
    ap.add_argument("--h-min", type=float, default=None,
                    help="临时覆盖 segment.brass.h_min")
    # ★ 现场调"分割阈值"（不是相机控件）—— 边缘闪烁/切不全时主要就调这几个。
    #   为什么放在这里：从**截图**量阈值不可靠（截图会被缩放/加窗框，实测差 8 倍），
    #   只有对着**实时画面**调才准。调完按 W 写进 workpiece.yaml。
    ap.add_argument("--v-min", type=float, default=None,
                    help="临时覆盖 segment.brass.v_min（★ 边缘闪烁多半是它卡边界）")
    ap.add_argument("--morph-close", type=int, default=None,
                    help="临时覆盖 segment.brass.morph_close（填边缘缝隙，默认 5）")
    args = ap.parse_args()

    steps = {k: (n, d * float(args.step_scale)) for k, (n, d) in KEY_STEPS.items()}

    from assemvision.config import load_config
    from assemvision.core.segment import Segmenter
    from assemvision.core.features import contour_features
    from assemvision.sources.factory import make_source

    cam = load_config(args.camera)
    wp = load_config(args.workpiece)
    # ★ 临时覆盖 brass 阈值（现场试"降 s_min / 降 h_min / 降 v_min"用；不改配置文件）
    #   ★★ 边缘闪烁多半是 `v_min` 卡边界：零件暗一点就掉出掩膜，下一帧又回来。
    #      把 v_min 降下来（黄铜 V 低也有色相）比调曝光去迁就阈值更稳。
    _ov = {"s_min": args.s_min, "h_min": args.h_min,
           "v_min": args.v_min, "morph_close": args.morph_close}
    _ov = {k: v for k, v in _ov.items() if v is not None}
    if _ov:
        seg_brass = dict(wp.get("segment.brass", {}) or {})
        seg_brass.update({k: (int(v) if k == "morph_close" else float(v))
                          for k, v in _ov.items()})
        wp.set("segment.brass", seg_brass)
        print("── 已临时覆盖 segment.brass：%s（只影响本次运行，不写配置）──"
              % "、".join("%s=%s" % (k, v) for k, v in _ov.items()))
    seg = Segmenter(wp)
    # ★ 分割阈值的"当前值"（现场可调；按 j 写回 workpiece.yaml）
    seg_cur = {
        "h_min": float(seg.brass.get("h_min", 10)),
        "h_max": float(seg.brass.get("h_max", 45)),
        "s_min": float(seg.brass.get("s_min", 30)),
        "v_min": float(seg.brass.get("v_min", 80)),
        "morph_close": float(seg.brass.get("morph_close", 5)),
    }

    def apply_seg():
        """把 seg_cur 写进 Segmenter（现场调的阈值立刻生效）。"""
        b = dict(seg.brass)
        for k in ("h_min", "h_max", "s_min", "v_min"):
            b[k] = int(round(seg_cur[k]))
        b["morph_close"] = int(round(seg_cur["morph_close"])) | 1     # 必须奇数
        seg.brass = b
        return b

    def bump_seg(name, delta):
        lo, hi = SEG_LIMITS[name]
        seg_cur[name] = float(np.clip(seg_cur[name] + delta, lo, hi))
        b = apply_seg()
        return "segment.brass.%s -> %s" % (name, b[name])

    def save_seg_to_config(wp_path: str) -> str:
        """把 seg_cur 写回 workpiece.yaml —— **同样只替换 brass 那一段，保留注释**。"""
        try:
            with open(wp_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines(keepends=True)
        except Exception as e:                                          # noqa: BLE001
            return "★ 读 %s 失败：%s" % (wp_path, e)
        i_br = next((i for i, s in enumerate(lines)
                     if s.strip().startswith("brass:")), -1)
        if i_br < 0:
            return "★ 在 %s 里找不到 `brass:`，未改动" % os.path.basename(wp_path)
        base = len(lines[i_br]) - len(lines[i_br].lstrip())
        ind = " " * (base + 2)
        j = i_br + 1
        while j < len(lines):
            s = lines[j]
            if s.strip() == "":
                j += 1
                continue
            if (len(s) - len(s.lstrip())) <= base:
                break
            j += 1
        cmt = {}
        order = []
        for ln in lines[i_br + 1:j]:
            s = ln.strip()
            if not s or s.startswith("#") or ":" not in s:
                continue
            k, _, rest = s.partition(":")
            k = k.strip()
            _, sep, c = rest.partition("#")
            order.append(k)
            cmt[k] = ("  #" + c) if sep else ""
        b = apply_seg()
        new = []
        for k in order:
            if k in b:
                new.append("%s%s: %s%s\n" % (ind, k, b[k], cmt[k]))
        bak = wp_path + ".bak"
        try:
            with open(bak, "w", encoding="utf-8") as f:
                f.writelines(lines)
            with open(wp_path, "w", encoding="utf-8") as f:
                f.writelines(lines[:i_br + 1] + new + lines[j:])
        except Exception as e:                                          # noqa: BLE001
            return "★ 写 %s 失败：%s" % (wp_path, e)
        return ("已写回 %s（brass 段第 %d~%d 行；备份 %s）：%s"
                % (os.path.basename(wp_path), i_br + 2, j, os.path.basename(bak),
                   "、".join("%s=%s" % (k, b[k]) for k in order if k in b)))

    cfg_ctrl = dict(cam.get("realsense.controls", {}) or {})

    images = None
    src = None
    if args.source in ("images", "video") or args.path:
        src = make_source(cam, args.source, args.path)
        try:
            src.open()
        except Exception as e:                                          # noqa: BLE001
            print("★ 离线源打开失败：%s" % str(e).strip())
            return 2
    else:
        try:
            src = make_source(cam)
            src.open()
            if not args.no_controls and hasattr(src, "apply_locked_controls"):
                print("── 先按 camera.yaml 应用一遍控件 ──")
                src.apply_locked_controls()
        except Exception as e:                                          # noqa: BLE001
            print("★ 相机打不开：%s" % str(e).strip())
            print("  ① 关掉 Windows 设置里的相机预览页 / RealSense Viewer / 别的 python")
            print("  ② 加 --reset 不行就 probe_d435.py --reset（★ 会冲掉手调值）")
            print("  ③ 受限进程打不开（实录 §2.39）")
            return 2

    ctrl = None
    if not args.no_controls and not args.source:
        try:
            import pyrealsense2 as rs                                   # noqa: PLC0415
            sensor = find_rgb_sensor(rs, src.profile)
            ctrl = Ctrl(rs, sensor, cfg_ctrl)
            print("── 相机控件（可键盘调，按 w 写回配置）──")
            for k in ("enable_auto_exposure", "exposure", "gain",
                      "enable_auto_white_balance", "white_balance", "brightness",
                      "contrast", "gamma", "saturation", "sharpness"):
                if k in ctrl.cur:
                    print("   %-26s = %.0f" % (k, ctrl.cur[k]))
        except Exception as e:                                          # noqa: BLE001
            print("★ 控件句柄拿不到（只能看画面）：%s" % str(e).strip())

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "分析叠加图"), exist_ok=True)

    print("\n" + "═" * 74)
    print("  D435 工位实时调试    按键说明看文件头（或窗口里按 d 看 HUD）")
    print("  ★ q/ESC 退出 · s 存图 · **f 切换「相机模式 / 分割模式」** · w 写配置 · r 恢复")
    print("     相机模式：调曝光/增益/白平衡/亮度/对比度…  → w 写 camera.yaml")
    print("     分割模式：调 brass 阈值（V/S/H 下界、闭运算） → w 写 workpiece.yaml")
    print("═" * 74)

    show_mask, show_contour, show_metal, show_glove = True, True, False, False
    show_hud, paused = True, False
    # ★ 精细调节状态：按 f 切换。
    #   两个倍率**相乘**，语义才不会打架：
    #     实际步长 = --step-scale（基准倍率） × （f 开着 ? 0.2 : 1.0）
    #   ⇒ `--step-scale 1.0` + 按 f  = 0.2×
    #      `--step-scale 0.2` + 按 f  = 0.04×（很细，微调用）
    #      `--step-scale 0.2` 不按 f  = 0.2×（等于启动就细）
    fine_mode = False
    base_scale = float(args.step_scale)
    steps = {k: (n, d * base_scale) for k, (n, d) in KEY_STEPS.items()}
    # ★ 键位模式：cam = 调相机控件；seg = 调 segment.brass 阈值（按 f 切）
    mode = "cam"
    seg_ms, fps, n = 0.0, 0.0, 0
    t_last = time.perf_counter()
    frame_i = 0
    last_msg = ""
    last_msg_t = 0.0
    cached = None
    last_info = None          # ★ 必须先初始化：第一帧之前 HUD 没有统计可显示

    while True:
        if not paused:
            f = src.read(want_full=False)
            frame = getattr(f, "left", None) if f is not None else None
            if frame is None:
                if args.source or args.path:
                    print("  离线源读完，退出")
                    break
                time.sleep(0.01)
                continue
            frame_i += 1
        else:
            frame = cached
            if frame is None:
                time.sleep(0.02)
                continue

        do_mask = (not paused) and (frame_i % max(1, args.mask_every) == 0)
        if do_mask:
            cached = frame
            t0 = time.perf_counter()
            res = seg.segment(frame, scale=1.0)
            seg_ms = (time.perf_counter() - t0) * 1000.0
            # ★ 轮廓口径：union = brass ∪ metal（见 --parts-mask 的说明）
            if args.parts_mask == "union" and res.metal_mask is not None:
                _pm = cv2.bitwise_or(
                    res.brass_mask if res.brass_mask is not None
                    else np.zeros_like(res.metal_mask), res.metal_mask)
                _pb = Segmenter._blobs(
                    _pm, min_area_px=float(seg.brass.get("min_area_px", 500)))
                for _b in _pb:
                    _b.features = contour_features(_b.contour, _pm)
                parts_mask, parts_blobs = _pm, _pb
            else:
                parts_mask, parts_blobs = res.brass_mask, res.brass_blobs
        else:
            res = None

        canvas = frame.copy()

        if res is not None:
            # ── 掩膜叠加（零件口径 绿 / metal 青 / glove 品红）──
            if show_mask and parts_mask is not None:
                m = parts_mask > 0
                canvas[m] = (canvas[m] * 0.45 + np.array([0, 255, 0]) * 0.55).astype(np.uint8)
            if show_metal and res.metal_mask is not None:
                m = res.metal_mask > 0
                canvas[m] = (canvas[m] * 0.6 + np.array([255, 255, 0]) * 0.4).astype(np.uint8)
            if show_glove and res.glove_mask is not None:
                m = res.glove_mask > 0
                canvas[m] = (canvas[m] * 0.5 + np.array([255, 0, 255]) * 0.5).astype(np.uint8)
            # ── 真实形状轮廓（不是方框）──
            if show_contour and parts_blobs:
                for b in parts_blobs:
                    cv2.drawContours(canvas, [b.contour], -1, (0, 0, 255), 2)
                for b in parts_blobs:
                    cx, cy = int(b.centroid[0]), int(b.centroid[1])
                    cv2.drawMarker(canvas, (cx, cy), (255, 255, 255),
                                   cv2.MARKER_CROSS, 12, 1)

        # ── HUD ──
        if do_mask and res is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            B, R = frame[..., 0].astype(np.int32), frame[..., 2].astype(np.int32)
            green = (H.astype(np.int32) >= 60) & (H.astype(np.int32) <= 100) & (S > 120)
            warm = (R > B + 20) & (V >= 60) & (~green)
            info = {
                "V均值": float(V.mean()),
                "过曝%": 100.0 * float((frame.min(axis=2) >= 250).mean()),
                "绿幕H": float(np.median(H[green])) if green.any() else float("nan"),
                "零件H": float(np.median(H[warm])) if warm.any() else float("nan"),
                "brass_px": int(np.count_nonzero(res.brass_mask))
                if res.brass_mask is not None else 0,
                "连通域": len(res.brass_blobs),
                "seg_ms": seg_ms,
            }
        else:
            info = None
        if info is not None:
            last_info = info
        else:
            info = last_info        # 非掩膜帧 / 暂停时沿用上一次的统计

        if show_hud:
            lines = []
            if info:
                lines.append("V %5.1f  过曝 %5.3f%%  绿幕H %5.1f  零件H %5.1f"
                             % (info["V均值"], info["过曝%"], info["绿幕H"], info["零件H"]))
                lines.append("brass %6d px   连通域 %d   seg %.1f ms   %.1f fps   %s"
                             % (info["brass_px"], info["连通域"], info["seg_ms"], fps,
                                "PAUSED" if paused else "LIVE"))
            if ctrl is not None and mode == "cam":
                cv_ = ctrl.cur
                ae = "AE" if cv_.get("enable_auto_exposure", 1) else "锁定"
                aw = "AWB" if cv_.get("enable_auto_white_balance", 1) else "手动"
                lines.append("【相机模式】曝光 %5.0f[%s] 增益 %4.0f WB %5.0f[%s] 亮度 %4.0f 对比 %4.0f"
                             % (cv_.get("exposure", 0), ae, cv_.get("gain", 0),
                                cv_.get("white_balance", 0), aw,
                                cv_.get("brightness", 0), cv_.get("contrast", 0)))
                lines.append("  伽马 %4.0f 饱和 %4.0f 锐度 %4.0f 精细[%s] → 满意就按 w 写 camera.yaml"
                             % (cv_.get("gamma", 0), cv_.get("saturation", 0),
                                cv_.get("sharpness", 0), "开" if fine_mode else "关"))
            elif mode == "seg":
                b = seg.brass
                lines.append("【分割模式】brass H∈[%d,%d] S≥%d V≥%d morph_close=%d → 满意就按 w 写 workpiece.yaml"
                             % (b.get("h_min", 0), b.get("h_max", 0), b.get("s_min", 0),
                                b.get("v_min", 0), b.get("morph_close", 0)))
                lines.append("  [/]V下界 -=S下界 ,/.闭运算 ;/'H下界 9/0H上界   ★ 边缘闪烁→降 V 下界")
            lines.append("按键: q退出 s存图 w写配置 r恢复 | a掩膜 c轮廓 g金属 h手套 d详情 空格暂停 f切换模式")
            lines.append("相机键: [/]曝光 -/=增益 ,/.白平衡 ;/'亮度 9/0对比 o/p伽马 k/l饱和 n/m锐度 b=AWB 1精细")
            y = 0
            for i, t in enumerate(lines):
                cv2.rectangle(canvas, (0, y), (canvas.shape[1], y + 20), (25, 25, 25), -1)
                cv2.putText(canvas, t, (6, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (230, 230, 230), 1, cv2.LINE_AA)
                y += 20
            if last_msg and (time.perf_counter() - last_msg_t) < 3.0:
                cv2.putText(canvas, last_msg, (6, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, last_msg, (6, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if args.scale and abs(args.scale - 1.0) > 1e-6:
            canvas = cv2.resize(canvas, None, fx=args.scale, fy=args.scale,
                                interpolation=cv2.INTER_NEAREST)
        cv2.imshow(WIN, canvas)

        now = time.perf_counter()
        dt = now - t_last
        t_last = now
        n += 1
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if n > 3 else 1.0 / dt

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            paused = not paused
            last_msg, last_msg_t = ("已暂停" if paused else "继续"), time.perf_counter()
        elif k == ord("s"):
            base = os.path.join(OUT_DIR, "live_%s.png" % time.strftime("%H%M%S"))
            ov = os.path.join(OUT_DIR, "分析叠加图", "live_%s.png" % time.strftime("%H%M%S"))
            ok1 = imwrite_safe(base, frame)
            ok2 = imwrite_safe(ov, canvas)
            last_msg = "原图 %s / 叠加图 %s" % ("已存" if ok1 else "★失败",
                                              "已存" if ok2 else "★失败")
            last_msg_t = time.perf_counter()
            print("  %s\n     %s\n     %s" % (last_msg, base, ov))
        elif k == ord("a"):
            show_mask = not show_mask
        elif k == ord("c"):
            show_contour = not show_contour
        elif k == ord("g"):
            show_metal = not show_metal
        elif k == ord("h"):
            show_glove = not show_glove
        elif k == ord("d"):
            show_hud = not show_hud
        elif k == ord("f"):
            # ★ 模式切换：相机模式 <-> 分割模式（同一批方向键，屏幕提示跟着变）
            mode = "seg" if mode == "cam" else "cam"
            last_msg = ("已切到**分割模式**（调 segment.brass：[/]v_min -=s_min "
                        ",/.morph_close ;/'h_min 9/0h_max；按 w 写 workpiece.yaml）"
                        if mode == "seg" else
                        "已切到**相机模式**（调曝光/增益/白平衡/亮度/对比度…；按 w 写 camera.yaml）")
            last_msg_t = time.perf_counter()
            print("  " + last_msg)
        elif k == ord("1"):
            # 精细调节（原 f 的功能；只在相机模式下有意义）
            fine_mode = not fine_mode
            scale = (0.2 if fine_mode else 1.0) * float(args.step_scale)
            steps = {kk: (nn, dd * scale) for kk, (nn, dd) in KEY_STEPS.items()}
            last_msg = "精细调节：%s（相机键步长 ×%.2f）" % ("开" if fine_mode else "关", scale)
            last_msg_t = time.perf_counter()
            print("  " + last_msg)
        elif mode == "seg" and chr(k) in SEG_STEPS:
            name, d = SEG_STEPS[chr(k)]
            last_msg = bump_seg(name, d)
            last_msg_t = time.perf_counter()
            print("  " + last_msg)
        elif ctrl is not None and mode == "cam" and chr(k) in steps:
            name, d = steps[chr(k)]
            last_msg = ctrl.bump(name, d)
            last_msg_t = time.perf_counter()
            print("  " + last_msg)
        elif ctrl is not None and k == ord("b"):
            awb = 1.0 if ctrl.cur.get("enable_auto_white_balance", 0) < 0.5 else 0.0
            ctrl.set("enable_auto_white_balance", awb)
            last_msg = "AWB -> %s" % ("自动" if awb else "手动")
            last_msg_t = time.perf_counter()
            print("  " + last_msg)
        elif mode == "seg" and k == ord("w"):
            last_msg = save_seg_to_config(args.workpiece)
            last_msg_t = time.perf_counter()
            print("  " + last_msg)
        elif ctrl is not None and mode == "cam" and k == ord("w"):
            last_msg = ctrl.save_to_config(args.camera, wp)
            last_msg_t = time.perf_counter()
            print("  " + last_msg)
        elif ctrl is not None and k == ord("r"):
            last_msg = ctrl.restore()
            last_msg_t = time.perf_counter()
            print("  " + last_msg)

    try:
        src.close()
    except Exception:                                                   # noqa: BLE001
        pass
    cv2.destroyAllWindows()
    print("\n  已退出。存下来的图在：%s" % OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
