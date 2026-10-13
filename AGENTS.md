# AGENTS.md — 双目装配流程检查系统

> 本文件是给**后续接手的 AI 代理 / 开发者**看的项目说明。
> 目标：让你在不读完整段历史对话的情况下，也能安全地改这个项目。
> 配套文件：`VERSION.md`（版本与变更）、`docs/对话存档_*.md`（历史决策过程）。

---

## 1. 一句话说明

树莓派 4B + HBV-1714-2 S2.0 双目相机，检查**黄铜船用舵轮摆件**的装配流程。
工件是 **4 类零件**，一料位一件；料盘是**黑色 3D 打印件（160 mm）**，工作面铺**绿色卡纸/绿幕**。

代码在 [assembly_vision/](assembly_vision/)，算法与硬件彻底解耦，离线就能跑通回归测试。

---

## 2. ★ 不要重新踩的坑（本会话实测结论，务必先读）

这些结论都来自真实画面的测量。**不要凭"常见做法"或厂家规格书推翻它们。**

### 2.1 背景是绿幕，不是黑底

| | 实测值 |
|---|---|
| 绿幕 | H 中值 **83**（p2–p98 = [79, 95]）、S 中值 **255**、V 中值 224~242 |
| 黄铜零件 | S 分布 **[52, 109]**，中值 80~92 |

- 黄铜的 `s_max` 取 **180**。**不能取 90/110** —— 那样会削掉零件自己的高光（实测零件 S 上限 109）。
- `s_max` 之所以能挡绿幕：绿幕 S≈255，与零件 S≤109 之间有 75+ 的余量。
- `segment.py::_mask()` **曾经不读 `s_max`**（配置写了不生效），已修好。改 `_mask()` 时注意保持
  默认值 `h_max=179 / s_max=255 / v_max=255`，否则会破坏老配置。

### 2.2 黄铜假阳性靠阈值切不干净

原木桌面与手臂肤色的 S 中值 **67~87**，与零件 S 区间完全重叠。
演示阶段画面里有笔记本和木桌时，实测假黄铜块达 17883 / 8443 / 7937 px。

**根治办法是给黄铜检测加"工位全局 ROI"**（托盘 + 装配位的包围盒），而不是继续调阈值。
正式工位简化成"绿卡纸 + 黑料盘 + 工件"后此问题基本消失，但**换环境要重新确认**。

### 2.3 蓝手套与笔记本键盘背光/屏幕的色相完全重叠

| | H | S | V | 面积 |
|---|---|---|---|---|
| 真手套 | 92~99 | 134~246 | 91~255 | ~110000 px |
| 键盘背光 | 91~104 | 121~174 | 81~146 | ≤ 9052 px |
| 笔记本屏幕 | 96~100 | — | 81~148 | 47652 px |

**色相分不开，只能靠 V 下限 + 面积闸门。** 实测扫描后取 `glove.v_min: 130`：

```
v_min=110 -> 真手套 83316 px | 空场景最大蓝块 7882 px  ✗ 会过面积闸门
v_min=130 -> 真手套 79638 px | 空场景最大蓝块 1230 px  ✓
```

> 正式工位不放笔记本后此风险消失，但 `v_min=130` 保留（真手套只损失 4%），
> 用来兜住"操作人员的手/袖子/其它蓝色物体"。

### 2.4 深度只用 SGBM，StereoBM 不可用

料盘表面几乎无纹理（料盘区 Laplacian 方差 **156**，只有整幅 2490 的 6%），
局部块匹配器没有可比对的特征：

| 匹配器 | 整幅有效像素 | 料盘区有效 | 过 `min_valid_ratio` 闸门? | 料盘区视差散布 |
|---|---|---|---|---|
| StereoBM（原实现） | **7.8%** | 9.3% | **✗ 否**（闸门 0.2） | **75 px（纯噪声）** |
| **SGBM** nd=256 bs=5 | **22.7%** | **45.8%** | ✓ 是 | **6.6 px** |

已改为 SGBM，并把 `min_valid_ratio` 降到 **0.12**。**不要在 480×360 以上算深度**：
实测 SGBM 1280×960 要 476 ms，按树莓派 4B 慢 6 倍估约 2.9 秒，不可用。

### 2.5 相机的分辨率与拼接方式（实测，反直觉）

| 请求 | 实际 | 帧率 | 单眼 | 真双目? |
|---|---|---|---|---|
| 2560×960 | 2560×960 | **4.5 fps** | 1280×960 | ✅ |
| 2560×720 | 2560×720 | 4.5 fps | 1280×720 | ✅ |
| 1920×720 | →1280×720 | 48.7 fps | 640×720 | ✅ |
| **1280×720** | 1280×720 | **48.7 fps** | **640×720** | ✅ |
| 1280×960 | 1280×960 | 49.7 fps | — | ❌ **单眼画面** |
| 960×960 | 960×960 | 1.0 fps | — | ❌ 单眼画面 |

**两个反直觉点**：
1. **`1280×960` 不是双目拼接图**（中缝无跳变，是单眼 4:3 画面）。别被它的宽高比骗了。
2. **2560 宽只有 4.5 fps，且不是 USB 带宽问题**：实测 1280×960 能到 48.9 MB/s，
   而 2560×960 只有 8.8 MB/s —— 瓶颈在**相机内部 ISP/固件**。
   **换 USB 3.0 口、换线、换树莓派都不会改善。**（相机是 USB 2.0 设备）

**当前选型：全程 1280×720（方案"丁"）** —— 帧率够、且是真双目，深度可用。
以后可升级到"丙"：平时 1280×720 跑流程，终检前**在线切到 2560×960** 抓一帧算深度
（实测同一 `VideoCapture` 句柄可在线切模式，无需释放重开）。

### 2.6 Windows 上必须用 DSHOW 后端

```
cv2.VideoCapture(0, cv2.CAP_V4L2)  -> isOpened() = False   （V4L2 是 Linux 专用）
cv2.VideoCapture(0, cv2.CAP_DSHOW) -> isOpened() = True
```

`config/camera.yaml` 的 `capture.backend: dshow`。代码按平台自动取 `backend_windows` 覆盖。
`apply_locked_controls()` 在 Windows 走 `cv2.VideoCapture.set()`（DSHOW 下部分控件驱动会拒绝，
返回 False 只提示不报错），Linux 仍走 `v4l2-ctl`。

### 2.7 其他已知问题（改动时别踩回去）

| 坑 | 现象 | 处理 |
|---|---|---|
| 深度基线字段路径不一致 | c5 写 `assembly.wheel_height_mm`，checks.py 读 `assembly.baseline.wheel_height_mm` → **立柱判据静默失效** | 已让 checks.py 两个路径都认，c5 两处都写 |
| 合成图颜色过饱和 | 合成黄铜原来 S=190，一旦 `s_max=180` 就整个被滤掉 → 测试失败 | 合成色改为 `_hsv(22,150,200)`，贴近实测 |
| 判定是装配位不是料盘 | 写成"料盘槽里有没有"，而零件被拿走正是为了装上去 → 永远等不到下一步 | 谓词一律判装配位 |
| 立柱谓词会翻回 False | 轮盘装上后中心被遮住，`center_plug` 覆盖率 1.00→0.47，`column_present` 掉回 0 | `PreconditionNet` 用 `done` 集合记忆，结论不受影响；但界面会显示不准 |
| 辐条只判"少"不判"多" | `if spk.count < spk.required` | `spk.ok` 用 `==`，需判多时要用 ok |
| `wheel` 谓词门槛是 4 不是 8 | `out["wheel"] = spk.count >= 4` | 换到"辐条单独成步"的流程时要改成 8 |
| 齐套性不查外圈/轮毂 | `ring_coverage`/`hub_coverage` 算了但没比阈值 | 缺件可能漏判 |
| c4/c5 基线没人读 | `slots.*` 特征、`assembly.ring_coverage` 运行时不读取 | 合格性目前只判覆盖率，不判完整性 |

---

## 3. 架构与目录

```
视 觉/                          ← 工作区根
├── AGENTS.md                   ← 本文件
├── VERSION.md                  ← 版本与变更记录
├── docs/对话存档_*.md          ← 历史决策与实测数据
├── 方案总纲_v2(2).docx          ← 初版方案（6 类零件口径，已作废，留档）
├── 方案总纲_v3_4类零件.docx      ← ★ 当前方案基线（4 类零件）
└── assembly_vision/            ← 代码
```

```
assembly_vision/
├── app.py                      入口（命令行 + 无界面模式）
├── config/
│   ├── camera.yaml             采集与相机参数（★ 项必须实测确认）
│   ├── workpiece.yaml          工件：4 个槽位 / 阈值 / 流程 / 位姿标记
│   ├── calibration.example.yaml 标定产物结构说明（程序不读）
│   └── calibration.synthetic.yaml 合成图标定（自动生成，仅离线用）
├── assemvision/                核心包
│   ├── types.py                数据类：Frame / SegResult / SlotResult / Verdict ...
│   ├── config.py               YAML/JSON 配置加载（YAML 缺失自动退回同名 .json）
│   ├── imgio.py                ★ 兼容中文路径的图片读写（np.fromfile + imdecode）
│   ├── pipeline.py             线程编排 + UIState 组装 + 去抖方式选择
│   ├── sources/                base / video_file / uvc_stereo / factory
│   ├── core/                   纯算法，只吃 numpy 与数据类
│   │   ├── segment.py          黄铜分割（S 上限挡绿幕）+ 蓝手套分割（V 下限）
│   │   ├── features.py         旋转不变形状特征（圆度/凸包实度/内孔数/弯曲/两端宽度比）
│   │   ├── polar.py            极坐标展开 + 自适应阈值数峰（辐条计数）
│   │   ├── roi.py              料盘 mm ↔ 像素、槽位 ROI、覆盖率
│   │   ├── checks.py           合格性(①) + 齐套性(④) + 装配位布尔量
│   │   ├── fsm.py              前置条件表(③) + 去抖 + 遮挡冻结
│   │   ├── events.py           取件/装配事件
│   │   ├── tracker.py          IoU 跟踪 + 补帧（补帧不参与判定）
│   │   ├── depth.py            SGBM + 平面拟合 + 高度图
│   │   └── pose_check.py       位姿自检（aruco 新旧接口兼容）
│   └── ui/                     base(UIState+Renderer) / opencv_renderer / qt_*
├── calib/                      c1 测GSD → c2 双目标定 → c3 料盘单应 → c4 槽位基线 → c5 装配位基线
├── tools/
│   ├── make_synthetic.py       合成图 / 装配序列 / 合成标定
│   ├── probe_camera.py         ★ 探相机能力（分辨率/帧率/是否真双目）
│   ├── ui_preview.py           离屏渲染界面预览图（不需要显示器）
│   └── record.py               录真实素材
├── tests/                      _common / test_features / test_segment / test_polar / test_fsm
└── data/                       synthetic / synthetic_seq / ui_preview / probe / refs / videos
```

**分层铁律**：`sources/` 只管图像从哪来；`core/` 只吃 numpy 与数据类，**不碰相机、不碰界面**；
`ui/` 只吃 `UIState`，不碰算法。改代码时不要打破这条线 —— 离线测试能跑全靠它。

---

## 4. 常用命令

```bash
cd assembly_vision

# ── 离线（不需要相机）────────────────────────────
python tools/make_synthetic.py --out data/synthetic --seq data/synthetic_seq \
        --calib config/calibration.synthetic.yaml
python tests/run_all.py                       # 4 个模块，应全绿
python app.py --source images --path data/synthetic_seq \
        --calibration config/calibration.synthetic.yaml --no-ui --max-frames 200
python tools/ui_preview.py --renderer qt --out data/ui_preview

# ── 真机 ────────────────────────────────────────
python tools/probe_camera.py                  # ★ 先跑这个，确认相机能力
python tools/record.py --seconds 60 --out data/videos/real1.mp4
python calib/c1_measure_fov.py --distance-mm 400      # 钢尺量 GSD
python calib/c2_stereo.py --cols 9 --rows 6 --square 25   # 棋盘格 → 基线 B
python calib/c3_tray_homography.py --manual   # 料盘单应
python calib/c4_slot_baseline.py --frames 20
python calib/c5_assembly_baseline.py --frames 20
python app.py --windowed                      # 正式跑（窗口模式方便调试）
```

**运行环境（本机实测）**：
- Python 要用 **`D:\miniconda\python.exe`**（有 OpenCV 4.12）。DSH 自带运行时**没有 cv2**。
- PyYAML 需要（配置是 YAML）。缺了会自动退回同名 `.json`。

---

## 5. 当前状态与待办

见 `VERSION.md` 的「待办」一节。要点：

- ✅ 代码离线全绿；4 类零件/4 工序口径已对齐；绿幕阈值、SGBM、DSHOW 后端、方案丁选型已完成
- ⬜ **相机未标定**：`config/calibration.yaml` 不存在，所有 ROI 与深度参数都还没实测值
- ⬜ 料盘几何要按实物重测（旧值是 160×160 四圆槽的估算，实物是 2×2 矩形槽）
- ⬜ 装配位坐标未定；按实测视野需确认摆位（视野内"料盘下缘"往下只剩有限空间）
- ⬜ c4/c5 产出的基线**尚未接入** `checks.py`，合格性/齐套性目前只判覆盖率
- ⬜ 未在树莓派上跑过任何一次

---

## 6. 改代码时的约定

- **中文注释**，风格与现有代码一致；关键处用 `★` 标出"必须实测确认"的项。
- 改 `config/*.yaml` 时**保留 `★` 注释并更新依据**，不要让注释与值脱节。
- 阈值/尺寸**不要硬编码在算法里**，一律走配置。
- 加新功能后**必须跑 `python tests/run_all.py`** 确认 4 个模块仍全绿。
- 提交前更新 `VERSION.md` 的变更记录。
