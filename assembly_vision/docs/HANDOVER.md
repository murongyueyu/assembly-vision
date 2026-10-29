# HANDOVER —— 树莓派 rpi4-assembly / ABCDE 检测

> 当前六步入口和验证边界见仓库根目录 `docs/Pi快照_20261029_启动与交接.md`，本文件其余内容保留历史口径。

> 写于 2026-10-25，**2026-10-29 更新**（新增 2/4 立柱长度检测、顶盖窗口改为 68.5±1）。
> 目标读者：**接手这台树莓派和 ABCDE 检测链路的人**。
>
> 阅读顺序：
> 1. 本文件（现状 / 怎么跑 / 配置链 / 已知问题 / 待办）；
> 2. `docs/2026-10-29_项目实现说明_报告素材.md` —— **系统结构 + 两处测量项 + 标定与验证数据**（写报告看这份）；
> 3. `docs/2026-10-29_立柱长度检测.md` —— 2/4 立柱为什么从"测高"改成"量长度"；
> 4. `docs/2026-10-28_立柱测高与深度探针.md` —— 深度法失败的完整证据；
> 5. `docs/2026-10-25_cap_baseline_65mm.md`、`docs/2026-10-24_测高规范与提速记录.md` —— 历史基线由来；
> 6. `DEPLOY_PI.md` —— 环境安装层面的补充。

---

## 1. 一句话现状

ABCDE 集成链路已在 Pi4B + D435 上跑通并**正在运行**：`./deploy/pi/run_abcde.sh` 启动后，
操作员按「底座 → 立柱 → 轮盘 → 顶盖」取件，进度到 4/4 后按 `i` 人工框选两块测区，
系统输出顶盖高度并按 **67.500～69.500 mm** 判合格（2026-10-29 由 62.000～68.000 改）。检测帧率约 **7.8 fps**，
A 身份识别是性能下限。31 个回归模块 **2026-10-25 全部通过**。剩下的事见第 9 节。

---

## 2. 机器与环境（2026-10-25 实测，不是文档转抄）

| 项 | 值 |
|---|---|
| 硬件 | Raspberry Pi 4B，**1.8 GiB 内存（2GB 版）**，4 核 |
| 系统 | Raspberry Pi OS（Debian 13 trixie）64-bit 桌面，Wayland + **labwc**，Xwayland 在 `:0` |
| 面板 | **1024×768**（app 用 `--display 960x540`，窗口被 WM 显示为 770×432） |
| 主机名 / 账号 | `rpi4-assembly` / `pi`（密码在提示时输入，不入库） |
| Python | `~/miniforge3/envs/assembly/bin/python`，**3.12.15** |
| 依赖 | pyrealsense2 **2.59.1** · opencv-contrib **4.14.0** · numpy 2.2.6 · PyYAML 6.0.2 · Pillow 11.0.0 |
| 相机 | D435，SN **262322070302**，固件 5.17.3.10，**USB 3.2**，640×480@30 |
| 代码 | `/home/pi/assembly_vision`（**没有 git**，靠 `*.bak_*` 备份） |
| 磁盘 | 57 G，已用 8.2 G（`data/` 占 354 M） |
| 空载 / 满载温度 | 46 ℃ / 70～73 ℃（Pi4 软降频 80 ℃，长时间跑要留意散热） |

**必须知道的三个环境事实**

1. 系统 Python 是 3.13，**装不上 pyrealsense2**（ARM64 只有 cp39/310/312 wheel），
   所以一定要用 `~/miniforge3/envs/assembly`；
2. `~/miniforge3/envs/assembly/lib/python3.12/site-packages/sitecustomize.py` 里
   删掉了 `cv2.TrackerKCF` / `cv2.TrackerCSRT` 直接构造 —— 否则 **段错误**（try/except 抓不到）。
   重新装环境必须重装这个补丁（`~/pi_setup.sh` 已自动化）；
3. `config/calibration.yaml` **现在是存在的**（13.5 KB）。`DEPLOY_PI.md` §4.4 说"缺该文件"
   那段是 2026-10-22 的历史状态，**已过期**。

---

## 3. 怎么跑

### 3.1 唯一推荐入口

```bash
cd ~/assembly_vision
./deploy/pi/run_abcde.sh          # 桌面终端里直接跑
```

SSH 下必须给显示环境：

```bash
DISPLAY=:0 XAUTHORITY=/home/pi/.Xauthority ./deploy/pi/run_abcde.sh
```

后台常驻（SSH 断开不影响，日志可读）：

```bash
cd ~/assembly_vision
PYTHONUNBUFFERED=1 setsid nohup ./deploy/pi/run_abcde.sh > /tmp/abcde_run.log 2>&1 &
```

> `PYTHONUNBUFFERED=1` 很重要：不加的话 stdout 被重定向后**日志几乎永远是空的**。

### 3.2 快捷键

| 键 | 作用 |
|---|---|
| `q` / `Esc` | 退出 |
| `r` | 复位（清 4/4 与测量结果） |
| `i` | 高度测量（**按进度生效**：默认候选 4/4 顶盖；分阶段候选 2/4 立柱 + 4/4 顶盖） |

### 3.3 完整操作流程

1. 四件零件放回料盘对应槽，手离开，停 1～2 秒，等右上角从 `等待D稳定放行` 变成
   `已通过D并开工`；
2. 按 **底座 → 立柱 → 轮盘 → 顶盖** 顺序取走并装好，每件要离开原槽 ≥8 px，进度到 `4/4`；
3. **停约 2 秒**（2026-10-25 实测：刚放稳就测会出 8 mm 级离群），按 `i`；
4. 弹出 `Reference: bare green mat only` → 框**工件旁 3～5 cm 的裸绿垫**，**60×60 px**，Enter；
5. 弹出 `Target: current assembly CAP top only` → 框**顶盖中心圆盘**，**12×12 px**，Enter；
6. 面板出现 `快照测量：有效 xx.xxxx mm` + `高度符合性`；**连测 2 次取中位数**；
7. `C` / `Esc` 取消选区；重测再按 `i`；`r` 复位整轮。

> **2026-10-28 新增：分阶段测高候选 `abcde_integration_pi_column_candidate.yaml`**
> （入口 `./deploy/pi/run_abcde_column.sh`）在原有 4/4 顶盖之外，增加 **2/4 测立柱**：
> 取完底座+立柱、立柱拧到底座上之后按 `i`，参考区同上，目标区标题为
> `Target: current assembly COLUMN top only`，判据 **61.000～67.000 mm（64±3）**。
> 该阶段目前受物理条件限制**测不出读数**（杆顶无有效深度），
> 详见 `docs/2026-10-28_立柱测高与深度探针.md`；默认候选与老入口不受影响。

测区禁区（2026-10-24 实测结论，别再踩）：参考区不能放黑料盘/零件/辐条/阴影/拼接缝，
也不能放在"画面另一头"（30 cm 外会让平面外推误差放大到 ±15 mm）；顶盖区要避开螺丝槽、
轮毂翻边、辐条和绿垫。

---

## 4. ABCDE 是什么（代码地图）

| 字母 | 职责 | 代码 | 关键判据 |
|---|---|---|---|
| **A** 身份 | 冻结 v5：分割 variant + 特征 + 分类 + KCF 追踪 | `assemvision/core/part_identity.py` → `tools/part_dataset.py` 的 `segment_variant/predict` | 只有「本帧可见、`lost_frames==0`、且与某个 blob 轮廓**完全唯一相等**」的轨迹才给身份 |
| **B** 高度 | 人工同帧测高 | `assemvision/core/depth.py` `NativeHeightEstimator` | 参考区拟合桌面平面 → 取目标区法向高度**中位数**；质量门禁：`min_points 50`、`min_valid_ratio .6`、`max_height_spread 8 mm`、`max_plane_residual 2 mm` |
| **C** 观测 | 空盘槽形模板解方向 + 逐槽占用 | `assemvision/core/tray_observer.py` `TrayObserver.observe` | 4 朝向（`angle_offsets_deg: [0]`）NCC 选朝向 → 把 `metal\|brass` 掩膜 warp 到料盘坐标 → 逐槽覆盖率 → `empty/occupied/unknown` |
| **D** 开工联检 | 四槽「当前 A 身份 × C 占用」关联 | `assemvision/core/checks.py` `evaluate_prestart` / `PrestartGate` | 轮廓 fillPoly 与 C 槽多边形求交，「唯一槽 + 覆盖达标」才赋身份；四槽全 match 且连续 3 帧 / 0.3 s 才放行 |
| **E** 取件顺序 | 四工序顺序推定 | `assemvision/core/fsm.py` `PickSequence` + `core/events.py` | 只认 `match→missing` 边沿（手退出后才确认）；越序/多件取走不推进；放回则回退；4/4 只代表取件顺序，**不代表装好** |

### 一帧的线程模型（`assemvision/pipeline.py`）

```
D-acq(唯一读相机, 30fps) ──► D-prep(预分割 v2, ~98ms) ──► D-check(取 prepared)
                                       └──────────────► ├─ D-pair[0] A 身份(v5, ~106ms)
                                                        └─ D-pair[1] C 观测(4 朝向, ~70ms)
                                     关联/门禁/E(~20ms) → 原子快照 → UI
```

2026-10-25 实测单帧 128 ms（检测 7.8 fps，采集 29.7 fps，4 核平均占用仅 60%）：
**A 是硬下限**，再提速必须动 A 的 v5 分割/特征复用（会改身份确认节奏，要单独候选 + 验收）。

---

## 5. 配置链与冻结来源

判据全部由一条**显式 hash 绑定**的链决定，任何一环被改都会拒绝启动：

```
config/abcde_integration_pi_performance_candidate.yaml      ← 现场正式候选（run_abcde.sh 指向它）
 ├─ sequence → config/process_sequence_size_candidate.yaml         (+sha256)
 │   └─ prestart → config/prestart_check_size_candidate.yaml
 │       ├─ identity.model  → ~/ai_测试工具/A轮盘碎孔/20261019/candidate_size_relaxed_01/model.json  (+sha256)
 │       ├─ identity.policy → 同目录 policy.json                                   (+sha256)
 │       ├─ identity.observer → config/tray_observation_v2.yaml                    (+sha256)
 │       ├─ workpiece_sha256 → config/workpiece.yaml
 │       └─ camera_sha256    → config/camera.yaml
 ├─ native_reference → ~/ai_测试工具/B模块深度/20261018_full_baseline_03/camera.native-depth.json (+sha256)
 └─ height_stages（立柱 2/4 量长度 61.000～67.000；顶盖 4/4 量高度 67.500～69.500，只覆盖判定，不动 B 源文件）
```

`tray_observation_v2.yaml` 里面又有 1 张空盘模板 + 8 张方向参考图的 sha256。
**改动任何一张图或任何一个 yaml 数值都要同步更新对应的 hash 字段。**

当前生效参数（`Integration` 反解得到，不是手算）：

| 项 | 值 |
|---|---|
| 高度窗口 | **67.500～69.500 mm**（顶盖 68.5±1；2026-10-29 由 62～68 改）；立柱长度另为 **61.000～67.000 mm** |
| 关联方式 | `manual_same_frame_cap_region`（人工同帧，单帧快照） |
| C 方向 | 4 朝向，`correlation_backend: box_sum_cached` |
| 调度 | `observer_workers 4` / `opencv_threads 1` / `ui_fps 8` |
| 时序 | `mode functional_only`，`max_age/gap/identity_gap = 3.0 s`（**非实时性能验收**） |

自检命令（**改配置、换机器、启动失败时第一步**）：

```bash
cd ~/assembly_vision
~/miniforge3/envs/assembly/bin/python tools/pi_diagnostics/preflight_pi.py
# 输出「结果：全部通过」才可以启动；MISS/HASH 先按备份回退
```

---

## 6. 文件权威性分级（哪些能删、哪些不能动）

### 6.1 权威（唯一真相，别覆盖）

- `assemvision/**`（除 `.bak_*`）、`app.py`、`deploy/pi/run_abcde.sh`
- `config/` 里**不带 `.bak_`** 的文件（尤其 `camera.yaml`、`workpiece.yaml`、
  `tray_observation_v2.yaml`、各 `*_candidate.yaml`）
- `~/ai_测试工具/**` 下的冻结产物（model/policy/native-depth）—— **不在仓库内，删了无法重建**

### 6.2 备份（交接保留，不要清理）

| 备份文件 | 回到哪个状态 |
|---|---|
| `config/abcde_integration_pi_performance_candidate.yaml.bak_20261025` | **基线 70、窗口 67～73（2026-10-25 改动的直接回退点）** |
| `config/abcde_integration_pi_performance_candidate.yaml.bak_20261024` | 基线 68.025、上限 68.5 |
| `assemvision/pipeline.py.bak_20261024_pre_ab` | 无 A/C 并行、无预分割（~2.2 fps） |
| `assemvision/pipeline.py.bak_20261024_prediv` | 有 A/C 并行、无预分割 |
| `assemvision/integration.py.bak_20261024` | 无 `baseline_mm` 覆盖 |
| `config/tray_observation_v2.yaml.bak_20261024` | C 细搜 ±3°（28 位姿） |
| `deploy/pi/run_abcde.sh.bak_20261024` | 指向功能候选 |
| `DEPLOY_PI.md.bak_20261024` | 未加 §7 |
| `app.py.bak_20261023_gc` | 未加 GC 调优 |
| `tests/*.py.bak_20261025`（4 个） | 修「prep 线程」适配前的注入式测试 |
| `tests/test_abcde_integration.py.bak_20261025b` | 修「写死 68.5」断言前 |

### 6.3 可以不带（非代码）

- `data/`（354 MB 合成与回放素材，可由 `tools/make_synthetic.py` 重建）
- `__pycache__/`、`~/runall*.log`、`~/pi_setup.log`
- `~/assembly_g_*_20261022.zip`（10-22 的两份部署快照，历史归档）
- `~/¬`（一个意外产生的怪名目录，可删）

---

## 7. 回归测试现状

```bash
cd ~/assembly_vision
PYTHONIOENCODING=utf-8 ~/miniforge3/envs/assembly/bin/python tests/run_all.py
# 2026-10-25 结果：全部通过 ✓（31 个测试模块）
```

交接前有 **4 个长期红项**，根因和对法如下（**只改测试装配方式与过期断言，
产品代码、判据、断言意图都没动**）：

| 模块 | 失败断言 | 根因 | 处理 |
|---|---|---|---|
| `test_prestart_integration` | `assert pipe._c_process_once()` | 2026-10-24 加了 D-prep 预分割线程后，`_d_process_once` 改从 `self._prepared` 取帧；这些用例是直接写 `pipe._c_input` 的注入式测试，且不启动 prep 线程 → 永远返回 False | 测试里加 `pipe._prep_thread = None`，走回 `_c_input` 分支 |
| `test_process_integration` | 同上 | 同上 | 同上 |
| `test_process_candidate` | 同上 | 同上 | 同上 |
| `test_abcde_integration` | 同上 **+** `height_upper_limit == 68.5` | 上限在 10-24 改 73、10-25 改 68，断言写死没跟 | 前 3 行同上；上限改为**读配置**再校验，并加一条「覆盖值必须偏离冻结基线 68.025」证明覆盖生效 |

> 注意 `run_all.py` 把 traceback 打到 stderr、模块标题打到 stdout，
> 重定向时**两者会交错**，排查失败归属时要加 `PYTHONUNBUFFERED=1`。

---

## 8. 已知问题（接手必读）

1. **「刚放稳就按 i」会出离群读数**：2026-10-25 实测同装配 4 次为 63.10/63.31/63.44/71.93 mm，
   其中 71.93 出现在 4/4 刚完成那一刻。**连测 2 次取中位数**可避开；三次稳定读数极差 0.34 mm，
   背靠背两次差 0.21 mm —— 系统本身不抖，是"时机"问题。
2. **D435 的 UVC 可能挂死**：连续多次开关进程后出现 `取帧失败：Frame didn't arrive within 1000`，
   检测 0 fps 但进程不退出、也不报错。**不用拔线**：
   ```bash
   ~/miniforge3/envs/assembly/bin/python -c "import pyrealsense2 as rs,time; [d.hardware_reset() for d in rs.context().query_devices()]; time.sleep(6)"
   ```
3. **`--no-ui` 不能测性能**：它是 `while: sleep(0.02)` 轮询计数，`--max-frames` 数的是轮询次数，
   会给出「53 fps」这种假数。要测就停掉 app 用 `tools/pi_diagnostics/measure_fps.py`。
4. **顶层 README 原先把读者指向了不存在的文件**（`../README.md`、`../docs/现状与交接.md`、
   `../docs/架构与命令.md`、`../任务卡_*.md` —— 那些在 Windows 开发机上，这台机器上没有）。
   README 已按"部署副本"改写，`tests` 数量也由「16 个」更正为 31 个。
5. **面板不显示深度质量**：B 的结果里其实已经有 `target_points / *_valid_ratio /
   height_spread_mm / plane_rms_mm`，但 UI 只显示高度和合格性。离群时无法判断是"点太少"
   还是"参考区被污染"——建议优先补这个（第 9 节第 2 项）。
6. **检测帧率比记录低**：`docs/2026-10-24` 记录无界面 11.06 fps，2026-10-25 同一候选实测 7.8 fps。
   引用旧数字前先重测。
7. **窗口尺寸/坐标系有两套**：Xwayland 里主窗口是 `(0,0) 956×492`，Wayland 屏幕上显示为
   `(92,38) 770×432`。用 `tools/pi_diagnostics/xin.py` 远程注入时**必须用 X 坐标**；
   截图用 `grim`（`scrot` 在 Xwayland 上只有黑屏）。
8. **`/tmp` 会丢**：本次诊断脚本已归到 `tools/pi_diagnostics/`，重启后从那里跑。

---

## 9. 待办（按优先级，前两条是纯软件改动）

1. **测高重复性流程化**：把「等 2 秒 + 连测 2 次取中位数」写进程序（或至少在面板提示），
   现在靠人工记规矩；
2. **面板显示深度质量**（点数/有效率/高度散布/平面残差）—— 结果字典里已有这些字段，
   只是没渲染。**2026-10-28 已做一半**：`测量：无效` 时多一行
   `深度质量：目标 N 点 / 有效率 x.xx / 参考 M 点`，`stats` 里常驻 `深度质量`；
   高度散布与平面残差仍未渲染；
3. **确认基线最终值**：稳态实测 63.1～63.4，基线设 65，下限 62 只剩 1.2 mm 余量。
   建议改 63.5（窗口 60.5～66.5）或先消离群再重标 —— 需要现场重新授权；
4. **收紧窗口前先换测区**：要 ±1.5 mm 就必须避开顶盖中心凸台、改多帧中位数；
5. **A 模块提速**：动 `tools/part_dataset.py` 的 v5 分割/特征复用会改身份确认节奏，
   必须单独候选 + 独立验收，不能顺手改；
6. **补测试**：`height_assessment` 的 `baseline_mm` 覆盖逻辑仍无专门用例（当前只有现场验证）。
   （2026-10-28：`height_stages` 全链已有 `tests/test_height_stages.py`，覆盖阶段解析、违规拒绝、
   按进度取结果、界面不串台；单阶段 `baseline_mm` 覆盖本身仍无专门用例。）
7. **2/4 立柱检测：高度法不可用，已改为「横放量长度」**（2026-10-28 → 10-29）：
   立柱顶面 Ø15 mm≈17 px、高反光，D435 在那块**一个有效深度点都没有**
   （实测 40×20 px 区域有效点 = 0，周围满格；全图最大高度 18.9 mm），
   所以 10-29 用户改口径为**把装配好的立柱横放、量长度约 64 mm**，
   检测改为 `mode: length`（2D 轮廓轴向跨度，不依赖深度）：
   **现场实测 64.38 mm，卡尺 64.7 mm，差 0.3 mm**。
   操作约定：测区**只框杆身、框到圆盘内侧面为止**，框进圆盘会多算厚度（读数 ~73 mm → 误判）；
   手必须在测区外（皮肤也会过黄铜阈值）。详见 `docs/2026-10-29_立柱长度检测.md`。

---

## 10. 交接验收清单

接手后按顺序做一遍，4 条都过就说明环境没坏：

```bash
cd ~/assembly_vision
PY=~/miniforge3/envs/assembly/bin/python

# 1) 冻结来源全对
$PY tools/pi_diagnostics/preflight_pi.py                # 期望：结果：全部通过

# 2) 判据是 65.5～71.5（顶盖）
$PY -c "
import sys; sys.path.insert(0,'.')
from assemvision.config import load_config
from assemvision.integration import Integration
i=Integration('config/abcde_integration_pi_performance_candidate.yaml', load_config('config/camera.yaml'))
print(i.height_rule)"                                   # 期望：67.500～69.500 mm

# 3) 相机与链路能起来（不需要真装配）
#    ★ 相机是独占的：先停掉正在跑的 app，否则这里会「取帧失败」
pkill -f 'app.py --integration'; sleep 2
DISPLAY=:0 XAUTHORITY=/home/pi/.Xauthority $PY -u app.py \
  --integration config/abcde_integration_pi_performance_candidate.yaml \
  --windowed --no-ui --max-frames 400                     # 期望：D/A/C 出槽位状态，无 Traceback

# 4) 回归测试
PYTHONIOENCODING=utf-8 $PY tests/run_all.py              # 期望：全部通过 ✓（31 个）
```

---

## 11. 常用命令速查

```bash
PY=~/miniforge3/envs/assembly/bin/python
cd ~/assembly_vision

# 启动 / 停止
cd ~/assembly_vision && ./deploy/pi/run_abcde.sh
pkill -f 'app.py --integration'

# 后台 + 可读日志
PYTHONUNBUFFERED=1 setsid nohup ./deploy/pi/run_abcde.sh > /tmp/abcde_run.log 2>&1 &
tail -f /tmp/abcde_run.log

# 帧率 / 分项耗时（会独占相机，先停 app）
$PY tools/pi_diagnostics/measure_fps.py --config config/abcde_integration_pi_performance_candidate.yaml --seconds 20 --warmup 8
$PY tools/pi_diagnostics/where_time2.py 15

# 温度 / 进程
vcgencmd measure_temp; ps -o pid,nlwp,%cpu,rss -p $(pgrep -f 'app.py --integration')

# 界面截图（Wayland）
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 grim /tmp/shot.png

# 远程按键/拖拽（没有 xdotool）
DISPLAY=:0 XAUTHORITY=/home/pi/.Xauthority $PY tools/pi_diagnostics/xin.py windows
```
