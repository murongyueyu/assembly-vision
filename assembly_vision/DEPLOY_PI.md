# 树莓派 4B 部署说明

> 当前六步入口和验证边界见仓库根目录 `docs/Pi快照_20261029_启动与交接.md`，本文件其余内容保留历史口径。

本文件记录这台树莓派上的实际环境。**改动前先看「已知坑」一节**，其中两条是
ARM64 特有的，在 Windows 开发机上永远看不到。

## 1. 环境

| 项 | 值 |
|---|---|
| 硬件 | Raspberry Pi 4B，**内存 1.8GiB（2GB 版）** |
| 系统 | Raspberry Pi OS（Debian 13 trixie）64-bit 桌面版，镜像 2026-10-06 |
| 主机名 | `rpi4-assembly` |
| 账号 | `pi`（登录密码在提示时输入，不入库） |
| 存储 | 57G（首次启动已自动扩展分区） |
| Python | `~/miniforge3/envs/assembly`，Python **3.12.15** |
| 代码 | `/home/pi/assembly_vision` |

## 2. 已装依赖

```
pyrealsense2          2.59.1.11053
opencv-contrib-python 4.14.0.94      ← 不是 5.0.0，见第 4 节
numpy                 2.2.6
PyYAML                6.0.2
Pillow                11.0.0
```

系统库：`libusb-1.0-0`、`libgl1`、`libglib2.0-0`、`libsm6`、`libxext6`、
`libxrender1`、`libxcb-xinerama0`、`libxkbcommon-x11-0`、`libdbus-1-3`、`libgomp1`、
`fonts-wqy-zenhei`（界面中文标签用）。

D435 的 udev 规则：`/etc/udev/rules.d/99-realsense-libusb.rules`。

## 3. 启动

在树莓派桌面的终端里：

```bash
cd ~/assembly_vision

# 默认兼容入口（真实相机）
./run_app.sh --windowed

# 集成候选（ABCDE 全流程）
./run_app.sh --integration config/abcde_integration_height_3mm_candidate.yaml --windowed

# 无界面，只打印结论
./run_app.sh --no-ui

# 用合成序列离线验（不需要相机）
./run_app.sh --source images --path data/synthetic_seq \
             --calibration config/calibration.synthetic.yaml --no-ui --max-frames 200
```

`run_app.sh` 会切到本目录并调用 `~/miniforge3/envs/assembly/bin/python`。
窗口模式需要桌面会话（XWayland 上 `DISPLAY=:0`，实测可用）。

## 4. ★ 已知坑（ARM64 特有）

### 4.1 pyrealsense2 没有 cp311 / cp313 的 ARM64 wheel

PyPI 上 `pyrealsense2` 的 aarch64 wheel 只有 `cp39` / `cp310` / `cp312`。
系统的 Python 是 **3.13**，装不上。所以必须用 Miniforge 另建 **3.12** 环境。
**不要**图省事把项目跑在系统 Python 上。

### 4.2 `cv2.TrackerKCF()` 直接构造 → 段错误

OpenCV 的 ARM64 wheel（5.0.0.93 和 4.14.0.94 都一样）里，用类直接构造：

```python
t = cv2.TrackerKCF()      # 构造"成功"
t.init(img, (x, y, w, h)) # ← Segmentation fault，进程直接死
```

段错误不是 `Exception`，`try/except` 抓不到。而
`part_tracker.make_tracker()` 的候选顺序恰好把这个写法排在第一位。

**处理**：本机在 conda 环境里放了一个 `sitecustomize.py`
（`~/miniforge3/envs/assembly/lib/python3.12/site-packages/sitecustomize.py`），
启动时把 `cv2.TrackerKCF` / `cv2.TrackerCSRT` 从模块里删掉，
于是 `make_tracker` 的 `except` 会捕获 `AttributeError` 并回落到
`cv2.TrackerKCF_create()`（实测完全正常）。

这是**环境补丁，不是项目代码改动**。如果在别处重装环境，必须重新装它
（`pi_setup.sh` 里已自动化）。

### 4.3 OpenCV 用 4.14 而不是 5.0

除了 4.2 的段错误，实测 5.0.0.93 即使打了补丁，`test_tracker` 也会在
「有像零件但认不出的块时不能判丢」这条断言上失败 —— 5.0 的追踪行为与 4.x 有差异。
4.14.0.94 下 30 个测试模块的表现与开发机一致。

### 4.4 缺 `config/calibration.yaml`

这个文件是 D435 工位标定产物，被 `.gitignore` 排除，**当前工作区里本来就没有**
（开发机上同样缺，`test_tracking_roi` / `test_process_candidate` /
`test_abcde_integration` 在两边都失败）。

后果：真实相机模式下只能用兜底 ROI，位姿/槽位判据不成立。
要用真实工位流程，必须先在树莓派上重跑标定（`calib/c1` → `calib/c3`），
或从备份恢复该文件。

## 5. 重新安装

```bash
cd ~
bash pi_setup.sh            # 幂等：已装的会跳过
```

脚本会：apt 运行库 → Miniforge → Python 3.12 环境 → pip 依赖 →
sitecustomize 补丁 → D435 udev 规则 → 冒烟测试 → 30 模块回归。

## 6. 实测结果（2026-10-22）

| 项 | 结果 |
|---|---|
| 30 个测试模块 | **27 通过**；3 个失败均因缺 `calibration.yaml`（开发机同样失败） |
| 合成序列端到端 | 装配流程 **4/4**，齐套性 OK |
| `--windowed` | 12 帧正常，中文标签正常 |
| D435 | **枚举成功**：SN 262322070302、固件 5.17.3.10、**USB 3.2**；13 项控件全部写入成功 |
| 纯 SDK 读帧 | 640×480 彩色+深度 **30.0 fps**（单帧 33.37ms），相机侧达标 |
| 真实相机端到端 | `app.py --no-ui --max-frames 100` 4.7s；内存 used 471Mi / available 1.3Gi，**2GB 够用** |
| **带界面帧率** | **1920×1080 → 5.5 fps**；1280×720 → 9.1；**960×540 → 12.5**；800×480 → 11.0 |

### 6.1 ★ `--no-ui` 不能用来测性能

`app.py` 的 `--no-ui` 分支是 `while ... time.sleep(0.02)` 的轮询循环
（第 92-109 行），`--max-frames` 数的是**轮询次数**而不是处理帧数。
实测会给出「53fps」这种数字，那是空转，**不反映处理能力**。
真实帧率只有带界面时的 5.5~12.5 fps。

### 6.2 性能结论

相机侧完全达标，**CPU 侧是瓶颈**：最乐观 12.5 fps，距相机 30fps 差 2~5 倍。
缩小界面分辨率只能小幅改善（1080p 5.5 → 960×540 12.5），
说明瓶颈在分割/识别/追踪的计算，不在绘制。

## 7. 2026-10-24 现场改动（必读）

- **测高取样**：参考区必须取**工件旁边 3～5 cm** 的裸绿垫（60×60），顶盖取
  **12×12** 的中心圆盘平面。参考区放远处（30 cm）会让平面外推误差放大到 ±15 mm，
  实测同一件在 59.6～75.3 mm 之间乱跳；就近取参考后稳定在 69.0 mm。
- **C 取向**：只做 4 朝向（去掉 ±1° 微调），方向角稳定偏 −1.000°，槽位状态一致；
  这是 9.65 fps 的主要来源。详见
  `docs/2026-10-24_测高规范与提速记录.md`。
- **判定窗口**：`62.000～68.000 mm`（基线 65 mm ±3，2026-10-25 改判）；
  2026-10-24 的 70 mm 基线 / 73 mm 上限、以及更早的 68.5 mm 上限均已作废；
  B 源文件与其 hash 未动。详见
  `docs/2026-10-25_cap_baseline_65mm.md`。
- **帧率**：同一场景带界面 **2.22 → 9.65 fps**（C 只做 4 朝向、A/C 并行、预分割线程），
  视频侧仍 29～30 fps。
- **部署入口**：`deploy/pi/run_abcde.sh` 已指向
  `config/abcde_integration_pi_performance_candidate.yaml`（性能候选）。
- 备份文件与回退命令见上述文档第 4、6 节。
