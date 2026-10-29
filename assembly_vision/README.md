# assembly_vision —— 树莓派部署副本

**这是一份"跑在现场"的部署副本，不是开发主仓。** 开发机在 Windows 上，
完整工程、标定产物和项目级文档在那边；本目录只保证"在这台树莓派上能跑、且判据可追溯"。

👉 **接手先读 [docs/HANDOVER.md](docs/HANDOVER.md)**（环境实况、怎么跑、配置链与
冻结 hash、已知问题、待办、回滚）。

## 最快上手

```bash
cd ~/assembly_vision
./deploy/pi/run_abcde_column.sh          # 现场推荐：2/4 测立柱长度 + 4/4 测顶盖高度
./deploy/pi/run_abcde.sh                 # 只测顶盖（旧候选，不建立柱阶段）
```

窗口里：`q` 退出 · `r` 复位 · `i` **按进度生效**：`2/4` 测立柱长度（61.000～67.000 mm）、
`4/4` 测顶盖高度（67.500～69.500 mm）。

## 目录

| 目录 | 内容 |
|---|---|
| `assemvision/` | 库：`sources`(D435/视频源) · `core`(分割/识别/追踪/几何/判据/流程) · `ui`(OpenCV·可选Qt) · `pipeline.py`(线程编排) · `integration.py`(ABCDE 接线) |
| `config/` | 全部判据入口；现场正式候选 `abcde_integration_pi_column_candidate.yaml`（分阶段：2/4 立柱长度 + 4/4 顶盖高度） |
| `deploy/pi/` | `run_abcde_column.sh`(分阶段正式入口) · `run_abcde.sh`(只测顶盖) · `profile_runtime.py` · `benchmark.py` |
| `tools/` | 采集/标注/离线评测；`tools/pi_diagnostics/` 是只读诊断脚本（帧率、耗时、hash 自检、远程按键） |
| `calib/` | c1～c6 标定工具，按需使用 |
| `tests/` | **33 个回归模块**，2026-10-29 **全绿**（含 `test_height_stages`）；红项历史见 HANDOVER 第 7 节 |
| `docs/` | 现场记录：`HANDOVER.md`、`2026-10-29_项目实现说明_报告素材.md`、`2026-10-29_立柱长度检测.md`、`2026-10-28_立柱测高与深度探针.md` 等 |
| `data/` | 合成与回放素材（354 MB，交接时可不带） |

## 现场三个硬约束

1. **冻结来源不能改**：不少配置里有 `*_sha256`，改文件而不改 hash 会直接启动失败
   （这是特性，不是 bug）。改动前先跑
   `python tools/pi_diagnostics/preflight_pi.py`，改完再跑一遍。
2. **不要用 `--no-ui` 测性能**：那条分支是 `sleep(0.02)` 轮询计数，会给出 50 fps 的假数。
3. **ABCDE 只在 `--integration` 入口下成立**：不传该参数就退回旧主流程兼容模式。

## 文档索引

- `docs/HANDOVER.md` —— **交接主文档**（先读它）\n- `docs/2026-10-29_项目实现说明_报告素材.md` —— **系统结构、两处测量项、标定与验证数据**（写报告看这份）\n- `docs/2026-10-29_立柱长度检测.md` —— 2/4 立柱为什么从"测高"改成"量长度"+ 现场标定 0.638 mm/px\n- `docs/2026-10-28_立柱测高与深度探针.md` —— 深度法失败的完整证据与两个探针工具
- `docs/2026-10-25_cap_baseline_65mm.md` —— 顶盖高度基线 70→65 mm 的改动、证据与离群分析
- `docs/2026-10-24_测高规范与提速记录.md` —— B 测高取样规范、检测帧率四步提速、备份回退
- `DEPLOY_PI.md` —— 环境安装、ARM64 三个坑、实测结果、2026-10-24/25 现场改动
- `tools/pi_diagnostics/README.md` —— 诊断脚本用法与界面坐标备忘
