# ABCDE集成队友交接（2026-10-22，rev.3y.2）

## 先看这几条

本轮已将相机→A身份/追踪→C位姿/槽位→D稳定开工→E取件四步→B人工关联高度接入一个app入口。2026-10-22用户收紧高度上限：大于68.5mm不合格，等于可通过；原下限65.025114mm保留，实际范围65.025114～68.500000mm。配置文件名保留height_3mm以兼容原启动命令，但不再按对称±3mm放行。67.654270mm的真实已保存顶盖快照落在范围内，高度项合格。

**紧固性检测由队友负责，本轮未实现，也没有用视觉高度或E4/4替代紧固判断。** E4/4仅取件顺序推定；安装/辐条/其他质量亦不能据此全部判合格。本轮不做G树莓派部署。

本次真实首次整轮停3/4；顶盖只在槽内可见，取出阶段没有有效当前离槽轨迹。只补了未确认的顶盖回槽/再取，最终4/4，前三件未重做。这是带补充步骤的受控通过，不能对外称一次无补充全流程100%通过。详细结果见[集成交付](ABCDE_主程序集成.md)。

## 分工与冻结边界

| 工作 | 负责人/状态 |
|---|---|
| A/C/D/E识别、追踪、几何、开工和取件顺序 | 本程序已集成，现有算法/阈值/模型冻结 |
| 原生深度、顶盖测表面人工关联、高度±3mm | 本程序受控候选；原B正式源文件未改 |
| 紧固性检测方法、实现、有效性与通过/不通过证据 | 队友负责，另立任务范围并独立验证 |
| 紧固结果回传本主程序/联合结论 | 尚无正式接口实现；按下方协议提案对齐后接线，不能写成已经接好 |
| G部署 | 未开始，本次不推进 |

不要为接紧固性修改A身份门槛、分割、追踪、C标定/槽位、D开工或E顺序；不要把「顶盖存在」「高度在范围」「E4/4」当拧紧证据。需要改这些内容时，先提供具体失败证据和最小方案，另获授权。

## 获取代码与运行依赖

仓库：ssh://git@ssh.github.com:443/murongyueyu/assembly-vision.git。更新main后再开自己的分支；本次会把此前32个本地前序提交与本轮交付一起普通push。最终提交号以GitHub main及本轮交付消息为准。

```powershell
git pull --ff-only origin main
git switch -c codex/tightness
```

**GitHub代码不包含模型及硬件绑定文件。只pull仓库无法直接运行本候选。** 必需文件及SHA256见[运行依赖清单](ABCDE_运行依赖清单.json)：

- ai_测试工具/A轮盘碎孔/20261019/candidate_size_relaxed_01/model.json
- 同目录policy.json
- assembly_vision/config/calibration.yaml（当前D435工位C标定）
- ai_测试工具/B模块深度/20261018_full_baseline_03/camera.native-depth.json（原B来源，baseline_accepted=false）

已在交付机器另生成ai_测试工具/ABCDE集成/20261019/队友运行依赖_20261019.zip，包含上述四文件及hash清单，约18KB。请向项目负责人取这份ZIP，在新克隆的同版仓库内按相对路径放置；先核对hash，不覆盖另一工位已有标定。完整原始证据另行备份，未塞进这个运行包，也不随Git推送。

Windows当前解释器固定D:\miniconda\envs\yolov11\python.exe（Python3.11.9、OpenCV5.0.0、numpy2.2.6、pyrealsense2 2.59.1、PyYAML，UI用Pillow）。D435用SDK、USB3，普通桌面权限运行；不要换成cv2.VideoCapture。当前设备SN262322070302，640×480@30。换相机、分辨率或工位，原标定/参考/高度基线不能算新工位已验收；先按原规则复核，不按分辨率外推。

## 启动与最少现场操作

以下从仓库根目录PowerShell运行：

```powershell
& 'D:\miniconda\envs\yolov11\python.exe' assembly_vision/tests/run_all.py
& 'D:\miniconda\envs\yolov11\python.exe' assembly_vision/app.py --integration config/abcde_integration_height_3mm_candidate.yaml --windowed
```

显式选择height_3mm候选。旧abcde_integration_candidate.yaml不判高度合格；不传--integration是旧兼容入口，会提示，不能拿它当本次集成演示。缺依赖/hash不符应停止，不能修改hash绕过，也不能退回旧模型。

1. 四件各在正确槽，料盘完整居中，手退出；等待D稳定放行。
2. 按底座→立柱→轮盘组件→顶盖正常逐步取件/装配。等每一步确认再继续。顶盖小，捏边时保留金属可见；完全遮挡且从未看到离槽不会补记。不要为追求4/4降低门槛。
3. 4/4后保持装配体不动、手退出，点击主窗口按i。在冻结彩色图先框裸绿垫参考、Enter；再框当前装配体中央顶盖顶部小片、Enter。避开边缘、轮盘和背景；看不清就C取消，保持待测。
4. 结果只属于标出的测量帧。高度数值有效并在原下限65.025114mm至上限68.5mm（边界包含）内才显示高度合格；超差显示高度不合格，空洞/错误设备/取消等不判通过。它不是持续自动表面定位或整机全部合格。
5. r复位清进度/事件/测量，重新过D；q正常退出并释放设备。

若需留证而非只看窗口，先确认实际摆放，再运行（--output必须为新目录）：

```powershell
& 'D:\miniconda\envs\yolov11\python.exe' assembly_vision/tools/validate_abcde_integration.py --output 'ai_测试工具/ABCDE集成/teammate_next' --initial-truth '填写实际摆放确认原文'
```

窗口运行中会持续写observations.jsonl和progress.json；状态变化保存UI及同帧深度/XYZ。往输出目录control.json写不同id的`{"id":"cp_01","action":"checkpoint"}`可保存检查点，不必关闭窗口。最终report.json才包含完整退出审计。不要覆盖旧证据目录，不按目录名充当真值，不把连续帧当独立试验。

## 紧固性结果交回提案（尚未接入）

当前接线文件为app.py、assemvision/pipeline.py和assemvision/integration.py；core/depth.py只负责冻结高度计算。测量结果在pipe.integration.result，流程代次为pipe._e_generation（当前内部字段，正式回传前应封装稳定接口）。证据工具输出目录可用作本次session标识；不要跨进程仅凭帧号或不同机器的monotonic数值硬拼。

建议紧固模块先交结构化结果，人工描述和原始证据一起保留：

```json
{
  "inspection": "tightness",
  "session_id": "双方约定的本次运行ID",
  "cycle_generation": 0,
  "evidence_frame_index": 9965,
  "status": "unknown",
  "reason": "not_measured",
  "evidence_paths": [],
  "method_version": "队友检测方法版本"
}
```

status使用pass/fail/unknown三态；未测、超时、失效或证据关联不足必须unknown，不能默认pass。若使用独立传感器，记录它的原始时间、单位、设备和与装配体关联方法；不能伪称相机同帧。结果必须对应当前运行及装配轮次，复位/换装配体后旧结果失效。pass标准和误差依据由队友实测给出，不能从本程序的±3mm高度标准推导紧固标准。

先独立验证紧固模块，再提出回传接口的具体文件与最小接线改动；本交付只保留「紧固性：队友负责」，不消费以上JSON，不自动合并整机合格。双方对齐接口后另做受控联合验证。

## 交回给项目负责人的材料

- 独立分支、提交hash及修改清单；说明为什么需要每处改动。
- 一条可运行命令、环境/设备/参数、测试日志；原30模块回归结果。
- 正常、真实不紧固/缺陷、无效/超时等场景的用户确认和原始证据，分别记录，不造标签。
- 结果协议样例、有效期/复位行为、未完成事项；若需改主程序集成先提交具体方案。
- 新增依赖另打包并给hash，不提交凭据、模型全集或巨大原始数据。

可在自己的codex/tightness分支提交并交PR给负责人审核，保持main的现有算法/配置冻结；不要强推共享分支。紧固性接好后再开展一次联合验收，G仍另立专项。
