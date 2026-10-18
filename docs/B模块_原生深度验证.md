# B 模块：D435 原生深度与高度验证

2026-10-18。当前处于现场验证阶段，不声明完整装配或缺立柱验收完成。

## 实现与边界

- 显式开启 depth.enabled=true、depth.backend=realsense 才开彩色和 Z16 双流；默认正式 camera.yaml 仍仅开彩色，A 模块来源 hash 不变。
- SDK 将深度对齐彩色，并按设备实际 depth_scale 换算毫米；同帧 SDK pointcloud 反投影为 XYZ 毫米，不使用旧 HBV 内参。0/空洞为 nan，right/right_full 始终为空。
- 显式裸桌面参考区拟合三维平面，测量目标到平面的法向距离。目标不参与拟合；不再从整幅图猜桌面。平面残差、内点比例、目标有效比例/点数/高度散布和跨帧散布均有门禁，参数在 config/depth_d435.yaml。
- 目标区选轮盘可见平坦金属顶面的小片，不能框整个零件、背景、孔洞或中心顶盖。当前为固定机位、固定测量区，不自动定位轮盘；移动相机、目标位置或改测量表面须重新选区及验证。
- 运行配置绑定设备序列号、对齐内参和 depth_scale，不符时拒绝高度。pipeline 不初始化旧双目匹配器，深度无效或超过2秒返回未知。
- 高度基线/容差只注入判据的内存配置副本。实际终检从 workpiece 的 assembly 字段取基线，仅写 calibration 字段不足以接通；此路径有回归覆盖。c5生成的候选默认baseline_accepted=false，原图表面核验及独立实测通过前不向终检提供基线，防止稳定的错选区读数生效。
- 立柱实时流程的高度辅助阈值 column_threshold_mm 暂不自动生成；需采底座与完整装配对照后决定。不修改工序或合格性算法；原主程序的旧几何与完整流程验收仍待对应模块处理。

SDK 机制依据：[RealSense 投影与单位说明](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/)。SDK 点云保留实际反投影模型，不以理想针孔近似替代设备模型。

## 实验步骤

从仓库根目录 PowerShell 使用指定解释器。相机必须 USB3，开流和写盘需要普通桌面权限。以下输出目录必须是新的，工具拒绝覆盖。

```powershell
$py = 'D:\miniconda\envs\yolov11\python.exe'
& $py 'assembly_vision/tools/probe_native_depth.py' --out 'ai_测试工具/B模块深度/stream_next'
```

基线：先把四件组装成完整装配体，放在料盘左侧绿垫上、完整入画，手退出。启动后第一个窗口只选平整空绿垫作为参考；第二个窗口选轮盘顶面一小片金属，避开孔洞与顶盖。每次鼠标拖矩形后按回车；C或ESC取消选区，此次不能形成有效基线，需用新目录重开。随后自动采30个有效样本。

```powershell
& $py 'assembly_vision/calib/c5_assembly_baseline.py' --native-depth calibrate --frames 30 --out 'ai_测试工具/B模块深度/full_next'
```

成功才生成 camera.native-depth.json；过低高度、空洞或不稳定不会生成基线。report.json 含全部尝试与拒绝原因，首/末样本保存彩色PNG和深度/XYZ NPZ，可独立重算。只在退出后检查报告，关闭窗口或口头说完成不能代替结果。

独立正常检查：保留机位和目标位置，先不拆装，重新开流采样；随后另做重新摆放的重复性验证。无需再选区。

```powershell
& $py 'assembly_vision/calib/c5_assembly_baseline.py' --native-depth check --native-reference 'ai_测试工具/B模块深度/full_next/camera.native-depth.json' --native-label full --frames 30 --no-preview --out 'ai_测试工具/B模块深度/full_check_next'
```

缺立柱检查：按现场确认的安全摆放方式去掉立柱，保留底座/轮盘等、使被测轮盘表面仍落在原目标区域，再运行同样check，native-label改missing-column、out换新目录。先保存现场原图核对，不能拿测到桌面或错误零件的高度冒充缺立柱检测。高度不符合仅说明高度偏离，不单独证明具体缺件；空洞必须是未知。

只有 B 高度数据和独立场景验证就绪、记录证据并将运行副本的baseline_accepted置true，才可通过 app.py --camera 指向该运行配置向终检提供原生基线。未接受的候选仍可用于独立check，但不能当已验收标定；这不代表旧料盘标定有效或已集成 A 模块 v5。

## 已验证证据

- test_depth 覆盖设备非默认单位、倾斜三维桌面、离群点、空洞/共线/重叠/粗糙参考拒绝、目标高度混合拒绝、设备绑定、SDK双流/对齐/XYZ、pipeline基线实际消费与过期数据，以及 c5 标定→独立检查→失败清理的完整离线流程。
- 17个回归模块通过；主程序合成200次UI轮询达到4/4、终检OK；--windowed合成12帧正常退出。这些不是原生深度现场精度验收。
- 20261018_native_stream_01：受限进程0x80070005拒绝访问，不是相机损坏；_02在普通权限通过。USB3.2，彩色/深度640×480，XYZ 480×640×3，5帧有效率0.9997135～0.9998893，全画面深度中位440～441mm；XYZ-Z与毫米深度最大差约0.00003052mm，仅说明单位/管线一致，不代表绝对测量误差如此小。
- 20261018_full_baseline_01：用户未看清选区要求，中止进程；experiment_status.json明确aborted_by_user、valid_baseline=false，无有效基线。
- 20261018_full_baseline_02：30次全部通过数值门禁，高度中位66.3147mm，帧间P90-P10=1.0233mm；参考矩形[52,20,290,213]，目标矩形[248,360,28,28]。原图复核发现目标包含中心顶盖及周边结构，不是纯轮盘顶面，不能据此确认轮盘高度基线。用户尺量真实高度与轮盘水平状态待核对，独立正常/缺立柱检查未做。自动生成的camera.native-depth.json仅是待核验候选。

实验目录在 ai_测试工具/B模块深度/，被Git忽略，交接需另备份。A v5模型、核心算法、正式相机及工件配置不改。
