# assembly_vision

装配检查系统的运行代码。项目入口、当前进度和技术路线请先看[根README](../README.md)、[现状与交接](../docs/现状与交接.md)和[阶段1任务卡](../任务卡_阶段1_分割识别基准.md)。

## 结构

- assemvision/sources：D435 SDK、视频/图片及兼容UVC源。
- assemvision/core：分割、特征、识别、追踪、几何、判据和流程逻辑。
- assemvision/ui：OpenCV及可选Qt渲染；pipeline.py进行线程编排。
- tools：当前阶段1的数据采集、标注、离线评测，以及合成素材和必要的运行诊断。
- calib：c1～c6标定工具；保留供对应模块使用，不作为当前阶段1的自动执行顺序。
- config：正式工位配置及实验入口；tests：16个回归模块。

## 运行

从仓库根目录的PowerShell使用指定环境：

```powershell
$py = 'D:\miniconda\envs\yolov11\python.exe'
& $py 'assembly_vision/tests/run_all.py'
Push-Location 'assembly_vision'
& $py app.py --source images --path data/synthetic_seq --calibration config/calibration.synthetic.yaml --no-ui --max-frames 200
Pop-Location
```

若合成素材缺失，先在assembly_vision目录运行：

```powershell
& $py tools/make_synthetic.py --out data/synthetic --seq data/synthetic_seq --calib config/calibration.synthetic.yaml
```

D435使用pyrealsense2、640×480、USB3；不要用cv2.VideoCapture打开D435。
当前优先传统视觉，持续无实质进展时对照验证神经网络。开发图同批拟合/回放结果不是独立验收；Pi4B仍未部署。

完整标注、评测、现场和窗口检查命令统一见[架构与命令](../docs/架构与命令.md)，避免在多个README维护互相过期的操作步骤。
