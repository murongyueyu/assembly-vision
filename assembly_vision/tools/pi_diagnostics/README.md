# tools/pi_diagnostics —— 只读诊断脚本

这些脚本**只测量、不改变判据、不改配置、不写业务文件**。原本散落在 `/tmp`
（重启就丢），2026-10-25 交接时归到这里。

统一用现场环境跑：

```bash
cd ~/assembly_vision
PY=~/miniforge3/envs/assembly/bin/python
```

| 脚本 | 用途 | 典型用法 | 会不会抢相机 |
|---|---|---|---|
| `preflight_pi.py` | 启动前自检：逐级核对配置链与**所有冻结 hash**，并打印生效判据 | `$PY tools/pi_diagnostics/preflight_pi.py` | 不碰相机 |
| `measure_fps.py` | 测采集/检测帧率（计数包装，不改算法） | `$PY tools/pi_diagnostics/measure_fps.py --config config/abcde_integration_pi_performance_candidate.yaml --seconds 20 --warmup 8` | **会**，必须先停掉正在跑的 app |
| `where_time2.py` | 按线程拆分单帧耗时（A / C / D-prep / 关联尾），并给出 4 核占用 | `$PY tools/pi_diagnostics/where_time2.py 15` | **会**，同上 |
| `xin.py` | X11 输入注入（ctypes + libXtst，无需装 xdotool）：列窗口/点击/拖拽/按键/移窗 | 见下 | 不碰相机（只是操作 UI） |

## preflight_pi.py

ABCDE 启动失败的**第一大原因**是某个冻结文件被改过或被删，hash 不符。
这个脚本把 `integration → sequence → prestart → identity model/policy → observer
→ 参考图 → workpiece/camera` 整条链的 sha256 全核一遍：

```
OK    sequence        /home/pi/assembly_vision/config/process_sequence_size_candidate.yaml
OK    model           /home/pi/ai_测试工具/A轮盘碎孔/20261019/candidate_size_relaxed_01/model.json
...
结果：全部通过
```

有 `MISS` / `HASH` 就**不要启动**，先按第 5 节备份回退。

## measure_fps.py / where_time2.py

`--no-ui` **不能**用来测性能（那是 `sleep(0.02)` 轮询计数，会给出 50 fps 的假数）。
要测就停掉 app 再跑这两个，或者用 `deploy/pi/profile_runtime.py`（带窗口）。

2026-10-25 实测（空盘、性能候选）：

```
采集 29.7 fps / 检测 7.8 fps / 单帧 128 ms
A 身份 106 ms（内部 v5 分割 66 ms）  C 观测 70 ms  D-prep 分割 98 ms  关联尾 ~20 ms
4 核平均占用 60%
```

## xin.py（远程操作 UI）

树莓派没装 `xdotool`，这个脚本用 ctypes 直接调 `libXtst` 的 XTEST 扩展。
只对 Xwayland（`DISPLAY=:0`）里的 cv2 窗口有效。

```bash
export DISPLAY=:0 XAUTHORITY=/home/pi/.Xauthority
$PY tools/pi_diagnostics/xin.py windows                 # 列顶层窗口+几何
$PY tools/pi_diagnostics/xin.py movewin 'Target:' 0 0   # 把选区窗口挪到左上
$PY tools/pi_diagnostics/xin.py drag 474 419 534 479    # 左键拖拽（画选区）
$PY tools/pi_diagnostics/xin.py key i                   # 按键（keysym 名）
$PY tools/pi_diagnostics/xin.py key Return
```

### 界面坐标备忘（**换窗口尺寸/换屏后必须重测**）

- 选区窗口（`cv2.selectROI`）是 **640×480 图像 + 39 px 工具栏**，
  图像原点在窗口客户区 `(0, 39)`；
- 主窗口在 Xwayland 里是 `(0,0) 956×492`，但在 Wayland 屏幕上显示为
  `(92, 38) 770×432` —— **两套坐标不一致**，注入点必须用 X 坐标，
  看画面用 grim 截图；
- Wayland 截图：`grim` 可用（`scrot` 在 Xwayland 上只能拍到黑屏）；
- 选区窗口默认位置会被 WM 放到屏幕下半部（图像被裁掉），先用 `movewin` 挪到 `(0,0)`。

## 别做的事

- 不要用这些脚本去改判据或"顺手"调阈值；判据只在 `config/` 里改，并且要留备份；
- 不要为了看清结果去改 `app.py` 的 --no-ui 分支（它本来就不是性能通道）；
- `measure_fps.py` / `where_time2.py` 会独占相机，跑之前先 `pkill -f 'app.py --integration'`。
