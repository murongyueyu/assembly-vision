# run_calib.ps1 —— 标定脚本一键运行（自动切目录 + 用对解释器）
#
# 为什么需要它：标定脚本在 assembly_vision 目录下，而必须用 yolo 环境的
# Python（OpenCV 5.0）才能拿到正确的相机模式。手工敲很容易搞错目录或解释器。
#
# 用法（在 assembly_vision 目录下）：
#     .\run_calib.ps1 c1            # 测 GSD（默认工作距离 360mm）
#     .\run_calib.ps1 c2            # 双目标定（拿基线 B）
#     .\run_calib.ps1 c3            # 料盘单应（手工点四点）
#     .\run_calib.ps1 c4            # 逐槽标准件基线
#     .\run_calib.ps1 c5            # 装配位基线
#     .\run_calib.ps1 live          # 实时预览（调曝光/摆放用）
#     .\run_calib.ps1 probe         # 相机能力探针
#     .\run_calib.ps1 eyes          # 双眼遮挡检查
#     .\run_calib.ps1 app           # 正式跑（窗口模式）
#
# 如果提示"禁止运行脚本"，先执行一次：
#     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

param(
    [Parameter(Position = 0)][string]$Target = "c1",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = "Stop"

# ── 配置：解释器与工作目录 ─────────────────────────────────────
$Py = "D:\miniconda\envs\yolov11\python.exe"     # ★ 必须是有 OpenCV 5.0 的那个
if (-not (Test-Path $Py)) {
    Write-Host "找不到 Python: $Py" -ForegroundColor Red
    Write-Host "请改成你环境里的 python.exe（需要有 cv2 的那个）" -ForegroundColor Yellow
    exit 1
}

$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location $Root

# ── 前置检查：相机是否被占用 ───────────────────────────────────
function Test-CameraBusy {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -match 'c1_|c2_|c3_|c4_|c5_|probe|stereo_calib|live_preview|app\.py' }
    return $procs
}
$busy = Test-CameraBusy
if ($busy) {
    Write-Host "★ 检测到可能占用相机的进程：" -ForegroundColor Yellow
    foreach ($b in $busy) { Write-Host ("   PID {0}: {1}" -f $b.ProcessId, $b.CommandLine) }
    Write-Host "  如果打不开相机，请先关掉它们的窗口。" -ForegroundColor Yellow
    Write-Host ""
}

# ── 按目标选择脚本 ─────────────────────────────────────────────
switch ($Target.ToLower()) {
    "c1"    { $script = "calib\c1_measure_fov.py"; $default = @("--distance-mm", "360") }
    "c2"    { $script = "calib\c2_stereo.py";      $default = @("--cols", "9", "--rows", "6", "--square", "19.0") }
    "c3"    { $script = "calib\c3_tray_homography.py"; $default = @("--manual") }
    "c4"    { $script = "calib\c4_slot_baseline.py"; $default = @("--frames", "20") }
    "c5"    { $script = "calib\c5_assembly_baseline.py"; $default = @("--frames", "20") }
    "live"  { $script = "tools\live_preview.py";   $default = @("--no-auto", "--exposure", "-5") }
    "probe" { $script = "tools\probe_camera.py";   $default = @("--seconds", "0.8") }
    "eyes"  { $script = "tools\check_eyes.py";     $default = @() }
    "app"   { $script = "app.py";                  $default = @("--windowed") }
    default {
        Write-Host "未知目标: $Target" -ForegroundColor Red
        Write-Host "可用: c1 c2 c3 c4 c5 live probe eyes app" -ForegroundColor Yellow
        exit 1
    }
}

if (-not (Test-Path $script)) {
    Write-Host "找不到脚本: $script" -ForegroundColor Red
    Write-Host "当前目录: $(Get-Location)" -ForegroundColor Yellow
    exit 1
}

# 用户额外参数优先：给了就不加默认参数
if ($Rest -and $Rest.Count -gt 0) {
    $argv = $Rest
} else {
    $argv = $default
}

Write-Host "==> $Py $script $($argv -join ' ')" -ForegroundColor Cyan
& $Py $script @argv
exit $LASTEXITCODE
