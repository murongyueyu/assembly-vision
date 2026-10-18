[CmdletBinding()]
param(
    [switch]$Replay,
    [ValidateRange(0, 1000000)][int]$MaxFrames = 0
)

# 仅D开工演示；沿用指定解释器，不激活或修改其他环境。
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
$env:PYTHONIOENCODING = 'utf-8'
$demoRoot = $PSScriptRoot
$repoRoot = Split-Path -Parent $demoRoot
$pythonExe = 'D:\miniconda\envs\yolov11\python.exe'
$demoVideo = Join-Path $demoRoot 'D-prestart-demo.mp4'

if ($Replay) {
    if (-not (Test-Path -LiteralPath $demoVideo -PathType Leaf)) {
        throw 'Demo video is missing. Copy D-prestart-demo.mp4 into this folder.'
    }
    Invoke-Item -LiteralPath $demoVideo
    exit 0
}

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw 'Required Python was not found: D:\miniconda\envs\yolov11\python.exe'
}
Write-Host 'D prestart demo: correct slots -> missing/wrong part -> glove occlusion -> recovery.'
Write-Host 'Parts show their current contours; the glove shows a rectangle only.'
Write-Host 'This demo does not start assembly steps. Press Q or Esc to exit.'
Push-Location -LiteralPath (Join-Path $repoRoot 'assembly_vision')
try {
    & $pythonExe app.py --prestart-check config/prestart_check.yaml --windowed --max-frames $MaxFrames
    $demoExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $demoExitCode
