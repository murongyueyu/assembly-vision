@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Demo.ps1" -Replay
if errorlevel 1 pause
