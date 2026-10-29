#!/usr/bin/env bash
# F六步操作界面；不改变既有两个启动入口。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
exec "${ASSEMBLY_PYTHON:-$HOME/miniforge3/envs/assembly/bin/python}" app.py \
  --integration config/abcde_integration_pi_ui6_candidate.yaml \
  --renderer opencv --display auto "$@"
