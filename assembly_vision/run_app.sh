#!/usr/bin/env bash
# 树莓派上启动主程序。参数原样透传给 app.py。
#   ./run_app.sh --windowed
#   ./run_app.sh --no-ui --max-frames 300
cd "$(dirname "$0")" || exit 1
exec "$HOME/miniforge3/envs/assembly/bin/python" app.py "$@"
