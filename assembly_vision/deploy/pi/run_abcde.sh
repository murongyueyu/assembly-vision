#!/usr/bin/env bash
# G部署入口：显式选择已验证的环境及ABCDE候选，不落回兼容流程。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
assembly_python="${ASSEMBLY_PYTHON:-$HOME/miniforge3/envs/assembly/bin/python}"
if [[ ! -x "$assembly_python" ]]; then
    printf 'Python environment missing: %s\n' "$assembly_python" >&2
    exit 1
fi
# 2026-10-24 切到性能候选：检测 1.15→2.2fps（仅运行配置，算法/判据不变）
exec "$assembly_python" app.py \
    --integration config/abcde_integration_pi_performance_candidate.yaml \
    --windowed --display 960x540 "$@"
