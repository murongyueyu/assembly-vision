#!/usr/bin/env bash
# 2026-10-28 新增入口：分阶段测高（2/4 测立柱 64±3、4/4 测顶盖 62～68）。
# 与 run_abcde.sh 的唯一差别是--integration 指向分阶段候选；算法/闸门不变。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
assembly_python="${ASSEMBLY_PYTHON:-$HOME/miniforge3/envs/assembly/bin/python}"
if [[ ! -x "$assembly_python" ]]; then
    printf 'Python environment missing: %s\n' "$assembly_python" >&2
    exit 1
fi
exec "$assembly_python" app.py \
    --integration config/abcde_integration_pi_column_candidate.yaml \
    --windowed --display 960x540 "$@"
