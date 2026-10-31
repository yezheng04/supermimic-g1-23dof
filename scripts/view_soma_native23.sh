#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
soma_python="${SOMA_PYTHON:-$project_dir/../SOMA-Retargeter/.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1
exec "$soma_python" "$project_dir/scripts/view_soma_native23.py" "$@"
