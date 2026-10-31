#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$project_dir/data/retargeted/bones_seed_native23_v4"
exec "$project_dir/../SOMA-Retargeter/.venv/bin/python" \
  "$project_dir/scripts/prepare_bones_seed_batch.py" \
  --work-dir "$work_dir" --reuse-retargeted-root "$work_dir" "$@"
