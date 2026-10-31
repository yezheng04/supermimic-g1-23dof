#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task_python="${1:-python3.11}"
if [ ! -d .venv ]; then
  "$task_python" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install --no-deps -e . pytest pluggy iniconfig
.venv/bin/python -c 'import torch, mujoco, mjlab, mujoco_warp; print("torch",torch.__version__,"mujoco",mujoco.__version__,"CUDA",torch.cuda.is_available())'
.venv/bin/sm23 demo
