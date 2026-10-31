#!/usr/bin/env bash
# Install the tested CUDA stack in an isolated environment on the training host.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports
task_uv="${SM23_UV:-$HOME/.local/bin/uv}"
if [ ! -d .venv ]; then
  "$task_uv" venv --python 3.11 .venv
fi
"$task_uv" pip install --python .venv/bin/python \
  'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' --index-url https://download.pytorch.org/whl/cu128
"$task_uv" pip install --python .venv/bin/python \
  --constraint configs/remote_environment_constraints.txt -e '.[test]'
"$task_uv" pip check --python .venv/bin/python
if [ ! -f .runtime/nccl_2_31_2/nvidia/nccl/lib/libnccl.so.2 ]; then
  "$task_uv" pip install --python .venv/bin/python --target .runtime/nccl_2_31_2 \
    --no-deps 'nvidia-nccl-cu12==2.31.2'
fi
source scripts/training_runtime.sh
.venv/bin/python - <<'PY'
import sys, torch, mujoco, mujoco_warp, warp, mjlab
from importlib.metadata import version
print('Python', sys.version)
for package in ('torch', 'mjlab', 'mujoco', 'mujoco-warp', 'warp-lang', 'numpy', 'scipy'):
    print(package, version(package))
assert torch.cuda.device_count() == 2, 'Expected two visible training GPUs'
for i in range(2):
    with torch.cuda.device(i):
        x = torch.randn(256, 256, device=f'cuda:{i}')
        assert torch.isfinite(x @ x).all()
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory)
PY
"$task_uv" pip freeze --python .venv/bin/python > reports/remote_environment_freeze.txt
