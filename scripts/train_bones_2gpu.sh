#!/usr/bin/env bash
# One process/GPU; all configured sizes are per GPU. Run from a copied project root.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
source scripts/training_runtime.sh
task_python="${SM23_PYTHON:-$project_dir/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
"$task_python" - "$@" <<'PY'
import sys
from pathlib import Path
import torch
from supermimic.scripts.train import load_train_config
cfg, check_only = load_train_config(['--config', 'configs/bones_full_2gpu.json', *sys.argv[1:]])
if check_only:
    raise SystemExit('--check-config: use python -m supermimic.scripts.train --config configs/bones_full_2gpu.json --check-config instead')
if torch.cuda.device_count() != 2:
    raise SystemExit('Two visible CUDA GPUs are required; set CUDA_VISIBLE_DEVICES to the intended pair.')
from supermimic.data.packed_library import load_pack
if not Path(cfg.data.manifest).is_file() or cfg.data.library is None or not Path(cfg.data.library).is_file():
    raise SystemExit('Missing full-corpus manifest or packed .pt library; build/copy both before launching.')
pack = load_pack(cfg.data.library, cfg.data.manifest)
print(f'Full corpus: {len(pack["names"])} clips, {len(pack["ref"])} frames, {pack["ref"].dtype}')
del pack
if cfg.resume is not None and not Path(cfg.resume).is_file():
    raise SystemExit(f'Missing checkpoint: {cfg.resume}')
log = Path(cfg.log_root) / cfg.run_name
if cfg.resume is None and log.exists() and any(log.iterdir()):
    raise SystemExit(f'Refusing to overwrite {log}; use --resume or a new --run-name.')
for i in range(2):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name}, {p.total_memory/2**30:.1f} GiB')
print('Per GPU: 4096 envs, Replay 1000, WM 10 x 2048, Actor 4096 x 32.')
print('Base LR 2e-4; trainer applies sqrt(2) exactly once.')
print('PyTorch NCCL build version:', torch.cuda.nccl.version())
# torch.cuda.nccl.version() can report compile-time macros even with LD_PRELOAD.
import ctypes
try:
    nccl_version = ctypes.c_int()
    get_version = ctypes.CDLL(None).ncclGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    if get_version(ctypes.byref(nccl_version)) != 0:
        raise RuntimeError('ncclGetVersion failed')
    print('Preloaded NCCL runtime version:', nccl_version.value)
except AttributeError:
    print('Using bundled NCCL; no process-global override detected.')
PY
exec "$task_python" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  -m supermimic.scripts.train --config configs/bones_full_2gpu.json "$@"
