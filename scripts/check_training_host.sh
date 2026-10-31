#!/usr/bin/env bash
# Preliminary diagnostics while the full reference cache is still transferring.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/training_runtime.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN
.venv/bin/python -m pytest -q tests/test_g1_fk.py tests/test_time_alignment.py \
  tests/test_replay_labels.py tests/test_compact_reference.py tests/test_aligned_env.py \
  tests/test_training_loop.py tests/test_training_launch.py tests/test_compiled_parity.py \
  > reports/remote_runtime_tests.log 2>&1
.venv/bin/sm23 demo > reports/remote_demo.log
.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  -m supermimic.scripts.train --config reports/remote_kernelcheck_config.json \
  > logs/bones_2gpu_kernelcheck_20261029.console.log 2>&1
.venv/bin/python - <<'PY'
import json, math
from pathlib import Path
from supermimic.checkpoints import load_checkpoint
p = Path('logs/bones_2gpu_kernelcheck_20261029')
ck = load_checkpoint(p / 'ckpt_latest.pt')
assert ck['iteration'] == 110 and ck['num_updates'] >= 4
rows = [json.loads(x) for x in (p / 'updates.jsonl').read_text().splitlines()]
assert all(math.isfinite(v) for r in rows for g in ('wm', 'policy') for v in r[g].values())
Path('reports/remote_kernelcheck_passed.json').write_text(json.dumps(dict(
    iteration=ck['iteration'], updates=ck['num_updates'], finite_metrics=True,
    scope='Two ranks, paper batch sizes, synthetic standing only. Full corpus preflight still required.'), indent=2))
PY
