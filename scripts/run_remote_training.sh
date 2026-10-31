#!/usr/bin/env bash
# Run inside tmux on the authorized two-GPU training host after installation/transfer.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN
mkdir -p reports logs
status_file=reports/remote_training_status.txt
trap 'printf "FAILED at %s (line %s)\n" "$(date -u +%FT%TZ)" "$LINENO" > "$status_file"' ERR
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
  --format=csv -l 10 > reports/remote_gpu_monitor.csv &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
printf 'VERIFYING_DATA\n' > "$status_file"
(cd data/training/bones_seed_v4_full_v1 && sha256sum -c SHA256SUMS) > reports/remote_data_checksums.log
printf 'HARDWARE_PREFLIGHT\n' > "$status_file"
bash scripts/train_bones_2gpu.sh --config configs/bones_2gpu_preflight.json \
  > logs/bones_2gpu_preflight_20261029.console.log 2>&1
.venv/bin/python - <<'PY'
import json, math
from pathlib import Path
from supermimic.checkpoints import load_checkpoint
p = Path('logs/bones_2gpu_preflight_20261029')
ck = load_checkpoint(p / 'ckpt_latest.pt')
assert ck['iteration'] == 110 and ck['num_updates'] >= 4
rows = [json.loads(x) for x in (p / 'updates.jsonl').read_text().splitlines()]
assert rows and all(math.isfinite(x) for row in rows for group in ('wm', 'policy') for x in row[group].values())
assert any(row['policy']['policy_grad_norm'] > 0 for row in rows)
c = ck['config']
assert c['env']['num_envs'] == 4096 and c['policy_trainer']['batch_size'] == 4096
assert math.isclose(c['wm_trainer']['lr'], 2e-4 * math.sqrt(2))
Path('reports/remote_hardware_preflight.json').write_text(json.dumps(dict(
    iteration=ck['iteration'], updates=ck['num_updates'], finite_metrics=True,
    actor_gradients_nonzero=True, world_size_lr_verified=2,
    config=c, note='Full corpus, paper per-GPU sizes; this checkpoint is diagnostic only.'), indent=2, default=str))
PY
printf 'TRAINING\n' > "$status_file"
# Start fresh; do not initialize from the diagnostic checkpoint.
bash scripts/train_bones_2gpu.sh > logs/bones_full_2gpu.console.log 2>&1
printf 'COMPLETED\n' > "$status_file"
