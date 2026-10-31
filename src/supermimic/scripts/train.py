"""Train SuperMimic.  Example:

  .venv/bin/python -m supermimic.scripts.train --run-name native23 --env.num-envs 256
"""

from __future__ import annotations

import os
import sys

# Under torchrun, pin each rank to its own GPU *before* CUDA/warp initialise: MuJoCo Warp and
# mjlab's simulation always run on the first visible device, so cfg.device alone is not enough.
# Skipped on a box with fewer GPUs than ranks (single-GPU smoke tests share cuda:0 via gloo).
def _n_gpus() -> int:
    try:
        return len(os.listdir("/proc/driver/nvidia/gpus"))
    except OSError:
        return 0


# A preset CUDA_VISIBLE_DEVICES list (e.g. "0,2,3" to skip a bad card) is indexed by LOCAL_RANK.
_preset = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
_avail = _preset if _preset else [str(i) for i in range(_n_gpus())]
if "LOCAL_RANK" in os.environ and int(os.environ.get("LOCAL_WORLD_SIZE", "1")) <= len(_avail) and (not _preset or len(_preset) > 1):
    os.environ["CUDA_VISIBLE_DEVICES"] = _avail[int(os.environ["LOCAL_RANK"])]
    os.environ["SM_GPU_PINNED"] = "1"
    # Give each rank its own slice of cores (the training loop is single-thread CPU bound;
    # sharing cores between ranks adds jitter that lockstep all-reduces turn into waiting).
    try:
        cores = sorted(os.sched_getaffinity(0))
        n_local = int(os.environ["LOCAL_WORLD_SIZE"])
        per = len(cores) // n_local
        if per >= 2:
            r = int(os.environ["LOCAL_RANK"])
            os.sched_setaffinity(0, set(cores[r * per : (r + 1) * per]))
    except (OSError, ValueError):
        pass

import tyro

from supermimic.config import TrainConfig


def load_train_config(argv: list[str] | None = None) -> tuple[TrainConfig, bool]:
    """Keep tyro flags and add the project's JSON configs to the GPU-pinned entrypoint."""
    argv = sys.argv[1:] if argv is None else argv
    if not any(x == "--config" or x.startswith("--config=") for x in argv):
        return tyro.cli(TrainConfig, args=argv), False
    import argparse
    import json
    from pathlib import Path
    from supermimic.cli import apply_config

    parser = argparse.ArgumentParser(description="Train from JSON, including under torchrun")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args(argv)
    cfg = apply_config(TrainConfig(), json.loads(args.config.read_text()))
    if args.resume is not None:
        cfg.resume = args.resume
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.max_hours is not None:
        if args.max_hours <= 0:
            parser.error("--max-hours must be positive")
        cfg.schedule.max_hours = args.max_hours
    if cfg.env.future_frames != cfg.policy_trainer.future_frames or cfg.policy.history != cfg.policy_trainer.history:
        parser.error("Environment and imagination history/goal settings must match")
    if cfg.resume is not None and cfg.initialize_from is not None:
        parser.error("Use resume or initialize_from, not both")
    return cfg, args.check_config


def main() -> None:
    cfg, check_only = load_train_config()
    if check_only:
        import json
        from dataclasses import asdict
        print(json.dumps(asdict(cfg), indent=2, default=str))
        return
    from supermimic.training.loop import Trainer

    trainer = Trainer(cfg)
    try:
        trainer.run()
    finally:
        trainer.env.close()


if __name__ == "__main__":
    main()
