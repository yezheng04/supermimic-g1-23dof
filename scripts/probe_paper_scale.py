"""One isolated, real update at paper scale; report CUDA OOM without shrinking batches.

Uses 12 existing motions to obtain simulator replay. Failure with this small library
already rules out the full library for this unmodified implementation on this GPU.
It does not start a long training job or change the formal data manifest.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from supermimic.cli import apply_config
from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("reports/paper_scale_probe_v1.json"))
    p.add_argument("--reserve-full-library", action="store_true")
    p.add_argument("--compile", action="store_true", help="Enable the current compiled path; default preserves the historical uncompiled probe")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(4)
    cfg = apply_config(TrainConfig(), json.loads(Path("configs/bones_full_v1.json").read_text()))
    cfg.run_name = a.output.stem
    cfg.data.npz_dir = Path("data/training/bones_seed_v4_split_v1/pilot_train")
    cfg.data.manifest = None
    cfg.data.library = None
    cfg.wm_trainer.compile = cfg.policy_trainer.compile = "default" if a.compile else None
    assert (cfg.env.num_envs, cfg.env.replay_capacity, cfg.wm_trainer.batch_size,
            cfg.wm_trainer.batches_per_update, cfg.policy_trainer.batch_size) == (4096, 1000, 2048, 10, 4096)
    result = dict(config=dataclasses.asdict(cfg), scope=f"12 real reference motions; paper environment, replay, WM and Actor batch sizes; compile={a.compile}; no checkpointing/microbatching", stages=[])
    stage = "initialization"
    tr = None
    reservation = None
    started = time.monotonic()

    def snapshot(name):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        result['stages'].append(dict(stage=name, elapsed_s=time.monotonic()-started,
            allocated_gib=torch.cuda.memory_allocated()/2**30,
            reserved_gib=torch.cuda.memory_reserved()/2**30,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            device_free_gib=free/2**30, device_total_gib=total/2**30))
        print(json.dumps(result['stages'][-1]), flush=True)

    try:
        result['device'] = torch.cuda.get_device_name()
        torch.cuda.reset_peak_memory_stats()
        tr = Trainer(cfg)
        snapshot(stage)
        if a.reserve_full_library:
            stage = "reserve_full_reference_state_and_frame_index"
            meta = json.loads(Path("data/training/bones_seed_v4_split_v1/summary.json").read_text())
            extra = (meta['frames'] - tr.lib.num_frames) * (59 * 2 + 8)
            reservation = torch.empty(extra, dtype=torch.uint8, device='cuda:0')
            result['reservation_bytes'] = extra
            result['reservation_note'] = "Memory-only lower bound: fp16 state + int64 frame index. Does not load full dataset or include extra bins."
            snapshot(stage)
        stage = "collect_100_vector_steps"
        obs, _ = tr.env.reset()
        for _ in range(100):
            with torch.no_grad():
                obs, *_ = tr.env.step(tr.act(obs))
        snapshot(stage)
        stage = "world_model_10_batches_of_2048_windows"
        result['wm_metrics'] = tr.wm_trainer.update(tr.buffer)
        snapshot(stage)
        stage = "actor_4096_rollouts_x_32_steps_and_critic_update"
        result['policy_metrics'] = tr.policy_trainer.update(tr.buffer)
        snapshot(stage)
        result['status'] = "one_update_passed"
    except torch.OutOfMemoryError as exc:
        result.update(status="cuda_out_of_memory", failed_stage=stage, error=str(exc))
        result['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
        result['allocated_at_failure_gib'] = torch.cuda.memory_allocated()/2**30
        result['reserved_at_failure_gib'] = torch.cuda.memory_reserved()/2**30
        print(f"CUDA OOM at {stage}: {exc}", flush=True)
    except Exception as exc:
        result.update(status="error", failed_stage=stage, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result['elapsed_s'] = time.monotonic()-started
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2, default=str, allow_nan=False)+'\n')
        if tr is not None:
            tr.env.close()
            if tr.writer is not None:
                tr.writer.close()
        print(f"Report: {a.output}", flush=True)


if __name__ == '__main__':
    main()
