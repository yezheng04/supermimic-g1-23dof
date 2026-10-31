"""Bounded real-motion integration run, fixed validation dynamics probe, evaluation.

Run from project root. Test split remains untouched for later final evaluation.
The short pilot is a diagnostic, not a convergence or paper-performance claim.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from supermimic.cli import apply_config
from supermimic.config import TrainConfig
from supermimic.data.quality import validate_reference
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark, save_results, summarize
from supermimic.training.loop import Trainer


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


@torch.no_grad()
def make_probe(policy, files):
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=files, num_envs=len(files), replay_capacity=128,
        obs_noise=False, reset_noise=False, domain_randomization=False,
        action_noise_std=0., sampling_mode="start", compact_library=True,
        seed=20261029,
    ), device="cuda:0")
    try:
        cmd = env.command_manager.get_term("motion")
        cmd.forced_frame[:] = cmd.library.clip_start
        obs, _ = env.reset()
        for _ in range(64):
            obs, *_ = env.step(policy(obs["policy"], obs["goal"]))
        batch = env.recorder_manager.get_term("replay").buffer.sample_windows(64, 8)
        return batch
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("train", "resume", "evaluate"))
    ap.add_argument("--config", type=Path, default=Path("configs/bones_pilot_v1.json"))
    ap.add_argument("--split-root", type=Path, default=Path("data/training/bones_seed_v4_split_v1"))
    a = ap.parse_args()
    torch.set_num_threads(4)
    cfg = apply_config(TrainConfig(), json.loads(a.config.read_text()))
    log = Path(cfg.log_root) / cfg.run_name
    report = Path("reports") / cfg.run_name
    report.mkdir(parents=True, exist_ok=True)
    files = {s: sorted(map(str, (a.split_root / f"pilot_{s}").glob("*.npz"))) for s in ("train", "validation")}
    if not all(files.values()):
        raise ValueError("Missing pilot dataset")
    if a.mode == "train":
        if log.exists():
            raise FileExistsError(f"Refusing to overwrite {log}")
        validation = []
        for split, paths in files.items():
            for path in paths:
                with np.load(path, allow_pickle=False) as d:
                    validate_reference(d)
                    validation.append(dict(split=split, name=Path(path).stem, frames=len(d["joint_pos"])))
        write(report / "array_validation.json", validation)
        tr = Trainer(cfg)
        try:
            tr.save("ckpt_initial.pt")
            # Held-out dynamics probe is collected in another environment and is
            # never inserted into the training replay or used for an update.
            batch = make_probe(tr.policy, files["validation"])
            torch.save({f.name: getattr(batch, f.name).cpu() for f in dataclasses.fields(batch)}, report / "validation_dynamics_probe.pt")
            before = tr.wm.rollout_error(batch.states, batch.a_exec, batch.xi).tolist()
            initial = {k: v.detach().clone() for k, v in tr.policy.state_dict().items()}
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            tr.run()
            elapsed = time.monotonic() - start
            after = tr.wm.rollout_error(batch.states, batch.a_exec, batch.xi).tolist()
            updates = [json.loads(s) for s in (log / "updates.jsonl").read_text().splitlines()]
            finite = all(math.isfinite(v) for u in updates for g in ("wm", "policy") for v in u[g].values())
            changed = any(not torch.equal(v, tr.policy.state_dict()[k]) for k,v in initial.items())
            assert finite and changed and tr.num_updates > 0
            write(report / "training.json", dict(
                iteration=tr.iteration, updates=tr.num_updates, elapsed_s=elapsed,
                peak_torch_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_torch_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                memory_note="PyTorch counters exclude MuJoCo/Warp allocations; these are not whole-device memory peaks.",
                all_update_metrics_finite=finite, policy_parameters_changed=changed,
                fixed_validation_probe_nse_before=before, fixed_validation_probe_nse_after=after,
                probe_note="64 fixed 8-step windows from 6 held-out motions under initial policy; diagnostic distribution, never trained on.",
                limitations="Bounded integration pilot, not a converged controller; small environment and batch counts. See saved config for optimizer schedule and model-only initialization.",
            ))
        finally:
            tr.env.close()
    elif a.mode == "resume":
        cfg.resume = log / "ckpt_latest.pt"
        cfg.initialize_from = None
        cfg.run_name += "_resume_check"
        cfg.schedule.log_interval = 8
        if (Path(cfg.log_root) / cfg.run_name).exists():
            raise FileExistsError("Resume-check output already exists")
        tr = Trainer(cfg)
        try:
            start_it, start_updates = tr.iteration, tr.num_updates
            import math
            refill = max(cfg.schedule.warmup_iterations, cfg.wm_trainer.window + 1, cfg.policy.history + 1)
            fresh_steps = refill + math.ceil(1 / cfg.schedule.updates_per_env_step) + 1
            decay_steps = cfg.schedule.total_iterations * cfg.schedule.lr_decay_fraction
            cfg.schedule.total_iterations = start_it + fresh_steps
            cfg.schedule.lr_decay_fraction = decay_steps / cfg.schedule.total_iterations
            assert tr.buffer.size == 0
            tr.run()
            assert tr.iteration == start_it + fresh_steps and tr.num_updates > start_updates
            write(report / "resume.json", dict(start_iteration=start_it, end_iteration=tr.iteration,
                  start_updates=start_updates, end_updates=tr.num_updates,
                  fresh_replay_refilled=True, checkpoint=str(tr.log_dir / "ckpt_latest.pt"),
                  fresh_steps=fresh_steps,
                  note="Separate restart diagnostic with full warmup. Evaluation uses original pilot checkpoint; not bitwise continuation."))
        finally:
            tr.env.close()
    else:
        output = {}
        for label, filename in (("initial", "ckpt_initial.pt"), ("trained", "ckpt_latest.pt")):
            policy, saved_cfg = load_policy_from_checkpoint(log / filename, "cuda:0")
            output[label] = {}
            for split, paths in files.items():
                torch.manual_seed(20261029)
                result = run_benchmark(policy, paths, num_envs=6,
                    future_frames=saved_cfg["env"]["future_frames"],
                    anchor_obs=saved_cfg["env"]["anchor_obs"],
                    paper_protocol=saved_cfg["env"]["paper_protocol"])
                save_results(result, report / f"{label}_{split}.json")
                output[label][split] = summarize(result)
            del policy
            gc.collect()
            torch.cuda.empty_cache()
        write(report / "evaluation_comparison.json", output)
        print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
