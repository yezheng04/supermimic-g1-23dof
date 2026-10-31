"""Fixed training-coverage subsets and isolated checkpoint evaluation.

Sampling is independent of checkpoint performance. Diagnostic groups are reported
separately from the uniform sample and are not population-weighted benchmark claims.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import torch

from supermimic.data.packed_library import load_pack, sha256, validate_pack

ITERATIONS = (20000, 40000, 60000, 80000, 100000)
PATTERNS = {
    "push_up": r"push[_ -]*up",
    "inchworm": r"inch[_ -]*worm",
    "elbow_plank": r"elbow[_ -]*plank",
    "crawl": r"crawl",
    "flip": r"flip",
    "side_roll": r"side.*roll|roll.*side",
    "broad_jump": r"broad.*jump",
    "lying_recovery": r"lying|lie_down|get_up",
}


def sample_names(names, count, seed, label):
    """Stable pseudo-random order; independent of file order and Python hash seed."""
    return sorted(sorted(names, key=lambda n: hashlib.sha256(
        f"{seed}:{label}:{n}".encode()).digest())[:count])


def select_groups(names, lengths, seed=20261030, count=2000):
    if len(names) != len(set(names)) or len(names) != len(lengths):
        raise ValueError("Expected unique names and one length per name")
    if len(names) < count:
        raise ValueError("Population smaller than requested uniform sample")
    groups = {"uniform": sample_names(names, count, seed, "uniform")}
    for group, pattern in PATTERNS.items():
        candidates = [n for n in names if re.search(pattern, n, re.I)]
        groups[group] = (sample_names(candidates, 128, seed, group)
                         if group == "crawl" else sorted(candidates))
    # A duration-tail diagnostic, not a second unbiased population sample.
    cutoff = sorted(lengths)[int(0.9 * (len(lengths) - 1))]
    groups["long_duration"] = sample_names(
        [n for n, length in zip(names, lengths) if length >= cutoff], 64, seed, "long_duration")
    groups["challenge_union"] = sorted(set().union(*(set(v) for k, v in groups.items() if k != "uniform")))
    return groups, cutoff


def write_json(path, obj):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    pending.replace(path)


def prepare(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selection.json").exists() or (out / "library.pt").exists():
        raise FileExistsError("Use the existing frozen selection or a new output directory")
    source = load_pack(args.library, args.manifest)
    groups, cutoff = select_groups(source["names"], source["lengths"], args.seed)
    names = sorted(set(groups["uniform"]) | set(groups["challenge_union"]))
    selected_manifest = out / "manifest.csv"
    with selected_manifest.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "keep", "groups"])
        writer.writerows((n, 1, "|".join(g for g, members in groups.items() if n in members)) for n in names)
    index = {n: i for i, n in enumerate(source["names"])}
    chunks, starts, lengths = [], [], []
    for name in names:
        i = index[name]
        start, length = int(source["starts"][i]), int(source["lengths"][i])
        starts.append(sum(lengths))
        lengths.append(length)
        chunks.append(source["ref"][start:start + length])
    pack = {**source, "names": names, "starts": starts, "lengths": lengths,
            "ref": torch.cat(chunks), "manifest_sha256": sha256(selected_manifest)}
    validate_pack(pack, selected_manifest)
    torch.save(pack, out / "library.pt")
    selection = dict(seed=args.seed, population=len(source["names"]), unique_clips=len(names),
        groups=groups, group_counts={k: len(v) for k, v in groups.items()}, patterns=PATTERNS,
        long_duration_min_frames=cutoff, source_library=str(args.library.resolve()),
        source_manifest_sha256=sha256(args.manifest), model_sha256=pack["model_sha256"],
        manifest_sha256=sha256(selected_manifest), library_sha256=sha256(out / "library.pt"),
        checkpoints=list(ITERATIONS), selection_method="SHA256 seeded ordering, uniform over clip IDs; mirrors retained",
        interpretation="Training-set coverage, not held-out generalization. Keyword groups are local diagnostics, not official labels.",
        protocol=dict(reference="https://arxiv.org/pdf/2610.28480 Appendix A/F and Section 4",
            fps=50, start="first frame", horizon="full clip or failure", foot_friction=1.0,
            obs_noise=False, action_noise=False, reset_noise=False, domain_randomization=False,
            seed=0, height_error_threshold_m=0.25, gravity_z_difference_threshold=0.8,
            normalization="sum rewards / (0.1 * (reference frames - 1)); zero remaining reward after failure"))
    write_json(out / "selection.json", selection)
    print(json.dumps({k: selection[k] for k in ("population", "unique_clips", "group_counts")}, indent=2))


def group_summary(rows):
    from supermimic.eval.benchmark import summarize
    out = summarize(rows)
    n = len(rows)
    out["failure_rate"] = sum(r.failed for r in rows) / max(n, 1)
    out["mean_completion_fraction"] = sum(min(r.steps / max(r.frames - 1, 1), 1.) for r in rows) / max(n, 1)
    out["failure_reasons"] = {reason: sum(r.failed and r.failure_reason == reason for r in rows)
                              for reason in sorted({r.failure_reason for r in rows if r.failed}, key=str)}
    return out


def one(args):
    from supermimic.checkpoints import load_checkpoint
    from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark
    out = args.output
    selection = json.loads((out / "selection.json").read_text())
    if sha256(out / "library.pt") != selection["library_sha256"]:
        raise ValueError("Evaluation subset changed")
    policy_file = out / "policies" / f"ckpt_{args.iteration:07d}.pt"
    manifest = json.loads((out / "policies" / "manifest.json").read_text())
    expected = next(r for r in manifest if r["iteration"] == args.iteration)
    if sha256(policy_file) != expected["sha256"]:
        raise ValueError("Checkpoint transfer checksum mismatch")
    metadata = load_checkpoint(policy_file)
    if metadata["iteration"] != args.iteration:
        raise ValueError("Checkpoint iteration mismatch")
    source_hash = metadata["source_checkpoint_sha256"]
    del metadata
    policy, cfg = load_policy_from_checkpoint(policy_file, args.device)
    if not cfg["env"].get("paper_protocol") or cfg["env"].get("anchor_obs"):
        raise ValueError("Expected the current paper-observation native23 run")
    started = time.monotonic()
    rows = run_benchmark(policy, [str(out / "library.pt")], device=args.device,
        num_envs=args.num_envs, future_frames=cfg["env"]["future_frames"],
        history_length=cfg["policy"]["history"], anchor_obs=False, paper_protocol=True)
    if {r.name for r in rows} != set(load_pack(out / "library.pt")["names"]):
        raise ValueError("Missing or extra evaluated motions")
    by_name = {r.name: r for r in rows}
    report = dict(iteration=args.iteration, elapsed_s=time.monotonic()-started,
        source_checkpoint_sha256=source_hash, policy_sha256=expected["sha256"],
        selection_sha256=sha256(out / "selection.json"), num_envs=args.num_envs,
        device=torch.cuda.get_device_name(),
        versions={k: importlib.metadata.version(k) for k in ("torch", "mujoco", "mujoco-warp", "warp-lang", "mjlab")},
        groups={g: group_summary([by_name[n] for n in names]) for g, names in selection["groups"].items()},
        clips=[asdict(r) for r in rows])
    write_json(out / f"eval_{args.iteration:07d}.json", report)
    print(json.dumps({"iteration": args.iteration, "uniform": report["groups"]["uniform"],
                      "challenge_union": report["groups"]["challenge_union"]}, indent=2), flush=True)


def run(args):
    out = args.output
    (out / "status.txt").write_text("RUNNING\n")
    reports = []
    try:
        for iteration in ITERATIONS:
            report_path = out / f"eval_{iteration:07d}.json"
            if not report_path.exists():
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "one", "--output", str(out),
                           "--iteration", str(iteration), "--num-envs", str(args.num_envs), "--device", args.device]
                with (out / f"eval_{iteration:07d}.log").open("a") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads(report_path.read_text())
            if report["selection_sha256"] != sha256(out / "selection.json") or report["num_envs"] != args.num_envs:
                raise ValueError("Existing report uses a different selection or environment count")
            reports.append({k: v for k, v in report.items() if k != "clips"})
            write_json(out / "comparison.json", reports)
            print(f"Finished {iteration}: {report['groups']['uniform']['mean_norm_return']:.4f}, "
                  f"{report['groups']['uniform']['failed']}/2000 failed", flush=True)
        (out / "status.txt").write_text("COMPLETED\n")
    except BaseException:
        (out / "status.txt").write_text("FAILED\n")
        raise
    if args.wandb:
        import wandb
        selection = json.loads((out / "selection.json").read_text())
        with wandb.init(project="supermimic-g1-23dof", entity="1406157281-", job_type="checkpoint_evaluation",
            name="bones_20k_to_100k_fixed_eval_20261030", dir=str(out), save_code=False,
            config={"source_training_run": "opv9t7zf", "selection_sha256": sha256(out / "selection.json"),
                    "group_counts": selection["group_counts"], "protocol": selection["protocol"]},
            settings=wandb.Settings(disable_git=True)) as run:
            run.define_metric("checkpoint_iteration")
            run.define_metric("eval/*", step_metric="checkpoint_iteration")
            for report in reports:
                metrics = {"checkpoint_iteration": report["iteration"]}
                for group, summary in report["groups"].items():
                    for key in ("mean_norm_return", "failure_rate", "failed", "mean_completion_fraction",
                                "frame_weighted_full_episode_return", "mean_executed_step_norm_return"):
                        metrics[f"eval/{group}/{key}"] = summary[key]
                run.log(metrics)
            write_json(out / "wandb.json", {"id": run.id, "url": run.url})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "one", "run"))
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation_100k_20261030"))
    parser.add_argument("--library", type=Path, default=Path("data/training/bones_seed_v4_full_v1/library_native23_fp16.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("data/training/bones_seed_v4_full_v1/train.csv"))
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--iteration", type=int, help="Checkpoint iteration, verified against the policy manifest")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    if args.command == "one" and args.iteration is None:
        parser.error("one requires --iteration")
    if args.num_envs < 1:
        parser.error("num-envs must be positive")
    {"prepare": prepare, "one": one, "run": run}[args.command](args)
