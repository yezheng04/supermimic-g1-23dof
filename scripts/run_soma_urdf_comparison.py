#!/usr/bin/env python3
"""Run a bounded, isolated 10-motion comparison; never writes production outputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "SOMA-Retargeter"
NEW = ROOT.parent / "SOMA-Retargeter-official"
OUT = ROOT / "data/retargeted/soma_urdf_comparison_v1"
TARGET = ROOT / "assets/robotics/unitree/g1_23dof_urdf_v1"
MOTIONS = [
    ("walk", "Neutral_walk_forward_002__A057.bvh", "calibration"),
    ("wave", "wave_R_001__A428.bvh", "calibration"),
    ("hiphop", "dance_hiphop_shuffle_square_R_fast_002__A318.bvh", "calibration"),
    ("turn", "dance_basic_chaines_180_R_001__A306.bvh", "calibration"),
    ("jump", "high_jump_R_001__A277.bvh", "holdout"),
    ("pickup", "item_pick_up_standing_R_001__A410.bvh", "holdout"),
    ("throw", "Neutral_throw_ball_001__A057.bvh", "holdout"),
    ("stretch", "body_stretch_1_004__A069.bvh", "holdout"),
    ("pickup_big", "big_light_one_hand_pick_up_front_low_R_005__A508.bvh", "holdout"),
    ("pickup_small", "small_light_one_hand_pick_up_front_low_002__A507.bvh", "holdout"),
]


def run(command, log):
    print("Running", log.stem, flush=True)
    with log.open("w") as f:
        subprocess.run(list(map(str, command)), stdout=f, stderr=subprocess.STDOUT,
                       env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"), check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    rows = []
    for slug, filename, split in MOTIONS:
        source = NEW / "assets/motions/bvh" / filename
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        old_result = ROOT / "data/retargeted/soma_examples_v1" / slug
        if not (old_result / "reference_50hz.npz").exists():
            old_result = OUT / "current" / slug
        for label, dest, python, script in (
            ("current", old_result, OLD / ".venv/bin/python", "retarget_soma.py"),
            ("official", OUT / "official" / slug, NEW / ".venv/bin/python", "retarget_soma_official.py")):
            if (dest / "reference_50hz.json").exists():
                meta = json.loads((dest / "reference_50hz.json").read_text())
                if meta["source_sha256"] != sha:
                    raise ValueError(f"Stale result: {dest}")
                # These outputs belong to this immutable configuration experiment.
            else:
                run([python, ROOT / "scripts" / script, source, dest], logs / f"{label}_{slug}.log")
        old_overlay = ROOT / "data/retargeted/soma_examples_overlay_v1" / slug
        if not (old_overlay / "overlay.npz").exists():
            old_overlay = OUT / "current_overlays" / slug / old_result.name
            if not (old_overlay / "overlay.npz").exists():
                run([OLD / ".venv/bin/python", ROOT / "scripts/render_soma_overlay.py",
                     "--reference", old_result / "reference_50hz.npz", "--output", old_overlay.parent, "--prepare-only"],
                    logs / f"current_overlay_{slug}.log")
        row = dict(motion=slug, source=str(source), source_sha256=sha, split=split,
                   current=str(old_result), current_overlay=str(old_overlay), official=str(OUT / "official" / slug))
        rows.append(row)
        (OUT / "manifest.json").write_text(json.dumps(rows, indent=2))
        print("Completed pair", slug, flush=True)
    # Hold calibration fixed: diagnose the effect of the official experimental foot-contact pass.
    for slug in ("walk", "jump", "turn"):
        row = next(x for x in rows if x["motion"] == slug)
        dest = OUT / "no_contact" / slug
        if not (dest / "reference_50hz.npz").exists():
            run([NEW / ".venv/bin/python", ROOT / "scripts/retarget_soma_official.py",
                 row["source"], dest, "--no-contact"], logs / f"no_contact_{slug}.log")
    # Separate optimization corpus; six held-out clips are never optimizer inputs.
    train = OUT / "optimizer_corpus"
    train.mkdir(exist_ok=True)
    for row in rows:
        if row["split"] == "calibration":
            link = train / Path(row["source"]).name
            if not link.exists():
                link.symlink_to(row["source"])
    config = json.loads((NEW / "app/tools/assets/ik_optimizer/ik_weight_optimizer_config.json").read_text())
    config.update(target_type=TARGET.name, input_folder=str(train), output_folder=str(OUT / "optimizer"),
                  extra_robot_paths=[str(TARGET)], outer_iterations=2, motion_batch_size=2, fk_frame_batch_size=25,
                  plot=dict(enabled=False, live=False), save_probe_diagnostics=True)
    config_path = OUT / "optimizer_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    optimized = OUT / "optimizer" / ("optimized_" + TARGET.name) / "optimized_retargeter.json"
    if not optimized.exists():
        run([NEW / ".venv/bin/python", ROOT / "scripts/optimize_soma_urdf.py", config_path], logs / "optimizer.log")
    for row in rows:
        dest = OUT / "optimized" / row["motion"]
        if not (dest / "reference_50hz.npz").exists():
            run([NEW / ".venv/bin/python", ROOT / "scripts/retarget_soma_official.py",
                 row["source"], dest, "--config", optimized], logs / f"optimized_{row['motion']}.log")
    print("All comparison and optimizer outputs complete", flush=True)


if __name__ == "__main__":
    main()
