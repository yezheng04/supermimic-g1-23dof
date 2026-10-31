#!/usr/bin/env python3
"""Compare resident GPU batches against completed serial motions, including exports."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from bones_batch_worker import finalize
from prepare_bones_seed import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/retargeted/batch_validation/full64")
    args = parser.parse_args()
    from supermimic.retarget.soma_batch import ResidentSoma
    old = ROOT / "data/retargeted/bones_seed_native23_v3/clips"
    available = []
    for directory in old.iterdir():
        if (directory / "raw.npz").exists() and (directory / "screening.json").exists():
            report = json.loads((directory / "reference_50hz.json").read_text())
            available.append((int(report["input_frames"]), directory, report))
    available.sort(key=lambda x: (x[0], x[1].name))
    selected = [available[i] for i in np.linspace(0, len(available)-1, args.count, dtype=int)]
    if len({p.name for n, p, r in selected}) != args.count:
        raise ValueError("Not enough distinct serial clips")
    if args.work_dir.exists():
        raise ValueError("Use an empty benchmark work directory")
    args.work_dir.mkdir(parents=True)
    config = json.loads((ROOT / "configs/bones_seed_preparation.json").read_text())
    sources = [Path(r["source"]) for n, p, r in selected]
    engine = ResidentSoma(args.work_dir / "calibration")
    started = time.monotonic()
    raws, metas, stats = engine.retarget(sources)
    raw_errors = []
    for raw, (n, directory, report) in zip(raws, selected):
        with np.load(directory / "raw.npz", allow_pickle=False) as z:
            expected = np.c_[z["root_pos"], z["root_quat"][:, [1, 2, 3, 0]], z["joint_pos"]]
            if raw.shape != expected.shape:
                raise ValueError("Changed source frame count")
            raw_errors.append(float(np.max(np.abs(raw - expected))))
    if max(raw_errors) > 1e-6:
        raise AssertionError(f"Raw IK parity failed: {max(raw_errors)}")
    print("RAW_PARITY", max(raw_errors), "STATS", json.dumps(stats), flush=True)
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn")) as pool:
        futures = []
        for raw, meta, source, (n, directory, report) in zip(raws, metas, sources, selected):
            row = dict(name=directory.name, source=str(source.relative_to(ROOT / "data/raw/bones_seed")),
                       package="benchmark", source_frames=n)
            futures.append(pool.submit(finalize, row, str(ROOT / "data/raw/bones_seed"),
                str(args.work_dir), config, raw, meta))
        results = [f.result() for f in futures]
    elapsed = time.monotonic() - started
    fields = ("timestamps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    errors = {key: 0. for key in fields}
    decisions_equal = True
    for n, directory, report in selected:
        with np.load(directory / "reference_50hz.npz", allow_pickle=False) as old_ref, \
                np.load(args.work_dir / "clips" / directory.name / "reference_50hz.npz", allow_pickle=False) as new_ref:
            for key in fields:
                if old_ref[key].shape != new_ref[key].shape:
                    raise ValueError(f"Shape mismatch: {directory.name}/{key}")
                errors[key] = max(errors[key], float(np.max(np.abs(old_ref[key] - new_ref[key]))))
        old_screen = json.loads((directory / "screening.json").read_text())
        new_screen = json.loads((args.work_dir / "clips" / directory.name / "screening.json").read_text())
        decisions_equal &= all(old_screen[key] == new_screen[key] for key in ("keep", "appendix_e_reasons", "diagnostics"))
    if max(errors.values()) > 1e-5 or not decisions_equal:
        raise AssertionError(f"Export/screening parity failed: {errors}, decisions {decisions_equal}")
    result = dict(count=args.count, source_frames=sum(n for n,p,r in selected),
        raw_max_abs_difference=max(raw_errors), reference_max_abs_differences=errors,
        screening_decisions_equal=decisions_equal, timing=stats,
        end_to_end_s=elapsed, end_to_end_clips_per_s=args.count / elapsed,
        end_to_end_source_frames_per_s=sum(n for n,p,r in selected) / elapsed,
        cpu_workers=8, independent_clip_names=[p.name for n,p,r in selected])
    write_json(ROOT / "reports/soma_batch_benchmark.json", result)
    print("BENCHMARK", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
