#!/usr/bin/env python3
"""Resident CUDA SOMA retargeting with bounded parallel CPU exports and resume."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time

# Set before spawning CPU workers or importing numerical runtimes.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from supermimic.data.batch_schedule import batches
from prepare_bones_seed import digest, write_json, select, reuse_retargeted
from bones_batch_worker import finalize

FIELDS = ["name", "keep", "appendix_e_keep", "native23_quality_keep", "native23_quality_mode",
          "reasons", "diagnostics", "dynamic_feasibility"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/bones_seed")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/retargeted/bones_seed_native23_v4")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bones_seed_preparation.json")
    parser.add_argument("--reuse-retargeted-root", type=Path, default=ROOT / "data/retargeted/bones_seed_native23_v3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frame-budget", type=int, default=131072)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--max-new-clips", type=int, help="Optional bounded run; never marks the whole dataset complete")
    args = parser.parse_args()
    if min(args.batch_size, args.frame_budget, args.cpu_workers) < 1:
        parser.error("Batch size, frame budget and CPU worker count must be positive")
    if args.max_new_clips is not None and args.max_new_clips < 1:
        parser.error("max-new-clips must be positive")
    work, raw = args.work_dir.resolve(), args.raw.resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads(args.config.read_text())
    if config["output_fps"] != 50 or config["native23_quality_mode"] != "diagnostic":
        raise ValueError("Batch preparation currently requires the selected 50 Hz, diagnostic-only policy")
    dependencies = [Path(__file__), ROOT / "scripts/bones_batch_worker.py", ROOT / "scripts/prepare_bones_seed.py",
        ROOT / "src/supermimic/retarget/soma_batch.py", ROOT / "src/supermimic/data/batch_schedule.py",
        ROOT / "src/supermimic/data/motion_convert.py", ROOT / "src/supermimic/data/quality.py",
        ROOT / "src/supermimic/data/appendix_e.py", ROOT / "src/supermimic/robots/g1_23dof.xml",
        ROOT / "src/supermimic/retarget/bvh.py", ROOT / "reports/bones_seed_archive_identity.json", args.config]
    soma = Path(importlib.util.find_spec("soma_retargeter").origin).parent
    dependencies.extend(soma / name for name in (
        "pipelines/newton_pipeline.py", "robotics/human_to_robot_scaler.py",
        "assets/bvh.py", "utils/newton_utils.py",
        "configs/unitree_g1/soma_to_g1_23dof_retargeter_config.json",
        "configs/unitree_g1/soma_to_g1_scaler_config.json"))
    fingerprint = {str(p.resolve()): digest(p) for p in dependencies}
    fingerprint["runtime_versions"] = {name: importlib.metadata.version(name)
        for name in ("newton", "warp-lang", "numpy", "scipy", "mujoco")}
    fingerprint["metadata_sha256"] = digest(raw / "metadata/seed_metadata_v004.csv")
    provenance = work / "provenance.json"
    if provenance.exists() and json.loads(provenance.read_text()) != fingerprint:
        raise ValueError("Code or inputs changed; choose a new work directory")
    write_json(provenance, fingerprint)
    stop_requested = False
    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    done, pending = set(), {}
    counters = Counter()
    completed_source_frames = 0
    new_frames = 0
    compute_start = None
    total_frames = 0
    rows = []

    def update(state, **extra):
        value = dict(state=state, pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
            completed=len(done), total=len(rows), passed=counters["kept"],
            reused_completed=counters["reused"], new_completed=counters["new"],
            cpu_pending=len(pending), native23_quality_mode="diagnostic",
            collision_refinement_enabled=False, batch_size=args.batch_size, cpu_workers=args.cpu_workers,
            elapsed_s=round(time.monotonic()-started, 2), **extra)
        if compute_start is not None and new_frames:
            elapsed = time.monotonic() - compute_start
            value.update(source_frames_per_s=round(new_frames / elapsed, 2),
                remaining_hours_estimate=round(max(0, total_frames-completed_source_frames) * elapsed / new_frames / 3600, 2),
                estimate_basis="completed new source frames / end-to-end elapsed; provisional early in run")
        write_json(work / "status.json", value)
        print(json.dumps(value), flush=True)

    try:
        rows, summary = select(raw / "metadata/seed_metadata_v004.csv", config, work)
        by_name = {r["name"]: r for r in rows}
        total_frames = sum(int(r["source_frames"]) for r in rows)
        identity = json.loads((ROOT / "reports/bones_seed_archive_identity.json").read_text())
        source_marker = args.reuse_retargeted_root / "extraction.json"
        if not source_marker.exists() or json.loads(source_marker.read_text()).get("archive") != identity:
            raise ValueError("A successfully verified and extracted prior run is required")
        if (raw / "soma_uniform.tar.gz").stat().st_size != identity["bytes"]:
            raise ValueError("Archive size changed")
        if not all((raw / r["source"]).is_file() and (raw / r["source"]).stat().st_size > 0 for r in rows):
            raise ValueError("Extracted candidate sources are missing")
        write_json(work / "extraction.json", dict(extracted=len(rows), archive=identity,
            reused_existing_extraction=True, checksum_evidence=str(source_marker.resolve())))
        update("loading_existing_results")
        with (work / "manifest.csv").open("w", newline="") as manifest, \
                ProcessPoolExecutor(max_workers=args.cpu_workers, mp_context=mp.get_context("spawn")) as pool:
            writer = csv.DictWriter(manifest, fieldnames=FIELDS)
            writer.writeheader()

            def record(result, is_new):
                nonlocal completed_source_frames, new_frames
                name = result["name"]
                if name in done:
                    raise ValueError(f"Duplicate completion: {name}")
                done.add(name)
                frames = int(by_name[name]["source_frames"])
                completed_source_frames += frames
                if is_new:
                    new_frames += frames
                counters["new" if is_new else "reused"] += 1
                counters["kept"] += int(result["keep"])
                counters["diagnostic_findings"] += bool(result["diagnostics"])
                line = {k: result[k] for k in FIELDS}
                for key in ("keep", "appendix_e_keep", "native23_quality_keep"):
                    line[key] = int(line[key])
                for key in ("reasons", "diagnostics"):
                    line[key] = "|".join(line[key])
                writer.writerow(line)
                manifest.flush()

            def drain(block=False):
                if not pending:
                    return
                ready, _ = wait(pending, timeout=None if block else 0, return_when=FIRST_COMPLETED)
                for future in ready:
                    row, is_new = pending.pop(future)
                    try:
                        result, elapsed = future.result()
                    except Exception as exc:
                        raise RuntimeError(f"CPU export failed for {row['name']}: {exc}") from exc
                    record(result, is_new)

            todo = []
            for row in rows:
                directory = work / "clips" / row["name"]
                marker = directory / "screening.json"
                if marker.exists():
                    record(json.loads(marker.read_text()), False)
                    continue
                if directory.exists() and not (directory / "reference_50hz.json").exists():
                    directory.rename(directory.with_name(directory.name + f".incomplete-{time.time_ns()}"))
                reuse_retargeted(row, raw, work, args.reuse_retargeted_root)
                if (directory / "reference_50hz.json").exists():
                    pending[pool.submit(finalize, row, str(raw), str(work), config)] = (row, False)
                else:
                    todo.append(row)
                while len(pending) >= args.cpu_workers * 4:
                    drain(True)
            while pending:
                drain(True)
            update("existing_results_loaded", remaining=len(todo))
            if not stop_requested and todo:
                from supermimic.retarget.soma_batch import ResidentSoma
                engine = ResidentSoma(work / "calibration")
                compute_start = time.monotonic()
                scheduled = 0
                with (work / "batch_timings.jsonl").open("a") as timing_file:
                    for batch in batches(todo, args.batch_size, args.frame_budget):
                        if stop_requested:
                            break
                        if args.max_new_clips is not None:
                            batch = batch[:args.max_new_clips-scheduled]
                            if not batch:
                                break
                        while len(pending) >= args.batch_size * 2:
                            drain(True)
                        update("gpu_batch", current_batch_size=len(batch), scheduled_new=scheduled,
                               current_max_source_frames=max(int(r["source_frames"]) for r in batch))
                        raws, metadata, timing = engine.retarget([raw / r["source"] for r in batch])
                        timing.update(names=[r["name"] for r in batch])
                        timing_file.write(json.dumps(timing) + "\n")
                        timing_file.flush()
                        for row, values, meta in zip(batch, raws, metadata):
                            pending[pool.submit(finalize, row, str(raw), str(work), config, values, meta)] = (row, True)
                        scheduled += len(batch)
                        drain(False)
                        update("batch_submitted", scheduled_new=scheduled, last_batch=timing)
                while pending:
                    drain(True)
                    update("finishing_cpu_exports")
            state = "kinematic_screening_complete" if len(done) == len(rows) else (
                "stopped_between_batches" if stop_requested else "bounded_run_complete")
            update(state, diagnostic_clips=counters["diagnostic_findings"], dynamic_feasibility="not_evaluated")
    except Exception as exc:
        update("error", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
