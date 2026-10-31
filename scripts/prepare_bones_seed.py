#!/usr/bin/env python3
"""SONIC + SuperMimic Appendix E selection and native23 preparation.

Selection uses only the standard library. --run needs SOMA's Python environment.
Appendix E and additional native23 quality decisions are reported separately.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return path.as_posix()


def name_reasons(checked, config):
    sonic = [k for k in config["exclude_keywords"] if k.lower() in checked.lower()]
    normalized = re.sub(r"[^a-z0-9]+", "_", PurePosixPath(checked).stem.lower()).strip("_")
    added = [rule["term"] for rule in config.get("appendix_e_name_rules", [])
             if re.search(rule["pattern"], normalized)]
    return sonic, added


def select(metadata, config, output):
    rows, ids, sources = [], set(), set()
    counts, hits = Counter(), Counter()
    with metadata.open(newline="") as stream:
        for row in csv.DictReader(stream):
            source = safe_member(row["move_soma_uniform_path"])
            path = PurePosixPath(source)
            if not source.startswith("soma_uniform/bvh/") or path.suffix != ".bvh":
                raise ValueError(f"Unexpected uniform source: {source}")
            name = path.parent.name + "__" + path.stem
            if name in ids or source in sources:
                raise ValueError(f"Duplicate motion identity: {source}")
            ids.add(name)
            sources.add(source)
            # SONIC checks the immediate parent plus individual PKL filename.
            g1 = PurePosixPath(row["move_g1_path"])
            checked = f"{g1.parent.name}/{g1.stem}.pkl"
            sonic, extra = name_reasons(checked, config)
            reasons = ["sonic:" + k for k in sonic] + ["appendix_e_example:" + k for k in extra]
            hits.update(reasons)
            keep = not reasons
            counts[(row["package"], keep)] += 1
            rows.append(dict(name=name, source=source, package=row["package"],
                take_group=path.parent.name + "/" + row["take_name"],
                is_mirror=row["is_mirror"], source_frames=row["move_duration_frames"],
                checked_name=checked, sonic_keep=int(not sonic), keep=int(keep),
                sonic_reasons="|".join(sonic), appendix_e_name_reasons="|".join(extra),
                reasons="|".join(reasons)))
    output.mkdir(parents=True, exist_ok=True)
    for filename, selected in (("selection.csv", rows),
                                ("candidates.csv", [r for r in rows if r["keep"]])):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)
    summary = dict(total=len(rows), candidates=sum(r["keep"] for r in rows),
        excluded=sum(not r["keep"] for r in rows), metadata_sha256=digest(metadata),
        reference=config["reference"], keyword_hits_overlapping=dict(hits),
        after_sonic_name_filter=sum(r["sonic_keep"] for r in rows),
        appendix_e_additional_name_excluded=sum(r["sonic_keep"] and not r["keep"] for r in rows),
        paper_reference=config.get("paper_reference"),
        exact_author_name_list_available=False,
        by_package={p: {"candidates": counts[p, True], "excluded": counts[p, False]}
                    for p in sorted({r["package"] for r in rows})},
        dynamic_feasibility="not_evaluated")
    write_json(output / "selection_summary.json", summary)
    return [r for r in rows if r["keep"]], summary


def pilot_rows(rows, per_package):
    groups = defaultdict(list)
    for row in rows:
        groups[row["package"]].append(row)
    selected = []
    for package in sorted(groups):
        # Fixed hash ordering is independent of metadata row order.
        ordered = sorted(groups[package], key=lambda r: hashlib.sha256(r["name"].encode()).digest())
        selected.extend(ordered[:per_package])
    return selected


def quality_reasons(report, config):
    reasons = []
    q = report.get("quality", {})
    if q.get("finite") is not True:
        reasons.append("nonfinite_or_missing_quality")
    if report.get("robot") != "g1_23dof" or report.get("output_fps") != config["output_fps"]:
        reasons.append("wrong_robot_or_timebase")
    for field, limit in config["quality_gate"].items():
        # 'max_joint_limit_excess_rad' is already the name of the statistic.
        key = field if field == "max_joint_limit_excess_rad" else field.removeprefix("max_")
        value = q.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value > limit:
            reasons.append(key)
    return reasons


def screening_decision(appendix_e, native_reasons, config):
    mode = config.get("native23_quality_mode", "strict")
    if mode not in ("strict", "diagnostic"):
        raise ValueError(f"Unknown native23 quality mode: {mode}")
    reasons = ["appendix_e:" + r for r in appendix_e["reasons"]]
    diagnostics = ["native23:" + r for r in native_reasons]
    if mode == "strict":
        reasons += diagnostics
    return dict(keep=not reasons, reasons=reasons, diagnostics=diagnostics,
                native23_quality_mode=mode)


def reuse_retargeted(row, raw, work, reuse_root):
    """Reuse verified original SOMA outputs; screening is always recomputed."""
    directory = work / "clips" / row["name"]
    source = reuse_root / "clips" / row["name"]
    if directory.exists() or not (source / "reference_50hz.json").is_file():
        return
    report = json.loads((source / "reference_50hz.json").read_text())
    if report.get("model_sha256") != digest(ROOT / "src/supermimic/robots/g1_23dof.xml"):
        raise ValueError(f"Cannot reuse changed robot model: {source}")
    if report.get("source_sha256") != digest(raw / row["source"]):
        raise ValueError(f"Cannot reuse changed BVH source: {source}")
    if report.get("hand_frame_shift_m", -.084) != -.084 or report.get("hand_rotation_weight_scale", 1.) != 1.:
        raise ValueError(f"Cannot reuse diagnostic calibration variant: {source}")
    if not report.get("feet_stabilizer_enabled", True) or "refinement" in report:
        raise ValueError(f"Cannot reuse modified reference variant: {source}")
    directory.mkdir(parents=True)
    for name in ("reference_50hz.npz", "reference_50hz.json", "raw.npz", "motion.csv",
                 "motion_soma.csv", "motion.json", "scaler_native23.json"):
        if (source / name).exists():
            os.link(source / name, directory / name)
    write_json(directory / "reuse.json", dict(source_directory=str(source),
        reference_sha256=digest(source / "reference_50hz.npz"), screening_recomputed=True))


def extract_selected(archive_path, destination, rows, update):
    wanted = {r["source"] for r in rows}
    found = set()
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            name = safe_member(member.name)
            if name not in wanted:
                continue
            if name in found or not member.isfile():
                raise ValueError(f"Duplicate or nonregular selected entry: {name}")
            target = destination / name
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Extraction escapes destination: {name}")
            if target.is_symlink():
                raise ValueError(f"Symlink target: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(".bvh.part")
            # Re-extract on restart: equal size alone does not verify contents.
            with archive.extractfile(member) as source, temp.open("wb") as out:
                shutil.copyfileobj(source, out, 1024 * 1024)
            if temp.stat().st_size != member.size:
                raise ValueError(f"Truncated archive member: {name}")
            temp.replace(target)
            found.add(name)
            if len(found) % 1000 == 0:
                update("extracting", extracted=len(found), expected=len(wanted))
    missing = wanted - found
    if missing:
        raise ValueError(f"Missing {len(missing)} candidates in archive; examples {sorted(missing)[:5]}")
    return len(found)


def process_clip(row, raw, work, soma_python, config):
    directory = work / "clips" / row["name"]
    result_path = directory / "screening.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    report_path = directory / "reference_50hz.json"
    if directory.exists() and not report_path.exists():
        # Preserve partial attempts for diagnosis instead of silently overwriting.
        directory.rename(directory.with_name(directory.name + f".incomplete-{time.time_ns()}"))
    if not report_path.exists():
        logs = work / "logs"
        logs.mkdir(exist_ok=True)
        with (logs / (row["name"] + ".log")).open("w") as log:
            completed = subprocess.run([str(soma_python), str(ROOT / "scripts/retarget_soma.py"),
                str(raw / row["source"]), str(directory)], cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, env={**os.environ, "MUJOCO_GL": "egl"})
        if completed.returncode:
            raise RuntimeError(f"Retarget failed for {row['name']}; inspect {logs}. Job stopped, not classified as impossible.")
    report = json.loads(report_path.read_text())
    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))
    from supermimic.data.quality import validate_reference
    from supermimic.data.appendix_e import screen_reference
    with np.load(directory / "reference_50hz.npz", allow_pickle=False) as reference:
        validate_reference(reference, expected_fps=config["output_fps"])
    expected_duration = (report["output_frames"] - 1) / config["output_fps"]
    tail = report["source_duration_s"] - expected_duration
    if not math.isclose(report["duration_s"], expected_duration, abs_tol=1e-8) or not (-1e-8 <= tail < .02 + 1e-8):
        raise ValueError(f"Physical duration mismatch for {row['name']}")
    appendix_e = screen_reference(directory / "reference_50hz.npz", config["appendix_e_geometry"])
    write_json(directory / "appendix_e.json", appendix_e)
    native_reasons = quality_reasons(report, config)
    decision = screening_decision(appendix_e, native_reasons, config)
    result = dict(name=row["name"], source=row["source"], package=row["package"],
        **decision, frames=report["output_frames"],
        duration_s=report["duration_s"], quality=report["quality"],
        appendix_e_keep=appendix_e["keep"], appendix_e_reasons=appendix_e["reasons"],
        appendix_e_metrics=appendix_e["metrics"], native23_quality_keep=not native_reasons,
        native23_quality_reasons=native_reasons,
        dynamic_feasibility="not_evaluated", training_approved=False)
    if result["keep"]:
        library = work / "library_50hz"
        library.mkdir(exist_ok=True)
        for suffix in (".npz", ".json"):
            dest = library / (row["name"] + suffix)
            if not dest.exists():
                os.link(directory / ("reference_50hz" + suffix), dest)
    write_json(result_path, result)
    return result


def run(args):
    config = json.loads(args.config.read_text())
    if config.get("native23_quality_mode", "strict") not in ("strict", "diagnostic"):
        raise ValueError("native23_quality_mode must be strict or diagnostic")
    if config["output_fps"] != 50:
        raise ValueError("Native23 converter currently targets 50 Hz")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = json.loads((ROOT / "reports/bones_seed_archive_identity.json").read_text())
    dependencies = [args.config, Path(__file__), ROOT / "scripts/retarget_soma.py",
        ROOT / "src/supermimic/data/quality.py", ROOT / "src/supermimic/data/motion_convert.py",
        ROOT / "src/supermimic/data/appendix_e.py",
        ROOT / "src/supermimic/robots/g1_23dof.xml"]
    fingerprint = {str(p.resolve()): digest(p) for p in dependencies}
    fingerprint["archive_sha256"] = identity["sha256"]
    fingerprint["metadata_sha256"] = digest(args.raw / "metadata/seed_metadata_v004.csv")
    provenance = work / "provenance.json"
    if provenance.exists() and json.loads(provenance.read_text()) != fingerprint:
        raise ValueError("Inputs/code changed: use a new --work-dir to avoid mixing results")
    write_json(provenance, fingerprint)

    def update(state, **extra):
        write_json(work / "status.json", dict(state=state, pid=os.getpid(),
            updated_at=datetime.now(timezone.utc).isoformat(), **extra))
        print(json.dumps(dict(state=state, **extra), ensure_ascii=False), flush=True)

    try:
        rows, summary = select(args.raw / "metadata/seed_metadata_v004.csv", config, work)
        update("selected", **summary)
        if not args.run:
            return 0
        started = time.monotonic()
        while True:
            try:
                download = json.loads((args.raw / "download_status.json").read_text())
            except json.JSONDecodeError:
                time.sleep(1)  # Downloader writes its status non-atomically.
                continue
            if download["state"] in ("downloaded", "extracting", "complete"):
                if download.get("revision") != identity["revision"]:
                    raise ValueError("Downloaded revision differs from pinned archive identity")
                break
            if download["state"] != "downloading" or not args.wait_download:
                raise RuntimeError(f"Download not ready: {download['state']}")
            if args.download_pid:
                os.kill(args.download_pid, 0)  # Exit if original download process has died.
            if time.monotonic() - started > 24 * 3600:
                raise TimeoutError("Download wait exceeded 24 hours; rerun to resume")
            update("waiting_for_download", candidates=len(rows))
            time.sleep(30)
        if any(digest(p) != fingerprint[str(p.resolve())] for p in dependencies):
            raise ValueError("Preparation code changed while waiting; restart with new work directory")
        archive = args.raw / "soma_uniform.tar.gz"
        update("verifying_archive")
        if archive.stat().st_size != identity["bytes"] or digest(archive) != identity["sha256"]:
            raise ValueError("Archive size/SHA256 mismatch")
        reuse_extraction = False
        if args.reuse_retargeted_root:
            marker = args.reuse_retargeted_root / "extraction.json"
            if marker.is_file() and json.loads(marker.read_text()).get("archive") == identity:
                reuse_extraction = all((args.raw / row["source"]).is_file()
                    and (args.raw / row["source"]).stat().st_size > 0 for row in rows)
        if reuse_extraction:
            count = len(rows)
            update("reused_verified_archive_extraction", extracted=count)
        else:
            count = extract_selected(archive, args.raw, rows, update)
        write_json(work / "extraction.json", dict(extracted=count, archive=identity,
            reused_existing_extraction=reuse_extraction))
        pilot = pilot_rows(rows, config["pilot_per_package"])
        write_json(work / "pilot_selection.json", pilot)
        results = []
        for i, row in enumerate(pilot):
            update("retargeting_pilot", completed=i, total=len(pilot), current=row["name"])
            if args.reuse_retargeted_root:
                reuse_retargeted(row, args.raw, work, args.reuse_retargeted_root)
            results.append(process_clip(row, args.raw, work, args.soma_python, config))
        passed = sum(r["keep"] for r in results)
        pilot_report = dict(total=len(results), passed=passed,
            appendix_e_passed=sum(r["appendix_e_keep"] for r in results),
            native23_quality_passed=sum(r["native23_quality_keep"] for r in results),
            native23_quality_mode=config.get("native23_quality_mode", "strict"),
            diagnostic_counts=dict(Counter(reason for r in results for reason in r["diagnostics"])),
            pass_fraction=passed / len(results),
            reason_counts=dict(Counter(reason for r in results for reason in r["reasons"])),
            clips=results, native23_criteria=config["quality_gate"],
            appendix_e_criteria=config["appendix_e_geometry"], dynamic_feasibility="not_evaluated")
        write_json(work / "pilot_report.json", pilot_report)
        if config.get("native23_quality_mode", "strict") == "strict" and pilot_report["pass_fraction"] < config["minimum_pilot_pass_fraction"]:
            update("needs_retarget_calibration", pilot_total=len(results), pilot_passed=passed,
                   unprocessed_candidates=len(rows)-len(pilot),
                   note="High geometric rejection; investigate retargeter/model before full conversion. Remaining motions are not classified as impossible.")
            return 3
        # Only run the full library after the pilot clears the local gate.
        passed = appendix_e_passed = native23_passed = 0
        reasons = Counter()
        manifest = work / "manifest.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["name", "keep", "appendix_e_keep",
                "native23_quality_keep", "native23_quality_mode", "reasons", "diagnostics", "dynamic_feasibility"])
            writer.writeheader()
            for i, row in enumerate(rows):
                update("retargeting_all", completed=i, total=len(rows), passed=passed, current=row["name"])
                if args.reuse_retargeted_root:
                    reuse_retargeted(row, args.raw, work, args.reuse_retargeted_root)
                result = process_clip(row, args.raw, work, args.soma_python, config)
                passed += result["keep"]
                appendix_e_passed += result["appendix_e_keep"]
                native23_passed += result["native23_quality_keep"]
                reasons.update(result["reasons"])
                writer.writerow(dict(name=row["name"], keep=int(result["keep"]),
                    appendix_e_keep=int(result["appendix_e_keep"]),
                    native23_quality_keep=int(result["native23_quality_keep"]),
                    native23_quality_mode=result["native23_quality_mode"],
                    diagnostics="|".join(result["diagnostics"]),
                    reasons="|".join(result["reasons"]), dynamic_feasibility="not_evaluated"))
                stream.flush()
        update("kinematic_screening_complete", candidates=len(rows), kept=passed,
               appendix_e_geometry_passed=appendix_e_passed, native23_quality_passed=native23_passed,
               excluded=len(rows)-passed, reason_counts=dict(reasons), dynamic_feasibility="not_evaluated")
        return 0
    except Exception as exc:
        update("error", error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/bones_seed")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bones_seed_preparation.json")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/retargeted/bones_seed_native23_v3")
    parser.add_argument("--reuse-retargeted-root", type=Path,
        help="Reuse compatible original SOMA outputs and verified extraction from an older preparation run")
    parser.add_argument("--soma-python", type=Path, default=ROOT.parent / "SOMA-Retargeter/.venv/bin/python")
    parser.add_argument("--run", action="store_true", help="Extract, retarget pilot, then all if pilot passes")
    parser.add_argument("--wait-download", action="store_true")
    parser.add_argument("--download-pid", type=int, help="Fail instead of waiting if this downloader exits early")
    args = parser.parse_args()
    if args.download_pid is not None and args.download_pid <= 0:
        parser.error("download-pid must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
