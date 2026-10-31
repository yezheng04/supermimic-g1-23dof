"""Index completed BONES references and make deterministic, family-disjoint splits.

Only the existing manifest's keep flag controls eligibility. Quality diagnostics
are retained, never used as another filter. This does not modify retarget outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def family_key(name: str) -> str:
    # Same semantic action/take family across recording dates, actors and mirrors.
    name = re.sub(r"^\d{6,8}__", "", name)
    name = re.sub(r"__A\d+(?:_M)?$", "", name, flags=re.I)
    name = re.sub(r"_M$", "", name)
    return re.sub(r"_\d{3}$", "", name).lower()


def split_for(family: str, seed: int) -> str:
    u = int(hashlib.sha256(f"{seed}:{family}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if u < .9 else "validation" if u < .95 else "test"


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("data/retargeted/bones_seed_native23_v4"))
    p.add_argument("--output", type=Path, default=Path("data/training/bones_seed_v4_split_v1"))
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"Use a new output directory: {a.output}")
    a.output.mkdir(parents=True)
    source = a.source.resolve()
    status = json.loads((source / "status.json").read_text())
    if status["state"] != "kinematic_screening_complete" or status["cpu_pending"]:
        raise ValueError("Retargeting must be complete")
    with (source / "manifest.csv").open() as f:
        all_rows = list(csv.DictReader(f))
    eligible = sorted((r for r in all_rows if r["keep"] == "1"), key=lambda r: r["name"])
    assert len(eligible) == status["passed"]
    assert len(all_rows) == status["completed"] == status["total"]
    assert len({r["name"] for r in all_rows}) == len(all_rows)
    model = Path(__file__).resolve().parents[1] / "src/supermimic/robots/g1_23dof.xml"
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()

    def inspect(row):
        name = row["name"]
        meta = json.loads((source / "clips" / name / "reference_50hz.json").read_text())
        path = source / "library_50hz" / (name + ".npz")
        st = path.stat()
        if st.st_size == 0 or meta["model_sha256"] != model_sha or meta["output_fps"] != 50:
            raise ValueError(f"Reference metadata mismatch: {name}")
        n = int(meta["output_frames"])
        if n < 2 or abs(meta["duration_s"] - (n - 1) / 50) > 1e-8:
            raise ValueError(f"Reference duration mismatch: {name}")
        if not -1e-8 <= meta["source_duration_s"] - meta["duration_s"] < .02000001:
            raise ValueError(f"Source duration changed: {name}")
        family = family_key(name)
        return dict(name=name, keep=1, family=family, split=split_for(family, a.seed),
                    frames=n, duration_s=(n - 1) / 50, bytes=st.st_size,
                    path=str(path), diagnostics=row["diagnostics"])

    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, row in enumerate(pool.map(inspect, eligible)):
            rows.append(row)
            if (i + 1) % 10000 == 0:
                print(f"indexed {i + 1}/{len(eligible)}", flush=True)
    actual_files = {f.stem for f in (source / "library_50hz").glob("*.npz")}
    assert actual_files == {r["name"] for r in rows}, "Library differs from keep manifest"
    write_csv(a.output / "all.csv", rows)
    subsets = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    families = {s: {r["family"] for r in rr} for s, rr in subsets.items()}
    assert all(not families[s] & families[t] for s in families for t in families if s != t)
    for s, rr in subsets.items():
        write_csv(a.output / f"{s}.csv", rr)

    # Pilot selection uses names/duration only, no diagnostic-quality thresholds.
    # Prefer simple, short descriptions and one clip per family, non-mirrored.
    pilot = {}
    for split, count in (("train", 4), ("validation", 2), ("test", 2)):
        chosen, used = [], set()
        for category in ("walk", "turn", "wave"):
            candidates = [r for r in subsets[split]
                          if re.search(rf"(?:^|_){category}(?:_|$)", r["family"])
                          and 2 <= r["duration_s"] <= 12 and not r["name"].endswith("_M")]
            candidates.sort(key=lambda r: (len(r["family"]), r["family"], r["name"]))
            selected = []
            for r in candidates:
                if r["family"] not in used:
                    selected.append(dict(r, category=category))
                    used.add(r["family"])
                if len(selected) == count:
                    break
            if len(selected) != count:
                raise ValueError(f"Not enough distinct {category} families in {split}")
            chosen.extend(selected)
        write_csv(a.output / f"pilot_{split}.csv", chosen)
        link_dir = a.output / f"pilot_{split}"
        link_dir.mkdir()
        for r in chosen:
            (link_dir / (r["name"] + ".npz")).symlink_to(r["path"])
        pilot[split] = [r["name"] for r in chosen]

    frames = sum(r["frames"] for r in rows)
    report = dict(
        source=str(source), seed=a.seed, ratios={"train": .9, "validation": .05, "test": .05},
        assignment="sha256(seed:family); fixed hash thresholds, approximate ratios",
        family_rule="strip recording date, __A<actor>[_M], and trailing three-digit take; lowercase",
        limitation="Filename-derived grouping; undocumented source aliases may require additional grouping.",
        checks="All kept files exist; sidecar model/FPS/duration checked. Array validation is performed for pilot clips and on training load, not a full corpus array scan.",
        manifest_sha256=hashlib.sha256((source / "manifest.csv").read_bytes()).hexdigest(),
        model_sha256=model_sha, processed=len(all_rows), kept=len(rows), rejected=len(all_rows)-len(rows),
        frames=frames, duration_hours=sum(r["duration_s"] for r in rows)/3600,
        compressed_reference_bytes=sum(r["bytes"] for r in rows),
        compact_fp16_state_gib=frames*59*2/2**30,
        compact_fp32_state_gib=frames*59*4/2**30,
        frame_index_gib=frames*8/2**30,
        memory_note="State + int64 frame index only; excludes bins, training networks, optimizer, replay, activations and simulator. Measure headroom before loading full corpus on 16GB GPU; use shards or CPU-backed reference storage if needed.",
        splits={s: dict(clips=len(rr), families=len(families[s]), frames=sum(r['frames'] for r in rr)) for s,rr in subsets.items()},
        pilot=pilot, diagnostic_policy="unchanged: diagnostics do not exclude motions",
    )
    (a.output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "pilot"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
