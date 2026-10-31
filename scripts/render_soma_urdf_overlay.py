#!/usr/bin/env python3
"""Render the six original examples with our URDF calibration and the same overlay layout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from render_soma_overlay import ROOT, render
import imageio.v2 as imageio
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["official", "optimized"], default="official")
    p.add_argument("--output", type=Path, default=ROOT / "data/retargeted/soma_examples_urdf_overlay_v1")
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error("Choose an empty output directory")
    a.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((ROOT / "data/retargeted/soma_examples_v1/summary.json").read_text())
    results = []
    for row in rows:
        source_dir = ROOT / "data/retargeted/soma_urdf_comparison_v1" / a.variant / row["motion"]
        reference = source_dir / "reference_50hz.npz"
        metadata = json.loads(reference.with_suffix(".json").read_text())
        overlay_meta = json.loads((source_dir / "overlay/overlay.json").read_text())
        if digest(reference) != overlay_meta["reference_sha256"]:
            raise ValueError("Overlay reference hash mismatch")
        if metadata["model_sha256"] != digest(ROOT / "src/supermimic/robots/g1_23dof.xml"):
            raise ValueError("Robot model changed")
        with np.load(reference) as ref, np.load(source_dir / "overlay/overlay.npz") as ov:
            if not np.array_equal(ref["timestamps"], ov["timestamps"]):
                raise ValueError("Overlay timebase mismatch")
            expected_frames = len(range(0, len(ref["timestamps"]), 2))
        destination = a.output / row["motion"]
        destination.mkdir()
        for name in ("overlay.npz", "overlay.json"):
            shutil.copy2(source_dir / "overlay" / name, destination / name)
        label = "URDF defaults" if a.variant == "official" else "URDF optimized"
        video = render(reference, destination, row["title"] + " | " + label)
        results.append(dict(motion=row["motion"], variant=a.variant, reference=str(reference),
                            reference_sha256=digest(reference), video=str(video), expected_frames=expected_frames))
    output = a.output / "all_motions_overlay.mp4"
    with imageio.get_writer(output, fps=25, codec="libx264", quality=8) as writer:
        for row in results:
            with imageio.get_reader(row["video"]) as reader:
                for frame in reader:
                    writer.append_data(frame)
    with imageio.get_reader(output) as reader:
        meta = reader.get_meta_data()
        frames = sum(1 for _ in reader)
    assert frames == sum(x["expected_frames"] for x in results)
    assert meta["fps"] == 25 and meta["size"] == (1920, 736)
    report = dict(variant=a.variant, motions=results, video=str(output), video_sha256=digest(output),
                  frames=frames, fps=25, duration_s=frames/25, resolution=[1920, 736],
                  layout="Raw source skeleton / calibrated targets view A / calibrated targets view B",
                  timing="Exact 50Hz reference timestamps, every second frame rendered at 25fps; no speed change",
                  existing_ik_results_reused=True, full_video_decode_passed=True)
    (a.output / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
