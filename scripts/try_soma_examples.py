#!/usr/bin/env python3
"""Run six bundled SOMA BVHs through native23 IK, QC and two-view rendering."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from render_reference import render

EXAMPLES = [
    ("wave", "Wave", "wave_R_001__A428.bvh"),
    ("hiphop", "Hip-hop shuffle", "dance_hiphop_shuffle_square_R_fast_002__A318.bvh"),
    ("jump", "High jump", "high_jump_R_001__A277.bvh"),
    ("pickup", "Standing pick-up", "item_pick_up_standing_R_001__A410.bvh"),
    ("throw", "Throw ball", "Neutral_throw_ball_001__A057.bvh"),
    ("stretch", "Body stretch", "body_stretch_1_004__A069.bvh"),
]


def main():
    project = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--soma-repo", type=Path, default=project.parent / "SOMA-Retargeter")
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    if any(a.output.iterdir()):
        p.error("Output directory must be empty")
    results = []
    for slug, title, filename in EXAMPLES:
        source = a.soma_repo / "assets/motions/bvh" / filename
        if source.read_bytes().startswith(b"version https://git-lfs.github.com"):
            raise ValueError(f"Download LFS asset first: {source}")
        output = a.output / slug
        with (a.output / f"{slug}.log").open("w") as log:
            subprocess.run([str(a.soma_repo / ".venv/bin/python"),
                str(project / "scripts/retarget_soma.py"), str(source), str(output)],
                cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((output / "reference_50hz.json").read_text())
        print(f"Retargeted {slug}: {report['output_frames']} frames, {report['duration_s']:.2f}s", flush=True)
        video = render(output / "reference_50hz.npz", output / "preview.mp4", title)
        results.append({"motion": slug, "title": title, "source": str(source),
            "source_fps": report["source_fps"], "source_frames": report["source_frames_used"],
            "frames": report["output_frames"], "duration_s": report["duration_s"],
            "quality": report["quality"], "video": video["video"], "training_approved": False})
        (a.output / "summary.json").write_text(json.dumps(results, indent=2))
        print(f"Rendered {slug}: {video['video']}", flush=True)
    # One continuous reel: full clips in the same order, each retaining its time scale.
    import imageio.v2 as imageio
    with imageio.get_writer(a.output / "all_motions.mp4", fps=25, codec="libx264", quality=8) as writer:
        for row in results:
            with imageio.get_reader(row["video"]) as reader:
                for frame in reader:
                    writer.append_data(frame)
    print(f"Completed {len(results)} examples: {a.output / 'all_motions.mp4'}")


if __name__ == "__main__":
    main()
