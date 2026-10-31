#!/usr/bin/env python3
"""Official optimizer with exact BVH header timing (no package source edits)."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "SOMA-Retargeter-official"))
sys.path.insert(0, str(ROOT / "src"))
from soma_retargeter.io import bvh
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from supermimic.retarget.bvh import read_bvh_timing

load = bvh.load_bvh


def exact_load(path, *args, **kwargs):
    skeleton, animation = load(path, *args, **kwargs)
    n, fps = read_bvh_timing(Path(path))
    assert n == animation.num_frames
    return skeleton, AnimationBuffer(skeleton, n, fps, animation.local_transforms)


if __name__ == "__main__":
    bvh.load_bvh = exact_load
    from app.tools.ik_weight_optimizer import run_optimizer
    path = Path(sys.argv[1]).resolve()
    run_optimizer(json.loads(path.read_text()), path.parent)
