"""Source BVH timing shared by retargeting and reference conversion."""
from pathlib import Path
import re


def read_bvh_timing(path):
    text = Path(path).read_text()
    count = re.search(r"(?m)^\s*Frames:\s*(\d+)", text)
    interval = re.search(r"(?m)^\s*Frame Time:\s*([\d.eE+-]+)", text)
    if not count or not interval or float(interval[1]) <= 0:
        raise ValueError("BVH requires Frames and positive Frame Time metadata")
    return int(count[1]), 1. / float(interval[1])
