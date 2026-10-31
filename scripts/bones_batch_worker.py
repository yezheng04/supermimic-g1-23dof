"""CPU-only export/screening worker. Never import SOMA, Newton or Warp here."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from supermimic.data.motion_convert import G1_JOINT_NAMES, load_g1_model, save_reference
from prepare_bones_seed import process_clip

_MODEL = None


def finalize(row, source_root, work, config, raw=None, metadata=None):
    global _MODEL
    started = time.monotonic()
    work, source_root = Path(work), Path(source_root)
    destination = work / "clips" / row["name"]
    if raw is not None:
        if destination.exists():
            raise ValueError(f"Unexpected existing export directory: {destination}")
        destination.mkdir(parents=True)
        if _MODEL is None:
            _MODEL = load_g1_model()
        n, fps = metadata["source_frames_used"], metadata["source_fps"]
        if raw.shape != (n, 30) or not np.isfinite(raw).all():
            raise ValueError("Invalid batch result shape or values")
        timestamps = np.arange(n, dtype=np.float64) / fps
        np.savez_compressed(destination / "raw.npz", root_pos=raw[:, :3], root_quat=raw[:, [6, 3, 4, 5]],
            joint_pos=raw[:, 7:], timestamps=timestamps, fps=np.array(fps),
            joint_names=np.array(G1_JOINT_NAMES), quaternion_order=np.array("wxyz"))
        save_reference(raw, destination / "reference_50hz.npz", input_fps=fps,
            timestamps=timestamps, source=str((source_root / row["source"]).resolve()),
            model=_MODEL, extra_metadata=metadata)
    if not (destination / "reference_50hz.json").exists():
        raise ValueError("CPU worker requires a completed reference; cannot start a serial GPU subprocess")
    result = process_clip(row, source_root, work, Path("/unused"), config)
    return result, time.monotonic() - started
