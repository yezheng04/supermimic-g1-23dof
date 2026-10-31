#!/usr/bin/env python3
"""SOMA uniform-skeleton BVH -> native23 robot, with explicit source timestamps.

Run with SOMA-Retargeter's Python 3.12 environment. Original LAFAN1 skeletons
require a separate adapter; sharing the BVH extension is not sufficient.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import warp as wp
from soma_retargeter.assets.bvh import load_bvh
from soma_retargeter.assets.csv import save_csv, UnitreeG123DOF_CSVConfig
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
from soma_retargeter.utils import io_utils
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str
from supermimic.retarget.bvh import read_bvh_timing
from supermimic.data.motion_convert import g1_xml_path, G1_JOINT_NAMES, save_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--hand-frame-shift", type=float, default=-.084,
                   help="Diagnostic wrist-yaw to native23 wrist-roll x offset, in meters")
    p.add_argument("--hand-rotation-scale", type=float, default=1.,
                   help="Diagnostic multiplier on hand orientation IK weights")
    p.add_argument("--no-feet-stabilizer", action="store_true")
    a = p.parse_args()
    if a.max_frames is not None and a.max_frames < 2:
        p.error("max-frames must be >= 2")
    if a.output.exists() and any(a.output.iterdir()):
        p.error("Output directory must be empty; choose a new run directory")
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    skeleton, animation = load_bvh(str(a.source))
    declared_frames, fps = read_bvh_timing(a.source)
    if animation.num_frames != declared_frames:
        raise ValueError("Loaded frame count differs from source header")
    # Upstream rounds BVH FPS to 2 decimals. Restore its physical header timing.
    n = min(declared_frames, a.max_frames or declared_frames)
    animation = AnimationBuffer(skeleton, n, fps, animation.local_transforms[:n])
    config = io_utils.load_json(io_utils.get_config_file(
        "unitree_g1", "soma_to_g1_23dof_retargeter_config.json"))
    missing = set(config["ik_map"]) - set(skeleton.joint_names)
    if missing:
        raise ValueError(f"Input is not a compatible SOMA skeleton; missing effectors: {sorted(missing)}")
    config["mjcf_path"] = str(g1_xml_path().resolve())
    if not np.isfinite([a.hand_frame_shift, a.hand_rotation_scale]).all() or a.hand_rotation_scale < 0:
        p.error("Diagnostic calibration values must be finite, with nonnegative rotation scale")
    if a.no_feet_stabilizer:
        config["enable_post_processing"] = False
    for side in ("Left", "Right"):
        config["ik_map"][side + "Hand"]["r_weight"] *= a.hand_rotation_scale
    scaler = io_utils.load_json(io_utils.get_config_file(config["human_robot_scaler_config"]))
    # Upstream 23-DoF fork renames wrist_yaw_link to wrist_roll_rubber_hand,
    # but reuses the 29-DoF effector origin. Native23's frame is 84 mm proximal.
    for side in ("Left", "Right"):
        scaler["joint_offsets"][side+"Hand"][0][0] += a.hand_frame_shift
    scaler_path = (a.output / "scaler_native23.json").resolve()
    scaler_path.write_text(json.dumps(scaler, indent=2))
    config["human_robot_scaler_config"] = str(scaler_path)
    pipeline = NewtonPipeline(skeleton, "soma", "unitree_g1", retarget_config=config)
    names = [label.rsplit("/", 1)[-1] for label in pipeline.robot_builder.joint_label]
    movable = [name for name in names if name in G1_JOINT_NAMES]
    if tuple(movable) != G1_JOINT_NAMES or pipeline.robot_builder.joint_coord_count != 30:
        raise ValueError(f"Unexpected Newton native23 joint layout: {names}")
    source = str(a.source.resolve())
    converter = SpaceConverter(get_facing_direction_type_from_str("Mujoco"))
    pipeline.add_input_motions([animation], [converter.transform(wp.transform_identity())], True)
    outputs = pipeline.execute()
    raw = np.stack(outputs[0].data).astype(np.float64)
    if raw.shape != (n, 30) or not np.isfinite(raw).all():
        raise ValueError(f"Unexpected output: {raw.shape}, expected {(n, 30)}")
    # Newton free root is xyz + xyzw, matching our numeric CSV loader.
    np.savetxt(a.output / "motion.csv", raw, delimiter=",", fmt="%.10g")
    save_csv(str(a.output / "motion_soma.csv"), outputs[0], UnitreeG123DOF_CSVConfig())
    timestamps = np.arange(n, dtype=np.float64) / fps
    np.savez_compressed(a.output / "raw.npz", root_pos=raw[:, :3],
        root_quat=raw[:, [6,3,4,5]], joint_pos=raw[:, 7:], timestamps=timestamps,
        fps=np.array(fps), joint_names=np.array(G1_JOINT_NAMES), quaternion_order=np.array("wxyz"))
    meta = {"pipeline": "SOMA Newton IK + feet stabilizer",
        "human_adapter": "upstream SOMA human scaler",
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "soma_commit": "6c0517236d98910a63a57b082c4a585b2776de91",
        "ik_config": config, "human_scaler_config": scaler,
        "source_fps": fps, "source_frames_used": n,
        "hand_frame_shift_m": a.hand_frame_shift,
        "hand_rotation_weight_scale": a.hand_rotation_scale,
        "feet_stabilizer_enabled": not a.no_feet_stabilizer,
        "training_approved": False, "elapsed_s": time.monotonic()-started}
    report = save_reference(raw, a.output / "reference_50hz.npz", input_fps=fps,
        timestamps=timestamps, source=source, extra_metadata=meta)
    (a.output / "motion.json").write_text(json.dumps({
        "fps": fps, "position_units": "m", "quaternion_order": "xyzw", "joint_units": "rad",
        "joint_names": list(G1_JOINT_NAMES), "soma_csv_units": "cm / Euler xyz degrees / joint degrees"
    }, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
