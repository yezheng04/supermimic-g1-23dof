#!/usr/bin/env python3
"""Run an isolated official SOMA target, preserving BVH time and native23 outputs."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import newton
import warp as wp
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from soma_retargeter.io.bvh import load_bvh
from soma_retargeter.io.csv import save_csv
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
from soma_retargeter.robotics.robot_registry import register_robots_path
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
from supermimic.retarget.bvh import read_bvh_timing
from supermimic.data.motion_convert import save_reference
from supermimic.robots.spec import JOINT_NAMES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_overlay(pipeline, animation, xform, reference, output, config):
    skeleton = animation.skeleton
    times = np.load(reference)["timestamps"]
    source_times = np.arange(animation.num_frames) / animation.sample_rate
    source = np.asarray([animation.compute_global_transforms(i, xform) for i in range(animation.num_frames)])
    targets = pipeline.human_robot_scaler.compute_effectors_from_buffer(animation, True, xform)
    names = list(config["ik_match_table"])
    targets = targets[:, [pipeline.human_robot_scaler.effector_names().index(x) for x in names]]
    source_names = [x for x in skeleton.joint_names if x in set(names) | {
        "Spine", "Spine1", "Spine2", "Neck1", "Neck2", "Head", "HeadEnd",
        "LeftShoulder", "RightShoulder", "LeftToeBase", "RightToeBase"}]
    def edges(display):
        result = []
        for i, name in enumerate(display):
            parent = skeleton.parent_indices[skeleton.joint_index(name)]
            while parent >= 0 and skeleton.joint_names[parent] not in display:
                parent = skeleton.parent_indices[parent]
            if parent >= 0:
                result.append((display.index(skeleton.joint_names[parent]), i))
        return np.asarray(result, dtype=np.int32).reshape(-1, 2)
    target_pos = interp1d(source_times, targets[:, :, :3], axis=0)(times)
    target_quat = np.stack([Slerp(source_times, Rotation.from_quat(targets[:, j, 3:]))(times).as_quat()
                            for j in range(len(names))], axis=1)
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / "overlay.npz", timestamps=times,
        source_pos=interp1d(source_times, source[:, [skeleton.joint_index(x) for x in source_names], :3], axis=0)(times),
        source_names=np.array(source_names), source_edges=edges(source_names),
        target_pos=target_pos, target_quat_xyzw=target_quat, target_names=np.array(names), target_edges=edges(names),
        target_bodies=np.array([config["ik_match_table"][x]["t_body"] for x in names]),
        orientation_bodies=np.array([config["ik_match_table"][x]["r_body"] for x in names]))
    (output / "overlay.json").write_text(json.dumps(dict(reference=str(reference), reference_sha256=digest(reference),
        source_fps=animation.sample_rate, output_fps=50,
        alignment="Original world coordinates, no fitted alignment; official SOMA URDF calibration",
        target_stage="Pre-IK scaler targets, before contact/limb post-processing"), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--robot-dir", type=Path, default=ROOT / "assets/robotics/unitree/g1_23dof_urdf_v1")
    p.add_argument("--config", type=Path, help="Optional retargeter override; never overwrites the registered config")
    p.add_argument("--no-contact", action="store_true", help="Only disable foot plant correction")
    p.add_argument("--no-post", action="store_true", help="Disable limb and contact post-processing for ablation")
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error("Use an empty experiment output directory")
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    register_robots_path(str(a.robot_dir))
    manifest = json.loads((a.robot_dir / "manifest.json").read_text())
    config_path = a.config or a.robot_dir / manifest["retarget_configs"]["soma"]
    config = json.loads(config_path.read_text())
    if a.no_contact or a.no_post:
        config["enable_contact_processing"] = False
    if a.no_post:
        config["enable_post_processing"] = False
    scaler = json.loads((a.robot_dir / config["human_robot_scaler_config"]).read_text())
    skeleton, animation = load_bvh(str(a.source))
    n, fps = read_bvh_timing(a.source)
    assert animation.num_frames == n and n >= 2
    animation = AnimationBuffer(skeleton, n, fps, animation.local_transforms)
    wp.set_device("cuda:0")
    pipeline = SomaRetargetingPipeline(skeleton, "soma", manifest["name"], retarget_config=config)
    names = [x.split("/")[-1] for x, t in zip(pipeline.robot_builder.joint_label, pipeline.robot_builder.joint_type)
             if t == newton.JointType.REVOLUTE]
    assert tuple(names) == JOINT_NAMES and pipeline.robot_builder.joint_coord_count == 30
    xform = SpaceConverter(FacingDirectionType.MUJOCO).transform(wp.transform_identity())
    pipeline.add_input_motions([animation], [xform], True)
    outputs = pipeline.execute()
    raw = np.stack(outputs[0].data).astype(np.float64)
    assert raw.shape == (n, 30) and np.isfinite(raw).all()
    times = np.arange(n, dtype=np.float64) / fps
    np.savez_compressed(a.output / "raw.npz", root_pos=raw[:, :3], root_quat=raw[:, [6, 3, 4, 5]],
        joint_pos=raw[:, 7:], timestamps=times, fps=np.array(fps), joint_names=np.array(JOINT_NAMES),
        quaternion_order=np.array("wxyz"))
    np.savetxt(a.output / "motion.csv", raw, delimiter=",", fmt="%.10g")
    save_csv(str(a.output / "motion_soma.csv"), names, outputs[0])
    reference = a.output / "reference_50hz.npz"
    meta = dict(pipeline="Official SOMA URDF-native calibration", target=manifest["name"],
        official_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent / "SOMA-Retargeter-official", text=True).strip(),
        runtime_versions={x: importlib.metadata.version(x) for x in ("newton", "warp-lang", "numpy", "scipy", "mujoco")},
        source_sha256=digest(a.source), source_fps=fps, source_frames_used=n,
        ik_config=config, human_scaler_config=scaler, target_manifest=manifest,
        imported_urdf_sha256=digest(a.robot_dir / manifest["desc"]["urdf_path"]),
        calibration_audit_sha256=digest(a.robot_dir / "calibration_audit.json"),
        training_approved=False, elapsed_s=time.monotonic()-started)
    report = save_reference(raw, reference, input_fps=fps, timestamps=times,
                            source=str(a.source.resolve()), extra_metadata=meta)
    save_overlay(pipeline, animation, xform, reference, a.output / "overlay", config)
    print(json.dumps(dict(output=str(reference), frames=report["output_frames"], quality=report["quality"]), indent=2))


if __name__ == "__main__":
    main()
