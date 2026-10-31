#!/usr/bin/env python3
"""Generate a native23 target using NVIDIA's Robot Configurator equations/defaults.

Run with SOMA-Retargeter-official/.venv/bin/python. Does not read community configs.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

os.environ.setdefault("PYGLET_HEADLESS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT.parent / "SOMA-Retargeter-official"
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import newton
import warp as wp
import mujoco
from scipy.spatial.transform import Rotation
from soma_retargeter.io.bvh import load_bvh
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
from soma_retargeter.robotics.robot_registry import register_robots_path
from soma_retargeter.pipelines.utils import create_robot_builder
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
from soma_retargeter.utils.newton_utils import compute_ground_offset
from supermimic.robots.spec import JOINT_NAMES
from supermimic.data.motion_convert import load_g1_model, g1_xml_path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf", type=Path, default=ROOT.parent / "unitree_ros/robots/g1_description/g1_23dof_rev_1_0.urdf")
    p.add_argument("--output", type=Path, default=ROOT / "assets/robotics/unitree/g1_23dof_urdf_v1")
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error("Choose an empty output directory; calibration versions are immutable")
    wp.set_device("cpu")
    spec = importlib.util.spec_from_file_location("official_configurator", OFFICIAL / "app/tools/robot_config_generator.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    defaults_path = OFFICIAL / "app/tools/assets/robot_configurator/robot_config_generator_config.json"
    defaults = json.loads(defaults_path.read_text())
    name = a.output.name
    desc, configs = a.output / "desc", a.output / "configs"
    (desc / "meshes").mkdir(parents=True)
    configs.mkdir()
    tree = ET.parse(a.urdf)
    movable = [j for j in tree.getroot().findall("joint") if j.get("type") != "fixed"]
    assert tuple(j.get("name") for j in movable) == JOINT_NAMES
    mesh_hashes = {}
    for mesh in tree.getroot().iter("mesh"):
        source = a.urdf.parent / mesh.get("filename")
        if not source.is_file():
            raise FileNotFoundError(source)
        target = desc / "meshes" / source.name
        if target.exists() and digest(target) != digest(source):
            raise ValueError("Conflicting mesh basename")
        shutil.copy2(source, target)
        mesh_hashes[source.name] = digest(source)
        mesh.set("filename", "meshes/" + source.name)
    imported = desc / a.urdf.name
    tree.write(imported, encoding="utf-8", xml_declaration=True)
    # Only packaging paths change; all link origins, axes, limits and geometry are kept.
    for lic in (a.urdf.parent / "README.md", ROOT.parent / "unitree_ros/LICENSE"):
        if lic.is_file():
            shutil.copy2(lic, a.output / ("SOURCE_" + lic.name))
    builder = newton.ModelBuilder()
    builder.add_urdf(str(imported), floating=True)
    ground_offset = compute_ground_offset(builder)
    manifest = dict(name=name, desc=dict(urdf_path="desc/" + imported.name, urdf_offset=ground_offset),
                    retarget_configs=dict(soma="configs/retargeter.json"))
    write(a.output / "manifest.json", manifest)
    register_robots_path(str(a.output))
    builder = create_robot_builder(name)
    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    robot_q = model.joint_q.numpy().copy()
    names = [x.split("/")[-1] for x in model.body_label]
    robot_tx = state.body_q.numpy().copy()
    coords = [x.split("/")[-1] for x, t in zip(builder.joint_label, builder.joint_type) if t == newton.JointType.REVOLUTE]
    assert tuple(coords) == JOINT_NAMES and len(robot_q) == 30

    # Independent articulated-model equivalence check at zero and 32 seeded poses.
    mj = load_g1_model()
    md = mujoco.MjData(mj)
    common = [mj.body(i).name for i in range(1, mj.nbody)]
    mj_ids, nt_ids = [mj.body(x).id for x in common], [names.index(x) for x in common]
    rng = np.random.default_rng(23)
    position_error = rotation_error = 0.0
    for k in range(33):
        q = robot_q.copy()
        if k:
            q[7:] = rng.uniform(mj.jnt_range[1:, 0], mj.jnt_range[1:, 1])
        model.joint_q.assign(q)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        got = state.body_q.numpy()[nt_ids]
        md.qpos[:] = np.r_[q[:3], q[[6, 3, 4, 5]], q[7:]]
        mujoco.mj_forward(mj, md)
        position_error = max(position_error, float(np.max(np.linalg.norm(got[:, :3] - md.xpos[mj_ids], axis=-1))))
        rotation_error = max(rotation_error, float(np.max((Rotation.from_quat(got[:, 3:]) .inv() * Rotation.from_quat(md.xquat[mj_ids][:, [1, 2, 3, 0]])).magnitude())))
    assert position_error < 1e-5 and rotation_error < 1e-5, (position_error, rotation_error)
    urdf_limits = np.array([[float(j.find("limit").get(x)) for x in ("lower", "upper")] for j in movable])
    limit_delta = float(np.max(np.abs(urdf_limits - mj.jnt_range[1:])))
    assert limit_delta < 5e-6, "Joint limits differ beyond XML decimal rounding"
    model.joint_q.assign(robot_q)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    zero_path = OFFICIAL / "soma_retargeter/assets/soma/soma_zero_frame0.bvh"
    skeleton, animation = load_bvh(str(zero_path))
    converter = SpaceConverter(FacingDirectionType.MUJOCO)
    # Exactly the configurator's reference transform; the zero BVH faces -Y.
    zero_xform = wp.mul(wp.transform(wp.vec3(0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.radians(90.))),
                        converter.transform(wp.transform_identity()))
    human = SkeletonInstance(skeleton, wp.vec3(1), zero_xform)
    human.set_local_transforms(animation.get_local_transforms(0))
    human_tx = human.compute_global_transforms(scale=1.)
    scale = float(robot_tx[names.index("pelvis"), 2] / human_tx[skeleton.joint_index("Hips"), 2])
    human.scale = scale
    human_tx = human.compute_global_transforms()
    mapping = {"Hips": "pelvis", "Chest": "torso_link"}
    for side, prefix in (("Left", "left"), ("Right", "right")):
        for source, body in (("Arm", "shoulder_roll_link"), ("ForeArm", "elbow_link"),
                             ("Hand", "wrist_roll_rubber_hand"), ("Leg", "hip_roll_link"),
                             ("Shin", "knee_link"), ("Foot", "ankle_roll_link")):
            mapping[side + source] = prefix + "_" + body
    mapping = {s: mapping[s] for s in defaults["soma_joints"] if s in mapping}
    offsets = {}
    for source, body in mapping.items():
        offset = module._compute_local_offset(wp.transform(*human_tx[skeleton.joint_index(source)]),
                                               wp.transform(*robot_tx[names.index(body)]))
        offsets[source] = [list(map(float, offset.p)), list(map(float, offset.q))]
    scaler = dict(human_type="soma", human_root="Hips", robot_type=name, robot_root="pelvis",
                  human_scale_ratio=scale, reference_joint_q=robot_q.tolist(), human_joint_offsets=offsets)
    write(configs / "scaler.json", scaler)
    weights = defaults["default_soma_weights"]
    config = dict(defaults["default_ik_params"], smooth_joint_filter_weight=0.0,
                  human_robot_scaler_config="configs/scaler.json",
                  post_processing=dict(robot_config="configs/post_processing.json"),
                  ik_match_table={s: dict(t_body=b, r_body=b, t_weight=weights[s][0], r_weight=weights[s][1]) for s, b in mapping.items()})
    write(configs / "retargeter.json", config)
    proxy = SimpleNamespace(body_link_names=["(none)"] + list(mapping.values()),
                            mapping_body_idx={s: i+1 for i, s in enumerate(mapping)})
    post = module.Viewer._build_post_processing_config(proxy)
    post["contact_correction"]["sole_normal_local"] = [0., 0., 1.]
    write(configs / "post_processing.json", post)
    # Runtime checkpoint: reload the serialized scaler, reproduce every calibration target.
    loaded = HumanToRobotScaler(skeleton, str(configs / "scaler.json"))
    actual = loaded.compute_effectors_from_buffer(animation, True, zero_xform)[0]
    expected = robot_tx[[names.index(mapping[s]) for s in loaded.effector_names()]]
    calibration_pos = float(np.max(np.linalg.norm(actual[:, :3] - expected[:, :3], axis=-1)))
    calibration_rot = float(np.max((Rotation.from_quat(actual[:, 3:]).inv() * Rotation.from_quat(expected[:, 3:])).magnitude()))
    assert calibration_pos < 2e-6 and calibration_rot < 2e-6
    bbox_scale = module.Viewer._default_scale_from_bbox(SimpleNamespace(soma_instance=human,
        _compute_robot_bbox_height=lambda: module.Viewer._compute_robot_bbox_height(SimpleNamespace(model=model, state=state))))
    audit = dict(official_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=OFFICIAL, text=True).strip(),
        source_urdf=str(a.urdf), source_urdf_sha256=digest(a.urdf), imported_urdf_sha256=digest(imported),
        canonical_mjcf_sha256=digest(g1_xml_path()), mesh_sha256=mesh_hashes,
        source_zero_bvh_sha256=digest(zero_path), configurator_defaults_sha256=digest(defaults_path),
        joint_names=coords, joint_coordinates=30, joint_dofs=29, mapped_effectors=mapping,
        ground_offset_m=ground_offset, human_scale_ratio=scale, bounding_box_scale_alternative=bbox_scale,
        scale_choice="Official pelvis-height option; avoids calibrating limb scale from head/decorative geometry",
        reference_pose="URDF default articulated q=0: upright legs, upper arms down, forearms already forward by URDF geometry",
        wrist_landmark="Native wrist_roll_rubber_hand body origin, zero robot-local point offset; transform measured from URDF FK, no 29DoF 84mm heuristic",
        source_zero_transform="Rz(90deg) * official Mujoco SpaceConverter, exactly as Robot Configurator",
        unsupported_source_landmarks=["Head", "LeftToeBase", "RightToeBase"],
        smooth_filter="No masks initially; global weight zero as official configurator does for an empty list",
        calibration_max_position_error_m=calibration_pos, calibration_max_rotation_error_rad=calibration_rot,
        urdf_vs_mjcf=dict(poses=33, seed=23, bodies=len(common), max_position_error_m=position_error,
                          max_rotation_error_rad=rotation_error, max_joint_limit_difference_rad=limit_delta,
                          joint_limits_match_within_xml_rounding=True),
        config_provenance="NVIDIA configurator defaults and local URDF FK only; no community config inputs",
        production_changed=False)
    write(a.output / "calibration_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
