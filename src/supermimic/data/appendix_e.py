"""SuperMimic Appendix E geometry rules, with explicit unpublished gate choices.

Thresholds come from arXiv:2610.28480 Appendix E. Numerical upright/standing/
stationary gates and the sample-count duration convention are local choices;
this module does not claim bitwise agreement with the unreleased author code.
"""
from functools import lru_cache
import re

import mujoco
import numpy as np

from .motion_convert import G1_JOINT_NAMES, g1_xml_path
from .quality import validate_reference


def longest_run(mask):
    """Number of consecutive control samples, never aggregate separated runs."""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return int(np.max(edges[1::2] - edges[::2], initial=0))


def floating_gate(pelvis_pos, pelvis_quat, pelvis_lin_vel, pelvis_ang_vel, joint_vel, settings):
    # R_zz for wxyz: +1 for upright, -1 for upside down; yaw does not matter.
    upright_cos = 1 - 2 * (pelvis_quat[:, 1] ** 2 + pelvis_quat[:, 2] ** 2)
    low, high = settings["pelvis_height_range_m"]
    return ((upright_cos > settings["upright_cos_min"])
        & (pelvis_pos[:, 2] > low) & (pelvis_pos[:, 2] < high)
        & (np.linalg.norm(pelvis_lin_vel, axis=1) < settings["pelvis_speed_max_m_s"])
        & (np.linalg.norm(pelvis_ang_vel, axis=1) < settings["pelvis_angular_speed_max_rad_s"])
        & (np.sqrt(np.mean(joint_vel ** 2, axis=1)) < settings["joint_speed_rms_max_rad_s"]))


def evaluate_geometry(zmin, foot_distance, gate, fps, config):
    """Evaluate already measured trajectories; foot_distance is (T, left/right).

    NaN distances outside gate are permitted because those frames cannot float
    under this rule. A NaN inside the gate is an error, never an implicit pass.
    """
    zmin = np.asarray(zmin)
    foot_distance = np.asarray(foot_distance)
    gate = np.asarray(gate, dtype=bool)
    if zmin.ndim != 1 or not len(zmin) or gate.shape != zmin.shape or foot_distance.shape != (len(zmin), 2):
        raise ValueError("Incompatible Appendix E trajectory shapes")
    if not np.isfinite(zmin).all() or not np.isfinite(foot_distance[gate]).all():
        raise ValueError("Nonfinite Appendix E measurements")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid sampling rate")
    elevated = zmin > config["elevated_height_m"]
    deep = zmin < config["penetration_hard_m"]
    shallow = zmin < config["penetration_soft_m"]
    floating = gate & np.all(foot_distance > config["floating_foot_height_m"], axis=1)
    elevated_s = longest_run(elevated) / fps
    floating_s = longest_run(floating) / fps
    reasons = []
    if elevated_s >= config["elevated_duration_s"]:
        reasons.append("elevated_support")
    if deep.any() or shallow.mean() > config["penetration_soft_fraction"]:
        reasons.append("ground_penetration")
    if floating_s >= config["floating_duration_s"]:
        reasons.append("floating_feet")
    valid_distances = foot_distance[gate]
    return dict(keep=not reasons, reasons=reasons,
        metrics=dict(min_body_link_z_m=float(zmin.min()),
            elevated_longest_frames=longest_run(elevated), elevated_longest_s=elevated_s,
            hard_penetration_frames=int(deep.sum()), soft_penetration_fraction=float(shallow.mean()),
            floating_longest_frames=longest_run(floating), floating_longest_s=floating_s,
            stationary_upright_standing_frames=int(gate.sum()),
            min_foot_plane_distance_in_gate_m=float(valid_distances.min()) if valid_distances.size else None),
        duration_convention="consecutive_sample_count / fps; 30 samples = 0.6 s at 50 Hz")


@lru_cache(maxsize=1)
def distance_model():
    """Canonical native23 robot plus a measurement plane at world z=0."""
    spec = mujoco.MjSpec.from_file(str(g1_xml_path()))
    spec.worldbody.add_geom(name="appendix_e_measurement_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, .1], pos=[0, 0, 0], contype=0, conaffinity=0)
    model = spec.compile()
    if tuple(model.joint(i).name for i in range(1, model.njnt)) != G1_JOINT_NAMES:
        raise ValueError("Native23 measurement model joint order changed")
    floor = model.geom("appendix_e_measurement_floor").id
    feet = [[g for g in range(model.ngeom)
             if re.fullmatch(side + r"_foot[1-7]_collision", model.geom(g).name or "")]
            for side in ("left", "right")]
    if any(len(ids) != 7 for ids in feet):
        raise ValueError("Expected seven foot collision geoms per side")
    return model, floor, feet


def measure_foot_distances(root_pos, root_quat, joint_pos, gate):
    model, floor, feet = distance_model()
    data = mujoco.MjData(model)
    distances = np.full((len(gate), 2), np.nan)
    for t in np.flatnonzero(gate):
        data.qpos[:] = np.r_[root_pos[t], root_quat[t], joint_pos[t]]
        mujoco.mj_kinematics(model, data)
        for side, ids in enumerate(feet):
            distances[t, side] = min(mujoco.mj_geomDistance(model, data, floor, g, 1e6, None) for g in ids)
    return distances


def screen_reference(path, config):
    with np.load(path, allow_pickle=False) as ref:
        validate_reference(ref)
        names = ref["body_names"].tolist()
        if "world" in names:
            raise ValueError("World must be excluded from body-link minimum")
        pel = names.index("pelvis")
        position, quat = ref["body_pos_w"][:, pel], ref["body_quat_w"][:, pel]
        # The NPZ body states were computed by native23 MuJoCo FK at conversion.
        zmin = ref["body_pos_w"][:, :, 2].min(axis=1)
        gate = floating_gate(position, quat, ref["body_lin_vel_w"][:, pel],
            ref["body_ang_vel_w"][:, pel], ref["joint_vel"], config["floating_gate_local"])
        feet = measure_foot_distances(position, quat, ref["joint_pos"], gate)
        report = evaluate_geometry(zmin, feet, gate, float(ref["fps"].item()), config)
    report.update(rules=config, source="SuperMimic Appendix E, arXiv:2610.28480v1",
        foot_distance_method="mj_geomDistance, each collision geom to world z=0 plane; minimum per foot",
        exact_author_implementation=False, dynamic_feasibility="not_evaluated")
    return report
