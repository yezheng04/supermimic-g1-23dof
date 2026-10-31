#!/usr/bin/env python3
"""Audit native23 model geometry, pilot thresholds and SOMA calibration trials."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mujoco
import numpy as np
from supermimic.data.appendix_e import distance_model
from supermimic.data.motion_convert import g1_xml_path, load_g1_model


def short_report(path):
    r = json.loads(path.read_text())
    q = r["quality"]
    return dict(name=path.parent.name, frames=r["output_frames"],
        self_frames=q["self_collision_frames_2mm"], foot_frames=q["foot_penetration_frames_2cm"],
        max_self_depth_m=max((v["max_penetration_m"] for v in q["self_collision_pairs"].values()), default=0.),
        min_foot_z_m=q["minimum_foot_surface_z_m"],
        max_joint_speed_rad_s=q["max_joint_speed_rad_s"],
        max_joint_acceleration_rad_s2=q["max_joint_acceleration_rad_s2"])


def main():
    baseline = ROOT / "data/retargeted/bones_seed_native23_v2/clips"
    reports = sorted(baseline.glob("*/reference_50hz.json"))
    rows = [short_report(p) for p in reports]
    if len(rows) != 64:
        raise ValueError("Expected the original 64-clip pilot")
    vendor_xml = ROOT.parent / "unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1_23dof.xml"
    hash_file = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    model, floor, feet = distance_model()
    data = mujoco.MjData(model)
    distance_errors = []
    for path in reports:
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            pel = z["body_names"].tolist().index("pelvis")
            for t in np.linspace(0, len(z["joint_pos"])-1, 5, dtype=int):
                data.qpos[:] = np.r_[z["body_pos_w"][t, pel], z["body_quat_w"][t, pel], z["joint_pos"][t]]
                mujoco.mj_kinematics(model, data)
                for ids in feet:
                    for g in ids:
                        exact = mujoco.mj_geomDistance(model, data, floor, g, 1e6, None)
                        analytic = data.geom_xpos[g, 2] - model.geom_size[g, 0] - model.geom_size[g, 1] * abs(data.geom_xmat[g].reshape(3, 3)[2, 2])
                        distance_errors.append(abs(exact - analytic))
    # Compare wrist origins relative to each robot's common elbow frame: the
    # complete torso structures differ slightly, so do not conflate that with a wrist offset.
    m23 = load_g1_model()
    m29 = mujoco.MjModel.from_xml_path(str(vendor_xml.with_name("g1.xml")))
    d23, d29 = mujoco.MjData(m23), mujoco.MjData(m29)
    rng = np.random.default_rng(14)
    offsets = []
    for _ in range(50):
        for j in range(1, m23.njnt):
            value = rng.uniform(max(m23.jnt_range[j, 0], -.5), min(m23.jnt_range[j, 1], .5))
            d23.qpos[m23.jnt_qposadr[j]] = value
            d29.qpos[m29.joint(m23.joint(j).name).qposadr[0]] = value
        mujoco.mj_kinematics(m23, d23)
        mujoco.mj_kinematics(m29, d29)
        for side in ("left", "right"):
            h23, h29 = m23.body(side + "_wrist_roll_rubber_hand").id, m29.body(side + "_wrist_yaw_link").id
            e23, e29 = m23.body(side + "_elbow_link").id, m29.body(side + "_elbow_link").id
            r23, r29 = d23.xmat[e23].reshape(3, 3), d29.xmat[e29].reshape(3, 3)
            p23, p29 = r23.T @ (d23.xpos[h23]-d23.xpos[e23]), r29.T @ (d29.xpos[h29]-d29.xpos[e29])
            offsets.append((r23.T @ d23.xmat[h23].reshape(3, 3)).T @ (p29-p23))
    ablations = {}
    trials = ROOT / "data/retargeted/native23_calibration_ablation"
    for variant in ("no_wrist_shift", "hand_rotation_025", "no_stabilizer"):
        ablations[variant] = [short_report(p) for p in sorted((trials/variant).glob("*/reference_50hz.json"))]
    result = dict(model_sha256=hash_file(g1_xml_path()), unitree_model_sha256=hash_file(vendor_xml),
        model_identical_to_unitree=g1_xml_path().read_bytes() == vendor_xml.read_bytes(),
        foot_distance_comparisons=len(distance_errors), max_foot_distance_disagreement_m=max(distance_errors),
        wrist_common_elbow_frame=dict(poses=50, expected_delta_m=[.084, 0, 0],
            max_error_m=float(np.max(np.linalg.norm(np.asarray(offsets)-[.084, 0, 0], axis=1)))),
        baseline=rows, ablations=ablations,
        self_collision_threshold_sweep=[dict(tolerance_m=x, clips_with_no_exceedance=sum(r["max_self_depth_m"] <= x for r in rows))
            for x in (.002, .005, .01, .02, .05, .1)],
        foot_threshold_sweep=[dict(tolerance_m=x, clips_with_no_exceedance=sum(r["min_foot_z_m"] >= -x for r in rows))
            for x in (.02, .025, .03, .05)],
        production_decision="Appendix E filters candidates; extra native23 findings are diagnostic only. Preserve baseline calibration; arm refinement remains a separate experiment.",
        dynamic_feasibility="not_evaluated")
    dest = ROOT / "reports/native23_audit/summary.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("baseline", "ablations")}, indent=2))


if __name__ == "__main__":
    main()
