"""Reference validation. Kinematic checks do not certify dynamic trackability."""
import numpy as np
import mujoco


def quality_report(model, motion, bodies):
    limits = model.jnt_range[1:]
    excess = np.maximum(limits[:, 0] - motion.joint_pos, motion.joint_pos - limits[:, 1])
    data = mujoco.MjData(model)
    foot_ids = [i for i in range(model.ngeom)
                if "foot" in (model.geom(i).name or "") and "collision" in (model.geom(i).name or "")]
    floor_by_side = np.full((motion.num_frames, 2), np.nan)
    self_collision_frames = 0
    collision_pairs = {}
    # Capsule support distance to z=0: rotated half-axis + radius, not ankle origin height.
    for t in range(motion.num_frames):
        data.qpos[:] = np.r_[motion.root_pos[t], motion.root_quat[t], motion.joint_pos[t]]
        mujoco.mj_fwdPosition(model, data)
        hits = [c for c in data.contact if c.dist < -0.002
                and model.geom_bodyid[c.geom1] != 0 and model.geom_bodyid[c.geom2] != 0]
        if hits:
            self_collision_frames += 1
        frame_pairs = {}
        for c in hits:
            key = " | ".join(sorted((model.geom(c.geom1).name or str(c.geom1),
                                     model.geom(c.geom2).name or str(c.geom2))))
            frame_pairs[key] = max(frame_pairs.get(key, 0.), -float(c.dist))
        for key, depth in frame_pairs.items():
            stat = collision_pairs.setdefault(key, {"frames": 0, "max_penetration_m": 0.})
            stat["frames"] += 1
            stat["max_penetration_m"] = max(stat["max_penetration_m"], depth)
        lows = [[], []]
        for g in foot_ids:
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                half_z = abs(data.geom_xmat[g].reshape(3, 3)[2, 2]) * model.geom_size[g, 1]
                side = 0 if model.geom(g).name.startswith("left") else 1
                lows[side].append(data.geom_xpos[g, 2] - half_z - model.geom_size[g, 0])
        for side in range(2):
            if lows[side]:
                floor_by_side[t, side] = min(lows[side])
    floor_min = np.min(floor_by_side, axis=1)
    ankle_ids = [bodies.body_names.index(f"{s}_ankle_roll_link") for s in ("left", "right")]
    foot_speed = np.linalg.norm(bodies.lin_vel_w[:, ankle_ids, :2], axis=-1)
    near_floor = np.abs(floor_by_side) < 0.03
    return {
        "finite": bool(all(np.isfinite(x).all() for x in (motion.joint_pos, motion.joint_vel, bodies.pos_w, bodies.lin_vel_w))),
        "joint_limit_violation_frames": int(np.any(excess > 1e-5, axis=1).sum()),
        "max_joint_limit_excess_rad": float(max(0., excess.max())),
        "max_joint_speed_rad_s": float(np.abs(motion.joint_vel).max()),
        "max_joint_acceleration_rad_s2": float(np.abs(np.gradient(motion.joint_vel, 1/motion.fps, axis=0)).max()),
        "self_collision_frames_2mm": self_collision_frames,
        "self_collision_pairs": collision_pairs,
        "max_root_speed_m_s": float(np.linalg.norm(motion.root_lin_vel_w, axis=1).max()),
        "foot_penetration_frames_2cm": int(np.sum(floor_min < -0.02)),
        "minimum_foot_surface_z_m": float(np.nanmin(floor_min)) if np.isfinite(floor_min).any() else None,
        "near_floor_max_ankle_xy_speed_m_s": float(foot_speed[near_floor].max()) if near_floor.any() else None,
        "notes": "Foot speed is a screening statistic, not a contact label. No automatic clipping, ground shifts or dynamic feasibility guarantee.",
    }


def validate_reference(d, expected_fps=50.):
    from supermimic.data.motion_convert import G1_JOINT_NAMES
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    n = len(d["joint_pos"])
    if n < 2 or not np.isclose(fps, expected_fps):
        raise ValueError(f"Expected >=2 frames at {expected_fps} Hz, got {n} at {fps}")
    if tuple(d["joint_names"].tolist()) != G1_JOINT_NAMES:
        raise ValueError("Reference joint order does not match native23 model")
    if "timestamps" not in d or not np.allclose(d["timestamps"], np.arange(n)/fps, atol=1e-8, rtol=0):
        raise ValueError("Missing or misaligned reference timestamps; reconvert source data")
    nb = len(d["body_names"])
    shapes = {"joint_pos": (n,23), "joint_vel": (n,23), "body_pos_w": (n,nb,3),
              "body_quat_w": (n,nb,4), "body_lin_vel_w": (n,nb,3), "body_ang_vel_w": (n,nb,3)}
    for key, shape in shapes.items():
        if d[key].shape != shape or not np.isfinite(d[key]).all():
            raise ValueError(f"Invalid reference field {key}: expected finite {shape}")
    if not np.allclose(np.linalg.norm(d["body_quat_w"], axis=-1), 1., atol=1e-5):
        raise ValueError("Non-unit reference quaternions")
