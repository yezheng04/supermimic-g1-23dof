"""Bounded, offline arm-only collision refinement after SOMA IK.

Uses the unchanged canonical MuJoCo collision geometry. Root, waist, legs and
timestamps are immutable. This is an optional local preprocessing extension,
not a SOMA feature or a dynamic feasibility certificate.
"""
import time

import mujoco
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from supermimic.data.motion_convert import load_g1_model


def contact_constraints(model, data, dofs, margin):
    """Linear signed-distance derivatives using contact normals and Jacobians."""
    rows, gaps = [], []
    jac1, jac2 = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for contact in data.contact:
        b1, b2 = model.geom_bodyid[[contact.geom1, contact.geom2]]
        if not b1 or not b2 or contact.dist >= margin:
            continue
        mujoco.mj_jac(model, data, jac1, None, contact.pos, b1)
        mujoco.mj_jac(model, data, jac2, None, contact.pos, b2)
        derivative = contact.frame[:3] @ (jac2 - jac1)[:, dofs]
        if np.linalg.norm(derivative) < 1e-8:
            continue  # Leg-only collisions cannot be repaired by arm coordinates.
        rows.append(derivative.copy())
        gaps.append(margin - float(contact.dist))
    return np.asarray(rows).reshape(-1, len(dofs)), np.asarray(gaps)


def refine_arms(raw, *, margin_m=.005, max_joint_correction_rad=.6, iterations=12,
                trust_region_rad=.12, temporal_weight=.25):
    started = time.monotonic()
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 30 or not np.isfinite(raw).all():
        raise ValueError("Expected finite native23 xyz/xyzw/joints motion")
    if min(margin_m, max_joint_correction_rad, trust_region_rad) <= 0 or iterations < 1 or temporal_weight < 0:
        raise ValueError("Invalid collision refinement settings")
    model = load_g1_model()
    # Broad-phase contact margin only; actual collision geometry is unchanged.
    model.geom_margin[:] = margin_m
    data = mujoco.MjData(model)
    joints = [i for i in range(1, model.njnt) if any(token in model.joint(i).name
              for token in ("shoulder_", "elbow_", "wrist_"))]
    coords = model.jnt_qposadr[joints]
    dofs = model.jnt_dofadr[joints]
    if len(coords) != 10:
        raise ValueError("Expected exactly ten native23 arm joints")
    limits = model.jnt_range[joints]
    out = raw.copy()
    previous_delta = np.zeros(len(coords))
    optimizer_failures = 0
    for t, frame in enumerate(raw):
        pose = np.r_[frame[:3], frame[[6, 3, 4, 5]], frame[7:]]
        source_arm = pose[coords].copy()
        lower = np.maximum(limits[:, 0], source_arm - max_joint_correction_rad)
        upper = np.minimum(limits[:, 1], source_arm + max_joint_correction_rad)
        # Warm start with the previous correction, then softly return to the source.
        pose[coords] = np.clip(source_arm + previous_delta * temporal_weight / (1 + temporal_weight), lower, upper)
        target = source_arm + previous_delta * temporal_weight / (1 + temporal_weight)
        for _ in range(iterations):
            data.qpos[:] = pose
            mujoco.mj_fwdPosition(model, data)
            jac, gaps = contact_constraints(model, data, dofs, margin_m)
            if not len(gaps) or gaps.max() < 1e-5:
                break
            bias = pose[coords] - target
            lo = np.maximum(lower - pose[coords], -trust_region_rad)
            hi = np.minimum(upper - pose[coords], trust_region_rad)
            # Slacks allow progress from deep penetration within a trust region.
            # Their large cost favors separation while keeping an infeasible clip auditable.
            n, k = len(coords), len(gaps)
            matrix = np.c_[jac, np.eye(k)]
            slack_weight = 1e5
            def objective(x):
                return .5 * np.dot(x[:n] + bias, x[:n] + bias) + .5 * slack_weight * np.dot(x[n:], x[n:])
            def gradient(x):
                return np.r_[x[:n] + bias, slack_weight * x[n:]]
            result = minimize(objective, np.r_[np.zeros(n), np.maximum(gaps, 0)], jac=gradient,
                method="SLSQP", bounds=Bounds(np.r_[lo, np.zeros(k)], np.r_[hi, np.full(k, np.inf)]),
                constraints=[LinearConstraint(matrix, gaps, np.inf)],
                options={"maxiter": 50, "ftol": 1e-9})
            if not result.success or not np.isfinite(result.x).all():
                optimizer_failures += 1
                break
            step = result.x[:n]
            pose[coords] += step
            if np.max(np.abs(step)) < 1e-6:
                break
        previous_delta = pose[coords] - source_arm
        out[t, coords] = pose[coords]  # Raw and MuJoCo joint coordinates both start at column 7.
    delta = out[:, coords] - raw[:, coords]
    immutable = sorted(set(range(30)) - set(coords.tolist()))
    if not np.array_equal(out[:, immutable], raw[:, immutable]):
        raise AssertionError("Arm refinement changed immutable coordinates")
    return out, dict(method="bounded arm-only sequential quadratic collision refinement",
        margin_m=margin_m, max_joint_correction_rad=max_joint_correction_rad,
        iterations=iterations, trust_region_rad=trust_region_rad, temporal_weight=temporal_weight,
        corrected_frames=int(np.any(np.abs(delta) > 1e-6, axis=1).sum()),
        observed_max_joint_correction_rad=float(np.abs(delta).max(initial=0)),
        correction_rms_rad=float(np.sqrt(np.mean(delta ** 2))),
        optimizer_failures=optimizer_failures, elapsed_s=time.monotonic() - started,
        immutable="root position/orientation, waist, legs, frame count and timestamps",
        dynamic_feasibility="not_evaluated")
