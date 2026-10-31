"""CSV (G1 retarget) -> mjlab-compatible motion npz, batch friendly.

Conventions mirror ``mjlab/scripts/csv_to_npz.py`` (commit 8bb2bc9) exactly, but
everything is vectorised and the forward kinematics run on CPU MuJoCo instead of a
GPU Simulation so that hundreds of thousands of clips can be converted in parallel.

CSV layout (no header, one row per frame):
  cols 0:3   root position (m)
  cols 3:7   root quaternion in **xyzw** order (converted to wxyz here)
  cols 7:30  23 joint angles in :data:`G1_JOINT_NAMES` order

Output npz keys (same as mjlab ``MotionLoader`` expects):
  fps, joint_pos (T,23), joint_vel (T,23), body_pos_w (T,24,3), body_quat_w (T,24,4) wxyz,
  body_lin_vel_w (T,24,3), body_ang_vel_w (T,24,3)
plus extras: joint_names, body_names, source (str), input_fps.

Velocity definitions (as in mjlab):
  * joint_vel and root linear velocity: ``np.gradient`` of the resampled signal.
  * root angular velocity: central SO(3) difference of consecutive quaternions,
    expressed in the **world** frame, first/last sample repeated.
  * body link velocities: from MuJoCo ``cvel`` with
    ``lin_w = cvel_lin - ang x (subtree_com[root] - xpos)``, ``ang_w = cvel_ang``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from supermimic.robots.spec import JOINT_NAMES as G1_JOINT_NAMES
NUM_JOINTS = len(G1_JOINT_NAMES)


def g1_xml_path() -> Path:
    return Path(__file__).parents[1] / "robots/g1_23dof.xml"


def load_g1_model(xml_path: str | Path | None = None) -> mujoco.MjModel:
    """Standalone CPU MjModel of the G1. Asserts the joint order matches the CSV order."""
    model = mujoco.MjModel.from_xml_path(str(xml_path or g1_xml_path()))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "first joint must be the free joint"
    assert tuple(names[1:]) == G1_JOINT_NAMES, f"joint order mismatch: {names[1:]}"
    assert model.nq == 7 + NUM_JOINTS and model.nv == 6 + NUM_JOINTS
    return model


# ----------------------------------------------------------------------------- quaternions
# All quaternions below are wxyz (MuJoCo / mjlab convention).


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [1, 2, 3, 0]]


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    """Rotation vector (angle * axis) of wxyz quaternions, numerically stable near identity."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1)
    angle = 2.0 * np.arctan2(s, q[..., 0])
    scale = np.where(s > 1e-9, angle / np.where(s > 1e-9, s, 1.0), 2.0)
    return v * scale[..., None]


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vectors into the body frame: R(q)^T v."""
    r = Rotation.from_quat(quat_wxyz_to_xyzw(q))
    return r.inv().apply(v)


# ----------------------------------------------------------------------------- resampling


@dataclass
class ResampledMotion:
    fps: float
    root_pos: np.ndarray  # (T,3)
    root_quat: np.ndarray  # (T,4) wxyz
    joint_pos: np.ndarray  # (T,23)
    root_lin_vel_w: np.ndarray  # (T,3) world
    root_ang_vel_w: np.ndarray  # (T,3) world
    joint_vel: np.ndarray  # (T,23)

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])


BONES_HEADER_PREFIX = "Frame,root_translateX"


def csv_format(path: str | Path) -> str:
    """'lafan1' (no header: pos m, quat xyzw, joints rad) or 'bones' (header: Frame, pos cm, Euler xyz deg, joints deg)."""
    with open(path) as f:
        first = f.readline()
    return "bones" if first.startswith(BONES_HEADER_PREFIX) else "lafan1"


def load_csv(path: str | Path, project_29: bool = False, joint_names=None) -> np.ndarray:
    """CSV positions in m, xyzw quaternions, rad hinges; BONES units converted explicitly.

    Headerless inputs must have the documented native joint order. A 29->23 projection
    is opt-in: it is not IK retargeting and does not preserve end-effector positions.
    """
    import csv
    if csv_format(path) == "bones":
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            rows = list(reader)
        names = [c.removesuffix("_dof") for c in cols if c.endswith("_joint_dof")]
        get = lambda ns: np.array([[float(row[n]) for n in ns] for row in rows])
        pos = get(["root_translateX", "root_translateY", "root_translateZ"]) / 100
        quat = Rotation.from_euler("xyz", get(["root_rotateX", "root_rotateY", "root_rotateZ"]), degrees=True).as_quat()
        arr = np.concatenate([pos, quat, np.deg2rad(get([n+"_dof" for n in names]))], axis=1)
        frame_ids = get(["Frame"]).ravel()
        if len(frame_ids) > 1 and not np.allclose(np.diff(frame_ids), 1):
            raise ValueError("BONES Frame column is not contiguous; supply explicit timestamps via raw NPZ")
    else:
        arr = np.loadtxt(path, delimiter=",", ndmin=2)
        names = joint_names
        if names is None:
            if arr.shape[1] == 30:
                names = G1_JOINT_NAMES
            elif arr.shape[1] == 36 and project_29:
                names = G1_29_JOINT_NAMES
            else:
                raise ValueError(f"{path}: expected 30 columns; 29-DoF projection requires --project-29")
    return map_joints(arr, names, project_29)


G1_29_JOINT_NAMES = (
    *G1_JOINT_NAMES[:13], "waist_roll_joint", "waist_pitch_joint",
    *G1_JOINT_NAMES[13:18], "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    *G1_JOINT_NAMES[18:], "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def map_joints(raw, names, allow_projection=False):
    names = tuple(str(n) for n in names)
    if len(set(names)) != len(names) or raw.shape[1] != 7 + len(names):
        raise ValueError("Invalid or duplicate joint names / column count")
    missing = set(G1_JOINT_NAMES) - set(names)
    extra = set(names) - set(G1_JOINT_NAMES)
    if missing or (extra and not allow_projection):
        raise ValueError(f"Joint schema mismatch: missing={missing}, extra={extra}")
    return np.concatenate([raw[:, :7], raw[:, [7 + names.index(n) for n in G1_JOINT_NAMES]]], axis=1)


def resample_motion(raw: np.ndarray, input_fps: float | None, output_fps: float = 50.0,
                    timestamps: np.ndarray | None = None) -> ResampledMotion:
    """Interpolate by physical time, never by rounded frame ratio or stretched linspace.

    Output is t=k/fps <= source duration. An off-grid tail shorter than dt is
    reported, not appended as an irregular timestep. Input positions are m and rad;
    input quaternion xyzw; result quaternion wxyz. No guessed phase/heading offset.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 30 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError("Expected at least two finite native23 frames with 30 columns")
    if not np.isfinite(output_fps) or output_fps <= 0:
        raise ValueError("output_fps must be positive and finite")
    if timestamps is None:
        if input_fps is None or not np.isfinite(input_fps) or input_fps <= 0:
            raise ValueError("Source FPS is required; it cannot be inferred from frame count")
        ts = np.arange(len(raw), dtype=np.float64) / input_fps
    else:
        ts = np.asarray(timestamps, dtype=np.float64)
        if ts.shape != (len(raw),) or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
            raise ValueError("timestamps must be finite and strictly increasing, one per frame")
        ts = ts - ts[0]
        if input_fps is not None and not np.allclose(np.diff(ts), 1/input_fps, rtol=1e-5, atol=1e-8):
            raise ValueError("Source FPS conflicts with timestamps; omit FPS for irregular timestamps")
    dt = 1/output_fps
    times = np.arange(math.floor(ts[-1] * output_fps + 1e-9) + 1) / output_fps
    if len(times) < 2:
        raise ValueError("Clip duration must cover at least one output interval")
    interp = lambda x: np.stack([np.interp(times, ts, x[:, i]) for i in range(x.shape[1])], axis=-1)
    pos, joints = interp(raw[:, :3]), interp(raw[:, 7:])
    q = raw[:, 3:7]
    norms = np.linalg.norm(q, axis=-1)
    if np.any(norms < 1e-8):
        raise ValueError("Zero root quaternion")
    q = q / norms[:, None]
    # Slerp handles q/-q and shortest-path rotations across +/-pi.
    quat = Slerp(ts, Rotation.from_quat(q))(np.minimum(times, ts[-1])).as_quat()[:, [3,0,1,2]]
    for i in range(1, len(quat)):
        if quat[i] @ quat[i-1] < 0:
            quat[i] *= -1
    edge = 2 if len(times) >= 3 else 1
    return ResampledMotion(output_fps, pos, quat, joints,
                           np.gradient(pos, dt, axis=0, edge_order=edge),
                           _so3_derivative(quat, dt),
                           np.gradient(joints, dt, axis=0, edge_order=edge))


def _so3_derivative(q: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocity: one-sided endpoints, centered interior SO(3) log."""
    n = len(q)
    if n < 2:
        return np.zeros((n, 3))
    out = np.empty((n, 3))
    out[0] = rotvec_from_quat(quat_mul(q[1], quat_conjugate(q[0]))) / dt
    out[-1] = rotvec_from_quat(quat_mul(q[-1], quat_conjugate(q[-2]))) / dt
    if n > 2:
        out[1:-1] = rotvec_from_quat(quat_mul(q[2:], quat_conjugate(q[:-2]))) / (2*dt)
    return out


# ----------------------------------------------------------------------------- forward kinematics


@dataclass
class BodyStates:
    body_names: list[str]
    pos_w: np.ndarray  # (T, nb, 3) link frame origin
    quat_w: np.ndarray  # (T, nb, 4) wxyz
    lin_vel_w: np.ndarray  # (T, nb, 3) link frame origin velocity, world
    ang_vel_w: np.ndarray  # (T, nb, 3) world


def forward_kinematics(model: mujoco.MjModel, motion: ResampledMotion) -> BodyStates:
    """Body link poses and velocities for every frame via CPU MuJoCo (no contacts)."""
    data = mujoco.MjData(model)
    T = motion.num_frames
    nb = model.nbody - 1  # exclude world
    pos_w = np.empty((T, nb, 3))
    quat_w = np.empty((T, nb, 4))
    lin_w = np.empty((T, nb, 3))
    ang_w = np.empty((T, nb, 3))
    root_id = model.jnt_bodyid[0]

    # Free-joint qvel: world linear velocity, **body-frame** angular velocity.
    ang_vel_b = quat_rotate_inverse(motion.root_quat, motion.root_ang_vel_w)

    qpos = np.zeros(model.nq)
    qvel = np.zeros(model.nv)
    for t in range(T):
        qpos[0:3] = motion.root_pos[t]
        qpos[3:7] = motion.root_quat[t]
        qpos[7:] = motion.joint_pos[t]
        qvel[0:3] = motion.root_lin_vel_w[t]
        qvel[3:6] = ang_vel_b[t]
        qvel[6:] = motion.joint_vel[t]
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        pos_w[t] = data.xpos[1:]
        quat_w[t] = data.xquat[1:]
        cvel = data.cvel[1:]
        offset = data.subtree_com[root_id][None, :] - data.xpos[1:]
        lin_w[t] = cvel[:, 3:6] - np.cross(cvel[:, 0:3], offset)
        ang_w[t] = cvel[:, 0:3]

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    return BodyStates(names, pos_w, quat_w, lin_w, ang_w)


# ----------------------------------------------------------------------------- end to end


def save_reference(raw, out_path, input_fps=None, output_fps=50., timestamps=None,
                   source="", model=None, extra_metadata=None):
    import hashlib
    import json
    from supermimic.data.quality import quality_report
    model = model or load_g1_model()
    motion = resample_motion(raw, input_fps, output_fps, timestamps)
    bodies = forward_kinematics(model, motion)
    source_duration = float(timestamps[-1]-timestamps[0]) if timestamps is not None else (len(raw)-1)/input_fps
    report = quality_report(model, motion, bodies)
    metadata = {
        "schema_version": 1, "robot": "g1_23dof", "source": str(source),
        "input_fps": input_fps, "output_fps": output_fps, "input_frames": len(raw),
        "output_frames": motion.num_frames, "source_duration_s": source_duration,
        "duration_s": (motion.num_frames-1)/output_fps,
        "dropped_tail_s": max(0., source_duration-(motion.num_frames-1)/output_fps),
        "source_start_s": float(timestamps[0]) if timestamps is not None else 0.,
        "quaternion_order": "wxyz", "position_units": "m", "joint_units": "rad",
        "velocity_frame": "world", "reference_phase": "state_t_matches_reference_t",
        "model_sha256": hashlib.sha256(g1_xml_path().read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
        "quality": report, **(extra_metadata or {}),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, fps=np.array([output_fps]),
        timestamps=np.arange(motion.num_frames, dtype=np.float64)/output_fps,
        joint_pos=motion.joint_pos.astype(np.float32), joint_vel=motion.joint_vel.astype(np.float32),
        body_pos_w=bodies.pos_w.astype(np.float32), body_quat_w=bodies.quat_w.astype(np.float32),
        body_lin_vel_w=bodies.lin_vel_w.astype(np.float32), body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),
        joint_names=np.array(G1_JOINT_NAMES), body_names=np.array(bodies.body_names),
        metadata_json=np.array(json.dumps(metadata)))
    out_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def convert_csv_to_npz(csv_path, out_path, input_fps, output_fps=50., model=None, project_29=False):
    raw = load_csv(csv_path, project_29=project_29)
    return save_reference(raw, out_path, input_fps, output_fps, source=csv_path, model=model,
                          extra_metadata={"projected_from_29": bool(project_29)})
