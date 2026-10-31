# Portions adapted from SOMA-Retargeter NewtonPipeline and FeetStabilizer.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resident SOMA batch IK with targets and intermediate foot poses on the GPU.

Same objectives, initialisation frames, iteration counts and clamping as the
pinned upstream implementation. Each environment is an independent motion.
"""
import hashlib
import json
from pathlib import Path
import time

import newton
import newton.ik as ik
import numpy as np
import warp as wp
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.animation import ik as ik_utils
from soma_retargeter.assets.bvh import load_bvh
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
from soma_retargeter.utils import io_utils
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str
from supermimic.data.motion_convert import g1_xml_path, G1_JOINT_NAMES
from supermimic.retarget.bvh import read_bvh_timing


@wp.kernel
def gather_target(targets: wp.array3d(dtype=wp.transform), frame: wp.array(dtype=wp.int32),
                  effector: int, positions: wp.array(dtype=wp.vec3), rotations: wp.array(dtype=wp.vec4)):
    env = wp.tid()
    tx = targets[frame[0], env, effector]
    positions[env] = wp.transform_get_translation(tx)
    q = wp.transform_get_rotation(tx)
    rotations[env] = wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel
def ramp_weight(frame: wp.array(dtype=wp.int32), weights: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    out[0] = weights[wp.min(frame[0], weights.shape[0] - 1)]


@wp.kernel
def solve_feet(body_q: wp.array2d(dtype=wp.transform), pelvis: int,
               chains: wp.array2d(dtype=wp.int32), parents: wp.array(dtype=wp.int32),
               hints: wp.array(dtype=wp.int32), hint_offsets: wp.array(dtype=wp.vec3),
               targets: wp.array3d(dtype=wp.transform), frame: wp.array(dtype=wp.int32),
               foot_indices: wp.array(dtype=wp.int32), out: wp.array2d(dtype=wp.transform)):
    env = wp.tid()
    body = body_q[env]
    out[env, 0] = body[pelvis]
    for i in range(chains.shape[0]):
        hint_idx = hints[i]
        hint = wp.vec3(0.0, 0.0, 0.0)
        if hint_idx != -1:
            hint = wp.transform_point(body[hint_idx], hint_offsets[i])
        result = ik_utils.wp_solve_two_bone_ik(1.0, body[parents[i]],
            body[chains[i, 0]], body[chains[i, 1]], body[chains[i, 2]],
            targets[frame[0], env, foot_indices[i]], hint_idx != -1, hint)
        out[env, 1 + i * 3] = result.root
        out[env, 2 + i * 3] = result.mid
        out[env, 3 + i * 3] = result.tip


@wp.kernel
def gather_foot_target(targets: wp.array2d(dtype=wp.transform), effector: int,
                       positions: wp.array(dtype=wp.vec3), rotations: wp.array(dtype=wp.vec4)):
    env = wp.tid()
    tx = targets[env, effector]
    positions[env] = wp.transform_get_translation(tx)
    q = wp.transform_get_rotation(tx)
    rotations[env] = wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel
def save_frame(q: wp.array2d(dtype=wp.float32), frame: wp.array(dtype=wp.int32),
                output: wp.array3d(dtype=wp.float32)):
    env, coord = wp.tid()
    output[frame[0], env, coord] = q[env, coord]


@wp.kernel
def next_frame(frame: wp.array(dtype=wp.int32)):
    frame[0] += 1


class ResidentSoma:
    def __init__(self, calibration_dir):
        self.calibration_dir = Path(calibration_dir).resolve()
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = None
        self.shape = None
        self.setup_count = 0

    def _init_pipeline(self, skeleton):
        self.skeleton = skeleton
        config = io_utils.load_json(io_utils.get_config_file("unitree_g1", "soma_to_g1_23dof_retargeter_config.json"))
        config["mjcf_path"] = str(g1_xml_path().resolve())
        scaler = io_utils.load_json(io_utils.get_config_file(config["human_robot_scaler_config"]))
        for side in ("Left", "Right"):
            scaler["joint_offsets"][side + "Hand"][0][0] -= .084
        scaler_path = self.calibration_dir / "scaler_native23.json"
        scaler_path.write_text(json.dumps(scaler, indent=2))
        config["human_robot_scaler_config"] = str(scaler_path)
        self.config, self.scaler = config, scaler
        self.pipeline = NewtonPipeline(skeleton, "soma", "unitree_g1", retarget_config=config)
        names = [x.rsplit("/", 1)[-1] for x in self.pipeline.robot_builder.joint_label]
        if tuple(n for n in names if n in G1_JOINT_NAMES) != G1_JOINT_NAMES or self.pipeline.robot_builder.joint_coord_count != 30:
            raise ValueError("Unexpected native23 layout")
        self.converter = SpaceConverter(get_facing_direction_type_from_str("Mujoco"))

    def _setup(self, batch, capacity):
        p = self.pipeline
        self.model = p._build_model(batch)
        state = self.model.state()
        self.pos, self.rot, limit, self.smooth = p._create_ik_objectives(batch, self.model, state)
        objectives = [*self.pos, *self.rot, limit, self.smooth]
        self.solver = ik.IKSolver(model=p.ik_model, n_problems=batch, objectives=objectives,
            lambda_initial=.1, jacobian_mode=ik.IKJacobianType.ANALYTIC)
        self.q = wp.empty((batch, 30), dtype=wp.float32)
        self.initial_q = self.model.joint_q.numpy().reshape(batch, 30).copy()
        wp.copy(self.q, self.model.joint_q)
        p.feet_stabilizer.setup_num_envs(batch)
        self.targets = wp.empty((capacity, batch, len(self.pos)), dtype=wp.transform)
        self.output = wp.empty((capacity, batch, 30), dtype=wp.float32)
        self.frame = wp.zeros(1, dtype=wp.int32)
        self.foot_indices = wp.array(p.feet_effector_indices, dtype=wp.int32)
        warmup = p.num_initialization_frames + p.num_stabilization_frames
        self.weights = wp.array(np.array([p.smooth_joint_filter_weight * (i / float(warmup))
            for i in range(warmup + 1)], dtype=np.float32))
        self.shape = (batch, capacity)
        self.solver.reset()
        if not wp.get_device().is_cuda:
            raise RuntimeError("Resident batch retargeter requires CUDA")
        with wp.ScopedCapture() as capture:
            self._step()
        self.graph = capture.graph
        self.setup_count += 1

    def _step(self):
        p = self.pipeline
        batch = self.shape[0]
        wp.launch(ramp_weight, dim=1, inputs=[self.frame, self.weights, self.smooth._weight])
        for i, (pos, rot) in enumerate(zip(self.pos, self.rot)):
            wp.launch(gather_target, dim=batch, inputs=[self.targets, self.frame, i, pos.target_positions, rot.target_rotations])
        self.solver.step(self.q, self.q, iterations=p.ik_iterations)
        fs = p.feet_stabilizer
        fs.reset_state(self.q)
        wp.launch(solve_feet, dim=batch, inputs=[fs.state.body_q.reshape((batch, fs.num_body_count)),
            fs.pelvis_idx, fs.two_bone_ik_chains, fs.two_bone_ik_chain_parent,
            fs.two_bone_ik_hint_references, fs.two_bone_ik_hint_offsets,
            self.targets, self.frame, self.foot_indices, fs.out_effectors])
        for i, (pos, rot) in enumerate(zip(fs.position_objectives, fs.rotation_objectives)):
            wp.launch(gather_foot_target, dim=batch, inputs=[fs.out_effectors, i, pos.target_positions, rot.target_rotations])
        fs.ik_solver.step(fs.joint_q, fs.joint_q, iterations=fs.ik_iterations)
        result = p.joint_limit_clamper.apply(fs.current_state())
        wp.launch(save_frame, dim=(batch, 30), inputs=[result, self.frame, self.output])
        wp.launch(next_frame, dim=1, inputs=[self.frame])

    def retarget(self, sources):
        start = time.monotonic()
        buffers, timing = [], []
        for source in sources:
            skeleton, animation = load_bvh(str(source))
            n, fps = read_bvh_timing(source)
            if n != animation.num_frames:
                raise ValueError(f"Frame count mismatch: {source}")
            if self.pipeline is None:
                self._init_pipeline(skeleton)
            if (skeleton.joint_names != self.skeleton.joint_names
                or not np.array_equal(skeleton.parent_indices, self.skeleton.parent_indices)):
                raise ValueError(f"Incompatible uniform skeleton: {source}")
            # Some uniform BVHs have different reference offsets. Keep each
            # source's offsets and fully decoded local transforms, including
            # the reference pose used for initialization of unmapped joints.
            buffers.append(AnimationBuffer(skeleton, n, fps, animation.local_transforms))
            timing.append((n, fps))
        loaded = time.monotonic()
        p = self.pipeline
        p.clear()
        for buffer in buffers:
            # Scaler arrays depend only on the validated topology and config;
            # its identity check must point at this clip's original skeleton.
            p.human_robot_scaler.skeleton = buffer.skeleton
            p.add_input_motions([buffer], [self.converter.transform(wp.transform_identity())], True)
        target_end = time.monotonic()
        batch, steps = len(buffers), p.max_frames
        capacity = 1 << (steps - 1).bit_length()
        if self.shape is None or self.shape[0] != batch or self.shape[1] < steps:
            self._setup(batch, capacity)
        # Every clip restarts from the same model pose and solver state. Short
        # clips hold their final target only in padding, never in saved output.
        host = np.empty((self.shape[1], batch, len(self.pos), 7), dtype=np.float32)
        for env, targets in enumerate(p.input_targets):
            host[:len(targets), env] = targets
            host[len(targets):, env] = targets[-1]
        self.targets.assign(host)
        self.q.assign(self.initial_q)
        self.frame.zero_()
        self.solver.reset()
        p.feet_stabilizer.ik_solver.reset()
        setup_end = time.monotonic()
        for _ in range(steps):
            wp.capture_launch(self.graph)
        wp.synchronize()
        solved = time.monotonic()
        output = self.output.numpy()
        warmup = p.num_initialization_frames + p.num_stabilization_frames
        raws = [output[warmup:warmup+n, env].astype(np.float64) for env, (n, fps) in enumerate(timing)]
        if any(x.shape != (n, 30) or not np.isfinite(x).all() for x, (n, fps) in zip(raws, timing)):
            raise ValueError("Invalid batched IK output")
        elapsed = time.monotonic() - start
        metadata = [dict(pipeline="SOMA Newton IK + feet stabilizer", human_adapter="upstream SOMA human scaler",
            source_sha256=hashlib.sha256(Path(src).read_bytes()).hexdigest(),
            soma_commit="6c0517236d98910a63a57b082c4a585b2776de91", ik_config=self.config,
            human_scaler_config=self.scaler, source_fps=fps, source_frames_used=n,
            hand_frame_shift_m=-.084, hand_rotation_weight_scale=1., feet_stabilizer_enabled=True,
            training_approved=False, execution_backend="resident_gpu_batch", batch_size=batch)
            for src, (n, fps) in zip(sources, timing)]
        stats = dict(batch_size=batch, max_source_frames=steps-warmup,
            source_frames=sum(n for n, fps in timing), padded_utilization=sum(n for n, fps in timing)/(batch*(steps-warmup)),
            load_s=loaded-start, targets_s=target_end-loaded, setup_s=setup_end-target_end,
            gpu_s=solved-setup_end, total_s=elapsed, setup_count=self.setup_count)
        return raws, metadata, stats
