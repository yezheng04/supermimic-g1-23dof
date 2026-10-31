"""Fixed flip coverage and suffix-from-reference diagnostics, without training changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from supermimic.checkpoints import load_checkpoint
from supermimic.data.packed_library import load_pack, sha256, validate_pack
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.state_io import StateIO
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots.spec import JOINT_NAMES
from record_world_model_demo import TerminalCapture

TARGET = "230531__flip_360_001__A415"
# Chosen from reference kinematics before inspecting the newer policy's results.
# Labels describe reference pose phases; they do not assert measured contact events.
PHASES = [
    (0, "完整动作起点"), (100, "准备倾身"), (130, "下蹲阶段"),
    (140, "根部上升阶段"), (148, "接近水平阶段"), (157, "接近倒立阶段"),
    (166, "翻转后段"), (172, "根部最低点附近"), (190, "恢复直立阶段"),
    (240, "末段站立"),
]


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prepare(out):
    original = Path("reports/evaluation_100k_20261030")
    selection = json.loads((original / "selection.json").read_text())
    names = selection["groups"]["flip"]
    assert len(names) == 54 and TARGET in names
    source = load_pack(original / "library.pt")
    chunks, starts, lengths = [], [], []
    for name in names:
        i = source["names"].index(name)
        start, length = int(source["starts"][i]), int(source["lengths"][i])
        starts.append(sum(lengths)); lengths.append(length)
        chunks.append(source["ref"][start:start + length])
    pack = {**source, "names": names, "starts": starts, "lengths": lengths,
            "ref": torch.cat(chunks), "manifest_sha256": None}
    validate_pack(pack)
    torch.save(pack, out / "library.pt")
    cases = [dict(kind="full_clip", name=name, start_frame=0, phase="首帧") for name in names]
    cases += [dict(kind="phase", name=TARGET, start_frame=frame, phase=label) for frame, label in PHASES]
    save(out / "protocol.json", dict(source_selection_sha256=sha256(original / "selection.json"),
        library_sha256=sha256(out / "library.pt"), cases=cases, repeats=3, seed=0,
        thresholds=dict(height_m=.25, gravity_z_difference=.8), fps=50,
        initialization="Reference root pose and qvel; reference joints clipped to the existing simulator soft limits, as in training. Standard reset history and previous action; no reconstructed executed history",
        evaluation="Run to original clip end or original failure condition; no noise/domain randomization; friction 1.0",
        normalization="Reward / (0.1 * remaining reference transitions); zero reward for unexecuted suffix",
        caveat="Phase success means suffix completion from a supplied reference state/velocity, not reaching that state from the start or full-motion mastery",
        phase_selection="Manual timestamps from reference root height and gravity z, chosen before new-checkpoint evaluation; not contact-detector labels",
        repeatability="Three separate simulator runs with seed=0; empirical repeats do not establish strict determinism"))


def summarize_rows(rows):
    return dict(cases=len(rows), successful=sum(not r["failed"] for r in rows),
        failed=sum(r["failed"] for r in rows),
        mean_norm_return=float(np.mean([r["norm_return"] for r in rows])),
        mean_completion_fraction=float(np.mean([r["steps"]/r["remaining_steps"] for r in rows])))


def reached_metrics(actual, reference):
    def gz(q):
        q = q / q.norm(dim=-1, keepdim=True)
        return -(1 - 2*(q[..., 1]**2 + q[..., 2]**2))
    return dict(root_height_error_m=abs(float(actual[2]-reference[2])),
        gravity_z_difference=abs(float(gz(actual[3:7])-gz(reference[3:7]))))


@torch.no_grad()
def evaluate(args):
    out = args.output
    protocol = json.loads((out / "protocol.json").read_text())
    if sha256(out / "library.pt") != protocol["library_sha256"]:
        raise ValueError("Frozen library changed")
    checkpoint = load_checkpoint(args.checkpoint)
    iteration = int(checkpoint["iteration"])
    source_hash = checkpoint["source_checkpoint_sha256"]
    del checkpoint
    if (out / f"eval_{iteration:07d}_r{args.repeat}.json").exists():
        raise FileExistsError("Use a new repeat index to preserve completed evidence")
    policy, cfg = load_policy_from_checkpoint(args.checkpoint, args.device)
    assert cfg["env"]["paper_protocol"] and not cfg["env"]["anchor_obs"]
    cases = protocol["cases"]
    started = time.monotonic()
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(out / "library.pt")], num_envs=len(cases),
        future_frames=cfg["env"]["future_frames"], history_length=cfg["policy"]["history"],
        episode_length_s=1e5, action_noise_std=0., obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode="start", record_transitions=False,
        seed=0, anchor_obs=False, paper_protocol=True, compact_library=True,
        root_height_threshold=.25, root_tilt_threshold=.8), device=args.device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        names = [c.name for c in lib.clips]
        ids = torch.tensor([names.index(c["name"]) for c in cases], device=args.device)
        offsets = torch.tensor([c["start_frame"] for c in cases], device=args.device)
        frames = lib.clip_start[ids] + offsets
        remaining = lib.last_frame_of(frames) - frames
        if (remaining < 1).any():
            raise ValueError("Start must precede clip end")
        cmd.forced_frame[:] = frames
        io = StateIO(env)
        capture = TerminalCapture(io)
        env.recorder_manager.terms["diagnostic_terminal"] = capture
        obs, _ = env.reset()
        expected_qpos, expected_qvel = lib.ref_qpos_qvel(frames)
        state0 = io.read()
        limits = env.scene["robot"].data.soft_joint_pos_limits
        reference_joints = expected_qpos[:, 7:].clone()
        expected_qpos[:, 7:] = reference_joints.clamp(limits[:, :, 0], limits[:, :, 1])
        clamp_delta = expected_qpos[:, 7:] - reference_joints
        initial_error = (state0 - torch.cat([expected_qpos, expected_qvel], -1)).abs().amax(-1)
        if float(initial_error.max()) > 1e-4:
            expected = torch.cat([expected_qpos, expected_qvel], -1)
            differences = (state0 - expected).abs()
            details = [dict(case=cases[i], column=j, actual=float(state0[i,j]), expected=float(expected[i,j]))
                       for i,j in (differences > 1e-4).nonzero().tolist()]
            save(out / "initialization_mismatch.json", details)
            raise ValueError(f"Reference state/velocity reset mismatch: {float(initial_error.max())}")
        alive = torch.ones(len(cases), dtype=torch.bool, device=args.device)
        steps = torch.zeros(len(cases), dtype=torch.long, device=args.device)
        returns = torch.zeros(len(cases), device=args.device)
        results = [None]*len(cases)
        states, actions = [state0], []
        while alive.any():
            action = policy(obs["policy"], obs["goal"])
            obs, reward, terminated, timeout, _ = env.step(action)
            reached = io.read()
            done = terminated | timeout
            reached[done] = capture.states[done]
            reached[~alive] = states[-1][~alive]
            states.append(reached)
            actions.append(action)
            steps += alive.long()
            returns += reward*alive
            for i in (done & alive).nonzero().flatten().tolist():
                n, rem = int(steps[i]), int(remaining[i])
                if n > rem or (not bool(terminated[i]) and n != rem):
                    raise ValueError("Early timeout or reference boundary mismatch")
                ref = lib.reference_state(frames[i:i+1] + n)[0]
                results[i] = dict(**cases[i], iteration=iteration, repeat=args.repeat,
                    start_seconds=cases[i]["start_frame"]/50, remaining_steps=rem,
                    steps=n, elapsed_seconds=n/50, end_reference_seconds=(cases[i]["start_frame"]+n)/50,
                    failed=bool(terminated[i]),
                    failure_reason="+".join(k for k, v in env.failure_masks.items() if bool(v[i])) or None,
                    norm_return=float(returns[i])/(.1*rem), initial_state_max_abs_error=float(initial_error[i]),
                    initial_joint_clamp_max_deg=float(clamp_delta[i].abs().max()*180/torch.pi),
                    initial_joint_clamps=[dict(name=JOINT_NAMES[j], delta_rad=float(clamp_delta[i,j]))
                                          for j in (clamp_delta[i].abs()>1e-6).nonzero().flatten().tolist()],
                    **reached_metrics(reached[i], ref))
            alive &= ~done
            if len(actions) > int(remaining.max()):
                raise ValueError("Exceeded complete suffix duration")
        np.savez_compressed(out / f"trajectories_{iteration:07d}_r{args.repeat}.npz",
            states=torch.stack(states).cpu().numpy(), actions=torch.stack(actions).cpu().numpy(),
            steps=steps.cpu().numpy(), start_frames=offsets.cpu().numpy(),
            names=np.array([c["name"] for c in cases]))
        full = [r for r in results if r["kind"] == "full_clip"]
        phases = [r for r in results if r["kind"] == "phase"]
        report = dict(iteration=iteration, repeat=args.repeat, elapsed_s=time.monotonic()-started,
            source_checkpoint_sha256=source_hash, policy_sha256=sha256(args.checkpoint),
            protocol_sha256=sha256(out / "protocol.json"), summary=summarize_rows(full),
            full_clips=full, phases=phases)
        save(out / f"eval_{iteration:07d}_r{args.repeat}.json", report)
        print(json.dumps(dict(iteration=iteration, repeat=args.repeat, summary=report["summary"], phases=phases),
                         ensure_ascii=False, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--output", type=Path, default=Path("reports/flip_diagnostics_128k_20261030"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    if args.command == "prepare":
        if (args.output / "protocol.json").exists():
            raise FileExistsError("Use existing frozen protocol")
        prepare(args.output)
    else:
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        evaluate(args)
