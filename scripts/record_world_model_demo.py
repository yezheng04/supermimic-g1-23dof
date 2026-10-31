"""Record simulator execution and action-conditioned world-model predictions.

Both prediction modes use exactly the simulator's executed actions and context.
The 32-step mode reseeds its state every 32 steps; the continuous mode uses only
the initial simulator state. Neither predicted trajectory is the reference motion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.checkpoints import load_checkpoint
from supermimic.cli import apply_config
from supermimic.config import TrainConfig
from supermimic.data.motion_convert import load_g1_model
from supermimic.data.packed_library import load_pack, sha256, validate_pack
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.sim_context import SimContext
from supermimic.envs.state_io import StateIO
from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX, ROOT_BODY, TRACKED_BODIES
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Policy
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel


CLIPS = [
    ("230125__walk_ff_loop_315_001__A146", "行走"),
    ("230906__jog_ff_loop_180_R_slow_002__A448", "慢跑"),
    ("230524__high_jump_R_001__A395", "跳跃"),
    ("230413__dance_hiphop_prep_R_fast_002__A317", "街舞"),
    ("230509__push_up_loop_002__A359", "俯卧撑"),
    ("230120__crawl_ff_loop_270_001__A133", "爬行"),
    ("230509__inchworm_001__A359", "尺蠖式动作"),
    ("230509__inchworm_001__A361", "尺蠖式动作 · 失败诊断"),
]


class TerminalCapture:
    """Save reached terminal states before mjlab automatically resets them."""
    def __init__(self, io):
        self.io = io
        self.states = io.read()

    def record_pre_reset(self, ids):
        self.states[ids] = self.io.read()[ids]

    def record_post_reset(self, ids):
        pass

    def record_post_step(self):
        pass


@torch.no_grad()
def predict(wm, states, actions, contexts, window=32):
    if states.shape[0] != actions.shape[0] + 1 or contexts.shape[:2] != actions.shape[:2]:
        raise ValueError("One state more than transitions is required")
    short = states[0].clone()
    continuous = states[0].clone()
    result_short, result_cont = [short.clone()], [continuous.clone()]
    valid = torch.ones(states.shape[1], dtype=torch.bool, device=states.device)
    for t, action in enumerate(actions):
        if t % window == 0:
            short = states[t].clone()
        short = wm.step(short, action, contexts[t])
        continuous = wm.step(continuous, action, contexts[t])
        valid &= torch.isfinite(continuous).all(-1) & (continuous.abs().amax(-1) < 1e6)
        result_short.append(short.clone())
        result_cont.append(torch.where(valid[:, None], continuous, torch.full_like(continuous, float("nan"))))
        # Invalid trajectories stay marked invalid; no hidden reseeding of the output.
        continuous = torch.where(valid[:, None], continuous, torch.zeros_like(continuous))
    return torch.stack(result_short), torch.stack(result_cont)


@torch.no_grad()
def main(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    artifact = json.loads((out / "checkpoint.json").read_text())
    checkpoint = out / "models_0080000.pt"
    if sha256(checkpoint) != artifact["models_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    ck = load_checkpoint(checkpoint)
    assert ck["iteration"] == 80000
    cfg = apply_config(TrainConfig(), ck["config"])
    assert cfg.env.paper_protocol and not cfg.env.anchor_obs
    source = load_pack(args.library)
    index = {name: i for i, name in enumerate(source["names"])}
    clips = json.loads(args.clips_json.read_text()) if args.clips_json else CLIPS
    if not clips or any(len(clip) != 2 for clip in clips):
        raise ValueError("Clip selection must contain [name, title] pairs")
    names, titles = map(list, zip(*clips))
    chunks, starts, lengths = [], [], []
    for name in names:
        i = index[name]
        start, length = int(source["starts"][i]), int(source["lengths"][i])
        starts.append(sum(lengths)); lengths.append(length)
        chunks.append(source["ref"][start:start+length])
    pack = {**source, "names": names, "starts": starts, "lengths": lengths,
            "ref": torch.cat(chunks), "manifest_sha256": None}
    validate_pack(pack)
    torch.save(pack, out / "motions.pt")
    del source, chunks, pack

    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(out / "motions.pt")], num_envs=len(names),
        future_frames=cfg.env.future_frames, history_length=cfg.policy.history,
        episode_length_s=1e5, action_noise_std=0., obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode="start", record_transitions=False,
        seed=0, anchor_obs=False, paper_protocol=True, compact_library=True), device=args.device)
    try:
        robot = env.scene["robot"]
        io = StateIO(env)
        ctx = SimContext(env, FOOT_GEOM_REGEX)
        capture = TerminalCapture(io)
        env.recorder_manager.terms["demo_terminal"] = capture
        cmd = env.command_manager.get_term("motion")
        cmd.forced_frame[:] = cmd.library.clip_start
        obs, _ = env.reset()
        policy = Policy(cfg.policy).to(args.device)
        policy.load_state_dict(ck["policy"]); policy.eval()
        fk = G1Kinematics(load_g1_model(), device=args.device)
        heads = AnalyticHeads(fk, TRACKED_BODIES, ROOT_BODY, ROOT_BODY,
            robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        wm = WorldModel(cfg.world_model, heads, args.device)
        wm.load_state_dict(ck["wm"]); wm.eval()
        del ck
        alive = torch.ones(len(names), dtype=torch.bool, device=args.device)
        steps = torch.zeros(len(names), dtype=torch.long, device=args.device)
        returns = torch.zeros(len(names), device=args.device)
        failed = torch.zeros_like(alive)
        reasons = [None] * len(names)
        states, actions, contexts = [io.read()], [], []
        while alive.any():
            action = policy(obs["policy"], obs["goal"])
            contexts.append(ctx.read())
            actions.append(action.clone())  # zero execution noise: equals a_exec
            obs, reward, term, timeout, _ = env.step(action)
            reached = io.read()
            done = term | timeout
            reached[done] = capture.states[done]
            reached[~alive] = states[-1][~alive]
            states.append(reached)
            returns += reward * alive
            steps += alive.long()
            for e in (done & alive).nonzero().flatten().tolist():
                failed[e] = term[e]
                reasons[e] = "+".join(k for k, v in env.failure_masks.items() if bool(v[e])) or None
            alive &= ~done
            if len(actions) > max(lengths):
                raise RuntimeError("Playback exceeded full clip length")
        states, actions, contexts = torch.stack(states), torch.stack(actions), torch.stack(contexts)
        short, continuous = predict(wm, states, actions, contexts)
        e_short = wm.spec.normalized_sq_error(short, states)
        e_cont = wm.spec.normalized_sq_error(continuous, states)
        np.savez_compressed(out / "trajectories.npz", names=np.array(names), titles=np.array(titles),
            fps=50, states=states.cpu().numpy(), actions=actions.cpu().numpy(), contexts=contexts.cpu().numpy(),
            wm_32=short.cpu().numpy(), wm_continuous=continuous.cpu().numpy(),
            nse_32=e_short.cpu().numpy(), nse_continuous=e_cont.cpu().numpy(),
            steps=steps.cpu().numpy(), reference_frames=np.array(lengths), failed=failed.cpu().numpy())
        clips = []
        for i, name in enumerate(names):
            n = int(steps[i])
            row = dict(name=name, title=titles[i], steps=n, seconds=n/50, reference_frames=lengths[i],
                failed=bool(failed[i]), failure_reason=reasons[i],
                full_clip_norm_return=float(returns[i])/(.1*(lengths[i]-1)))
            for label, pred in (("32", short), ("continuous", continuous)):
                err = (pred[1:n+1, i, :3] - states[1:n+1, i, :3]).norm(dim=-1)
                valid = torch.isfinite(err)
                row[f"wm_{label}_root_mean_cm"] = float(err[valid].mean()*100) if valid.any() else None
                row[f"wm_{label}_root_last_cm"] = float(err[-1]*100) if valid[-1] else None
                row[f"wm_{label}_invalid_frames"] = int((~valid).sum())
            clips.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        report = dict(checkpoint=artifact, model_sha256=load_pack(out / "motions.pt")["model_sha256"],
            input_library_sha256=sha256(args.library), selection=f"{len(names)} named examples; not a statistical benchmark; failures retained",
            protocol="Simulator closes the policy loop; both world-model modes replay its executed actions and context, no policy recomputation on predicted states",
            prediction_32="Autoregressive 32-step windows, reseed from reached simulator state every 32 steps (0.64 s)",
            prediction_continuous="Seed only at frame zero, no later ground-truth states, no spatial realignment",
            terminal_states="Captured before auto-reset; playback stops on failure or clip end",
            simulation="MuJoCo Warp native23, 50 Hz, friction 1.0, no noise or domain randomization",
            rendering="MuJoCo FK rendering of recorded poses; this is simulation, not physical robot footage",
            clips=clips)
        (out / "recording.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("reports/demo_80k_world_model_20261030"))
    p.add_argument("--library", type=Path, default=Path("reports/evaluation_100k_20261030/library.pt"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--clips-json", type=Path, help="JSON list of [motion name, display title] pairs")
    main(p.parse_args())
