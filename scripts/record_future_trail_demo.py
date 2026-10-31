"""Predict future pose trails with the actor closed-loop inside the world model.

Each video frame seeds a new 32-step prediction from the recorded simulator state
and past observation history. No recorded future actions or future states enter
the prediction; the known motion reference remains the goal.
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
from supermimic.data.packed_library import sha256
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.tracking_env_cfg import ROOT_BODY, TRACKED_BODIES
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Policy
from supermimic.robots.spec import NQ
from supermimic.world_model.heads import AnalyticHeads, analytic_observation, flatten_history_term_major
from supermimic.world_model.world_model import WorldModel


@torch.no_grad()
def rollout_actor(wm, policy, library, state, history, xi, frames, q0, horizon=32):
    """Return H+1 states and H freshly computed policy actions; no teacher forcing."""
    states, actions = [state.clone()], []
    last = library.last_frame_of(frames)
    history = history.clone()
    for offset in range(horizon):
        goal = library.goal_window(torch.minimum(frames+offset, last), 10)
        action = policy(flatten_history_term_major(history), goal)
        state = wm.step(state, action, xi)
        observation = analytic_observation(state, action, xi, q0)
        history = torch.cat([history[1:], observation[None]], 0)
        states.append(state.clone()); actions.append(action)
    return torch.stack(states), torch.stack(actions)


@torch.no_grad()
def main(args):
    source, out = args.source, args.output
    out.mkdir(parents=True, exist_ok=True)
    artifact = json.loads((source/"checkpoint.json").read_text())
    if sha256(source/"models_0080000.pt") != artifact["models_sha256"]:
        raise ValueError("Checkpoint changed")
    ck = load_checkpoint(source/"models_0080000.pt")
    cfg = apply_config(TrainConfig(),ck["config"])
    assert cfg.env.paper_protocol and not cfg.env.anchor_obs
    d = np.load(source/"trajectories.npz",allow_pickle=False)
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(source/"motions.pt")],num_envs=len(d["names"]),
        action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,
        record_transitions=False,compact_library=True,paper_protocol=True,seed=0),device="cuda:0")
    try:
        robot = env.scene["robot"]
        lib = env.command_manager.get_term("motion").library
        q0 = robot.data.default_joint_pos[0].clone()
        heads = AnalyticHeads(G1Kinematics(load_g1_model(),device="cuda:0"),TRACKED_BODIES,ROOT_BODY,ROOT_BODY,
                              robot.data.soft_joint_pos_limits[0].clone(),q0)
        wm = WorldModel(cfg.world_model,heads,"cuda:0");wm.load_state_dict(ck["wm"]);wm.eval()
        policy = Policy(cfg.policy).cuda();policy.load_state_dict(ck["policy"]);policy.eval()
        states = torch.tensor(d["states"],device="cuda:0")
        actions = torch.tensor(d["actions"],device="cuda:0")
        contexts = torch.tensor(d["contexts"],device="cuda:0")
        prev = torch.cat([torch.zeros_like(actions[:1]),actions],0)
        observations = analytic_observation(states,prev,torch.cat([contexts,contexts[-1:]],0),q0)
        entries = [(i,t) for i,steps in enumerate(d["steps"]) for t in range(0,int(steps)+1,2)]
        trail_offsets = np.arange(4,33,4)
        trails, counts = [], []
        max_action_error, sum_sq, n_values = 0., 0., 0
        for start in range(0,len(entries),128):
            batch = entries[start:start+128]
            ids = torch.tensor([i for i,t in batch],device="cuda:0")
            times = torch.tensor([t for i,t in batch],device="cuda:0")
            hist_indices = (times[None]+torch.arange(-9,1,device="cuda:0")[:,None]).clamp_min(0)
            hist = observations[hist_indices,ids[None]]
            xi = contexts[times.clamp_max(len(contexts)-1),ids]
            frames = lib.clip_start[ids]+times
            predicted, predicted_actions = rollout_actor(wm,policy,lib,states[times,ids],hist,xi,frames,q0)
            still_active = times < torch.as_tensor(d["steps"],device="cuda:0")[ids]
            if still_active.any():
                error = predicted_actions[0,still_active]-actions[times[still_active],ids[still_active]]
                max_action_error=max(max_action_error,float(error.abs().max()))
                sum_sq += float((error*error).sum());n_values+=error.numel()
            qpos = predicted[torch.tensor(trail_offsets,device="cuda:0"),:,:NQ].permute(1,0,2).cpu().numpy()
            if not np.isfinite(qpos).all():
                raise ValueError("Nonfinite future pose in 32-step model rollout")
            trails.extend(qpos)
            for i,t in batch:
                # At the final reached state, including a terminal state, no future ghosts.
                remaining = int(d["reference_frames"][i])-1-t if t < int(d["steps"][i]) else 0
                counts.append(int(np.sum(trail_offsets<=remaining)))
            print(f"Predicted {min(start+128,len(entries))}/{len(entries)} video frames",flush=True)
        # Analytic reconstruction should agree with the recorded simulator actor inputs.
        if max_action_error > .01:
            raise ValueError(f"History reconstruction differs too much: {max_action_error}")
        np.savez_compressed(out/"future_trails.npz", entries=np.array(entries),
            qpos=np.array(trails),visible_counts=np.array(counts),offset_steps=trail_offsets)
        report = dict(source=str(source),checkpoint_iteration=80000,checkpoint_sha256=artifact["models_sha256"],
            actual_trajectory_sha256=sha256(source/"trajectories.npz"),frames=len(entries),horizon_steps=32,
            horizon_s=.64,ghost_offsets_s=(trail_offsets/50).tolist(),
            mode="Policy on: actor recomputes actions on predicted-state histories at each model step",
            reseeding="New future prediction at each displayed simulator frame; no real-state correction inside the 32-step rollout",
            inputs="Current actual state, past observation history, known motion reference, simulation context; no future actual actions/states",
            start_action_max_abs_difference=max_action_error,start_action_rmse=(sum_sq/max(n_values,1))**.5,
            style_reference="https://supermimic.github.io/#closer",
            differences="Native23 checkpoint, 0.64s chosen horizon, own motion examples; Policy on only, no policy-off/slippery-floor experiment")
        (out/"prediction.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
    finally:
        env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path("reports/demo_80k_world_model_20261030"))
    p.add_argument("--output",type=Path,default=Path("reports/demo_80k_future_trail_20261030"))
    main(p.parse_args())
