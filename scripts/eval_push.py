"""Native23 adaptation of CuboiLeo's velocity-push diagnostic (§4.3).

The settling thresholds below are reproduction choices, not paper parameters.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.data.motion_convert import G1_JOINT_NAMES, save_reference
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.eval.benchmark import load_policy_from_checkpoint


def make_tpose_clip(out, seconds=20., fps=50):
    raw = np.zeros((round(seconds * fps) + 1, 7 + len(G1_JOINT_NAMES)))
    raw[:, 2], raw[:, 6] = .793, 1.  # root xyzw layout at converter input
    raw[:, 7 + G1_JOINT_NAMES.index("left_shoulder_roll_joint")] = 1.5
    raw[:, 7 + G1_JOINT_NAMES.index("right_shoulder_roll_joint")] = -1.5
    save_reference(raw, out, fps, fps, source="synthetic native23 T-pose push diagnostic")
    return out


@torch.no_grad()
def main(ckpt: Path, speeds: tuple[float, ...] = (0., .5, 1., 2., 4.), num_envs: int = 256,
         device: str = "cuda:0", push_time_s: float = 1., horizon_s: float = 8.,
         seed: int = 0, out: Path | None = None):
    if not 0 <= push_time_s < horizon_s or num_envs < 1 or any(s < 0 for s in speeds):
        raise ValueError("Require 0 <= push_time < horizon, positive env count and nonnegative speeds")
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    rows = []
    with tempfile.TemporaryDirectory(prefix="sm23_push_") as tmp:
        path = make_tpose_clip(Path(tmp) / "tpose.npz", seconds=horizon_s + 1)
        env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(path)], num_envs=num_envs,
            action_noise_std=0., obs_noise=False, reset_noise=False, domain_randomization=False,
            sampling_mode="start", record_transitions=False, episode_length_s=1e5,
            future_frames=cfg["env"]["future_frames"], history_length=cfg["policy"]["history"],
            anchor_obs=cfg["env"].get("anchor_obs", False), paper_protocol=cfg["env"].get("paper_protocol", True),
            compact_library=True, seed=seed), device=device)
        try:
            robot = env.scene["robot"]
            cmd = env.command_manager.get_term("motion")
            dt = env.step_dt
            # Identical directions across speeds, including a no-push baseline.
            rng = torch.Generator(device=device).manual_seed(seed)
            angles = torch.rand(num_envs, device=device, generator=rng) * 2 * np.pi
            for speed in speeds:
                obs, _ = env.reset()
                fallen = torch.zeros(num_envs, dtype=torch.bool, device=device)
                calm = torch.zeros(num_envs, dtype=torch.long, device=device)
                settle = torch.full((num_envs,), -1., device=device)
                pushed = False
                prepush_falls = 0
                actual_push_time = 0.
                for step in range(round(horizon_s / dt)):
                    if not pushed and step * dt + 1e-8 >= push_time_s:
                        prepush_falls = int(fallen.sum())
                        actual_push_time = step * dt
                        v = robot.data.root_link_vel_w.clone()
                        v[:, 0] += speed * angles.cos()
                        v[:, 1] += speed * angles.sin()
                        robot.write_root_link_velocity_to_sim(v)
                        env.sim.forward()
                        # The push changes linear velocity, absent from the actor's proprioception.
                        pushed = True
                    obs, _, term, _, _ = env.step(policy(obs["policy"], obs["goal"]))
                    fallen |= term  # An auto-reset can never turn failure into success.
                    if pushed:
                        err = (cmd.root_pos_w - cmd.robot_root_pos_w).norm(dim=-1)
                        spd = robot.data.root_link_lin_vel_w.norm(dim=-1)
                        ok = (err < .05) & (spd < .1) & ~fallen
                        calm = torch.where(ok, calm + 1, torch.zeros_like(calm))
                        newly = (calm >= round(.5 / dt)) & (settle < 0)
                        settle[newly] = (step + 1) * dt - actual_push_time
                settled = (settle >= 0) & ~fallen
                row = dict(speed_m_s=speed, envs=num_envs, fallen=int(fallen.sum()),
                    fell_before_push=prepush_falls, settled=int(settled.sum()),
                    settle_mean_s=float(settle[settled].mean()) if settled.any() else None)
                rows.append(row)
                print(json.dumps(row), flush=True)
        finally:
            env.close()
    out = out or ckpt.parent / f"push_{ckpt.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(seed=seed, push_time_s=push_time_s, horizon_s=horizon_s,
        criterion="speed < 0.1 m/s and root position error < 0.05 m for 0.5 s; community diagnostic",
        checkpoint=str(ckpt), results=rows), indent=2) + "\n")


if __name__ == "__main__":
    tyro.cli(main)
