"""Compare simulator rewards/FK/termination with analytic heads on real motions."""
import argparse
import json
from pathlib import Path

import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.tracking_env_cfg import TRACKED_BODIES
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import reference_at


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    files = sorted(map(str, a.data.glob("*.npz")))
    policy, _ = load_policy_from_checkpoint(a.checkpoint, "cuda:0")
    env = make_env(SuperMimicEnvCfgArgs(npz_files=files, num_envs=len(files),
        compact_library=True, replay_capacity=160, obs_noise=False,
        reset_noise=False, domain_randomization=False, action_noise_std=0.,
        sampling_mode="start", seed=20261029), device="cuda:0")
    try:
        cmd = env.command_manager.get_term("motion")
        rec = env.recorder_manager.get_term("replay")
        robot = env.scene["robot"]
        heads = AnalyticHeads(G1Kinematics(load_g1_model(), env.device), TRACKED_BODIES,
            "pelvis", "pelvis", robot.data.soft_joint_pos_limits[0], robot.data.default_joint_pos[0])
        cmd.forced_frame[:] = cmd.library.clip_start
        obs, _ = env.reset()
        errors = {n: 0. for n in ["total", *env.reward_manager.active_terms]}
        checked = resets = 0
        for step in range(100):
            previous_frame = cmd.frame.clone()
            obs, reward, term, tout, _ = env.step(policy(obs["policy"], obs["goal"]))
            alive = ~(term | tout)
            resets += int((~alive).sum())
            assert torch.equal(cmd.frame[alive], previous_frame[alive] + 1)
            state = rec.state_io.read()
            ref = reference_at(cmd.library, cmd.frame)
            analytic = heads.reward(state, env.action_manager.action, env.action_manager.prev_action, ref)
            if alive.any():
                checked += int(alive.sum())
                expected = dict(total=reward)
                expected.update({n: env.step_reward_terms[:, i] for i, n in enumerate(env.reward_manager.active_terms)})
                for n in errors:
                    errors[n] = max(errors[n], float((analytic[n][alive] - expected[n][alive]).abs().max()))
                    torch.testing.assert_close(analytic[n][alive], expected[n][alive], atol=3e-6, rtol=3e-5)
                assert not heads.terminated(state, ref)[alive].any()
            # Terminal states are preserved in replay before automatic reset.
            if (~alive).any():
                ids = (~alive).nonzero().squeeze(-1)
                b = rec.buffer
                # Recorder post-reset may write a new episode's current row in
                # future implementations. Locate the latest terminal explicitly.
                for e in ids.tolist():
                    slots = (b.ptr[e] - 1 - torch.arange(4, device=env.device)) % b.T
                    slots = slots[b.terminal[e, slots] & b.valid[e, slots]]
                    assert len(slots), "Missing terminal state"
                    t = slots[0]
                    pred = heads.terminated(b.s[e, t][None], reference_at(cmd.library, b.frame[e, t][None]))
                    assert bool(pred[0]) == bool(term[e]), "Terminal label differs from analytic condition"
        out = dict(clips=len(files), steps=100, checked_nonterminal_transitions=checked,
            reset_transitions_checked=resets, reward_max_abs_error=errors,
            phase_and_terminal_labels_match=True,
            note="Real-motion compact library, deterministic initial policy. Per-term rewards include differentiable FK; 3e-6 absolute/3e-5 relative tolerance.")
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(out, indent=2)+"\n")
        print(json.dumps(out, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
