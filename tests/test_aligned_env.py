"""GPU integration: time phase, exact post-step rewards, reset boundary and clip end."""
from pathlib import Path
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires MuJoCo Warp CUDA")


def test_reward_phase_replay_reset_and_end(tmp_path, motion_file):
    from supermimic.data.motion_convert import save_reference
    from supermimic.envs import make_env, SuperMimicEnvCfgArgs
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.wm_trainer import reference_at
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES
    with np.load(motion_file) as d:
        root=list(d["body_names"]).index("pelvis")
        raw=np.c_[d["body_pos_w"][:13,root],d["body_quat_w"][:13,root][:,[1,2,3,0]],d["joint_pos"][:13]]
    raw[:,0] += np.arange(13)*.02*.1
    path=tmp_path/"move.npz"
    save_reference(raw,path,50.)
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(path)],num_envs=2,replay_capacity=64,
        obs_noise=False, reset_noise=False,domain_randomization=False,action_noise_std=0.,sampling_mode="start"))
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        cmd=env.command_manager.get_term("motion")
        rec=env.recorder_manager.get_term("replay")
        rob=env.scene["robot"]
        heads=AnalyticHeads(G1Kinematics(load_g1_model(),env.device),TRACKED_BODIES,"pelvis","pelvis",
            rob.data.soft_joint_pos_limits[0],rob.data.default_joint_pos[0])
        obs,_=env.reset()
        from supermimic.world_model.heads import analytic_observation, flatten_history_term_major
        first=analytic_observation(rec.state_io.read(),env.action_manager.action,rec.ctx.read(),rob.data.default_joint_pos[0])
        history=[first.clone() for _ in range(10)]
        torch.testing.assert_close(obs["policy"],flatten_history_term_major(torch.stack(history)),atol=2e-6,rtol=2e-5)
        assert cmd.frame.tolist()==[0,0]
        assert obs["policy"].shape==(2,750) and obs["goal"].shape==(2,560)
        for step in range(12):
            obs,reward,term,tout,_=env.step(torch.zeros(2,23,device=env.device))
            assert not term.any()
            if step<11:
                assert cmd.frame.tolist()==[step+1]*2
                analytic=heads.reward(rec.state_io.read(),env.action_manager.action,env.action_manager.prev_action,
                                      reference_at(cmd.library,cmd.frame))
                torch.testing.assert_close(analytic["total"],reward,atol=2e-6,rtol=2e-5)
                for i,name in enumerate(env.reward_manager.active_terms):
                    torch.testing.assert_close(analytic[name],env.step_reward_terms[:,i],atol=2e-6,rtol=2e-5)
                current=analytic_observation(rec.state_io.read(),env.action_manager.action,rec.ctx.read(),rob.data.default_joint_pos[0])
                history=history[1:]+[current]
                torch.testing.assert_close(obs["policy"],flatten_history_term_major(torch.stack(history)),atol=2e-6,rtol=2e-5)
            else:
                assert tout.all()  # Exactly N-1 transitions, no held last frame.
                assert cmd.frame.tolist()==[0,0]
        assert (rec.buffer.terminal.sum(1)==1).all()
        w=rec.buffer.sample_windows(16,8)
        assert (w.frame[1:]-w.frame[:-1]==1).all()
        assert int(rec.buffer.ep_counter.min())==1
    finally:
        env.close()
