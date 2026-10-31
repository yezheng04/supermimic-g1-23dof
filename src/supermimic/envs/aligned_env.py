"""mjlab 1.2 adapter with explicit state/reference time alignment and replay hooks.

At observation time state and command are both at t. Reward and termination compare
the integrated state at t+dt with reference t+dt. Reset never advances a clip.
"""
import torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.envs.recorder_compat import RecorderManager


class AlignedTrackingEnv(ManagerBasedRlEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        if abs(self.step_dt - 0.02) > 1e-10:
            raise ValueError("50 Hz references require decimation * physics_dt = 0.02")
        self.recorder_manager = RecorderManager(cfg._supermimic_recorders, self)

    def reset(self, *, env_ids=None, **kwargs):
        obs, extras = super().reset(env_ids=env_ids, **kwargs)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        cmd = self.command_manager.get_term("motion")
        cmd.update_relative_body_poses()
        self.recorder_manager.call("record_post_reset", ids)
        return obs, extras

    def step(self, action):
        self._in_step = True
        self.action_manager.process_action(action.to(self.device))
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=self.physics_dt)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        # Refresh FK BEFORE reward and termination, not just before observation.
        self.sim.forward()
        cmd = self.command_manager.get_term("motion")
        cmd.advance_reference()
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self.step_reward_terms = self.reward_manager._step_reward.clone() * self.step_dt
        self.failure_masks = {n: self.termination_manager.get_term(n).clone() for n in ("root_height", "root_tilt")}
        self.metrics_manager.compute()
        ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # Log the reward exactly once: terminal reward is logged by _resample_command.
        cmd._log_step_rewards(~self.reset_buf)
        if ids.numel():
            self.recorder_manager.call("record_pre_reset", ids)
            self._reset_idx(ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
        if "step" in self.event_manager.available_modes:
            self.event_manager.apply(mode="step", dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        cmd.update_relative_body_poses()
        cmd._update_metrics()
        self.sim.sense()
        self.obs_buf = self.observation_manager.compute(update_history=True)
        if ids.numel():
            self.recorder_manager.call("record_post_reset", ids)
        self.recorder_manager.call("record_post_step")
        self._in_step = False
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
