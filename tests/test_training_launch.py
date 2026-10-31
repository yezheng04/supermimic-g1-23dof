import json
from pathlib import Path

import pytest

from supermimic.scripts.train import load_train_config


def test_json_dual_gpu_config_keeps_per_rank_paper_sizes():
    path = Path(__file__).parents[1] / 'configs/bones_full_2gpu.json'
    cfg, check = load_train_config(['--config', str(path), '--check-config'])
    assert check
    assert (cfg.env.num_envs, cfg.env.replay_capacity) == (4096, 1000)
    assert (cfg.wm_trainer.batch_size, cfg.wm_trainer.batches_per_update) == (2048, 10)
    assert (cfg.policy_trainer.batch_size, cfg.policy_trainer.horizon) == (4096, 32)
    assert cfg.wm_trainer.lr == cfg.policy_trainer.lr == cfg.policy_trainer.critic_lr == 2e-4
    assert not cfg.env.anchor_obs
    assert cfg.wm_trainer.compile == cfg.policy_trainer.compile == 'default'
    assert cfg.data.clips is None and cfg.data.limit is None


def test_wall_time_cap_does_not_compress_learning_rate_schedule(tmp_path):
    path = tmp_path / 'cfg.json'
    path.write_text(json.dumps({'schedule': {'total_iterations': 320000}}))
    cfg, _ = load_train_config(['--config', str(path), '--max-hours', '.25',
        '--run-name', 'hardware_check', '--resume', 'checkpoint.pt'])
    assert cfg.schedule.max_hours == .25
    assert cfg.schedule.total_iterations * cfg.schedule.lr_decay_fraction == 250000
    assert cfg.resume == Path('checkpoint.pt') and cfg.run_name == 'hardware_check'


def test_json_rejects_mismatched_target_windows(tmp_path):
    path = tmp_path / 'cfg.json'
    path.write_text(json.dumps({'env': {'future_frames': 9}}))
    with pytest.raises(SystemExit):
        load_train_config(['--config', str(path)])
