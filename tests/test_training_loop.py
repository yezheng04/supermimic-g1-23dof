import json
import math
import torch
import pytest
from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer


@pytest.mark.skipif(not torch.cuda.is_available(),reason="requires MuJoCo Warp")
def test_training_bptt_and_resume_refills_replay(motion_file,tmp_path):
    cfg=TrainConfig(run_name="test",log_root=tmp_path)
    cfg.data.npz_dir=motion_file.parent
    cfg.env.num_envs=4
    cfg.env.replay_capacity=64
    cfg.env.reset_noise=False
    cfg.env.domain_randomization=True
    cfg.env.sampling_mode="start"
    cfg.world_model.hidden=(64,64)
    cfg.policy.encoder_hidden=(64,32)
    cfg.policy.decoder_hidden=(64,32)
    cfg.critic.hidden=(32,16)
    cfg.wm_trainer.compile=None
    cfg.wm_trainer.batches_per_update=1
    cfg.wm_trainer.batch_size=4
    cfg.policy_trainer.compile=None
    cfg.policy_trainer.batch_size=2
    cfg.policy_trainer.critic_epochs=1
    cfg.policy_trainer.critic_minibatches=2
    cfg.schedule.total_iterations=16
    cfg.schedule.warmup_iterations=16
    cfg.schedule.updates_per_env_step=1.
    cfg.schedule.log_interval=16
    tr=Trainer(cfg)
    initial=[p.detach().clone() for p in tr.policy.parameters()]
    try:
        tr.run()
        assert tr.num_updates==1
        metrics=json.loads((tmp_path/"test/updates.jsonl").read_text().splitlines()[-1])
        assert all(math.isfinite(x) for group in ("wm","policy") for x in metrics[group].values())
        assert metrics["policy"]["policy_grad_norm"]>0
        assert any(not torch.equal(old,new) for old,new in zip(initial,tr.policy.parameters()))
        assert all(p.requires_grad for p in tr.wm.parameters())
        assert all(p.requires_grad for p in tr.critic.parameters())
        rec=tr.recorder
        assert (rec.buffer.a[rec.buffer.valid]-rec.buffer.a_exec[rec.buffer.valid]).abs().sum()>0
        assert (rec.ctx.read()[:,1:24].abs()<=.010001).all()
        cfg.resume=tmp_path/"test/ckpt_latest.pt"
    finally:
        tr.env.close()
    cfg.run_name="resume"
    cfg.schedule.total_iterations=33  # Full configured warmup repeats after resume.
    resumed=Trainer(cfg)
    try:
        assert resumed.iteration==16
        resumed.run()
        assert resumed.num_updates==3  # Updates after 16 and 17 fresh collection steps.
        assert resumed.iteration==33
        rows=[json.loads(x) for x in (tmp_path/"resume/updates.jsonl").read_text().splitlines()]
        assert [r["iteration"] for r in rows]==[32,33]
    finally:
        resumed.env.close()
