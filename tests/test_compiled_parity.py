"""Verify native23 compiled FK/reward/observation and action/state gradients."""
import numpy as np
import pytest
import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.envs.tracking_env_cfg import TRACKED_BODIES
from supermimic.imagination.compiled import ImagStep
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Inductor")
def test_compiled_native23_forward_and_backward(motion_file):
    torch.manual_seed(41)
    device = "cuda:0"
    model = load_g1_model()
    q0 = torch.zeros(23, device=device)
    lib = MotionLibrary([motion_file], TRACKED_BODIES, q0, device=device, compact=True)
    fk = G1Kinematics(model, device=device)
    heads = AnalyticHeads(fk, TRACKED_BODIES, "pelvis", "pelvis",
        torch.tensor(model.jnt_range[1:], device=device, dtype=torch.float32), q0)
    wm = WorldModel(WorldModelCfg(hidden=(64, 64)), heads, device)
    # Nonzero action-dependent dynamics, not just the zero-initialized last layer.
    for p in wm.dynamics.parameters():
        torch.nn.init.normal_(p, std=.02)
    wm.requires_grad_(False)
    frame = torch.tensor([0, 3], device=device)
    qpos, qvel = lib.ref_qpos_qvel(frame)
    s0 = torch.cat([qpos, qvel], -1)
    a0 = torch.randn(2, 23, device=device) * .1
    xi = torch.zeros(2, 27, device=device)
    xi[:, 0] = 1.
    ref = reference_at(lib, frame + 1)
    results = []
    for mode in (None, "default"):
        state, action = s0.clone().requires_grad_(), a0.clone().requires_grad_()
        out = ImagStep(wm, q0, mode)(state, action, torch.zeros_like(action), xi, ref)
        loss = sum(x.square().mean() for x in out)
        grads = torch.autograd.grad(loss, (state, action))
        results.append(([x.detach() for x in out], grads))
    for eager, compiled in zip(results[0][0] + list(results[0][1]), results[1][0] + list(results[1][1])):
        assert torch.isfinite(compiled).all()
        torch.testing.assert_close(eager, compiled, rtol=3e-4, atol=3e-5)
    assert results[1][1][1].abs().sum() > 0
