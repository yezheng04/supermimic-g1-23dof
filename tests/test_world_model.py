"""World-model unit tests: state spec consistency, identity at init, gradients, and a quick overfit."""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"


def _random_states(n, device, dtype=torch.float32):
    s = torch.randn(n, 59, device=device, dtype=dtype)
    s[:, 3:7] = s[:, 3:7] / s[:, 3:7].norm(dim=-1, keepdim=True)
    return s


def test_state_spec_identity_and_errors():
    from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

    dev = "cpu"
    spec = StateSpec(StateSpecCfg(explicit_integration=False), dev, torch.float64)
    s = _random_states(64, dev, torch.float64)
    s2 = spec.integrate(s, torch.zeros(64, 58, dtype=torch.float64))
    np.testing.assert_allclose(s2.numpy(), s.numpy(), atol=1e-12)
    assert spec.normalized_sq_error(s, s).max() < 1e-20
    # A pure yaw rotation + xy translation of the state leaves the canonical features unchanged.
    from supermimic.kinematics.rotations import quat_apply, quat_from_yaw, quat_mul

    yaw = torch.rand(64, dtype=torch.float64) * 6.28 - 3.14
    qy = quat_from_yaw(yaw)
    s_rot = s.clone()
    s_rot[:, 0:2] += torch.randn(64, 2, dtype=torch.float64)
    s_rot[:, 3:7] = quat_mul(qy, s[:, 3:7])
    s_rot[:, 30:33] = quat_apply(qy, s[:, 30:33])
    np.testing.assert_allclose(spec.canonical(s_rot).numpy(), spec.canonical(s).numpy(), atol=1e-9)
    # Explicit integration with zero delta = constant-velocity extrapolation.
    spec2 = StateSpec(StateSpecCfg(explicit_integration=True), dev, torch.float64)
    s3 = spec2.integrate(s, torch.zeros(64, 58, dtype=torch.float64))
    np.testing.assert_allclose(s3[:, 0:3].numpy(), (s[:, 0:3] + 0.02 * s[:, 30:33]).numpy(), atol=1e-12)
    np.testing.assert_allclose(s3[:, 7:30].numpy(), (s[:, 7:30] + 0.02 * s[:, 36:59]).numpy(), atol=1e-12)

