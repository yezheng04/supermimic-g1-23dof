import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from supermimic.data.appendix_e import (
    distance_model, evaluate_geometry, floating_gate, measure_foot_distances, screen_reference,
)
from supermimic.data.motion_convert import save_reference
from supermimic.data.synthetic import standing_raw

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/bones_seed_preparation.json").read_text())["appendix_e_geometry"]


def evaluate(z, feet=None, gate=None):
    n = len(z)
    return evaluate_geometry(z, np.zeros((n, 2)) if feet is None else feet,
        np.zeros(n, dtype=bool) if gate is None else gate, 50., CONFIG)


def test_elevated_support_requires_contiguous_duration():
    assert "elevated_support" not in evaluate(np.full(29, .21))["reasons"]
    assert "elevated_support" in evaluate(np.full(30, .21))["reasons"]
    assert "elevated_support" not in evaluate(np.r_[np.full(20, .21), 0., np.full(20, .21)])["reasons"]
    assert evaluate(np.full(50, .20))["keep"]  # strictly above .20


def test_body_penetration_hard_and_fraction_boundaries():
    assert evaluate(np.array([-.12, 0.]))["keep"]
    assert "ground_penetration" in evaluate(np.array([-.12001, 0.]))["reasons"]
    assert evaluate(np.r_[np.full(80, -.09), np.zeros(20)])["keep"]
    assert "ground_penetration" in evaluate(np.r_[np.full(81, -.09), np.zeros(19)])["reasons"]
    assert evaluate(np.full(100, -.08))["keep"]


def test_floating_requires_both_feet_and_simultaneous_gates():
    n = 50
    feet, gate = np.full((n, 2), .03), np.ones(n, dtype=bool)
    assert "floating_feet" in evaluate(np.zeros(n), feet, gate)["reasons"]
    feet[:, 0] = 0
    assert evaluate(np.zeros(n), feet, gate)["keep"]
    feet[:] = .02
    assert evaluate(np.zeros(n), feet, gate)["keep"]
    feet[:] = .03
    gate[24:26] = False  # two 24-sample runs, neither reaches 0.5 s
    assert evaluate(np.zeros(n), feet, gate)["keep"]
    gate[:] = False
    gate[:25] = True
    assert "floating_feet" in evaluate(np.zeros(n), feet, gate)["reasons"]


def test_ballistic_motion_and_crouch_do_not_pass_stationary_gate():
    n = 5
    pos = np.tile([0., 0., .75], (n, 1))
    quat = np.tile([1., 0., 0., 0.], (n, 1))
    lin, ang, jv = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 23))
    lin[1, 2] = .5  # ascending/descending flight
    ang[2, 2] = 1.  # rotating pelvis
    jv[3] = 1.
    pos[4, 2] = .4  # not standing
    gate = floating_gate(pos, quat, lin, ang, jv, CONFIG["floating_gate_local"])
    np.testing.assert_array_equal(gate, [True, False, False, False, False])
    quat[0] = [0., 1., 0., 0.]  # upside down
    assert not floating_gate(pos, quat, lin, ang, jv, CONFIG["floating_gate_local"]).any()


def test_invalid_geometry_is_not_silently_accepted():
    with pytest.raises(ValueError, match="Nonfinite"):
        evaluate([np.nan])
    with pytest.raises(ValueError, match="Nonfinite"):
        evaluate([0.], [[np.nan, .03]], [True])
    assert evaluate([0.], [[np.nan, np.nan]], [False])["keep"]


def test_mujoco_signed_distance_matches_rotated_capsule_support():
    model, floor, foot_ids = distance_model()
    data = mujoco.MjData(model)
    raw = standing_raw(3)
    raw[:, 2] += [-.04, 0., .1]
    # Tilt a robot: foot geometry orientation must participate in the distance.
    angle = .22
    q = np.array([[1., 0., 0., 0.], [1., 0., 0., 0.], [np.cos(angle/2), 0., np.sin(angle/2), 0.]])
    actual = measure_foot_distances(raw[:, :3], q, raw[:, 7:], np.ones(3, dtype=bool))
    expected = np.zeros((3, 2))
    for t in range(3):
        data.qpos[:] = np.r_[raw[t, :3], q[t], raw[t, 7:]]
        mujoco.mj_kinematics(model, data)
        for side, ids in enumerate(foot_ids):
            assert all(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE for g in ids)
            expected[t, side] = min(data.geom_xpos[g, 2] - model.geom_size[g, 0]
                - model.geom_size[g, 1] * abs(data.geom_xmat[g].reshape(3, 3)[2, 2]) for g in ids)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert np.all(actual[0] < 0)  # signed penetration, not unsigned separation


def test_end_to_end_reference_screening(tmp_path):
    raw = standing_raw(51, fps=50.)
    grounded = tmp_path / "grounded.npz"
    save_reference(raw, grounded, input_fps=50.)
    assert screen_reference(grounded, CONFIG)["keep"]
    raw[:, 2] += .04
    floating = tmp_path / "floating.npz"
    save_reference(raw, floating, input_fps=50.)
    report = screen_reference(floating, CONFIG)
    assert report["reasons"] == ["floating_feet"]
    assert report["metrics"]["floating_longest_frames"] == 51
    assert report["exact_author_implementation"] is False
