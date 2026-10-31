"""Optional experimental arm refinement must preserve unmodified motion fields."""
import mujoco
import numpy as np

from supermimic.data.motion_convert import load_g1_model
from supermimic.retarget.collision_refine import contact_constraints, refine_arms


def colliding_pose():
    model = load_g1_model()
    data = mujoco.MjData(model)
    data.qpos[:3] = [0, 0, .8]
    data.qpos[3:7] = [1, 0, 0, 0]
    for side, angle in (("left", -.2), ("right", .2)):
        data.qpos[model.joint(side + "_shoulder_roll_joint").qposadr[0]] = angle
    mujoco.mj_fwdPosition(model, data)
    return model, data


def test_contact_distance_jacobian_matches_finite_differences():
    model, data = colliding_pose()
    joints = [i for i in range(1, model.njnt) if any(s in model.joint(i).name
        for s in ("shoulder_", "elbow_", "wrist_"))]
    dofs, coords = model.jnt_dofadr[joints], model.jnt_qposadr[joints]
    jac, gaps = contact_constraints(model, data, dofs, .005)
    assert len(jac)
    c = data.contact[0]
    a, b = int(c.geom1), int(c.geom2)
    baseline = data.qpos.copy()
    finite = np.zeros(len(coords))
    for j, coord in enumerate(coords):
        distances = []
        for sign in (1, -1):
            data.qpos[:] = baseline
            data.qpos[coord] += sign * 1e-6
            mujoco.mj_kinematics(model, data)
            distances.append(mujoco.mj_geomDistance(model, data, a, b, 1., None))
        finite[j] = (distances[0] - distances[1]) / 2e-6
    np.testing.assert_allclose(jac[0], finite, atol=2e-5)


def test_bounded_refinement_changes_only_arms_and_reduces_collision():
    model, data = colliding_pose()
    before = min(c.dist for c in data.contact)
    raw = np.tile(np.r_[data.qpos[:3], data.qpos[[4, 5, 6, 3]], data.qpos[7:]], (8, 1))
    refined, report = refine_arms(raw)
    np.testing.assert_array_equal(refined[:, :20], raw[:, :20])
    assert report["observed_max_joint_correction_rad"] <= .6 + 1e-9
    assert report["optimizer_failures"] == 0
    for frame in refined:
        data.qpos[:] = np.r_[frame[:3], frame[[6, 3, 4, 5]], frame[7:]]
        mujoco.mj_fwdPosition(model, data)
        after = min((c.dist for c in data.contact), default=0.)
        assert after > before and after >= -.002
        assert np.all(data.qpos[7:] >= model.jnt_range[1:, 0] - 1e-9)
        assert np.all(data.qpos[7:] <= model.jnt_range[1:, 1] + 1e-9)
