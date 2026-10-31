import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from supermimic.data.motion_convert import resample_motion, map_joints, G1_JOINT_NAMES, G1_29_JOINT_NAMES, save_reference
from supermimic.data.quality import validate_reference


def linear_clip(n=31, fps=30):
    t = np.arange(n)/fps
    raw = np.zeros((n,30))
    raw[:,0], raw[:,2] = 2*t, .8
    raw[:,3:7] = Rotation.from_rotvec(t[:,None]*np.array([[0.,0.,.7]])).as_quat()
    raw[:,7:] = t[:,None]*.2
    return raw


@pytest.mark.parametrize("source_fps,n", [(30,31),(60,61),(50,51),(120,121)])
def test_physical_duration_velocity_and_rotation(source_fps, n):
    m = resample_motion(linear_clip(n, source_fps), source_fps, 50)
    assert m.num_frames == 51  # 1 second, 50 intervals, not 31 steps at 50Hz.
    t = np.arange(51)/50
    np.testing.assert_allclose(m.root_pos[:,0], 2*t, atol=1e-12)
    np.testing.assert_allclose(m.root_lin_vel_w[:,0], 2, atol=1e-12)
    np.testing.assert_allclose(m.joint_vel, .2, atol=1e-12)
    np.testing.assert_allclose(m.root_ang_vel_w[:,2], .7, atol=1e-12)


def test_off_grid_tail_never_stretches_motion():
    m = resample_motion(linear_clip(30), 30, 50)
    assert m.num_frames == 49
    assert m.root_pos[-1,0] == pytest.approx(1.92)  # duration .96, source ends .966666...


def test_slerp_sign_flips_and_pi_wrap():
    raw = linear_clip()
    raw[:,3:7] = Rotation.from_euler("z", np.linspace(170,190,31)[:,None], degrees=True).as_quat()
    raw[::2,3:7] *= -1
    m = resample_motion(raw,30,50)
    np.testing.assert_allclose(m.root_ang_vel_w[:,2], np.deg2rad(20), atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(m.root_quat,axis=1),1,atol=1e-12)


def test_irregular_timestamps_and_nonzero_start():
    ts = np.array([100.,100.01,100.065,100.10])
    raw = linear_clip(4)
    raw[:,0] = 3*(ts-ts[0])
    m = resample_motion(raw, None,50,ts)
    np.testing.assert_allclose(m.root_pos[:,0],3*np.arange(6)/50,atol=1e-12)
    with pytest.raises(ValueError,match="conflicts"):
        resample_motion(raw,30,50,ts)


@pytest.mark.parametrize("bad", [0,-30,float("nan")])
def test_bad_fps_rejected(bad):
    with pytest.raises(ValueError):
        resample_motion(linear_clip(),bad,50)


def test_projection_requires_explicit_flag_and_uses_names():
    names = list(reversed(G1_29_JOINT_NAMES))
    raw = np.zeros((2,36))
    raw[:,7:] = np.arange(29)
    with pytest.raises(ValueError):
        map_joints(raw,names)
    result = map_joints(raw,names,True)
    np.testing.assert_array_equal(result[0,7:], [names.index(n) for n in G1_JOINT_NAMES])


def test_native_reference_schema_and_bad_timing(tmp_path):
    path=tmp_path/"a.npz"
    info=save_reference(linear_clip(),path,30)
    assert info["dropped_tail_s"] == pytest.approx(0.)
    with np.load(path) as d:
        validate_reference(d)
        broken=dict(d)
    broken["timestamps"] = np.arange(51)/30
    with pytest.raises(ValueError,match="timestamps"):
        validate_reference(broken)
