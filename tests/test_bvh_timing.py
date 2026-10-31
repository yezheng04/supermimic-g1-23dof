import numpy as np
import pytest
from supermimic.retarget.bvh import read_bvh_timing


def test_header_precision_is_preserved(tmp_path):
    from supermimic.data.motion_convert import resample_motion
    p=tmp_path/"timing.bvh"
    p.write_text("Frames: 301\nFrame Time: 0.033333\n")
    n,fps=read_bvh_timing(p)
    raw=np.zeros((n,30)); raw[:,6]=1
    raw[:,0]=np.arange(n)/fps
    motion=resample_motion(raw,fps,50)
    assert motion.num_frames == 500
    assert motion.root_pos[-1,0] == pytest.approx(9.98)
    np.testing.assert_allclose(motion.root_lin_vel_w[:,0],1.,atol=1e-10)
