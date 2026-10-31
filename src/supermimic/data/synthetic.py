"""Synthetic standing fixture for installation and integration tests, not mocap."""
import re
import numpy as np
from supermimic.data.motion_convert import G1_JOINT_NAMES, load_g1_model, resample_motion, forward_kinematics
from supermimic.data.quality import quality_report


def standing_raw(n=151, fps=30.):
    from supermimic.robots.config import HOME_KEYFRAME
    raw = np.zeros((n, 30))
    raw[:, 2], raw[:, 6] = .8, 1.
    for pattern, value in HOME_KEYFRAME.joint_pos.items():
        for i, name in enumerate(G1_JOINT_NAMES):
            if re.fullmatch(pattern, name):
                raw[:, 7+i] = value
    model=load_g1_model()
    m=resample_motion(raw[:2],fps,fps)
    report=quality_report(model,m,forward_kinematics(model,m))
    raw[:,2]-=report["minimum_foot_surface_z_m"]
    return raw
