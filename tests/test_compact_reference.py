import numpy as np
import torch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.envs.tracking_env_cfg import TRACKED_BODIES


def test_compact_reference_and_future_window(motion_file):
    q0=np.zeros(23)
    full=MotionLibrary([motion_file],TRACKED_BODIES,q0,device="cpu")
    comp=MotionLibrary([motion_file],TRACKED_BODIES,q0,device="cpu",compact=True,store_dtype=torch.float32)
    frame=torch.tensor([0,1,100,249,250])
    bf,bc=full.bodies(frame),comp.bodies(frame)
    torch.testing.assert_close(bf.pos_w,bc.pos_w,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(bf.lin_vel_w,bc.lin_vel_w,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(full.goal_window(frame,10),comp.goal_window(frame,10))
    expected=torch.stack([full.goal_feat[torch.clamp(f+torch.arange(1,11),max=250)] for f in frame])
    torch.testing.assert_close(full.goal_window(frame,10),expected.reshape(5,560))
