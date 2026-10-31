"""Wall-clock-scheduled kinematic reference viewer; does not simulate a controller."""
import time
import numpy as np
import mujoco
import mujoco.viewer
from supermimic.data.motion_convert import load_g1_model
from supermimic.data.quality import validate_reference


def play_reference(path):
    with np.load(path) as d:
        validate_reference(d)
        root = list(d["body_names"]).index("pelvis")
        poses = np.c_[d["body_pos_w"][:, root], d["body_quat_w"][:, root], d["joint_pos"]]
        times = d["timestamps"].copy()
    model = load_g1_model()
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.monotonic()
        while viewer.is_running():
            elapsed = time.monotonic() - start
            index = min(np.searchsorted(times, elapsed, side="right") - 1, len(times)-1)
            data.qpos[:] = poses[max(index, 0)]
            mujoco.mj_forward(model, data)
            viewer.sync()
            if elapsed >= times[-1]:
                break
            # Deadlines are relative to start; render overhead never changes motion speed.
            time.sleep(max(0., min(0.01, times[min(index+1, len(times)-1)]-elapsed)))
