import numpy as np
import pytest
import torch

from supermimic.data.motion_convert import save_reference
from supermimic.data.packed_library import build_library
from supermimic.eval.benchmark import run_benchmark


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires MuJoCo Warp")
def test_packed_benchmark_queue_and_failure_denominator(motion_file, tmp_path):
    source = tmp_path / "clips"
    source.mkdir()
    with np.load(motion_file) as d:
        root = d["body_names"].tolist().index("pelvis")
        raw = np.c_[d["body_pos_w"][:, root], d["body_quat_w"][:, root][:, [1,2,3,0]], d["joint_pos"]]
    save_reference(raw[:13], source / "a_short.npz", 50.)
    # Physically unsupported reference deliberately tests early failure, not quality selection.
    raw[:, 2] += 1.
    save_reference(raw, source / "b_airborne.npz", 50.)
    cache = tmp_path / "packed.pt"
    build_library(source, cache, workers=1)

    def zero_policy(obs, goal):
        assert obs.shape[-1] == 750 and goal.shape[-1] == 560
        return torch.zeros((obs.shape[0], 23), device=obs.device)

    rows = run_benchmark(zero_policy, [str(cache)], num_envs=1, verbose=False)
    assert [r.name for r in rows] == ["a_short", "b_airborne"]
    assert rows[0].steps == 12 and not rows[0].failed
    fail = rows[1]
    assert fail.failed and fail.steps < fail.frames - 1
    assert fail.failure_reason
    assert fail.norm_return == pytest.approx(fail.executed_step_norm_return * fail.steps / (fail.frames - 1))
