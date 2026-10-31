import json
import shutil

import numpy as np
import pytest
import torch

from supermimic.data.packed_library import build_library, load_pack, validate_pack
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.envs.tracking_env_cfg import TRACKED_BODIES


def test_pack_manifest_fp16_and_clip_boundaries(motion_file, tmp_path):
    shutil.copy(motion_file, tmp_path / "second.npz")
    shutil.copy(motion_file, tmp_path / "excluded.npz")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("name,keep,native23_quality_keep\nstand,1,0\nsecond,1,1\nexcluded,0,1\n")
    cache = tmp_path / "library.pt"
    report = build_library(tmp_path, cache, manifest, workers=2)
    p = load_pack(cache, manifest)
    assert p["names"] == ["second", "stand"]
    assert p["ref"].dtype == torch.float16
    assert report["clips"] == 2 and report["frames"] == 502
    lib = MotionLibrary([cache], TRACKED_BODIES, np.zeros(23), device="cpu", compact=True)
    original = MotionLibrary([motion_file], TRACKED_BODIES, np.zeros(23), device="cpu", compact=True)
    idx = torch.tensor([0, 249, 250, 251, 500, 501])
    torch.testing.assert_close(lib.goal_window(idx, 10), original.goal_window(idx % 251, 10))
    torch.testing.assert_close(lib.bodies(idx).pos_w, original.bodies(idx % 251).pos_w)
    manifest.write_text("name,keep\nstand,1\n")
    with pytest.raises(ValueError, match="differ"):
        load_pack(cache, manifest)
    p["lengths"][0] -= 1
    with pytest.raises(ValueError, match="contiguous"):
        validate_pack(p)


def test_pack_missing_manifest_motion_is_error(motion_file, tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("name,keep\nmissing,1\n")
    with pytest.raises(FileNotFoundError):
        build_library(tmp_path, tmp_path / "out.pt", manifest)
    assert not (tmp_path / "out.pt").exists()


@pytest.mark.parametrize("corruption", ["timestamps", "joint_names", "model", "overflow"])
def test_pack_rejects_incompatible_data_without_publishing(motion_file, tmp_path, corruption):
    with np.load(motion_file) as z:
        data = dict(z)
    if corruption == "timestamps":
        data["timestamps"] *= 50 / 30
    elif corruption == "joint_names":
        data["joint_names"] = data["joint_names"][::-1]
    elif corruption == "model":
        meta = json.loads(data["metadata_json"].item())
        meta["model_sha256"] = "wrong_robot"
        data["metadata_json"] = np.array(json.dumps(meta))
    else:
        data["joint_vel"][0, 0] = 1e6
    np.savez(motion_file, **data)
    with pytest.raises(ValueError):
        build_library(tmp_path, tmp_path / "out.pt", workers=1)
    assert not (tmp_path / "out.pt").exists()
