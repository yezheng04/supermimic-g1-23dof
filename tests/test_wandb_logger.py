import json
from pathlib import Path
from types import SimpleNamespace

from supermimic.training.wandb_logger import read_updates, tensorboard_rows


def test_partial_write_resume_and_training_iteration_rollback(tmp_path):
    path = tmp_path / "updates.jsonl"
    row = dict(iteration=102, wm={"loss": 2.}, policy={"critic_loss": .3})
    line = json.dumps(row).encode() + b"\n"
    path.write_bytes(line[:-4])
    assert read_updates(path, 0) == ([], 0)
    with path.open("ab") as f:
        f.write(line[-4:])
    rows, offset = read_updates(path, 0)
    assert rows == [{"iteration": 102, "wm/loss": 2., "policy/critic_loss": .3}]
    assert read_updates(path, offset) == ([], offset)
    row["iteration"] = 90  # Resume from an older checkpoint; do not skip new data.
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    assert read_updates(path, offset)[0][0]["iteration"] == 90


def test_tensorboard_delayed_events_and_duplicate_poll():
    events = {"train/episode_norm_return": [SimpleNamespace(step=50, wall_time=10., value=.5)],
              "wm/loss": [SimpleNamespace(step=50, wall_time=10., value=2.)]}
    tb = SimpleNamespace(Reload=lambda: None, Tags=lambda: {"scalars": list(events)}, Scalars=events.__getitem__)
    rows, cursors = tensorboard_rows(tb, {})
    assert rows == [{"iteration": 50, "train/episode_norm_return": .5, "periodic/wm/loss": 2.}]
    assert tensorboard_rows(tb, cursors)[0] == []
    events["wm/loss"].append(SimpleNamespace(step=40, wall_time=11., value=1.))
    assert tensorboard_rows(tb, cursors)[0] == [{"iteration": 40, "periodic/wm/loss": 1.}]
