import importlib.util
from pathlib import Path

import pytest

from supermimic.eval.benchmark import ClipResult

spec = importlib.util.spec_from_file_location("checkpoint_series", Path(__file__).parents[1] / "scripts/evaluate_checkpoint_series.py")
series = importlib.util.module_from_spec(spec)
spec.loader.exec_module(series)


def test_fixed_selection_is_order_independent_and_preserves_diagnostics():
    names = [f"walk_{i:03d}" for i in range(250)] + ["push_up_1", "inchworm_1"]
    lengths = list(range(2, len(names) + 2))
    a, cutoff = series.select_groups(names, lengths, count=100)
    b, other_cutoff = series.select_groups(names[::-1], lengths[::-1], count=100)
    assert a == b and cutoff == other_cutoff
    assert len(a["uniform"]) == len(set(a["uniform"])) == 100
    assert a["push_up"] == ["push_up_1"] and a["inchworm"] == ["inchworm_1"]
    assert all(n in names for values in a.values() for n in values)
    assert set(a["push_up"]) <= set(a["challenge_union"])
    assert series.select_groups(names, lengths, seed=7, count=100)[0]["uniform"] != a["uniform"]


def test_completion_and_failure_are_independent_of_executed_step_return():
    rows = [ClipResult("full", 101, 100, .8, False, -1, executed_step_norm_return=.8),
            ClipResult("short_failure", 101, 20, .16, True, 20, "root_height", executed_step_norm_return=.8)]
    summary = series.group_summary(rows)
    assert summary["mean_norm_return"] == pytest.approx(.48)
    assert summary["mean_executed_step_norm_return"] == pytest.approx(.8)
    assert summary["failure_rate"] == .5
    assert summary["mean_completion_fraction"] == .6
    assert summary["failure_reasons"] == {"root_height": 1}
