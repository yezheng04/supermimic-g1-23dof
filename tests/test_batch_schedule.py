import pytest
from supermimic.data.batch_schedule import batches


def test_length_buckets_cover_each_clip_once_and_bound_padding():
    rows = [dict(name=str(i), source_frames=n) for i, n in enumerate([9, 100, 5, 8, 120, 10000, 2])]
    grouped = list(batches(rows, batch_size=3, frame_budget=300))
    flat = [r for group in grouped for r in group]
    assert sorted(r["name"] for r in flat) == sorted(r["name"] for r in rows)
    assert [int(r["source_frames"]) for r in flat] == sorted(int(r["source_frames"]) for r in rows)
    for group in grouped:
        assert len(group) <= 3
        assert len(group) == 1 or max(r["source_frames"] for r in group) * len(group) <= 300
    assert len(grouped[-1]) == 1  # an unusually long clip gets its own batch


@pytest.mark.parametrize("size,budget", [(0, 100), (10, 0), (-1, 100)])
def test_invalid_batch_limits(size, budget):
    with pytest.raises(ValueError):
        list(batches([], size, budget))


def test_invalid_motion_is_not_silently_dropped():
    with pytest.raises(ValueError, match="fewer than two"):
        list(batches([dict(name="bad", source_frames=1)]))
