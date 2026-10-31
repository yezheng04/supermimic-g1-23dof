import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("splits", Path(__file__).parents[1] / "scripts/prepare_training_splits.py")
splits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splits)


def test_source_variants_never_cross_splits():
    names = ["240918__walk_ff_001__A548", "240918__walk_ff_001__A548_M",
             "230101__walk_ff_002__A012", "230101__walk_ff_002__A012_M"]
    families = {splits.family_key(n) for n in names}
    assert families == {"walk_ff"}
    assert len({splits.split_for(splits.family_key(n), 20261029) for n in names}) == 1
    assert splits.family_key("240918__turn_180_R_001__A548_M") == "turn_180_r"


def test_split_is_order_independent():
    families = [f"action_{i}" for i in range(200)]
    a = {s: splits.split_for(s, 20261029) for s in families}
    b = {s: splits.split_for(s, 20261029) for s in reversed(families)}
    assert a == b
    assert set(a.values()) == {"train", "validation", "test"}
