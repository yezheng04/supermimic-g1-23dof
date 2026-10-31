"""Selection parity, extraction integrity, and conservative screening semantics."""
import ast
import csv
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_bones", ROOT / "scripts/prepare_bones_seed.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
CONFIG = json.loads((ROOT / "configs/bones_seed_preparation.json").read_text())


def test_selection_matches_pinned_sonic_and_preserves_mirrors(tmp_path):
    upstream = ROOT / "references/gr00t_data/gear_sonic/data_process/filter_and_copy_bones_data.py"
    assert prep.digest(upstream) == CONFIG["reference_script_sha256"]
    tree = ast.parse(upstream.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "should_filter_out")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(upstream), "exec"), namespace)
    defaults = [ast.literal_eval(k.value) for n in ast.walk(tree) if isinstance(n, ast.Call)
                for k in n.keywords if k.arg == "default" and isinstance(k.value, ast.List)]
    assert CONFIG["exclude_keywords"] in defaults
    names = ["walk", "walk_M", "TABLE_TURN", "Form_Box", "sitDown"]
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["move_soma_uniform_path", "move_g1_path",
            "package", "take_name", "is_mirror", "move_duration_frames"])
        writer.writeheader()
        for name in names:
            writer.writerow(dict(move_soma_uniform_path=f"soma_uniform/bvh/240918/{name}.bvh",
                move_g1_path=f"g1/csv/240918/{name}.csv", package="Locomotion",
                take_name="walk" if name.startswith("walk") else name,
                is_mirror=str(name.endswith("_M")), move_duration_frames=121))
    sonic_config = {**CONFIG, "appendix_e_name_rules": []}
    kept, summary = prep.select(metadata, sonic_config, tmp_path / "selected")
    all_rows = list(csv.DictReader((tmp_path / "selected/selection.csv").open()))
    for row in all_rows:
        assert bool(int(row["keep"])) == (not namespace["should_filter_out"](
            row["checked_name"], CONFIG["exclude_keywords"]))
    assert kept[0]["take_group"] == kept[1]["take_group"]
    assert kept[0]["name"] != kept[1]["name"]
    assert summary["candidates"] == 3  # sitDown does not match the upstream sit_ keyword.
    assert prep.pilot_rows(kept, 2) == prep.pilot_rows(list(reversed(kept)), 2)


@pytest.mark.parametrize("name,extra", [
    ("jump_OFF_50cm_R_001__A001_M.pkl", "jump_off_50cm"),
    ("look-over-fence-180.pkl", "look_over_fence"),
    ("jump_on_box_R.pkl", "box"),
    ("shadow_boxing_R.pkl", None),
    ("fetal_position_R.pkl", None),
    ("sandbox_R.pkl", None),
])
def test_appendix_e_examples_without_unrelated_substrings(name, extra):
    sonic, additions = prep.name_reasons(name, CONFIG)
    if extra is None:
        assert not additions and not sonic
    else:
        assert extra in additions


def test_archive_selection_and_missing_detection(tmp_path):
    archive = tmp_path / "source.tar.gz"
    selected = "soma_uniform/bvh/240918/walk.bvh"
    with tarfile.open(archive, "w:gz") as out:
        for name in [selected, "soma_uniform/bvh/240918/table.bvh"]:
            info = tarfile.TarInfo("./" + name)
            info.size = 5
            out.addfile(info, io.BytesIO(b"hello"))
    target = tmp_path / "raw"
    target.mkdir()
    assert prep.extract_selected(archive, target, [{"source": selected}], lambda *a, **kw: None) == 1
    assert (target / selected).read_bytes() == b"hello"
    assert not (target / "soma_uniform/bvh/240918/table.bvh").exists()
    with pytest.raises(ValueError, match="Missing"):
        prep.extract_selected(archive, target, [{"source": "absent.bvh"}], lambda *a, **kw: None)


@pytest.mark.parametrize("path", ["../escape.bvh", "/tmp/escape.bvh", "soma_uniform/../../escape", "x\\escape"])
def test_unsafe_paths_rejected(path):
    with pytest.raises(ValueError, match="Unsafe"):
        prep.safe_member(path)


def test_local_gate_rejects_bad_and_missing_metrics():
    report = dict(robot="g1_23dof", output_fps=50, quality=dict(finite=True,
        max_joint_limit_excess_rad=0., self_collision_frames_2mm=0, foot_penetration_frames_2cm=0))
    assert prep.quality_reasons(report, CONFIG) == []
    report["quality"]["self_collision_frames_2mm"] = 1
    report["quality"]["max_joint_limit_excess_rad"] = float("nan")
    assert set(prep.quality_reasons(report, CONFIG)) == {"self_collision_frames_2mm", "max_joint_limit_excess_rad"}
    del report["quality"]["foot_penetration_frames_2cm"]
    assert "foot_penetration_frames_2cm" in prep.quality_reasons(report, CONFIG)


def test_diagnostics_do_not_reject_or_hide_appendix_e_failures():
    ok = dict(keep=True, reasons=[])
    bad = dict(keep=False, reasons=["floating_feet"])
    findings = ["self_collision_frames_2mm", "foot_penetration_frames_2cm"]
    diagnostic = {**CONFIG, "native23_quality_mode": "diagnostic"}
    result = prep.screening_decision(ok, findings, diagnostic)
    assert result["keep"] and not result["reasons"] and len(result["diagnostics"]) == 2
    result = prep.screening_decision(bad, findings, diagnostic)
    assert not result["keep"] and result["reasons"] == ["appendix_e:floating_feet"]
    assert not prep.screening_decision(ok, findings, {**CONFIG, "native23_quality_mode": "strict"})["keep"]
    with pytest.raises(ValueError, match="Unknown"):
        prep.screening_decision(ok, findings, {**CONFIG, "native23_quality_mode": "typo"})
