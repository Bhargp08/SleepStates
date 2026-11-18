"""Tests for scripts/run_all.py: pipeline ordering, argument plumbing, files on disk."""

import json
from pathlib import Path

from scripts import run_all


def test_plan_runs_every_stage_in_order():
    plan = run_all.build_plan()

    names = [Path(command[0]).name for _, command in plan]
    assert names == [
        "download.py",
        "extract_features.py",
        "select_k.py",
        "fit_models.py",
        "evaluate.py",
        "analyze.py",
        "baseline.py",
        "viewer.py",
    ]


def test_every_script_in_the_plan_exists():
    for _, command in run_all.build_plan():
        assert Path(command[0]).exists()


def test_plan_forwards_subject_and_night_options():
    plan = run_all.build_plan(subjects="3-5", night=2)
    download = plan[0][1]
    options = dict(zip(download[1::2], download[2::2], strict=True))

    assert options["--subjects"] == "3-5"
    assert options["--night"] == "2"


def test_plan_forwards_chosen_k_to_the_fit_step():
    default = run_all.build_plan()[3][1]
    explicit = run_all.build_plan(best_k=6)[3][1]

    assert "--n-states" not in default
    assert explicit[explicit.index("--n-states") + 1] == "6"


def test_skip_baseline_drops_the_yasa_step():
    plan = run_all.build_plan(include_baseline=False)

    assert all("baseline.py" not in command[0] for _, command in plan)
    assert len(plan) == 7


def test_viewer_writes_into_the_output_dir():
    plan = run_all.build_plan(out_dir="custom")
    viewer = plan[-1][1]

    assert viewer[viewer.index("--out") + 1] == "custom/hypnogram.html"


def test_read_best_k_reads_the_bic_choice(tmp_path):
    path = tmp_path / "best_k.json"
    path.write_text(json.dumps({"k_by_bic": 4, "k_by_heldout": 6}))

    assert run_all.read_best_k(path) == 4


def test_read_best_k_returns_none_when_missing(tmp_path):
    assert run_all.read_best_k(tmp_path / "absent.json") is None
