import json

import pytest

from hiringcue import runner


def test_run_refuses_to_overwrite_raw_output(tmp_path):
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "variants.jsonl").write_text(
        json.dumps({"variant_id": "variant-1"}) + "\n"
    )
    (plan_dir / "trials.jsonl").write_text("")

    out_dir = tmp_path / "responses"
    out_dir.mkdir()
    existing = out_dir / "responses__qwen3-32b__pilot01.jsonl"
    existing.write_text("preserve me\n")

    with pytest.raises(FileExistsError, match="would overwrite"):
        runner.run("qwen3-32b", plan_dir, out_dir, "pilot01")

    assert existing.read_text() == "preserve me\n"


def test_local_snapshot_must_match_configured_revision(tmp_path):
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "variants.jsonl").write_text("")
    (plan_dir / "trials.jsonl").write_text("")

    with pytest.raises(ValueError, match="configured revision"):
        runner.run(
            "qwen3-32b",
            plan_dir,
            tmp_path / "responses",
            "pilot01",
            model_path=str(tmp_path / "moving-target"),
        )
