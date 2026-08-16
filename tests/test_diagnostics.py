"""The diagnostics are checked against two simulated scorers.

One scores purely from the qualification arithmetic. The other also responds to
the soft criteria. Both produce wide, many-valued score distributions, so a
histogram cannot tell them apart - which is the whole reason the free-parameter
control exists. The gates must separate them.
"""

import json
import random

import pytest

from hiringcue import derive, diagnostics, pilot, plan, scenarios


def _response(row, score, decision):
    return {
        **row,
        "model_key": row["model_key"],
        "initial_raw": json.dumps(
            {
                "decision": decision,
                "suitability_score": int(max(0, min(100, round(score)))),
                "decision_confidence": 80,
                "justification": "Assessed against the stated requirements.",
            }
        ),
        "reflection_raw": json.dumps(
            {
                "revision_made": False,
                "revised_decision": decision,
                "revised_suitability_score": int(max(0, min(100, round(score)))),
                "reconsideration_confidence": 80,
                "reason": "No change on reconsideration.",
            }
        ),
        "reflection_schema": "reflection_generic_output.schema.json",
    }


def _simulate(model_key: str, responds_to_soft: bool, noise: float, seed: int):
    rng = random.Random(seed)
    families = {f.family_id: f for f in scenarios.validated_families()}
    twin_profiles = {
        family.occupation_slug: [
            dict(entry, position={"above": "below", "below": "above"}.get(entry["position"], "close"))
            for entry in family.soft_profile
        ]
        for family in families.values()
    }
    variants, trials = plan.build(twins=twin_profiles, allow_provisional=True)

    rows = []
    for trial in trials:
        family = families[trial.family_id]
        passed = family.gold_decision == "advance"
        margin = family.minimum_gate_margin or 0.0
        score = 45 + 20 * passed + 3 * margin
        if responds_to_soft:
            profile = (
                twin_profiles[family.occupation_slug]
                if trial.soft_variant == "twin"
                else family.soft_profile
            )
            weights = {"above": 4.0, "close": 0.0, "below": -4.0}
            score += sum(weights[entry["position"]] for entry in profile)
        score += rng.gauss(0, noise)
        record = {
            **trial.__dict__,
            "model_key": model_key,
            "model_id": model_key,
            "run_label": "sim",
        }
        rows.append(_response(record, score, "advance" if passed else "do_not_advance"))
    return derive.parse_records(rows)


@pytest.fixture(scope="module")
def deterministic():
    return _simulate("deterministic", responds_to_soft=False, noise=2.5, seed=1)


@pytest.fixture(scope="module")
def discretionary():
    return _simulate("discretionary", responds_to_soft=True, noise=2.5, seed=2)


def test_both_scorers_look_the_same_on_a_histogram(deterministic, discretionary):
    a = diagnostics.granularity(deterministic)["deterministic"]
    b = diagnostics.granularity(discretionary)["discretionary"]
    assert a["effective_distinct_values"] > 8
    assert b["effective_distinct_values"] > 8


def test_free_parameter_control_separates_them(deterministic, discretionary):
    a = diagnostics.d2_free_parameter(deterministic)["deterministic"]
    b = diagnostics.d2_free_parameter(discretionary)["discretionary"]
    assert a < 2.0
    assert b > 5.0


def test_rule_determinacy_is_higher_for_the_arithmetic_scorer(deterministic, discretionary):
    a = diagnostics.d1_rule_determinacy(deterministic)["deterministic"]
    b = diagnostics.d1_rule_determinacy(discretionary)["discretionary"]
    assert a > b


def test_the_arithmetic_scorer_is_blocked_from_the_confirmatory_run(deterministic):
    report = diagnostics.evaluate(deterministic, temperature=0.7)
    assert report["models"]["deterministic"]["confirmatory_run_authorised"] is False


def test_the_discretionary_scorer_is_authorised(discretionary):
    report = diagnostics.evaluate(discretionary, temperature=0.7)
    entry = report["models"]["discretionary"]
    assert entry["gates"]["D2_free_parameter_response"]["verdict"] == diagnostics.PASS


def test_run_to_run_gate_is_skipped_at_temperature_zero(discretionary):
    report = diagnostics.evaluate(discretionary, temperature=0)
    verdict = report["models"]["discretionary"]["gates"]["D4_run_to_run_sd"]["verdict"]
    assert verdict == diagnostics.SKIP


def test_counterfactual_differences_pair_within_scenario(discretionary):
    differences = derive.counterfactual_differences(discretionary)
    assert differences
    for record in differences:
        if "score_shift_concealed" in record and "score_shift_direct" in record:
            assert record["cue_mode_interaction"] == (
                record["score_shift_concealed"] - record["score_shift_direct"]
            )


def test_paired_difference_sd_is_reported_per_model(discretionary):
    differences = derive.counterfactual_differences(discretionary)
    summary = derive.paired_difference_sd(differences, "score_shift_concealed")
    assert set(summary) == {"discretionary"}
    assert summary["discretionary"]["n"] > 0


def test_positive_control_rows_do_not_enter_base_instrument_gates(discretionary):
    base = [row for row in discretionary if row["soft_variant"] != "twin"]
    assert diagnostics.d1_rule_determinacy(discretionary) == diagnostics.d1_rule_determinacy(base)
    assert diagnostics.d3_conditional_dispersion(discretionary) == diagnostics.d3_conditional_dispersion(base)
    assert diagnostics.granularity(discretionary) == diagnostics.granularity(base)


def test_pilot_summary_and_sizing_cover_required_instrument_outputs(discretionary):
    differences = derive.counterfactual_differences(discretionary)
    manifest = {
        "model_key": "discretionary",
        "started_at_utc": "2026-08-16T00:00:00+00:00",
        "finished_at_utc": "2026-08-16T00:10:00+00:00",
        "trial_count": len(discretionary),
        "tensor_parallel_size": 2,
    }
    report = pilot.summary(discretionary, differences, [manifest])
    model = report["models"]["discretionary"]
    assert report["counterfactual_integrity"]["pass"] is True
    assert model["paired_score_difference"]["score_shift_concealed"]["n"] > 0
    assert set(model["initial_accuracy_by_margin_band"]) == {
        "clear_fail", "near_fail", "near_pass", "clear_pass"
    }
    assert report["execution"]["discretionary"]["accelerator_hours"] == pytest.approx(1 / 3)

    sizing = pilot.confirmatory_sizing(differences)
    assert "only if the pilot diagnostics authorise" in sizing["interpretation"]
    assert sizing["minimum_meaningful_score_difference"] == 2.0
    assert sizing["models"]["discretionary"]["recommended_families"] >= 48
    assert sizing["models"]["discretionary"]["recommended_families"] % 4 == 0

    gated_sizing = pilot.confirmatory_sizing(
        differences, authorised_models={"discretionary"}
    )
    assert gated_sizing["authorised_models"] == ["discretionary"]
    assert gated_sizing["recommended_families_across_authorised_models"] == (
        sizing["models"]["discretionary"]["recommended_families"]
    )
