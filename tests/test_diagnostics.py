"""Checks on the blocking diagnostics.

Each criterion is exercised against a readout constructed to fail it and one
constructed to pass, because a gate that cannot fail carries no information.
Two of them - saturation and cross-batch stability - are the criteria the P0
instrument had no equivalent of and are the reason the round it authorised was
unreadable, so their failure paths are tested rather than their happy ones only.
"""

import math

import pytest

from hiringcue import diagnostics, plan, readout


def _row(**overrides):
    row = {
        "model_key": "checkpoint",
        "role": plan.PRIMARY,
        "soft_variant": "base",
        "family_id": "family_a",
        "margin_band": "near_pass",
        "condition": "concealed_white",
        "cue_mode": "concealed",
        "identity_group": "white",
        "context_level": "bare",
        "prestige_level": "modest",
        "counterfactual_pair_id": "pair_a",
        "gold_decision": "advance",
        "minimum_gate_margin": 1.0,
        "token_log_odds": 1.0,
        "implied_yes_probability": 1.0 / (1.0 + math.exp(-1.0)),
        "off_target_mass": 0.01,
    }
    row.update(overrides)
    return row


def _probability(value):
    return 1.0 / (1.0 + math.exp(-value))


def test_saturation_fails_when_most_cells_leave_the_readable_interval():
    rows = [_row(implied_yes_probability=0.999) for _ in range(10)]
    report = diagnostics.saturation(rows)
    assert report["outside_share"] == 1.0
    assert report["verdict"] == diagnostics.FAIL


def test_saturation_passes_on_a_readable_distribution():
    rows = [_row(implied_yes_probability=0.5) for _ in range(10)]
    assert diagnostics.saturation(rows)["verdict"] == diagnostics.PASS


def test_differential_off_target_mass_is_paired_within_counterfactual_sets():
    """Marginally the two arms differ only because of which families they sit in;
    within a set they share every fact but the identity block."""
    rows = []
    for index in range(6):
        rows.append(
            _row(
                counterfactual_pair_id=f"pair{index}",
                identity_group="white",
                off_target_mass=0.10 + index * 0.05,
            )
        )
        rows.append(
            _row(
                counterfactual_pair_id=f"pair{index}",
                identity_group="black",
                off_target_mass=0.11 + index * 0.05,
            )
        )
    report = diagnostics.differential_off_target_mass(rows)
    assert report["paired_arm_difference"] == pytest.approx(-0.01)
    assert report["verdict"] == diagnostics.PASS


def test_a_systematic_arm_difference_in_off_target_mass_fails():
    rows = []
    for index in range(6):
        rows.append(_row(counterfactual_pair_id=f"pair{index}", identity_group="white", off_target_mass=0.02))
        rows.append(_row(counterfactual_pair_id=f"pair{index}", identity_group="black", off_target_mass=0.30))
    assert diagnostics.differential_off_target_mass(rows)["verdict"] == diagnostics.FAIL


def _reading(prompt_id, value):
    return readout.Reading(
        prompt_id=prompt_id,
        token_log_odds=value,
        implied_yes_probability=_probability(value),
        off_target_mass=0.0,
        greedy_token_id=1,
        greedy_is_admitted=True,
        boundary_index=5,
    )


class _GatePrompt:
    """Minimal planned-prompt stand-in carrying the fields the gate keys on."""

    def __init__(self, prompt_id, family_id, context_level, arm, band="clear_pass"):
        self.prompt_id = prompt_id
        self.family_id = family_id
        self.context_level = context_level
        self.identity_group = arm
        self.margin_band = band
        self.cue_mode = "concealed"


def _gate_cell(family, context):
    return [
        _GatePrompt(f"{family}__{context}__b", family, context, "black"),
        _GatePrompt(f"{family}__{context}__w", family, context, "white"),
    ]


def test_stability_gates_the_estimand_not_the_worst_prompt():
    """A large single-prompt movement that cancels in the cell mean must pass.

    This is the case the retired gate got wrong. The two arms of a cell both
    move by 0.40 in the same direction, so every per-prompt movement is far past
    any plausible tolerance while the contrast the study estimates does not move
    at all.
    """
    prompts = _gate_cell("fam", "bare")
    readings = {
        "reference": [_reading("fam__bare__b", 1.00), _reading("fam__bare__w", 0.60)],
        "small_batch": [_reading("fam__bare__b", 1.40), _reading("fam__bare__w", 1.00)],
    }
    report = diagnostics.stability(readings, prompts)
    assert report["per_prompt_maximum_absolute_delta"] == pytest.approx(0.40)
    assert report["estimand_range"] == pytest.approx(0.0, abs=1e-12)
    assert report["verdict"] == diagnostics.PASS


def test_stability_fails_when_the_estimand_itself_moves():
    prompts = _gate_cell("fam", "bare")
    readings = {
        "reference": [_reading("fam__bare__b", 1.00), _reading("fam__bare__w", 0.60)],
        "small_batch": [_reading("fam__bare__b", 1.00), _reading("fam__bare__w", 0.50)],
    }
    report = diagnostics.stability(readings, prompts)
    assert report["estimand_range"] == pytest.approx(0.10)
    assert report["verdict"] == diagnostics.FAIL


def test_stability_averages_over_cells_rather_than_over_prompts():
    """Cells are weighted equally, so an unbalanced cell cannot dominate."""
    prompts = _gate_cell("a", "bare") + _gate_cell("b", "bare")
    prompts.append(_GatePrompt("a__bare__b2", "a", "bare", "black"))
    readings = {
        "reference": [
            _reading("a__bare__b", 1.0), _reading("a__bare__b2", 1.0),
            _reading("a__bare__w", 0.0),
            _reading("b__bare__b", 0.0), _reading("b__bare__w", 0.0),
        ],
        "small_batch": [
            _reading("a__bare__b", 1.0), _reading("a__bare__b2", 1.0),
            _reading("a__bare__w", 0.0),
            _reading("b__bare__b", 0.0), _reading("b__bare__w", 0.0),
        ],
    }
    report = diagnostics.stability(readings, prompts)
    assert report["cells"] == 2
    assert report["estimand_by_layout"]["reference"] == pytest.approx(0.5)
    assert report["verdict"] == diagnostics.PASS


def test_stability_without_prompts_reports_but_does_not_judge():
    """Per-prompt movement alone cannot decide the gate."""
    readings = {
        "reference": [_reading("a", 1.0)],
        "small_batch": [_reading("a", 1.4)],
    }
    report = diagnostics.stability(readings)
    assert report["verdict"] == diagnostics.SKIP
    assert report["estimand_range"] is None
    assert report["per_prompt_maximum_absolute_delta"] == pytest.approx(0.4)


def test_stability_refuses_to_compare_disjoint_prompt_sets():
    """One prompt across compositions is the comparison; two prompts is not."""
    with pytest.raises(ValueError, match="every layout"):
        diagnostics.stability(
            {"reference": [_reading("a", 1.0)], "shuffled": [_reading("b", 1.0)]}
        )


def test_batch_size_sensitivity_reports_the_estimand_without_a_verdict():
    prompts = _gate_cell("fam", "bare")
    readings = {
        "reference": [_reading("fam__bare__b", 1.00), _reading("fam__bare__w", 0.60)],
        "large": [_reading("fam__bare__b", 1.10), _reading("fam__bare__w", 0.50)],
    }
    report = diagnostics.batch_size_sensitivity(readings, prompts)
    assert report["estimand_range"] == pytest.approx(0.20)
    assert report["disclosure_only"] is True
    assert "verdict" not in report
    assert "limit" not in report


def test_d1_detects_a_readout_the_gate_arithmetic_already_determines():
    rows = [
        _row(
            gold_decision="advance" if margin > 0 else "do_not_advance",
            minimum_gate_margin=float(margin),
            token_log_odds=2.0 * margin,
        )
        for margin in range(-6, 7)
    ]
    assert diagnostics.d1_rule_determinacy(rows)["r_squared"] == pytest.approx(1.0)
    assert diagnostics.d1_rule_determinacy(rows)["verdict"] == diagnostics.FAIL


def test_d2_measures_movement_against_the_perturbed_twin():
    rows = []
    for index in range(4):
        rows.append(
            _row(
                family_id=f"family{index}",
                condition="neutral",
                cue_mode="neutral",
                identity_group=None,
                prestige_level="modest",
                context_level="bare",
                token_log_odds=0.0,
            )
        )
        rows.append(
            _row(
                family_id=f"family{index}",
                soft_variant="twin",
                role=plan.TWIN_CONTROL,
                token_log_odds=1.5,
            )
        )
    report = diagnostics.d2_free_parameter(rows)
    assert report["median_absolute_movement"] == pytest.approx(1.5)
    assert report["verdict"] == diagnostics.PASS


def test_d2_fails_when_the_readout_will_not_move_for_a_legitimate_change():
    """A readout that will not move for the criteria it reflects will not move
    for a name either, and no sample size repairs that."""
    rows = []
    for index in range(4):
        rows.append(
            _row(
                family_id=f"family{index}",
                condition="neutral",
                cue_mode="neutral",
                identity_group=None,
                token_log_odds=0.0,
            )
        )
        rows.append(
            _row(family_id=f"family{index}", soft_variant="twin", token_log_odds=0.05)
        )
    assert diagnostics.d2_free_parameter(rows)["verdict"] == diagnostics.FAIL


def test_d3_is_computed_within_band_and_on_qualified_candidates_only():
    rows = [
        _row(margin_band="near_pass", token_log_odds=value)
        for value in (-1.0, 0.0, 1.0)
    ] + [
        _row(
            margin_band="clear_fail",
            role=plan.RULE_CONTROL,
            gold_decision="do_not_advance",
            token_log_odds=value,
        )
        for value in (-20.0, 20.0)
    ]
    report = diagnostics.d3_conditional_dispersion(rows)
    assert list(report["standard_deviation_by_band"]) == ["near_pass"]
    assert report["verdict"] == diagnostics.PASS


def test_a_failed_criterion_withholds_authorisation():
    rows = [_row(implied_yes_probability=0.999) for _ in range(10)]
    report = diagnostics.evaluate(rows)
    entry = report["models"]["checkpoint"]
    assert "saturation" in entry["failed"]
    assert entry["authorised"] is False


def test_the_context_variants_are_evaluated_in_the_predeclared_order():
    """Taking whichever variant performs better would make the template a fitted
    parameter rather than a stimulus."""
    selection = diagnostics.context_selection(
        interaction_by_variant={
            "employer": {"estimate": 0.35, "interval_lower": 0.10, "interval_upper": 0.60},
            "employer_selectivity": {
                "estimate": 0.90,
                "interval_lower": 0.60,
                "interval_upper": 1.20,
            },
        },
        saturation_by_variant={"employer": 0.10, "employer_selectivity": 0.05},
    )
    assert selection["selected"] == "employer"


def test_the_fallback_is_rejected_when_it_increases_saturation():
    selection = diagnostics.context_selection(
        interaction_by_variant={
            "employer": {"estimate": 0.10, "interval_lower": -0.10, "interval_upper": 0.30},
            "employer_selectivity": {
                "estimate": 0.50,
                "interval_lower": 0.20,
                "interval_upper": 0.80,
            },
        },
        saturation_by_variant={"employer": 0.10, "employer_selectivity": 0.35},
    )
    assert selection["selected"] is None
    assert selection["decision"] == "kill"


def test_neither_variant_passing_is_the_design_s_own_falsifier():
    selection = diagnostics.context_selection(
        interaction_by_variant={
            "employer": {"estimate": 0.05, "interval_lower": -0.20, "interval_upper": 0.30},
            "employer_selectivity": {
                "estimate": 0.08,
                "interval_lower": -0.15,
                "interval_upper": 0.31,
            },
        },
        saturation_by_variant={"employer": 0.10, "employer_selectivity": 0.10},
    )
    assert selection["decision"] == "kill"


def test_a_failed_first_variant_requires_the_predeclared_fallback():
    selection = diagnostics.context_selection(
        interaction_by_variant={
            "employer": {
                "estimate": 0.10,
                "interval_lower": -0.10,
                "interval_upper": 0.30,
            }
        },
        saturation_by_variant={"employer": 0.10},
    )
    assert selection["selected"] is None
    assert selection["decision"] == "fallback_required"


def test_development_only_contexts_do_not_enter_primary_instrument_gates():
    rows = [
        _row(context_level="bare", implied_yes_probability=0.5),
        _row(context_level="realistic", implied_yes_probability=0.5),
        _row(context_level="realistic_named", implied_yes_probability=1.0),
        _row(context_level="realistic_matched", implied_yes_probability=1.0),
    ]
    report = diagnostics.evaluate(rows)["models"]["checkpoint"]["criteria"]["saturation"]
    assert report["cells"] == 2
    assert report["outside_share"] == 0.0
