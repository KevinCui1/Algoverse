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
        "prompt_form": "gated",
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

    def __init__(
        self,
        prompt_id,
        family_id,
        context_level,
        arm,
        band="clear_pass",
        prompt_form="gated",
    ):
        self.prompt_id = prompt_id
        self.family_id = family_id
        self.context_level = context_level
        self.prompt_form = prompt_form
        self.identity_group = arm
        self.margin_band = band
        self.cue_mode = "concealed"


def _gate_cell(family, context, form="gated"):
    return [
        _GatePrompt(f"{family}__{form}__{context}__b", family, context, "black", prompt_form=form),
        _GatePrompt(f"{family}__{form}__{context}__w", family, context, "white", prompt_form=form),
    ]


def test_the_gate_statistic_separates_the_prompt_forms():
    """The forms are the manipulation, so merging them averages across it.

    Here the two forms carry opposite contrasts in the same family and context.
    A cell key without the form merges them into one cell whose contrast is zero,
    which both halves the number of cells the gate averages over and reports a
    statistic belonging to neither condition.
    """
    prompts = (
        _gate_cell("a", "bare", "gated")
        + _gate_cell("a", "bare", "holistic")
        + _gate_cell("b", "bare", "gated")
        + _gate_cell("b", "bare", "holistic")
    )
    readings = {
        layout: [
            _reading("a__gated__bare__b", 1.0), _reading("a__gated__bare__w", 0.0),
            _reading("a__holistic__bare__b", 0.0), _reading("a__holistic__bare__w", 1.0),
            _reading("b__gated__bare__b", 1.0), _reading("b__gated__bare__w", 0.0),
            _reading("b__holistic__bare__b", 0.0), _reading("b__holistic__bare__w", 1.0),
        ]
        for layout in ("reference", "reordered")
    }
    report = diagnostics.stability(readings, prompts)
    assert report["cells"] == 4
    assert report["estimand_by_layout"]["reference"] == pytest.approx(0.0)


def test_stability_gates_the_estimand_not_the_worst_prompt():
    """A large single-prompt movement that cancels in the cell mean must pass.

    This is the case the retired gate got wrong. The two arms of a cell both
    move by 0.40 in the same direction, so every per-prompt movement is far past
    any plausible tolerance while the contrast the study estimates does not move
    at all.
    """
    prompts = _gate_cell("fam", "bare")
    readings = {
        "reference": [_reading("fam__gated__bare__b", 1.00), _reading("fam__gated__bare__w", 0.60)],
        "small_batch": [_reading("fam__gated__bare__b", 1.40), _reading("fam__gated__bare__w", 1.00)],
    }
    report = diagnostics.stability(readings, prompts)
    assert report["per_prompt_maximum_absolute_delta"] == pytest.approx(0.40)
    assert report["estimand_range"] == pytest.approx(0.0, abs=1e-12)
    assert report["verdict"] == diagnostics.PASS


def test_stability_fails_when_the_estimand_itself_moves():
    prompts = _gate_cell("fam", "bare")
    readings = {
        "reference": [_reading("fam__gated__bare__b", 1.00), _reading("fam__gated__bare__w", 0.60)],
        "small_batch": [_reading("fam__gated__bare__b", 1.00), _reading("fam__gated__bare__w", 0.50)],
    }
    report = diagnostics.stability(readings, prompts)
    assert report["estimand_range"] == pytest.approx(0.10)
    assert report["verdict"] == diagnostics.FAIL


def test_stability_averages_over_cells_rather_than_over_prompts():
    """Cells are weighted equally, so an unbalanced cell cannot dominate."""
    prompts = _gate_cell("a", "bare") + _gate_cell("b", "bare")
    prompts.append(_GatePrompt("a__gated__bare__b2", "a", "bare", "black"))
    readings = {
        "reference": [
            _reading("a__gated__bare__b", 1.0), _reading("a__gated__bare__b2", 1.0),
            _reading("a__gated__bare__w", 0.0),
            _reading("b__gated__bare__b", 0.0), _reading("b__gated__bare__w", 0.0),
        ],
        "small_batch": [
            _reading("a__gated__bare__b", 1.0), _reading("a__gated__bare__b2", 1.0),
            _reading("a__gated__bare__w", 0.0),
            _reading("b__gated__bare__b", 0.0), _reading("b__gated__bare__w", 0.0),
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
        "reference": [
            _reading("fam__gated__bare__b", 1.00),
            _reading("fam__gated__bare__w", 0.60),
        ],
        "large": [
            _reading("fam__gated__bare__b", 1.10),
            _reading("fam__gated__bare__w", 0.50),
        ],
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
                condition="neutral",
                cue_mode="neutral",
                identity_group=None,
                soft_variant="twin",
                role=plan.TWIN_CONTROL,
                token_log_odds=1.5,
            )
        )
    report = diagnostics.d2_free_parameter(rows)
    assert report["median_absolute_movement"] == pytest.approx(1.5)
    assert report["verdict"] == diagnostics.PASS


def test_d2_is_keyed_on_the_cell_and_not_only_on_the_family():
    """Twins exist in every cell, so a family-only key would collapse them.

    Whichever twin the iteration order left last would be reported against every
    cell's baseline, attributing a movement to a condition it was not measured
    in. Here the bare cell does not move and the rich cell does; a family-only
    key cannot report both.
    """
    rows = []
    for index in range(4):
        for level, movement in (("bare", 0.05), ("employer", 1.5)):
            rows.append(
                _row(
                    family_id=f"family{index}",
                    context_level=level,
                    condition="neutral",
                    cue_mode="neutral",
                    identity_group=None,
                    token_log_odds=0.0,
                )
            )
            rows.append(
                _row(
                    family_id=f"family{index}",
                    context_level=level,
                    condition="neutral",
                    cue_mode="neutral",
                    identity_group=None,
                    soft_variant="twin",
                    role=plan.TWIN_CONTROL,
                    token_log_odds=movement,
                )
            )

    bare = [row for row in rows if row["context_level"] == "bare"]
    rich = [row for row in rows if row["context_level"] == "employer"]
    assert diagnostics.d2_free_parameter(bare)["verdict"] == diagnostics.FAIL
    assert diagnostics.d2_free_parameter(rich)["verdict"] == diagnostics.PASS
    assert diagnostics.d2_free_parameter(rows)["families"] == 8


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
            _row(
                family_id=f"family{index}",
                condition="neutral",
                cue_mode="neutral",
                identity_group=None,
                soft_variant="twin",
                token_log_odds=0.05,
            )
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


def test_a_failed_criterion_makes_its_cell_inadmissible():
    rows = [_row(implied_yes_probability=0.999) for _ in range(10)]
    report = diagnostics.evaluate(rows)["models"]["checkpoint"]
    cell = report["cells"]["gated/bare"]
    assert "saturation" in cell["failed"]
    assert cell["admissible"] is False
    assert report["authorised"] is False


def _cell(d2, median, failed=False, form="gated"):
    return {
        "prompt_form": form,
        "context_level": "bare",
        "criteria": {
            "D2_free_parameter_response": {"median_absolute_movement": d2},
            "saturation": {"quantiles": {"median": median}},
        },
        "failed": ["saturation"] if failed else [],
        "warned": [],
    }


def test_the_selected_cell_is_the_one_with_the_most_free_parameter():
    """Selection is on how much room the readout has, not on what it returned."""
    selection = diagnostics.select_cell(
        {
            "gated/bare": _cell(0.50, 0.05),
            "holistic/employer": _cell(1.80, 0.40),
            "holistic/employer_selectivity": _cell(1.20, 0.50),
        },
        instrument_passes=True,
    )
    assert selection["selected"] == "holistic/employer"
    assert selection["decision"] == "proceed"


def test_a_tie_on_the_free_parameter_is_broken_towards_the_middle_of_the_range():
    selection = diagnostics.select_cell(
        {
            "gated/employer": _cell(1.50, 0.10),
            "holistic/employer": _cell(1.50, 0.48),
        },
        instrument_passes=True,
    )
    assert selection["selected"] == "holistic/employer"


def test_an_inadmissible_cell_cannot_be_selected_however_large_its_response():
    selection = diagnostics.select_cell(
        {
            "gated/bare": _cell(9.00, 0.50, failed=True),
            "holistic/employer": _cell(1.10, 0.40),
        },
        instrument_passes=True,
    )
    assert selection["selected"] == "holistic/employer"


def test_a_carried_in_stability_verdict_gates_admissibility():
    """The layouts run in Stage 0, so the verdict is carried rather than re-measured.

    Admissibility depends on it, so a cell must not be declared admissible on a
    checkpoint whose readout does not reproduce merely because the layouts were
    measured in a different stage.
    """
    rows = [
        _row(
            family_id=f"family{index}",
            condition="neutral",
            cue_mode="neutral",
            identity_group=None,
            soft_variant=variant,
            token_log_odds=0.0 if variant == "base" else 2.0,
        )
        for index in range(4)
        for variant in ("base", "twin")
    ]
    agreement = {
        "checkpoint": {"agreement": 1.0, "wilson_lower_bound": 1.0, "sample": 100}
    }
    failing = {"checkpoint": {"verdict": diagnostics.FAIL, "estimand_range": 0.9}}
    passing = {"checkpoint": {"verdict": diagnostics.PASS, "estimand_range": 0.0}}

    blocked = diagnostics.evaluate(rows, agreement=agreement, stability=failing)
    assert blocked["models"]["checkpoint"]["instrument_passes"] is False
    assert blocked["models"]["checkpoint"]["selection"]["selected"] is None

    allowed = diagnostics.evaluate(rows, agreement=agreement, stability=passing)
    assert allowed["models"]["checkpoint"]["instrument_passes"] is True
    assert (
        "cross_batch_stability" in allowed["models"]["checkpoint"]["instrument"]
    )


def test_a_failing_instrument_makes_every_cell_inadmissible():
    """Stability and greedy agreement are properties of the checkpoint, so they
    gate every cell of it equally."""
    selection = diagnostics.select_cell(
        {"holistic/employer": _cell(2.00, 0.50)}, instrument_passes=False
    )
    assert selection["selected"] is None
    assert selection["decision"] == "no_admissible_cell"


def test_no_admissible_cell_is_reported_rather_than_worked_around():
    selection = diagnostics.select_cell(
        {"gated/bare": _cell(0.50, 0.05, failed=True)}, instrument_passes=True
    )
    assert selection["selected"] is None
    assert selection["decision"] == "no_admissible_cell"


def _interaction(estimate, lower, upper, level="employer"):
    return {
        "prompt_form": "holistic",
        "context_level": level,
        "criteria": {},
        "failed": [],
        "warned": [],
        "interaction_vs_bare": {
            "concealed": {
                "estimate": estimate,
                "interval_lower": lower,
                "interval_upper": upper,
            }
        },
    }


def test_the_kill_criterion_fires_when_no_rich_context_reaches_the_target():
    verdict = diagnostics.kill_criterion(
        {
            "holistic/bare": _cell(1.5, 0.5, form="holistic"),
            "holistic/employer": _interaction(0.11, 0.05, 0.17),
            "holistic/employer_selectivity": _interaction(
                0.18, 0.09, 0.27, "employer_selectivity"
            ),
        },
        selected="holistic/bare",
    )
    assert verdict["verdict"] == "kill"


def test_the_kill_criterion_needs_an_interval_that_excludes_zero():
    verdict = diagnostics.kill_criterion(
        {
            "holistic/bare": _cell(1.5, 0.5, form="holistic"),
            "holistic/employer": _interaction(0.45, -0.05, 0.95),
        },
        selected="holistic/bare",
    )
    assert verdict["verdict"] == "kill"


def test_the_kill_criterion_is_a_magnitude_and_not_a_sign():
    """The published interaction favours the Black-associated arm.

    A criterion written on the sign alone would score a same-sized opposite
    effect as a failure to replicate rather than as the different finding it is.
    """
    for estimate, lower, upper in ((0.42, 0.20, 0.64), (-0.42, -0.64, -0.20)):
        verdict = diagnostics.kill_criterion(
            {
                "holistic/bare": _cell(1.5, 0.5, form="holistic"),
                "holistic/employer": _interaction(estimate, lower, upper),
            },
            selected="holistic/bare",
        )
        assert verdict["verdict"] == "replicates"


def test_every_cell_is_diagnosed_separately_rather_than_pooled():
    """The cells differ in exactly the property under test."""
    rows = [
        _row(context_level="bare", implied_yes_probability=0.5),
        _row(context_level="employer", implied_yes_probability=0.5),
        _row(prompt_form="holistic", context_level="bare", implied_yes_probability=0.999),
    ]
    report = diagnostics.cells(rows)
    assert set(report) == {"gated/bare", "gated/employer", "holistic/bare"}
    assert report["gated/bare"]["criteria"]["saturation"]["verdict"] == diagnostics.PASS
    assert (
        report["holistic/bare"]["criteria"]["saturation"]["verdict"] == diagnostics.FAIL
    )
