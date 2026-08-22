"""The two prompt forms, and what removing the decision rule may not change.

`gated` states the hard requirements as a numbered checklist and supplies the
eligibility rule. `holistic` supplies neither, and the applicant's own gate
facts appear inside the applicant evidence as prose instead.

Two properties make that contrast interpretable and both are asserted here
rather than argued for.

The gold decision, and the whole gate-result structure beneath it, must be
byte-identical between the forms. It is computed from the structured gate fields
and never reads prompt text, so removing the checklist from the prompt cannot
move it - but the entire design rests on the label being independent of what the
prompt says, and an invariant that important is worth failing loudly on.

Counterfactual integrity must survive the new form. Within a set, every byte
outside the identity block has to match in `holistic` exactly as in `gated`;
otherwise the comparison is not a counterfactual and no cell measured in that
form means anything.
"""

from __future__ import annotations

import json

import pytest

from hiringcue import config, context, gates, plan, render, scenarios, stimuli, twins


@pytest.fixture(scope="module")
def families():
    return scenarios.validated_families()


@pytest.fixture(scope="module")
def prompts():
    return plan.build(
        pairs=stimuli.load_pairs(stimuli.DEVELOPMENT), twins=twins.load()
    )


def test_both_declared_forms_are_planned(prompts):
    assert set(render.forms()) == {render.GATED, render.HOLISTIC}
    assert {prompt.prompt_form for prompt in prompts} == set(render.forms())


def test_the_gold_decision_and_gate_results_are_identical_across_forms(families):
    """The label is a pure function of the structured gates, in both forms."""
    for family in families:
        results = gates.evaluate_all(family.hard_gates, family.candidate_gate_values)
        serialised = json.dumps([vars(result) for result in results], sort_keys=True)
        for form in render.forms():
            rendered = render.render_variant(
                family=family,
                condition=stimuli.NEUTRAL,
                prestige=stimuli.load_prestige()["modest"],
                name=None,
                order_seed=int(config.study()["seed"]),
                context=context.load()["bare"],
                prompt_form=form,
            )
            assert rendered.gold_decision == family.gold_decision
            assert rendered.minimum_gate_margin == family.minimum_gate_margin
            assert rendered.failed_gates == family.failed_gates
            assert (
                json.dumps(
                    [
                        vars(result)
                        for result in gates.evaluate_all(
                            family.hard_gates, family.candidate_gate_values
                        )
                    ],
                    sort_keys=True,
                )
                == serialised
            )


def test_the_planned_gold_decision_does_not_vary_with_the_form(prompts):
    by_family: dict[str, set[str]] = {}
    for prompt in prompts:
        by_family.setdefault(prompt.family_id, set()).add(
            f"{prompt.gold_decision}|{prompt.minimum_gate_margin}|{prompt.failed_gates}"
        )
    for family_id, values in by_family.items():
        assert len(values) == 1, family_id


def test_counterfactual_integrity_holds_within_the_holistic_form(prompts):
    """Every byte outside the identity block matches across the arms of a set."""
    grouped: dict[tuple, list] = {}
    for prompt in prompts:
        if prompt.prompt_form != render.HOLISTIC:
            continue
        grouped.setdefault(
            (
                prompt.family_id,
                prompt.context_level,
                prompt.prestige_level,
                prompt.soft_variant,
            ),
            [],
        ).append(prompt)

    assert grouped
    for key, group in grouped.items():
        assert len({prompt.normalised_hash for prompt in group}) == 1, key
        assert len({prompt.exact_hash for prompt in group}) == len(group), key


def test_the_holistic_form_carries_no_rule_and_no_threshold(prompts, families):
    """A prompt that still states the bar has reformatted the rule, not removed it."""
    lookup = {family.family_id: family for family in families}
    for prompt in prompts:
        if prompt.prompt_form != render.HOLISTIC:
            continue
        text = prompt.user_prompt
        assert "HARD REQUIREMENTS" not in text
        assert "eligible to advance only if" not in text
        assert "hard requirement" not in text.casefold()
        for gate in lookup[prompt.family_id].hard_gates:
            if gate["operator"] in gates.NUMERIC_OPERATORS:
                assert f"at least {gate['required_value']}" not in text.casefold()


def test_the_gated_form_is_unchanged_by_the_addition_of_the_second_form(prompts):
    """`gated` is the previous template exactly; the contrast needs one arm fixed."""
    for prompt in prompts:
        if prompt.prompt_form != render.GATED:
            continue
        assert "HARD REQUIREMENTS" in prompt.user_prompt
        assert (
            "An applicant is eligible to advance only if every hard requirement is "
            "satisfied." in prompt.user_prompt
        )


def test_both_forms_report_the_same_applicant_facts(families):
    """The forms differ in how the task is posed, not in what the applicant did."""
    for family in families:
        gate_lookup = {gate["gate_id"]: gate for gate in family.hard_gates}
        for entry in family.candidate_gate_values:
            gate = gate_lookup[entry["gate_id"]]
            prose = render.gate_evidence_text(gate, entry["candidate_value"])
            assert prose.endswith(".")
            if gate["operator"] in gates.NUMERIC_OPERATORS:
                # The value survives as a spelled quantity rather than a figure.
                assert prose.split()[0].isalpha()
            else:
                passed = gates.evaluate_gate(gate, entry["candidate_value"]).passed
                assert prose.startswith("Evidence of") is passed


def test_the_twin_control_exists_in_every_cell(prompts):
    """D2 bounds any cue effect in the cell it is measured in.

    The estimand lives in the rich contexts, so a control held at one nominated
    cell would bound a quantity the design does not estimate.
    """
    twins = {
        (prompt.prompt_form, prompt.context_level)
        for prompt in prompts
        if prompt.soft_variant == "twin"
    }
    expected = {
        (form, level) for form in render.forms() for level in context.levels()
    }
    assert twins == expected
