import pytest

from hiringcue import plan, render, scenarios, stimuli


@pytest.fixture(scope="module")
def built():
    return plan.build(allow_provisional=True)


def test_every_counterfactual_set_shares_one_normalised_hash(built):
    variants, _ = built
    render.check_counterfactual_integrity(variants)


def test_conditions_differ_from_each_other_in_exact_text(built):
    variants, _ = built
    grouped = {}
    for variant in variants:
        grouped.setdefault(
            (variant.family_id, variant.prestige_level, variant.soft_variant), []
        ).append(variant)
    for group in grouped.values():
        assert len({v.exact_hash for v in group}) == len(group)


def test_only_the_identity_block_differs_within_a_set(built):
    variants, _ = built
    grouped = {}
    for variant in variants:
        grouped.setdefault((variant.family_id, variant.prestige_level), []).append(variant)
    for group in grouped.values():
        normalised = {
            variant.user_prompt.replace(variant.identity_block, "<ID>", 1)
            for variant in group
        }
        assert len(normalised) == 1


def test_prestige_changes_the_prompt_and_never_the_gold_decision(built):
    variants, _ = built
    by_family = {}
    for variant in variants:
        by_family.setdefault((variant.family_id, variant.condition), {})[
            variant.prestige_level
        ] = variant
    for levels in by_family.values():
        if len(levels) < 2:
            continue
        modest, elite = levels["modest"], levels["elite"]
        assert modest.user_prompt != elite.user_prompt
        assert modest.gold_decision == elite.gold_decision


def test_no_placeholders_survive_rendering(built):
    variants, _ = built
    for variant in variants:
        assert not render.PLACEHOLDER_PATTERN.findall(variant.user_prompt)


def test_neutral_prompts_carry_no_name_or_race_term(built):
    variants, _ = built
    render._screen_neutral(variants)


def test_concealed_prompts_contain_the_name_and_no_direct_statement(built):
    variants, _ = built
    for variant in variants:
        if variant.cue_mode != "concealed":
            continue
        assert variant.applicant_name in variant.user_prompt
        assert "self-identifies as" not in variant.user_prompt


def test_direct_prompts_use_the_neutral_identifier_not_a_name(built):
    variants, _ = built
    names = {v.applicant_name for v in variants if v.applicant_name}
    for variant in variants:
        if variant.cue_mode != "direct":
            continue
        assert "self-identifies as" in variant.user_prompt
        assert not any(name in variant.user_prompt for name in names)


def test_numeric_gate_text_always_states_its_threshold():
    for family in scenarios.validated_families():
        for gate in family.hard_gates:
            text = render.gate_requirement_text(gate)
            if gate["operator"] in (">=", ">", "<=", "<"):
                assert str(gate["required_value"]) in text, (family.family_id, text)


def test_soft_criterion_order_is_stable_across_conditions(built):
    variants, _ = built
    orders = {}
    for variant in variants:
        block = variant.user_prompt.split("ADDITIONAL JOB-RELATED CRITERIA")[1].split(
            "APPLICANT EVIDENCE"
        )[0]
        orders.setdefault(variant.family_id, set()).add(block)
    for family_id, blocks in orders.items():
        assert len(blocks) == 1, family_id


def test_every_gate_value_appears_in_the_prompt(built):
    variants, _ = built
    families = {f.family_id: f for f in scenarios.validated_families()}
    for variant in variants:
        if variant.soft_variant != "base":
            continue
        family = families[variant.family_id]
        for entry in family.candidate_gate_values:
            assert str(entry["candidate_value"]) in variant.user_prompt


def test_trials_cover_every_variant_the_configured_number_of_times(built):
    variants, trials = built
    from hiringcue import config

    runs = int(config.study()["runs_per_variant"])
    assert len(trials) == len(variants) * runs


def test_neutral_trials_never_take_the_identity_specific_arm(built):
    _, trials = built
    for trial in trials:
        if trial.condition == stimuli.NEUTRAL:
            assert trial.reflection_arm in (plan.GENERIC, plan.LATE_DISCLOSURE)
        else:
            assert trial.reflection_arm in (plan.GENERIC, plan.IDENTITY_SPECIFIC)


def test_identity_reflection_template_matches_the_cue_mode(built):
    _, trials = built
    for trial in trials:
        if trial.reflection_arm != plan.IDENTITY_SPECIFIC:
            continue
        assert trial.cue_mode in trial.reflection_template
