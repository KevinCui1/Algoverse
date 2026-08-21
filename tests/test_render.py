"""Counterfactual integrity of the rendered prompts.

The design's whole claim rests on one property: within a counterfactual set the
rendered text differs in the identity block and nowhere else. Adding employer
context is the change most likely to break it, so the checks below are run over
a plan that carries both context levels rather than over the bare stimuli the
earlier rounds used.
"""

import pytest

from hiringcue import config, context, plan, render, scenarios, stimuli


@pytest.fixture(scope="module")
def prompts():
    return plan.build(pairs=stimuli.load_pairs(stimuli.DEVELOPMENT))


def _sets(prompts):
    grouped: dict[tuple, list] = {}
    for prompt in prompts:
        grouped.setdefault(
            (
                prompt.family_id,
                prompt.context_level,
                prompt.prestige_level,
                prompt.soft_variant,
            ),
            [],
        ).append(prompt)
    return grouped


def test_every_counterfactual_set_shares_one_normalised_hash(prompts):
    for key, group in _sets(prompts).items():
        assert len({prompt.normalised_hash for prompt in group}) == 1, key


def test_conditions_differ_from_each_other_in_exact_text(prompts):
    for group in _sets(prompts).values():
        assert len({prompt.exact_hash for prompt in group}) == len(group)


def test_context_is_constant_within_a_counterfactual_set(prompts):
    """Context enters as a factor rather than as a repair, so it must not vary
    inside the comparison the identity effect is measured on."""
    for group in _sets(prompts).values():
        assert len({prompt.context_variant for prompt in group}) == 1


def test_the_two_context_levels_produce_different_prompts(prompts):
    by_level: dict[str, set[str]] = {}
    for prompt in prompts:
        if prompt.soft_variant != "base":
            continue
        by_level.setdefault(prompt.context_level, set()).add(prompt.normalised_hash)
    assert set(by_level) == set(context.levels())
    assert not by_level["bare"] & by_level["realistic"]


def test_the_realistic_context_mentions_no_identity_or_fairness_term():
    """Nothing in a first-turn prompt may name identity, fairness or bias: that
    primes the behaviour the study exists to observe."""
    primed = ("race", "ethnic", "bias", "fair", "diverse", "inclusi", "equity")
    for level in context.load().values():
        lowered = level.text.casefold()
        assert not any(term in lowered for term in primed), level.variant


def test_both_realistic_variants_are_renderable_in_the_predeclared_order():
    for variant in context.realistic_variants():
        loaded = context.load(variant)
        assert loaded[context.REALISTIC].variant == variant
        assert loaded[context.REALISTIC].text.strip()


def test_an_undeclared_context_variant_is_refused():
    with pytest.raises(context.ContextError, match="predeclared"):
        context.load("employer_with_a_quota")


def test_prestige_changes_the_prompt_and_never_the_gold_decision(prompts):
    by_key: dict[tuple, dict] = {}
    for prompt in prompts:
        by_key.setdefault(
            (prompt.family_id, prompt.context_level, prompt.condition, prompt.name_stimulus_id),
            {},
        )[prompt.prestige_level] = prompt
    for levels in by_key.values():
        if len(levels) < 2:
            continue
        modest, elite = levels["modest"], levels["elite"]
        assert modest.user_prompt != elite.user_prompt
        assert modest.gold_decision == elite.gold_decision


def test_no_placeholders_survive_rendering(prompts):
    for prompt in prompts:
        assert not render.PLACEHOLDER_PATTERN.findall(prompt.user_prompt)


def test_concealed_prompts_contain_the_name_and_no_direct_statement(prompts):
    for prompt in prompts:
        if prompt.cue_mode != "concealed":
            continue
        assert prompt.applicant_name in prompt.user_prompt
        assert "self-identifies as" not in prompt.user_prompt


def test_direct_prompts_use_the_neutral_identifier_not_a_name(prompts):
    names = {prompt.applicant_name for prompt in prompts if prompt.applicant_name}
    for prompt in prompts:
        if prompt.cue_mode != "direct":
            continue
        assert "self-identifies as" in prompt.user_prompt
        assert not any(name in prompt.user_prompt for name in names)


def test_neutral_prompts_carry_no_name_or_race_term(prompts):
    for prompt in prompts:
        if prompt.condition != stimuli.NEUTRAL:
            continue
        lowered = prompt.user_prompt.casefold()
        assert "self-identifies as" not in lowered
        for name in {p.applicant_name for p in prompts if p.applicant_name}:
            assert name.casefold() not in lowered


def test_numeric_gate_text_always_states_its_threshold():
    for family in scenarios.validated_families():
        for gate in family.hard_gates:
            text = render.gate_requirement_text(gate)
            if gate["operator"] in (">=", ">", "<=", "<"):
                assert str(gate["required_value"]) in text, (family.family_id, text)


def test_soft_criterion_order_is_stable_across_conditions(prompts):
    orders: dict[str, set[str]] = {}
    for prompt in prompts:
        block = prompt.user_prompt.split("ADDITIONAL JOB-RELATED CRITERIA")[1].split(
            "APPLICANT EVIDENCE"
        )[0]
        orders.setdefault(prompt.family_id, set()).add(block)
    for family_id, blocks in orders.items():
        assert len(blocks) == 1, family_id


def test_every_gate_value_appears_in_the_prompt(prompts):
    families = {family.family_id: family for family in scenarios.validated_families()}
    for prompt in prompts:
        if prompt.soft_variant != "base":
            continue
        for entry in families[prompt.family_id].candidate_gate_values:
            assert str(entry["candidate_value"]) in prompt.user_prompt


def test_the_prompt_asks_for_one_word_and_nothing_else(prompts):
    """The readout takes the contrast at the position the answer would occupy,
    so the prompt must not invite a preamble before it."""
    for prompt in prompts[:20]:
        assert prompt.user_prompt.rstrip().endswith("Yes or No.")
        assert "Yes or No" in prompt.system_prompt


def test_every_name_pair_is_crossed_with_every_family(prompts):
    """Name pair is a random factor, so it must vary within families rather than
    be assigned one per family."""
    concealed = [prompt for prompt in prompts if prompt.cue_mode == "concealed"]
    pairs = {prompt.name_pair_id for prompt in concealed}
    by_family: dict[str, set[str]] = {}
    for prompt in concealed:
        by_family.setdefault(prompt.family_id, set()).add(prompt.name_pair_id)
    assert len(pairs) == len(stimuli.load_pairs(stimuli.DEVELOPMENT))
    assert all(seen == pairs for seen in by_family.values())


def test_planned_roles_separate_the_estimand_from_its_controls(prompts):
    required = config.study()["qualification"]["primary_estimand_requires_gold"]
    for prompt in prompts:
        if prompt.role == plan.PRIMARY:
            assert prompt.gold_decision == required
        elif prompt.role == plan.RULE_CONTROL:
            assert prompt.gold_decision != required
        else:
            assert prompt.soft_variant == "twin"
