import pytest

from hiringcue import gates, scenarios


def test_committed_scenario_set_passes_every_structural_check():
    families = scenarios.validated_families()
    assert families
    assert {family.margin_band for family in families} == set(scenarios.MARGIN_BANDS)


def test_every_band_is_represented_for_every_occupation():
    families = scenarios.validated_families()
    by_occupation = {}
    for family in families:
        by_occupation.setdefault(family.occupation_slug, set()).add(family.margin_band)
    for slug, bands in by_occupation.items():
        assert bands == set(scenarios.MARGIN_BANDS), slug


def test_gold_decision_matches_the_declared_margin_band():
    for family in scenarios.validated_families():
        expected = (
            gates.ADVANCE
            if family.margin_band in ("near_pass", "clear_pass")
            else gates.DO_NOT_ADVANCE
        )
        assert family.gold_decision == expected, family.family_id


def test_excluded_occupations_are_absent_and_carry_a_reason():
    excluded = scenarios.excluded_occupations()
    assert excluded
    slugs = {family.occupation_slug for family in scenarios.validated_families()}
    for slug, reason in excluded.items():
        assert slug not in slugs
        assert reason.strip()


def test_permuting_the_soft_layer_cannot_change_the_gold_decision():
    for family in scenarios.validated_families():
        scenarios._validate_gold_independence(family)


def test_uniform_soft_profile_is_rejected():
    family = scenarios.validated_families()[0]
    uniform = [dict(entry, position="above") for entry in family.soft_profile]
    probe = scenarios.ScenarioFamily(
        **{**family.__dict__, "soft_profile": uniform}
    )
    with pytest.raises(scenarios.ScenarioError, match="not mixed"):
        scenarios._validate_structure(probe)


def test_near_threshold_families_are_at_or_within_one_unit_of_the_bar():
    for family in scenarios.validated_families():
        if family.margin_band not in scenarios.NEAR_BANDS:
            continue
        margin = family.minimum_gate_margin
        if margin is None:
            continue
        assert abs(margin) <= 1.0, family.family_id
