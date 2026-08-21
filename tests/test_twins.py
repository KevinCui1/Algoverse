"""Contract checks for the perturbed soft-criteria positive control."""

from __future__ import annotations

import copy
import json

import pytest

from hiringcue import scenarios, twins


@pytest.fixture
def family():
    return scenarios.validated_families()[0]


@pytest.fixture
def valid_profile(family):
    return [
        {
            "criterion_id": entry["criterion_id"],
            "candidate_evidence": entry["candidate_evidence"],
            "position": twins.POSITION_TARGET[entry["position"]],
        }
        for entry in family.soft_profile
    ]


def test_valid_profile_matches_every_requested_direction(family, valid_profile):
    twins.validate(family, valid_profile)


def test_reordered_criteria_are_rejected(family, valid_profile):
    reordered = list(reversed(valid_profile))
    with pytest.raises(twins.TwinError, match="reorder"):
        twins.validate(family, reordered)


def test_wrong_position_direction_is_rejected(family, valid_profile):
    wrong = copy.deepcopy(valid_profile)
    wrong[0]["position"] = family.soft_profile[0]["position"]
    with pytest.raises(twins.TwinError, match="requested directions"):
        twins.validate(family, wrong)


def test_personal_pronouns_are_rejected(family, valid_profile):
    with_pronoun = copy.deepcopy(valid_profile)
    words = family.soft_profile[0]["candidate_evidence"].split()
    with_pronoun[0]["candidate_evidence"] = " ".join(["They", *words[1:]])
    with pytest.raises(twins.TwinError, match="personal pronouns"):
        twins.validate(family, with_pronoun)


def test_large_length_change_is_rejected(family, valid_profile):
    too_long = copy.deepcopy(valid_profile)
    base_words = len(family.soft_profile[0]["candidate_evidence"].split())
    too_long[0]["candidate_evidence"] = " ".join(["relevant"] * (base_words * 2))
    with pytest.raises(twins.TwinError, match="word-count ratio"):
        twins.validate(family, too_long)


def test_an_occupation_keyed_pool_is_expanded_across_that_occupation_s_families(tmp_path):
    """The pool authored for the first pilot stays usable while profiles are shared.

    Re-authoring it would cost a run for no change: where every band of an
    occupation carries the same soft profile, one twin is the perturbation of
    all of them, which is why one twin per occupation was valid to begin with.
    """
    families = []
    for _slug, group in scenarios.iter_by_occupation(scenarios.validated_families()):
        shared = group[0].soft_profile
        families.extend(
            scenarios.ScenarioFamily(
                **{**family.__dict__, "soft_profile": shared}
            )
            for family in group
        )
    stored = {}
    for slug, group in scenarios.iter_by_occupation(families):
        stored[slug] = [
            dict(entry, position={"above": "below", "below": "above"}.get(entry["position"], "close"))
            for entry in group[0].soft_profile
        ]
    path = tmp_path / "soft_twins.json"
    path.write_text(json.dumps({"twins": stored}))

    expanded = twins.load(path, families=families)
    assert set(expanded) == {family.family_id for family in families}
    for family in families:
        assert expanded[family.family_id] == stored[family.occupation_slug]


def test_expansion_is_refused_once_profiles_differ_across_bands(tmp_path):
    families = scenarios.validated_families()
    slug, group = next(scenarios.iter_by_occupation(families))
    altered = [
        f if f.family_id != group[1].family_id
        else scenarios.ScenarioFamily(
            **{**f.__dict__, "soft_profile": [
                dict(entry, candidate_evidence=entry["candidate_evidence"] + " Additionally noted.")
                for entry in f.soft_profile
            ]}
        )
        for f in families
    ]
    path = tmp_path / "soft_twins.json"
    path.write_text(json.dumps({"twins": {slug: group[0].soft_profile}}))
    with pytest.raises(twins.TwinError, match="author twins per family"):
        twins.load(path, families=altered)


def test_a_family_keyed_pool_is_refused_when_the_profile_it_perturbs_has_changed(
    family, valid_profile, tmp_path
):
    """A twin authored for an earlier scenario set must not load silently.

    It carries the right family identifiers while perturbing evidence the family
    no longer has, so the positive control would compare the score against
    unrelated text and the failure would not appear in any downstream number.
    """
    path = twins.save({family.family_id: valid_profile}, path=tmp_path / "soft_twins.json")
    assert twins.load(path=path, families=[family])

    moved = copy.deepcopy(family)
    object.__setattr__(
        moved,
        "soft_profile",
        [
            dict(entry, position="close") if entry["position"] == "above" else dict(entry)
            for entry in family.soft_profile
        ],
    )
    with pytest.raises(twins.TwinError, match="not the perturbation"):
        twins.load(path=path, families=[moved])
