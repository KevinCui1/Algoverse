"""S8: soft profiles are crossed with margin band rather than nested in occupation.

A profile held fixed across an occupation's four bands gives a six-occupation
set six distinct evidence profiles rather than twenty-four, and makes the soft
layer perfectly collinear with occupation. The score then has nothing to be a
function of except the gate arithmetic and a per-job prior, which is the
condition the whole two-layer design exists to avoid.
"""

import math
from collections import Counter

import pytest

from hiringcue import scenarios


def _family(occupation, band, shape):
    profile = [
        {"criterion_id": f"SC{index + 1}", "position": position, "candidate_evidence": "."}
        for index, position in enumerate(shape)
    ]
    return scenarios.ScenarioFamily(
        family_id=f"{occupation}__{band}",
        occupation_slug=occupation,
        occupation=occupation,
        job_summary="",
        margin_band=band,
        hard_gates=[],
        soft_criteria=[],
        candidate_gate_values=[],
        soft_profile=profile,
        candidate_summary="",
    )


BANDS = ("clear_fail", "near_fail", "near_pass", "clear_pass")
SHAPES = [
    ("above", "close", "below", "close"),
    ("close", "above", "close", "below"),
    ("below", "close", "above", "close"),
    ("close", "below", "close", "above"),
]


def _balanced(occupations=6):
    families = []
    for index in range(occupations):
        for offset, band in enumerate(BANDS):
            families.append(_family(f"occ{index}", band, SHAPES[(index + offset) % 4]))
    return families


def _nested(occupations=6):
    """Every band of an occupation carries the same profile, as in the first set."""
    return [
        _family(f"occ{index}", band, SHAPES[index % 4])
        for index in range(occupations)
        for band in BANDS
    ]


def test_the_check_does_not_bind_on_the_scenario_set_frozen_for_the_first_pilot():
    assert scenarios.profile_balance_required("1.0.0") is False
    assert scenarios.profile_balance_required("2.0.0") is True
    assert scenarios.profile_balance_required("2.1.0") is True


def test_a_profile_nested_in_occupation_is_rejected(monkeypatch):
    monkeypatch.setattr(scenarios, "profile_balance_required", lambda *_: True)
    with pytest.raises(scenarios.ScenarioError, match="collinear with occupation"):
        scenarios._validate_profile_balance(_nested())


def test_a_balanced_assignment_is_accepted(monkeypatch):
    monkeypatch.setattr(scenarios, "profile_balance_required", lambda *_: True)
    scenarios._validate_profile_balance(_balanced())


def test_too_few_distinct_shapes_across_the_set_is_rejected(monkeypatch):
    monkeypatch.setattr(scenarios, "profile_balance_required", lambda *_: True)
    # Varies within each occupation, but the whole set collapses onto two
    # shapes, which reintroduces the same collinearity one level up.
    families = [
        _family(f"occ{index}", band, SHAPES[(index + offset) % 2])
        for index in range(6)
        for offset, band in enumerate(BANDS)
    ]
    with pytest.raises(scenarios.ScenarioError, match="distinct soft-profile shapes"):
        scenarios._validate_profile_balance(families)


def test_one_dominant_shape_is_rejected(monkeypatch):
    monkeypatch.setattr(scenarios, "profile_balance_required", lambda *_: True)
    # Each occupation varies, so the nesting check passes, but one shape
    # accounts for three quarters of the set.
    families = [
        _family(
            f"occ{index}",
            band,
            SHAPES[0] if offset < 3 else SHAPES[1 + index % 3],
        )
        for index in range(10)
        for offset, band in enumerate(BANDS)
    ]
    with pytest.raises(scenarios.ScenarioError, match="recur more than"):
        scenarios._validate_profile_balance(families)


def test_shape_ignores_evidence_wording():
    a = _family("occ", "near_pass", SHAPES[0])
    b = _family("occ", "near_pass", SHAPES[0])
    object.__setattr__(
        b,
        "soft_profile",
        [dict(entry, candidate_evidence="different wording") for entry in b.soft_profile],
    )
    assert scenarios.profile_shape(a) == scenarios.profile_shape(b)


def test_the_current_scenario_set_passes_the_balance_check(monkeypatch):
    families = scenarios.validated_families()
    monkeypatch.setattr(scenarios, "profile_balance_required", lambda *_: True)
    scenarios._validate_profile_balance(families)
    # The requirement is that shape is not determined by margin band and that no
    # shape dominates the set. Asserting one exact shape count would pin the test
    # to a single authored set rather than to the property being enforced.
    shapes = Counter(scenarios.profile_shape(family) for family in families)
    assert len(shapes) >= len(scenarios.MARGIN_BANDS)
    assert max(shapes.values()) <= 2 * math.ceil(len(families) / len(shapes))
