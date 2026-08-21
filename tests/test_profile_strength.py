"""Soft-evidence strength varies among applicants who clear the same bar.

Balancing which criterion sits above or below expectation is not the same as
varying how strong a profile is. A set whose families are all built from the same
multiset of positions differs only in position, and a score that forms an overall
impression from the evidence has no reason to separate such families. Within-band
dispersion is then held near zero by the stimuli rather than measured on the
model, and the conditional-dispersion diagnostic cannot mean what it claims.

These checks read the scenario set as configured, so they bind on whichever set
is frozen for the next run rather than on a builder that produced one of them.
"""

import statistics
from collections import Counter

import pytest

from hiringcue import scenarios


def _strength(family):
    positions = [entry["position"] for entry in family.soft_profile]
    counts = Counter(positions)
    return (counts["above"] - counts["below"]) / len(positions)


def _by_band(families):
    grouped = {}
    for family in families:
        grouped.setdefault(family.margin_band, []).append(family)
    return grouped


@pytest.fixture(scope="module")
def families():
    return scenarios.validated_families()


def test_strength_varies_within_every_margin_band(families):
    for band, group in _by_band(families).items():
        strengths = {_strength(family) for family in group}
        assert len(strengths) > 1, f"{band}: every family carries the same profile strength"


def test_strength_varies_within_every_occupation(families):
    for slug, group in scenarios.iter_by_occupation(families):
        strengths = {_strength(family) for family in group}
        assert len(strengths) > 1, f"{slug}: strength is a fixed property of the occupation"


def test_within_band_strength_spread_is_comparable_across_bands(families):
    """Strength must not be confounded with the qualification margin.

    If one band carried a wider range than another, part of any difference in
    conditional dispersion between bands would be a property of the stimuli.
    """
    spreads = {
        band: statistics.pstdev([_strength(family) for family in group])
        for band, group in _by_band(families).items()
    }
    assert min(spreads.values()) > 0.0
    assert max(spreads.values()) - min(spreads.values()) < 0.05


def test_every_profile_remains_mixed_and_singly_ambiguous(families):
    """The two properties the strength ladder must not spend to buy its range.

    A profile with no below-expectation criterion collapses to a scalar, and a
    near-threshold family short of an ambiguous criterion is the case the
    ambiguity floor exists to exclude.
    """
    for family in families:
        positions = [entry["position"] for entry in family.soft_profile]
        assert "above" in positions and "below" in positions, family.family_id
        assert positions.count("close") == 1, family.family_id


def test_the_ambiguity_floor_is_unchanged_by_the_ladder(families):
    scores = sorted(family.ambiguity_score for family in families)
    median = scores[len(scores) // 2]
    thin = [
        family.family_id
        for family in families
        if family.margin_band in scenarios.NEAR_BANDS and family.ambiguity_score < median
    ]
    assert not thin
