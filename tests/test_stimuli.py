"""Checks on the matched name-pair pool and the script that builds it.

The build reads the published replication data, which is not vendored here, so
these tests run against synthetic rosters with the same column structure. What
they protect is the property the rebuild exists to establish: that the two arms
are comparable on attribution accuracy rather than each thresholded on it, and
that a pool which cannot deliver that comparability stops the build instead of
being returned with the imbalance in it.
"""

import csv
import importlib.util

import pytest

from hiringcue import config, matching, paths, stimuli

_spec = importlib.util.spec_from_file_location(
    "build_name_stimuli", paths.ROOT / "scripts" / "build_name_stimuli.py"
)
build_name_stimuli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_name_stimuli)

RULES = config.stimuli()["names"]["matching"]
MINIMUM_PAIRS = int(RULES["minimum_pairs"])
DEVELOPMENT_PAIRS = int(RULES["development_pairs"])
MAXIMUM_SMD = float(RULES["maximum_standardised_mean_difference"])
LADDER = [float(value) for value in RULES["caliper_ladder"]]

POOLED_COLUMNS = [
    "name",
    "identity",
    "correct",
    "income",
    "income.ord",
    "education",
    "education.ord",
    "citizen",
]

# The two labels the roster actually uses for the groups under study, alongside
# two it uses for groups that must not enter the pool.
WHITE = "White"
BLACK = "Black or African American"
UNUSED = ("Hispanic", "Asian or Pacific Islander")


def _write(path, columns, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _judgements(name, identity, share_correct, judgements=100):
    correct = round(share_correct * judgements)
    return [
        {
            "name": name,
            "identity": identity,
            "correct": 1 if index < correct else 0,
            "income": "Middle income\t($41,000 - $120,400)",
            "income.ord": 2,
            "education": "Bachelor's degree",
            "education.ord": 3,
            "citizen": 1,
        }
        for index in range(judgements)
    ]


def _corpus(tmp_path, white_accuracies, black_accuracies, extra_identities=()):
    roster, pooled = [], []
    for identity, tag, accuracies in (
        (WHITE, "W", white_accuracies),
        (BLACK, "B", black_accuracies),
    ):
        for index, accuracy in enumerate(accuracies):
            name = f"{tag}{index} Surname"
            roster.append({"name": name, "identity": identity})
            pooled.extend(_judgements(name, identity, accuracy))
    for identity in extra_identities:
        name = f"X{identity[:3]} Surname"
        roster.append({"name": name, "identity": identity})
        pooled.extend(_judgements(name, identity, 0.99))
    return (
        _write(tmp_path / "pooled.csv", POOLED_COLUMNS, pooled),
        _write(tmp_path / "names.csv", ["name", "identity"], roster),
    )


def _balanced(count=MINIMUM_PAIRS + 4):
    """Two arms whose accuracies coincide, so every pair matches at any caliper."""
    return [0.60 + index * 0.002 for index in range(count)]


def _candidates(white, black):
    return [
        matching.Candidate(f"{group}_{index:03d}", f"{group} {index}", group, value)
        for group, values in (("white", white), ("black", black))
        for index, value in enumerate(values)
    ]


def test_matching_returns_pairs_within_the_caliper():
    pairs = matching.match(_candidates(_balanced(), _balanced()), caliper=0.001)
    assert len(pairs) == MINIMUM_PAIRS + 4
    assert max(abs(pair.accuracy_difference) for pair in pairs) <= 0.001


def test_arms_too_far_apart_yield_no_pairs():
    """The caliper is what makes a pair a pair; without it matching is arbitrary."""
    pairs = matching.match(
        _candidates([0.90] * 40, [0.40] * 40), caliper=0.01
    )
    assert pairs == []


def test_matching_prefers_the_assignment_with_the_smallest_total_distance():
    """A greedy sweep from either end is optimal in count but biased in which
    names it keeps; the retained floor should not depend on sweep direction."""
    white = [0.700, 0.710]
    black = [0.706, 0.712]
    pairs = matching.match(_candidates(white, black), caliper=0.01)
    assert len(pairs) == 2
    assert sum(abs(pair.accuracy_difference) for pair in pairs) == pytest.approx(0.008)


def test_a_pool_that_cannot_be_balanced_stops_the_build():
    """An unbalanced pool is the defect being removed, so it is never returned."""
    with pytest.raises(matching.MatchingError, match="no caliper on the ladder"):
        matching.build_pool(
            _candidates([0.90] * 40, [0.40] * 40),
            caliper_ladder=LADDER,
            minimum_pairs=MINIMUM_PAIRS,
            maximum_standardised_mean_difference=MAXIMUM_SMD,
        )


def test_a_pool_short_of_the_required_pairs_stops_the_build():
    with pytest.raises(matching.MatchingError, match=f"{MINIMUM_PAIRS} pairs"):
        matching.build_pool(
            _candidates(_balanced(4), _balanced(4)),
            caliper_ladder=LADDER,
            minimum_pairs=MINIMUM_PAIRS,
            maximum_standardised_mean_difference=MAXIMUM_SMD,
        )


def test_the_tightest_admissible_caliper_is_the_one_used():
    pool = matching.build_pool(
        _candidates(_balanced(), _balanced()),
        caliper_ladder=LADDER,
        minimum_pairs=MINIMUM_PAIRS,
        maximum_standardised_mean_difference=MAXIMUM_SMD,
    )
    assert pool.caliper == min(LADDER)


def test_development_and_confirmatory_pairs_are_disjoint():
    pool = matching.build_pool(
        _candidates(_balanced(), _balanced()),
        caliper_ladder=LADDER,
        minimum_pairs=MINIMUM_PAIRS,
        maximum_standardised_mean_difference=MAXIMUM_SMD,
    )
    development, confirmatory = matching.split(
        pool, development_pairs=DEVELOPMENT_PAIRS, seed=1
    )
    assert len(development) == DEVELOPMENT_PAIRS
    assert not {pair.pair_id for pair in development} & {
        pair.pair_id for pair in confirmatory
    }


def test_groups_outside_the_study_never_enter_the_pool(tmp_path):
    pooled, names = _corpus(
        tmp_path, _balanced(), _balanced(), extra_identities=UNUSED
    )
    record = build_name_stimuli.build(pooled, names)
    for entry in record["pairs"]:
        assert set(entry["names"]) == {"white", "black"}


def test_perception_statistics_come_from_the_ordinal_columns(tmp_path):
    """The answer-text column shares a stem with the ordinal one and parses to
    nothing, so preferring the wrong one yields a pool with an empty table."""
    pooled, names = _corpus(tmp_path, _balanced(), _balanced())
    record = build_name_stimuli.build(pooled, names)
    for entry in record["pairs"]:
        for side in entry["names"].values():
            assert side["perceived_income"] == 2
            assert side["perceived_education"] == 3
            assert side["perceived_citizenship"] == 1


def test_a_perception_column_that_parses_to_nothing_stops_the_build(tmp_path):
    pooled, names = _corpus(tmp_path, _balanced(), _balanced())
    rows = list(csv.DictReader(pooled.open(newline="")))
    for row in rows:
        row["income.ord"] = "NA"
    _write(pooled, POOLED_COLUMNS, rows)
    with pytest.raises(SystemExit, match="perceived_income"):
        build_name_stimuli.build(pooled, names)


def test_a_missing_perception_column_stops_the_build(tmp_path):
    pooled, names = _corpus(tmp_path, _balanced(), _balanced())
    columns = [column for column in POOLED_COLUMNS if not column.startswith("citizen")]
    rows = [
        {key: value for key, value in row.items() if key in columns}
        for row in csv.DictReader(pooled.open(newline=""))
    ]
    _write(pooled, columns, rows)
    with pytest.raises(SystemExit, match="perceived_citizenship"):
        build_name_stimuli.build(pooled, names)


def test_the_committed_pool_meets_its_declared_requirements():
    pairs = stimuli.load_pairs()
    assert len(pairs) >= MINIMUM_PAIRS
    summary = stimuli.pool_summary(pairs)
    white = summary["arms"]["white"]["attribution_accuracy"]
    black = summary["arms"]["black"]["attribution_accuracy"]
    spread = max(white["max"], black["max"]) - min(white["min"], black["min"])
    assert abs(white["mean"] - black["mean"]) < MAXIMUM_SMD * spread
    # The descriptive perception table is the whole reason these are carried
    # through; a null here means the pool reports nothing about its bundle.
    for pair in pairs:
        for name in (pair.white, pair.black):
            assert name.perceived_income is not None
            assert name.perceived_education is not None
            assert name.perceived_citizenship is not None


def test_the_committed_pools_are_disjoint_and_identifiers_unique():
    development = stimuli.load_pairs(stimuli.DEVELOPMENT)
    confirmatory = stimuli.load_pairs(stimuli.CONFIRMATORY)
    stimuli.disjoint(development, confirmatory)
    assert len(development) == DEVELOPMENT_PAIRS

    identifiers = [
        name.stimulus_id
        for pair in stimuli.load_pairs()
        for name in (pair.white, pair.black)
    ]
    assert len(set(identifiers)) == len(identifiers)
