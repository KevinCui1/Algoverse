"""Checks on the name-stimulus pool and the script that builds it.

The build reads the published replication data, which is not vendored here, so
these tests run against synthetic files with the same column structure. What
they protect is the mapping between that structure and the pool: which intended
groups are recognised, which columns the perception statistics come from, and
which conditions must stop the build rather than produce a thin or hollow pool.
"""

import csv
import importlib.util

import pytest

from hiringcue import config, paths, stimuli

_spec = importlib.util.spec_from_file_location(
    "build_name_stimuli", paths.ROOT / "scripts" / "build_name_stimuli.py"
)
build_name_stimuli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_name_stimuli)

SETTINGS = config.stimuli()["names"]
THRESHOLD = float(SETTINGS["minimum_mean_correct"])
MINIMUM = int(SETTINGS["minimum_per_group"])

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
# the two it uses for the groups that must not enter the pool.
WHITE = "White"
BLACK = "Black or African American"
UNUSED = ("Hispanic", "Asian or Pacific Islander")


def _write(path, columns, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _judgements(name, identity, share_correct, judgements=20):
    """One row per respondent judgement, `share_correct` of them correct."""
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


def _corpus(tmp_path, share_correct=0.95, per_group=None, extra_identities=()):
    per_group = per_group or MINIMUM + 2
    roster, pooled = [], []
    for identity, tag in ((WHITE, "w"), (BLACK, "b")):
        for index in range(per_group):
            name = f"{tag.upper()}{index} Surname"
            roster.append({"name": name, "identity": identity})
            pooled.extend(_judgements(name, identity, share_correct))
    for identity in extra_identities:
        name = f"X{identity[:3]} Surname"
        roster.append({"name": name, "identity": identity})
        pooled.extend(_judgements(name, identity, 0.99))
    return (
        _write(tmp_path / "pooled.csv", POOLED_COLUMNS, pooled),
        _write(tmp_path / "names.csv", ["name", "identity"], roster),
    )


def test_roster_label_for_black_names_is_recognised(tmp_path):
    pooled, names = _corpus(tmp_path)
    record = build_name_stimuli.build(pooled, names)
    assert record["counts"]["black"] == MINIMUM + 2
    assert record["counts"]["white"] == MINIMUM + 2


def test_groups_outside_the_study_never_enter_the_pool(tmp_path):
    pooled, names = _corpus(tmp_path, extra_identities=UNUSED)
    record = build_name_stimuli.build(pooled, names)
    assert {entry["group"] for entry in record["names"]} == {"white", "black"}


def test_perception_statistics_come_from_the_ordinal_columns(tmp_path):
    """The answer-text column shares a stem with the ordinal one and parses to
    nothing, so preferring the wrong one yields a pool with an empty table."""
    pooled, names = _corpus(tmp_path)
    record = build_name_stimuli.build(pooled, names)
    for entry in record["names"]:
        assert entry["perceived_income"] == 2
        assert entry["perceived_education"] == 3
        assert entry["perceived_citizenship"] == 1


def test_a_perception_column_that_parses_to_nothing_stops_the_build(tmp_path):
    pooled, names = _corpus(tmp_path)
    rows = list(csv.DictReader(pooled.open(newline="")))
    for row in rows:
        row["income.ord"] = "NA"
    _write(pooled, POOLED_COLUMNS, rows)
    with pytest.raises(SystemExit, match="perceived_income"):
        build_name_stimuli.build(pooled, names)


def test_a_missing_perception_column_stops_the_build(tmp_path):
    pooled, names = _corpus(tmp_path)
    columns = [column for column in POOLED_COLUMNS if not column.startswith("citizen")]
    rows = [
        {key: value for key, value in row.items() if key in columns}
        for row in csv.DictReader(pooled.open(newline=""))
    ]
    _write(pooled, columns, rows)
    with pytest.raises(SystemExit, match="perceived_citizenship"):
        build_name_stimuli.build(pooled, names)


def test_names_below_the_threshold_are_dropped(tmp_path):
    pooled, names = _corpus(tmp_path, share_correct=THRESHOLD - 0.1)
    with pytest.raises(SystemExit, match="clear mean_correct"):
        build_name_stimuli.build(pooled, names)


def test_a_pool_short_of_the_per_group_minimum_stops_the_build(tmp_path):
    pooled, names = _corpus(tmp_path, per_group=MINIMUM - 1)
    with pytest.raises(SystemExit, match=f"need {MINIMUM}"):
        build_name_stimuli.build(pooled, names)


def test_the_committed_pool_meets_its_declared_requirements():
    pool = stimuli.load_names(paths.STIMULI / "names.json")
    for group, entries in pool.items():
        assert len(entries) >= MINIMUM, group
        assert min(entry.mean_correct for entry in entries) >= THRESHOLD, group
        # The descriptive perception table is the whole reason these are carried
        # through; a null here means the pool reports nothing about its bundle.
        for entry in entries:
            assert entry.perceived_income is not None
            assert entry.perceived_education is not None
            assert entry.perceived_citizenship is not None


def test_stimulus_identifiers_are_unique():
    pool = stimuli.load_names(paths.STIMULI / "names.json")
    identifiers = [entry.stimulus_id for group in pool.values() for entry in group]
    assert len(set(identifiers)) == len(identifiers)
