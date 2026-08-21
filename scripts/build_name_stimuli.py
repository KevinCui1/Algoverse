"""Build the matched name-pair pool from the Validated Names replication data.

The published data is not redistributed in this repository. Two files are
needed, both distributed as CSV alongside their R serialisations:

    study123.csv  one row per respondent-name judgement, pooled over the three
                  surveys (44,170 evaluations of 600 names by 4,026 respondents)
    names.csv     the name roster, with the intended group in `identity`

    python scripts/build_name_stimuli.py --pooled study123.csv --names names.csv

`correct` is 1 when a respondent assigned a name its intended group, so the mean
of `correct` over a name's judgements is that name's attribution accuracy: the
share of respondents who read the name as intended. It is taken over all
respondents rather than a same-group subset, because the quantity the study
needs is how a name reads to a general audience.

The pool is a set of *pairs*, one name from each arm, matched on attribution
accuracy. The two arms do not share that distribution in the source data, so a
single threshold applied to both selects them differently and confounds cue
strength with the group signalled. Matching is what removes that confound; the
retained floor is whatever the matching implies.

Perceived income, education, and citizenship are carried through for descriptive
reporting and are deliberately not used to match or filter: those perceptions are
downstream of perceived race, so selecting on them would remove part of the
effect being estimated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiringcue import config, matching, paths  # noqa: E402

# Exact labels as they appear in the published roster. Matching is exact rather
# than by substring so that the groups this study does not use are dropped
# outright instead of being absorbed by a loose rule.
GROUP_ALIASES = {
    "white": matching.WHITE,
    "black": matching.BLACK,
    "african american": matching.BLACK,
    "black or african american": matching.BLACK,
}

# The published file carries each perception twice: once as the respondent's
# answer text and once as an ordinal encoding of it. The ordinal is the numeric
# one, and it shares a stem with the text column, so it has to be preferred
# explicitly or the text column wins and every value parses to nothing.
PERCEPTION_COLUMNS = {
    "perceived_income": ("income.ord", "avg_income", "perceived_income", "income"),
    "perceived_education": ("education.ord", "avg_education", "perceived_education", "education"),
    "perceived_citizenship": ("citizen", "avg_citizenship", "perceived_citizenship", "citizenship"),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _column(row: dict[str, str], *candidates: str) -> str | None:
    for candidate in candidates:
        for key in row:
            if key.strip().casefold() == candidate:
                return key
    return None


def _to_float(value: str | None) -> float | None:
    if value in (None, "", "NA", "NaN"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _roster(pooled_path: Path, names_path: Path, attributes: list[str]) -> list[dict]:
    """One record per rostered name, with its attribution accuracy and perceptions."""
    pooled = _read_csv(pooled_path)
    names = _read_csv(names_path)
    if not pooled or not names:
        raise SystemExit("pooled and names files must both be non-empty")

    name_key = _column(pooled[0], "name")
    correct_key = _column(pooled[0], "correct")
    if not name_key or not correct_key:
        raise SystemExit(f"pooled file needs 'name' and 'correct' columns, saw {list(pooled[0])}")

    perception_keys: dict[str, str] = {}
    for attribute in attributes:
        candidates = PERCEPTION_COLUMNS.get(attribute)
        if candidates is None:
            raise SystemExit(
                f"{attribute} is declared in configs/stimuli.yaml but this script has no "
                f"column mapping for it. Add one to PERCEPTION_COLUMNS."
            )
        key = _column(pooled[0], *candidates)
        if key is None:
            raise SystemExit(
                f"{attribute}: none of {list(candidates)} present in the pooled file, "
                f"saw {list(pooled[0])}"
            )
        perception_keys[attribute] = key

    judgements: dict[str, list[float]] = defaultdict(list)
    perceptions: dict[str, dict[str, list[float]]] = {
        attribute: defaultdict(list) for attribute in attributes
    }
    for row in pooled:
        name = row[name_key].strip()
        value = _to_float(row[correct_key])
        if not name or value is None:
            continue
        judgements[name].append(value)
        for attribute, key in perception_keys.items():
            parsed = _to_float(row.get(key))
            if parsed is not None:
                perceptions[attribute][name].append(parsed)

    identity_key = _column(names[0], "identity", "race", "intended_race")
    names_name_key = _column(names[0], "name")
    first_key = _column(names[0], "first")
    last_key = _column(names[0], "last")
    if not identity_key:
        raise SystemExit(f"names file needs an identity column, saw {list(names[0])}")

    records: list[dict] = []
    seen: set[str] = set()
    for row in names:
        group = GROUP_ALIASES.get(row[identity_key].strip().casefold())
        if group is None:
            continue
        if names_name_key:
            full_name = row[names_name_key].strip()
        elif first_key and last_key:
            full_name = f"{row[first_key].strip()} {row[last_key].strip()}"
        else:
            raise SystemExit("names file needs either a 'name' column or 'first' and 'last'")
        if not full_name or full_name in seen:
            continue
        scores = judgements.get(full_name)
        if not scores:
            continue
        seen.add(full_name)
        record = {
            "stimulus_id": f"vn_{group}_{len(records):03d}",
            "full_name": full_name,
            "group": group,
            "attribution_accuracy": round(_mean(scores), 4),
            "n_judgements": len(scores),
        }
        for attribute in attributes:
            record[attribute] = _mean(perceptions[attribute].get(full_name, []))
        records.append(record)

    # A perception column that resolves but parses to nothing produces a roster
    # that looks complete and reports an empty descriptive table. That failure is
    # invisible downstream, so it stops the build here.
    for attribute in attributes:
        blank = [record["full_name"] for record in records if record[attribute] is None]
        if blank:
            raise SystemExit(
                f"{attribute}: no numeric values in column {perception_keys[attribute]!r} for "
                f"{len(blank)} of {len(records)} rostered names (e.g. {blank[0]!r}). "
                "The column is probably the respondent's answer text rather than its "
                "ordinal encoding."
            )
    return records


def _roster_summary(records: list[dict]) -> dict:
    summary = {}
    for group in (matching.WHITE, matching.BLACK):
        values = sorted(
            record["attribution_accuracy"] for record in records if record["group"] == group
        )
        summary[group] = {
            "n": len(values),
            "min": values[0],
            "median": values[len(values) // 2],
            "max": values[-1],
            "mean": round(sum(values) / len(values), 4),
        }
    return summary


def build(pooled_path: Path, names_path: Path) -> dict:
    settings = config.stimuli()["names"]
    rules = settings["matching"]
    attributes = list(settings["report_only_attributes"])

    records = _roster(pooled_path, names_path, attributes)
    by_id = {record["stimulus_id"]: record for record in records}
    candidates = [
        matching.Candidate(
            stimulus_id=record["stimulus_id"],
            full_name=record["full_name"],
            group=record["group"],
            attribution_accuracy=record["attribution_accuracy"],
        )
        for record in records
    ]

    pool = matching.build_pool(
        candidates,
        caliper_ladder=[float(value) for value in rules["caliper_ladder"]],
        minimum_pairs=int(rules["minimum_pairs"]),
        maximum_standardised_mean_difference=float(
            rules["maximum_standardised_mean_difference"]
        ),
    )
    development, confirmatory = matching.split(
        pool,
        development_pairs=int(rules["development_pairs"]),
        seed=int(rules["split_seed"]),
    )
    reserved = {pair.pair_id for pair in development}

    return {
        "source": settings["source"],
        "roster": _roster_summary(records),
        "caliper": pool.caliper,
        "balance": pool.balance,
        "counts": {
            "pairs": len(pool.pairs),
            "development": len(development),
            "confirmatory": len(confirmatory),
        },
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "role": "development" if pair.pair_id in reserved else "confirmatory",
                "accuracy_difference": round(pair.accuracy_difference, 4),
                "names": {
                    arm: {
                        key: value
                        for key, value in by_id[candidate.stimulus_id].items()
                        if key != "group"
                    }
                    for arm, candidate in (
                        (matching.WHITE, pair.white),
                        (matching.BLACK, pair.black),
                    )
                },
            }
            for pair in pool.pairs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=paths.STIMULI / "name_pairs.json")
    args = parser.parse_args()

    record = build(args.pooled, args.names)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"wrote {args.out}: {record['counts']} at caliper {record['caliper']}, "
        f"standardised mean difference "
        f"{record['balance']['standardised_mean_difference']:+.4f}, "
        f"floor {record['balance']['floor']:.4f}"
    )


if __name__ == "__main__":
    main()
