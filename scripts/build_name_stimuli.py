"""Build the name-stimulus pool from the Validated Names replication data.

The published data is not redistributed in this repository. Two files are
needed, both distributed as CSV alongside their R serialisations:

    study123.csv  one row per respondent-name judgement, pooled over the three
                  surveys (44,170 evaluations of 600 names by 4,026 respondents)
    names.csv     the name roster, with the intended group in `identity`

    python scripts/build_name_stimuli.py --pooled study123.csv --names names.csv

`correct` is 1 when a respondent assigned a name its intended group, so the mean
of `correct` over a name's judgements is the share of respondents who read the
name as intended. That share is the only statistic used for selection, and it is
taken over all respondents rather than a same-group subset, because the quantity
the study needs is how a name reads to a general audience.

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

from hiringcue import config, paths  # noqa: E402

# Exact labels as they appear in the published roster. Matching is exact rather
# than by substring so that the groups this study does not use are dropped
# outright instead of being absorbed by a loose rule.
GROUP_ALIASES = {
    "white": "white",
    "black": "black",
    "african american": "black",
    "black or african american": "black",
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


def build(pooled_path: Path, names_path: Path) -> dict:
    pooled = _read_csv(pooled_path)
    names = _read_csv(names_path)
    if not pooled or not names:
        raise SystemExit("pooled and names files must both be non-empty")

    settings = config.stimuli()["names"]
    threshold = float(settings["minimum_mean_correct"])
    minimum = int(settings["minimum_per_group"])
    attributes = list(settings["report_only_attributes"])

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

    records = []
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
            "mean_correct": round(_mean(scores), 4),
            "n_judgements": len(scores),
        }
        for attribute in attributes:
            record[attribute] = _mean(perceptions[attribute].get(full_name, []))
        records.append(record)

    retained = [record for record in records if record["mean_correct"] >= threshold]

    counts = {group: sum(1 for r in retained if r["group"] == group) for group in ("white", "black")}
    for group, count in counts.items():
        if count < minimum:
            raise SystemExit(
                f"{group}: only {count} names clear mean_correct >= {threshold}, need {minimum}. "
                "Lower the threshold deliberately and record the change, or use a larger source file."
            )

    # A perception column that resolves but parses to nothing produces a pool
    # that looks complete and reports an empty descriptive table. That failure is
    # invisible downstream, so it stops the build here.
    for attribute in attributes:
        blank = [r["full_name"] for r in retained if r[attribute] is None]
        if blank:
            raise SystemExit(
                f"{attribute}: no numeric values in column {perception_keys[attribute]!r} for "
                f"{len(blank)} of {len(retained)} retained names (e.g. {blank[0]!r}). "
                "The column is probably the respondent's answer text rather than its "
                "ordinal encoding."
            )

    return {
        "source": settings["source"],
        "minimum_mean_correct": threshold,
        "counts": counts,
        "names": sorted(retained, key=lambda record: (record["group"], record["full_name"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=paths.STIMULI / "names.json")
    args = parser.parse_args()

    record = build(args.pooled, args.names)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.out} with {record['counts']}")


if __name__ == "__main__":
    main()
