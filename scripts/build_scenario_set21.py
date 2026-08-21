#!/usr/bin/env python3
"""Build scenario set 2.1.0: soft-evidence profiles that vary in overall strength.

Set 2.0.0 gave every family its own concrete evidence and crossed profile shape
with occupation and margin band, which removed the collinearity between the soft
layer and occupation. It did not vary how strong a family's evidence is. Its
eight shapes are permutations of one multiset per criterion-count stratum -
(above, above, close, below) at four criteria and (above, above, above, close,
below) at five - so every family in the set carries the same number of
above-expectation and below-expectation criteria, and the families differ only in
which criterion occupies which position.

Within-band dispersion of the suitability score is a statistic about how
differently the model rates applicants who all clear the same qualification bar.
An evaluator that forms an overall impression from the evidence has no reason to
separate profiles that are equal in composition, so a set built this way holds
that dispersion near zero by construction rather than measuring it.

Set 2.1.0 keeps everything 2.0.0 established - one profile per family, concrete
non-countable evidence, position rotated across occupation and band - and adds
the missing factor: the number of above- and below-expectation criteria varies
across the families within each margin band, so applicants who clear the same bar
differ in how strong their non-binding evidence is.

Nothing binding moves. Hard gates, candidate gate values, soft-criterion
definitions, occupations, and the gold decision are untouched, and the evidence
sentences are the ones already authored for 2.0.0, re-assigned rather than
rewritten. Every family keeps exactly one close-to-expectation criterion, which
holds the near-threshold ambiguity floor exactly where 2.0.0 left it, and keeps
at least one above and one below, which is the mixed-profile requirement.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from build_scenario_set2 import BANDS, EVIDENCE, SOURCE, _candidate_summary

ROOT = Path(__file__).resolve().parents[1]
STUDY_CONFIG = ROOT / "configs" / "study.yaml"
PREVIOUS_VERSION = "2.0.0"
SET_VERSION = "2.1.0"

# Strength levels, as the multiset of positions each carries. One criterion is
# close to expectation at every level, so the ambiguity floor sees the same value
# it saw in 2.0.0; the remaining criteria split between above and below. At four
# criteria the mixed-profile requirement admits two levels, at five it admits
# three.
LEVELS = {
    4: {
        "high": ("above", "above", "close", "below"),
        "low": ("above", "close", "below", "below"),
    },
    5: {
        "high": ("above", "above", "above", "close", "below"),
        "mid": ("above", "above", "close", "below", "below"),
        "low": ("above", "close", "below", "below", "below"),
    },
}

# Level per occupation and margin band. Every band carries more than one level,
# so within-band strength varies; every occupation carries more than one level,
# so strength is not a fixed property of the job; and the two-level and
# three-level strata are each spread evenly across the set.
LEVEL_PLAN = {
    "environmental-counselor": ("high", "low", "high", "low"),
    "software-engineer": ("low", "high", "low", "high"),
    "veterinarian": ("high", "high", "low", "low"),
    "ceo": ("high", "mid", "low", "high"),
    "chefs-and-head-cooks": ("mid", "low", "high", "mid"),
    "preschool-teacher": ("low", "high", "mid", "low"),
}

# Rotation offset per occupation. Applied to the level's position multiset so
# that which criterion sits above or below is crossed with occupation and band
# rather than fixed by the level, which is what keeps position and strength from
# being read as the same variable.
ROTATIONS = {
    "environmental-counselor": 0,
    "software-engineer": 1,
    "veterinarian": 2,
    "ceo": 0,
    "chefs-and-head-cooks": 1,
    "preschool-teacher": 2,
}

FORBIDDEN_LABELS = (
    "strong ability",
    "moderate ability",
    "limited ability",
    "above-level",
    "below-level",
    "close-level",
)


def _rotate(positions: tuple[str, ...], offset: int) -> tuple[str, ...]:
    shift = offset % len(positions)
    return positions[shift:] + positions[:shift]


def _strength(positions: tuple[str, ...]) -> float:
    """Above-expectation criteria minus below-expectation, as a share of the profile.

    The quantity a within-band dispersion statistic needs to have variance in.
    It is descriptive and is never rendered into a prompt.
    """
    counts = Counter(positions)
    return (counts["above"] - counts["below"]) / len(positions)


def _bump_scenario_set_version() -> bool:
    text = STUDY_CONFIG.read_text()
    current = f'scenario_set_version: "{SET_VERSION}"'
    if current in text:
        return False
    previous = f'scenario_set_version: "{PREVIOUS_VERSION}"'
    if text.count(previous) != 1:
        raise ValueError(
            f"expected exactly one {previous!r} in {STUDY_CONFIG.name}; refusing to guess"
        )
    STUDY_CONFIG.write_text(text.replace(previous, current))
    return True


def build() -> list[Path]:
    changed: list[Path] = []
    realised_shapes: Counter = Counter()
    realised_profiles: set = set()
    strength_by_band: dict[str, list[float]] = {band: [] for band in BANDS}
    levels_by_occupation: dict[str, set] = {}
    emitted_evidence: set = set()

    for slug, plan in LEVEL_PLAN.items():
        path = SOURCE / slug / "output.json"
        record = json.loads(path.read_text())
        candidates = {item["margin_band"]: item for item in record["candidate_scenarios"]}
        if set(candidates) != set(BANDS):
            raise ValueError(f"{slug}: expected exactly the four frozen margin bands")

        criterion_ids = [entry["criterion_id"] for entry in record["soft_criteria"]]
        if set(criterion_ids) != set(EVIDENCE[slug]):
            raise ValueError(f"{slug}: evidence map does not match source criteria")
        levels = LEVELS[len(criterion_ids)]
        levels_by_occupation[slug] = set(plan)
        if not set(plan) <= set(levels):
            raise ValueError(
                f"{slug}: level plan {plan} is not available at {len(criterion_ids)} criteria"
            )

        for band_index, band in enumerate(BANDS):
            candidate = candidates[band]
            positions = _rotate(levels[plan[band_index]], ROTATIONS[slug] + band_index)
            if "above" not in positions or "below" not in positions:
                raise ValueError(f"{slug}/{band}: profile is not mixed")
            if positions.count("close") != 1:
                raise ValueError(
                    f"{slug}/{band}: every profile carries exactly one close criterion "
                    "so that the near-threshold ambiguity floor is unchanged"
                )

            candidate["soft_profile"] = [
                {
                    "criterion_id": criterion_id,
                    "candidate_evidence": EVIDENCE[slug][criterion_id][position],
                    "position": position,
                }
                for criterion_id, position in zip(criterion_ids, positions)
            ]
            candidate["candidate_summary"] = _candidate_summary(record, candidate)

            emitted_evidence.update(
                entry["candidate_evidence"] for entry in candidate["soft_profile"]
            )
            realised_shapes[tuple(zip(criterion_ids, positions))] += 1
            realised_profiles.add(
                tuple(entry["candidate_evidence"] for entry in candidate["soft_profile"])
            )
            strength_by_band[band].append(_strength(positions))

        record["candidate_scenarios"] = [candidates[band] for band in BANDS]
        path.write_text(json.dumps(record, indent=2) + "\n")
        changed.append(path)

    if len(realised_profiles) != 24:
        raise ValueError(f"expected 24 distinct profiles, got {len(realised_profiles)}")

    # The property this set exists to establish: applicants who clear the same
    # bar differ in how strong their non-binding evidence is. A band whose
    # families are equal in strength would reproduce the defect being repaired.
    for band, strengths in strength_by_band.items():
        if len(set(strengths)) < 2:
            raise ValueError(f"{band}: soft-evidence strength does not vary within the band")

    for slug, levels in levels_by_occupation.items():
        if len(levels) < 2:
            raise ValueError(f"{slug}: strength is constant across this occupation's bands")

    limit = 2 * -(-len(realised_profiles) // len(realised_shapes))
    if max(realised_shapes.values()) > limit:
        raise ValueError(f"profile shapes are not spread across the set: {realised_shapes}")

    for statement in emitted_evidence:
        lowered = statement.casefold()
        found = [label for label in FORBIDDEN_LABELS if label in lowered]
        if found:
            raise ValueError(f"evaluative level labels remain in {statement!r}: {found}")
        # Countables belong to the hard-gate layer. A soft statement carrying one
        # would let the non-binding layer restate a gate the score already reads.
        if re.search(r"\b\d+\b", lowered):
            raise ValueError(f"soft evidence carries a countable quantity: {statement!r}")

    return changed


if __name__ == "__main__":
    files = build()
    bumped = _bump_scenario_set_version()
    print(f"wrote scenario set {SET_VERSION} soft profiles to {len(files)} source files")
    print(
        f"scenario_set_version {'set to ' + SET_VERSION if bumped else 'already ' + SET_VERSION}"
    )
