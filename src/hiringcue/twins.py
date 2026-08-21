"""Perturbed soft-criteria twins: the positive control for the score.

A twin is the same occupation, the same applicant, and the same hard gates, with
the non-binding evaluative evidence changed. If a model's score does not move
between a scenario and its twin, the score is not responding to the criteria it
is supposed to reflect, and no identity cue is going to move it either.

The perturbation was authored rather than templated because a template-written
sentence sitting among generated prose would differ in register as well as in
substance, and a movement in the outcome could then be attributed to the writing
rather than to the evidence. Nothing countable may differ between a scenario and
its twin: years, degrees, certifications, licences, and job titles all belong to
the gate layer and are byte-identical across the pair.

Twins exist once per scenario family. Soft profiles vary across the margin bands
of an occupation, so each family carries the perturbation of its own evidence
rather than of a representative band's.

The twin set is frozen and this module loads and re-validates it. Every contract
below is re-checked at load rather than trusted from the file, because a twin
that touches the gate layer would turn the positive control into a measurement
of the qualification rule.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import config, paths, scenarios

TWIN_FILE = "soft_twins.json"

# Direction each starting position is asked to move. Two changes is the declared
# minimum; asking for a full reversal of every criterion would produce a score
# response so large that the gate would pass on any model and stop discriminating.
POSITION_TARGET = {"above": "below", "below": "above", "close": "close"}


class TwinError(ValueError):
    """Raised when a twin profile violates its contract."""


def requested_changes(soft_profile: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "criterion_id": entry["criterion_id"],
            "from": entry["position"],
            "to": POSITION_TARGET[entry["position"]],
        }
        for entry in soft_profile
    ]


def validate(
    family: scenarios.ScenarioFamily, profile: list[dict[str, Any]]
) -> None:
    """Reject a twin that changed too little, or changed the wrong layer."""
    minimum = int(config.study()["soft_twin"]["minimum_positions_changed"])
    minimum_length = float(
        config.study()["soft_twin"]["minimum_word_count_ratio"]
    )
    maximum_length = float(
        config.study()["soft_twin"]["maximum_word_count_ratio"]
    )
    base = {entry["criterion_id"]: entry for entry in family.soft_profile}
    twin = {entry["criterion_id"]: entry for entry in profile}
    base_ids = [entry["criterion_id"] for entry in family.soft_profile]
    twin_ids = [entry["criterion_id"] for entry in profile]

    if base_ids != twin_ids:
        raise TwinError(
            f"{family.family_id}: twin criterion ids differ from or reorder "
            "the source profile"
        )

    wrong_directions = [
        key
        for key in base
        if twin[key]["position"] != POSITION_TARGET[base[key]["position"]]
    ]
    if wrong_directions:
        raise TwinError(
            f"{family.family_id}: twin positions do not match the requested "
            f"directions for {wrong_directions}"
        )

    changed = sum(
        1 for key in base if base[key]["position"] != twin[key]["position"]
    )
    if changed < minimum:
        raise TwinError(
            f"{family.family_id}: twin changes {changed} positions, need {minimum}"
        )

    positions = [entry["position"] for entry in profile]
    if "above" not in positions or "below" not in positions:
        raise TwinError(
            f"{family.family_id}: twin profile is not mixed, so it is scalarisable"
        )

    # Anything countable belongs to the gate layer and must not appear here,
    # otherwise the twin would move a gate and stop being a clean control.
    banned = ("year", "degree", "certif", "licen", "diploma", "bachelor", "master")
    personal_pronouns = {
        "he",
        "her",
        "hers",
        "him",
        "his",
        "she",
        "their",
        "theirs",
        "them",
        "they",
    }
    for entry in profile:
        lowered = entry["candidate_evidence"].casefold()
        base_length = len(
            re.findall(
                r"[a-z0-9]+",
                base[entry["criterion_id"]]["candidate_evidence"].casefold(),
            )
        )
        twin_length = len(re.findall(r"[a-z0-9]+", lowered))
        length_ratio = twin_length / base_length
        if not minimum_length <= length_ratio <= maximum_length:
            raise TwinError(
                f"{family.family_id}: twin evidence for {entry['criterion_id']} "
                f"has word-count ratio {length_ratio:.2f}, outside "
                f"[{minimum_length:.2f}, {maximum_length:.2f}]"
            )
        for token in banned:
            if token in lowered:
                raise TwinError(
                    f"{family.family_id}: twin evidence for {entry['criterion_id']} "
                    f"mentions {token!r}, which belongs to the hard-gate layer"
                )
        present_pronouns = sorted(
            personal_pronouns.intersection(re.findall(r"[a-z]+", lowered))
        )
        if present_pronouns:
            raise TwinError(
                f"{family.family_id}: twin evidence for {entry['criterion_id']} "
                f"uses personal pronouns {present_pronouns}"
            )


def load(
    path: Path | None = None,
    families: list[scenarios.ScenarioFamily] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Twin profiles keyed by scenario family.

    A pool authored before twins were keyed per family is keyed by occupation.
    Such a pool is expanded across that occupation's families rather than
    re-authored, but only where every one of those families carries the same
    soft profile - which is the condition under which one twin was a valid
    control for all of them in the first place. Where the profiles differ the
    expansion is refused, because the perturbation of one band's evidence is not
    the perturbation of another's.

    A family-keyed pool is re-checked against the profile it is loaded with. A
    twin authored from an earlier scenario set carries the right family
    identifiers while perturbing evidence that family no longer has, so it would
    load without complaint and turn the positive control into a comparison
    against unrelated text. Because the control is what establishes that the
    score responds to the criteria at all, that failure would be invisible in
    every downstream number.
    """
    path = path or paths.STIMULI / TWIN_FILE
    if not path.exists():
        return {}
    stored = json.loads(path.read_text())["twins"]

    families = families if families is not None else scenarios.validated_families()
    by_family = {family.family_id: family for family in families}
    if not set(stored) - set(by_family):
        stale = {}
        for family_id, profile in stored.items():
            try:
                validate(by_family[family_id], profile)
            except TwinError as exc:
                stale[family_id] = str(exc)
        if stale:
            raise TwinError(
                "the stored twin pool is not the perturbation of the scenario set "
                f"it was loaded with, for {sorted(stale)}; author twins again for "
                f"this set. First mismatch: {stale[sorted(stale)[0]]}"
            )
        return stored

    expanded: dict[str, list[dict[str, Any]]] = {}
    for slug, group in scenarios.iter_by_occupation(families):
        profile = stored.get(slug)
        if profile is None:
            continue
        shapes = {scenarios.profile_shape(family) for family in group}
        evidence = {
            tuple(entry["candidate_evidence"] for entry in family.soft_profile)
            for family in group
        }
        if len(shapes) > 1 or len(evidence) > 1:
            raise TwinError(
                f"{slug}: twin pool is keyed by occupation but this occupation's "
                "families no longer share one soft profile, so the stored twin is "
                "not the perturbation of each of them; author twins per family"
            )
        for family in group:
            expanded[family.family_id] = profile
    return expanded


def save(twins: dict[str, list[dict[str, Any]]], path: Path | None = None) -> Path:
    path = path or paths.STIMULI / TWIN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"twins": twins}, indent=2) + "\n")
    return path
