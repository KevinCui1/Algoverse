"""Perturbed soft-criteria twins: the positive control for the score.

A twin is the same occupation, the same applicant, and the same hard gates, with
the non-binding evaluative evidence changed. If a model's score does not move
between a scenario and its twin, the score is not responding to the criteria it
is supposed to reflect, and no identity cue is going to move it either.

The perturbation is authored rather than templated because a template-written
sentence sitting among generated prose would differ in register as well as in
substance, and a score change could then be attributed to the writing rather
than to the evidence. The authoring model is prohibited from touching anything
countable: years, degrees, certifications, licences, and job titles all belong
to the gate layer and must be byte-identical between a scenario and its twin.

Twins are built once per occupation. The soft profile in the source scenario set
does not vary across margin bands within an occupation, so one twin profile
serves all four bands of that occupation.
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


def build_prompt(family: scenarios.ScenarioFamily) -> tuple[str, str]:
    system = (paths.PROMPTS / "soft_twin_author_system_v1.txt").read_text().strip()
    template = (paths.PROMPTS / "soft_twin_author_user_v1.txt").read_text()

    evidence_lines = []
    for entry in family.soft_profile:
        criterion = family.soft_criterion(entry["criterion_id"])
        evidence_lines.append(
            f"- {entry['criterion_id']} | {criterion['criterion']} | "
            f"standing: {entry['position']} | evidence: {entry['candidate_evidence']}"
        )

    change_lines = [
        f"- {change['criterion_id']}: {change['from']} -> {change['to']}"
        for change in requested_changes(family.soft_profile)
    ]

    user = (
        template.replace("{{OCCUPATION_TITLE}}", family.occupation)
        .replace("{{CRITERIA_WITH_CURRENT_EVIDENCE}}", "\n".join(evidence_lines))
        .replace("{{REQUESTED_POSITION_CHANGES}}", "\n".join(change_lines))
    )
    if "{{" in user:
        raise TwinError(f"{family.family_id}: unfilled twin placeholder")
    return system, user


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
            f"{family.occupation_slug}: twin criterion ids differ from or reorder "
            "the source profile"
        )

    wrong_directions = [
        key
        for key in base
        if twin[key]["position"] != POSITION_TARGET[base[key]["position"]]
    ]
    if wrong_directions:
        raise TwinError(
            f"{family.occupation_slug}: twin positions do not match the requested "
            f"directions for {wrong_directions}"
        )

    changed = sum(
        1 for key in base if base[key]["position"] != twin[key]["position"]
    )
    if changed < minimum:
        raise TwinError(
            f"{family.occupation_slug}: twin changes {changed} positions, need {minimum}"
        )

    positions = [entry["position"] for entry in profile]
    if "above" not in positions or "below" not in positions:
        raise TwinError(
            f"{family.occupation_slug}: twin profile is not mixed, so it is scalarisable"
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
                f"{family.occupation_slug}: twin evidence for {entry['criterion_id']} "
                f"has word-count ratio {length_ratio:.2f}, outside "
                f"[{minimum_length:.2f}, {maximum_length:.2f}]"
            )
        for token in banned:
            if token in lowered:
                raise TwinError(
                    f"{family.occupation_slug}: twin evidence for {entry['criterion_id']} "
                    f"mentions {token!r}, which belongs to the hard-gate layer"
                )
        present_pronouns = sorted(
            personal_pronouns.intersection(re.findall(r"[a-z]+", lowered))
        )
        if present_pronouns:
            raise TwinError(
                f"{family.occupation_slug}: twin evidence for {entry['criterion_id']} "
                f"uses personal pronouns {present_pronouns}"
            )


def load(path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    path = path or paths.STIMULI / TWIN_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text())["twins"]


def save(twins: dict[str, list[dict[str, Any]]], path: Path | None = None) -> Path:
    path = path or paths.STIMULI / TWIN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"twins": twins}, indent=2) + "\n")
    return path
