"""Identity and credential-prestige stimuli.

Two manipulations are crossed with every scenario family.

*Name cues.* First-and-last-name stimuli drawn from the Validated Names
replication data, retained only above a perception threshold recorded in
`configs/stimuli.yaml`. A name is a bundled treatment: it carries perceived
race together with perceived class, citizenship, and name-specific
associations, and the published perception ratings for those attributes are
mediators of race rather than rival causes. They are therefore reported
descriptively and never used to match or adjust. The measured quantity is the
effect of a name cue as it would arrive in a deployed system.

*Credential prestige.* Two institution stimuli, identical in every fact that
bears on the qualification rule and differing only in institutional standing.
This manipulates perceived class directly instead of adjusting for it, and
supplies a within-study reference effect against which the name effect can be
sized.

The name file is not vendored. `data/stimuli/names.json` is produced by
`scripts/build_name_stimuli.py` from the published replication data and is
required before any prompt can be rendered.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from . import config, paths

NEUTRAL = "neutral"
CONCEALED_WHITE = "concealed_white"
CONCEALED_BLACK = "concealed_black"
DIRECT_WHITE = "direct_white"
DIRECT_BLACK = "direct_black"

CONDITIONS = (NEUTRAL, CONCEALED_WHITE, CONCEALED_BLACK, DIRECT_WHITE, DIRECT_BLACK)
CONCEALED_CONDITIONS = (CONCEALED_WHITE, CONCEALED_BLACK)
DIRECT_CONDITIONS = (DIRECT_WHITE, DIRECT_BLACK)

GROUP_OF_CONDITION = {
    CONCEALED_WHITE: "white",
    CONCEALED_BLACK: "black",
    DIRECT_WHITE: "white",
    DIRECT_BLACK: "black",
}

CUE_MODE_OF_CONDITION = {
    NEUTRAL: "neutral",
    CONCEALED_WHITE: "concealed",
    CONCEALED_BLACK: "concealed",
    DIRECT_WHITE: "direct",
    DIRECT_BLACK: "direct",
}


class StimulusError(ValueError):
    """Raised when the stimulus pool does not meet its declared requirements."""


@dataclass(frozen=True)
class Name:
    stimulus_id: str
    full_name: str
    group: str
    mean_correct: float
    perceived_income: float | None
    perceived_education: float | None
    perceived_citizenship: float | None


@dataclass(frozen=True)
class Prestige:
    level: str
    stimulus_id: str
    text: str


def name_pool_path(allow_provisional: bool = False):
    """Locate the name pool, preferring the validated file over the provisional one."""
    validated = paths.STIMULI / "names.json"
    if validated.exists():
        return validated
    provisional = paths.STIMULI / "names.provisional.json"
    if allow_provisional and provisional.exists():
        return provisional
    raise StimulusError(
        f"{validated} is missing. Build it with scripts/build_name_stimuli.py, or pass "
        "--allow-provisional to run a dry run against the provisional pool."
    )


def is_provisional(path) -> bool:
    return json.loads(path.read_text()).get("provisional", False)


def load_names(path=None, allow_provisional: bool = False) -> dict[str, list[Name]]:
    """Load the name pool and enforce the declared threshold and pool size."""
    path = path or name_pool_path(allow_provisional=allow_provisional)
    settings = config.stimuli()["names"]
    threshold = float(settings["minimum_mean_correct"])
    minimum_per_group = int(settings["minimum_per_group"])

    record = json.loads(path.read_text())
    pool: dict[str, list[Name]] = {"white": [], "black": []}
    for entry in record["names"]:
        group = entry["group"]
        if group not in pool:
            raise StimulusError(f"unexpected name group {group!r}")
        if float(entry["mean_correct"]) < threshold:
            continue
        pool[group].append(
            Name(
                stimulus_id=entry["stimulus_id"],
                full_name=entry["full_name"],
                group=group,
                mean_correct=float(entry["mean_correct"]),
                perceived_income=entry.get("perceived_income"),
                perceived_education=entry.get("perceived_education"),
                perceived_citizenship=entry.get("perceived_citizenship"),
            )
        )

    for group, names in pool.items():
        if len(names) < minimum_per_group:
            raise StimulusError(
                f"{group}: {len(names)} names clear mean_correct >= {threshold}, "
                f"need {minimum_per_group}. A pool this small lets one stimulus carry the result."
            )
        if len({name.full_name for name in names}) != len(names):
            raise StimulusError(f"{group}: duplicate names in the pool")
    return pool


def load_prestige() -> dict[str, Prestige]:
    settings = config.stimuli()["prestige"]
    levels = {}
    for level in ("modest", "elite"):
        entry = settings[level]
        levels[level] = Prestige(
            level=level, stimulus_id=entry["stimulus_id"], text=entry["text"]
        )
    return levels


def assign_names(
    family_ids: list[str], pool: dict[str, list[Name]], seed: int
) -> dict[str, dict[str, Name]]:
    """Assign one White-associated and one Black-associated name per family.

    Names are drawn round-robin from a shuffled pool so each stimulus appears a
    near-equal number of times across occupations and margin bands. Balance is
    what makes the leave-one-name-out check meaningful: if one name carried the
    effect and appeared only once, dropping it would look like noise.
    """
    rng = random.Random(seed)
    assignment: dict[str, dict[str, Name]] = {}
    cycles: dict[str, list[Name]] = {}
    for group, names in pool.items():
        shuffled = list(names)
        rng.shuffle(shuffled)
        cycles[group] = shuffled

    for index, family_id in enumerate(sorted(family_ids)):
        assignment[family_id] = {
            group: cycles[group][index % len(cycles[group])] for group in cycles
        }
    return assignment


def identity_block(condition: str, name: Name | None) -> str:
    """The only span of the prompt that differs within a counterfactual set."""
    labels = config.stimuli()["identity_labels"]
    if condition == NEUTRAL:
        return labels["neutral_identifier"]
    if condition in CONCEALED_CONDITIONS:
        if name is None:
            raise StimulusError(f"{condition} requires a name stimulus")
        return f"Applicant name: {name.full_name}"
    if condition in DIRECT_CONDITIONS:
        group = GROUP_OF_CONDITION[condition]
        return f"{labels['neutral_identifier']}\nThe applicant self-identifies as {labels[group]}."
    raise StimulusError(f"unknown condition {condition!r}")


def reflection_label(condition: str) -> str | None:
    """The group wording used in the second turn, matched to the first-turn cue."""
    labels = config.stimuli()["identity_labels"]
    group = GROUP_OF_CONDITION.get(condition)
    return labels[group] if group else None


def pool_summary(pool: dict[str, list[Name]]) -> dict[str, Any]:
    """Descriptive report of what each name group's bundle contains.

    Reported, never adjusted for: perceived income and education are downstream
    of perceived race, so conditioning on them would remove part of the effect
    being estimated.
    """

    def stats(values: list[float]) -> dict[str, float] | None:
        clean = [value for value in values if value is not None]
        if not clean:
            return None
        clean.sort()
        return {
            "n": len(clean),
            "min": clean[0],
            "median": clean[len(clean) // 2],
            "max": clean[-1],
            "mean": sum(clean) / len(clean),
        }

    return {
        group: {
            "n": len(names),
            "mean_correct": stats([name.mean_correct for name in names]),
            "perceived_income": stats([name.perceived_income for name in names]),
            "perceived_education": stats([name.perceived_education for name in names]),
            "perceived_citizenship": stats([name.perceived_citizenship for name in names]),
        }
        for group, names in pool.items()
    }
