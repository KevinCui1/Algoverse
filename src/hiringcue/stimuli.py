"""Identity and credential-prestige stimuli.

Two manipulations are crossed with every scenario family.

*Name cues.* First-and-last-name stimuli drawn from the Validated Names
replication data and organised as matched pairs, one name from each arm, whose
attribution accuracies differ by no more than a recorded caliper. Matching
rather than thresholding is what makes the arms comparable: the two rosters do
not share an attribution-accuracy distribution, so a single threshold applied to
both selects one near its centre and the other from its upper tail, which
confounds cue strength with the group signalled. A name remains a bundled
treatment - it carries perceived class, citizenship and name-specific
associations alongside perceived race - and those perceptions are mediators of
race rather than rival causes, so they are reported descriptively and never used
to match or adjust. The measured quantity is the effect of a name cue as it
would arrive in a deployed system.

*Credential prestige.* Two institution stimuli, identical in every fact that
bears on the qualification rule and differing only in institutional standing.
This manipulates perceived class directly instead of adjusting for it, and
supplies a within-study reference effect against which the name effect can be
sized.

The name pair is a random factor in the analysis, so pairs are crossed with
every scenario family rather than assigned one per family. A fixed set of pairs
reused across families would put every family mean on the same draw of the name
effect, where a family-clustered interval cannot see it.

The pair file is not vendored. `data/stimuli/name_pairs.json` is produced by
`scripts/build_name_stimuli.py` from the published replication data and is
required before any prompt can be rendered.
"""

from __future__ import annotations

import json
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


DEVELOPMENT = "development"
CONFIRMATORY = "confirmatory"


@dataclass(frozen=True)
class Name:
    stimulus_id: str
    full_name: str
    group: str
    attribution_accuracy: float
    perceived_income: float | None
    perceived_education: float | None
    perceived_citizenship: float | None


@dataclass(frozen=True)
class NamePair:
    """One matched White-associated and Black-associated stimulus."""

    pair_id: str
    role: str
    white: Name
    black: Name

    def arm(self, group: str) -> Name:
        if group == "white":
            return self.white
        if group == "black":
            return self.black
        raise StimulusError(f"unknown identity arm {group!r}")


@dataclass(frozen=True)
class Prestige:
    level: str
    stimulus_id: str
    text: str


def name_pair_path():
    path = paths.STIMULI / "name_pairs.json"
    if not path.exists():
        raise StimulusError(
            f"{path} is missing. Build it with scripts/build_name_stimuli.py from the "
            "published replication data; there is no substitute pool, because an "
            "unmatched pool reintroduces the arm asymmetry the matching removes."
        )
    return path


def _name(record: dict[str, Any], group: str) -> Name:
    return Name(
        stimulus_id=record["stimulus_id"],
        full_name=record["full_name"],
        group=group,
        attribution_accuracy=float(record["attribution_accuracy"]),
        perceived_income=record.get("perceived_income"),
        perceived_education=record.get("perceived_education"),
        perceived_citizenship=record.get("perceived_citizenship"),
    )


def load_pairs(role: str | None = None, path=None) -> list[NamePair]:
    """Load the matched pairs, optionally restricted to one role.

    Development and confirmatory pools are disjoint by construction and the
    restriction is applied here rather than by the caller, so a round cannot
    silently draw from the other pool. A deterministic readout returns the same
    value for the same prompt, so a development prompt recollected for
    confirmation reproduces the development measurement rather than producing an
    independent observation.
    """
    path = path or name_pair_path()
    record = json.loads(path.read_text())
    settings = config.stimuli()["names"]["matching"]

    pairs = [
        NamePair(
            pair_id=entry["pair_id"],
            role=entry["role"],
            white=_name(entry["names"]["white"], "white"),
            black=_name(entry["names"]["black"], "black"),
        )
        for entry in record["pairs"]
        if role is None or entry["role"] == role
    ]
    if role is None and len(pairs) < int(settings["minimum_pairs"]):
        raise StimulusError(
            f"{len(pairs)} matched pairs available, need {settings['minimum_pairs']}. "
            "Report the shortfall and re-run the sizing rule against the achievable "
            "pair count rather than relaxing the matching."
        )
    if not pairs:
        raise StimulusError(f"no name pairs with role {role!r} in {path}")

    identifiers = [name.stimulus_id for pair in pairs for name in (pair.white, pair.black)]
    if len(set(identifiers)) != len(identifiers):
        raise StimulusError(f"{path}: a name stimulus appears in more than one pair")
    return pairs


def disjoint(left: list[NamePair], right: list[NamePair]) -> None:
    """Hard stop on any overlap between two pools."""
    shared = {pair.pair_id for pair in left} & {pair.pair_id for pair in right}
    names = {name.full_name for pair in left for name in (pair.white, pair.black)} & {
        name.full_name for pair in right for name in (pair.white, pair.black)
    }
    if shared or names:
        raise StimulusError(
            f"development and confirmatory name pools overlap: pairs {sorted(shared)}, "
            f"names {sorted(names)}"
        )


def load_prestige() -> dict[str, Prestige]:
    settings = config.stimuli()["prestige"]
    levels = {}
    for level in ("modest", "elite"):
        entry = settings[level]
        levels[level] = Prestige(
            level=level, stimulus_id=entry["stimulus_id"], text=entry["text"]
        )
    return levels


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


def pool_summary(pairs: list[NamePair]) -> dict[str, Any]:
    """Descriptive report of what each arm's bundle contains.

    Reported, never adjusted for: perceived income and education are downstream
    of perceived race, so conditioning on them would remove part of the effect
    being estimated. Attribution accuracy is different in kind - it is how
    strongly the stimulus delivers the cue at all - which is why it is the one
    quantity the arms are matched on.
    """

    def stats(values: list[float]) -> dict[str, float] | None:
        clean = sorted(value for value in values if value is not None)
        if not clean:
            return None
        return {
            "n": len(clean),
            "min": clean[0],
            "median": clean[len(clean) // 2],
            "max": clean[-1],
            "mean": sum(clean) / len(clean),
        }

    arms = {
        "white": [pair.white for pair in pairs],
        "black": [pair.black for pair in pairs],
    }
    return {
        "pairs": len(pairs),
        "arms": {
            group: {
                "n": len(names),
                "attribution_accuracy": stats([name.attribution_accuracy for name in names]),
                "perceived_income": stats([name.perceived_income for name in names]),
                "perceived_education": stats([name.perceived_education for name in names]),
                "perceived_citizenship": stats(
                    [name.perceived_citizenship for name in names]
                ),
            }
            for group, names in arms.items()
        },
    }
