"""Employer-context richness: the manipulated stimulus factor.

Scenarios authored for this study supply an occupation, a requirement list and a
candidate profile, and nothing else - no employer, no location, no
organisational detail. Published work on the same decision task reports that
under stimuli of that description all evaluated models show identity gaps under
two percent, and that adding a named employer with organisational context to
otherwise identical material raises those gaps by roughly a factor of five, with
anti-bias instructions present in both conditions. A measured null on bare
stimuli is therefore consistent with a property of the stimuli rather than with
an absence of identity sensitivity.

Context richness is consequently a factor crossed with every other factor, and
the estimand is its interaction with the identity cue:

    [(Black - White)_realistic - (Black - White)_bare]

Estimating the interaction rather than a simple identity effect is what makes
the bare arm a within-design control: it is what allows an observed identity
effect to be attributed to context richness rather than to any property the two
arms share.

Three levels are measured together rather than one realistic template being
bound per round. `bare` is the control. `employer` supplies organisational
detail alone. `employer_selectivity` supplies the same organisation and adds the
constraint that the posting has drawn many applications for one opening and that
only the strongest applicants advance.

The selectivity level is a first-class level rather than a fallback because of
its mechanism. Under a selectivity constraint, clearing the stated requirements
stops being sufficient for the answer, so whatever discretion the evidence
carries has to decide the case. The published condition this design is anchored
on combined organisational context with exactly such a constraint, so measuring
organisational context alone reproduces only half of it.

The employer is fictitious and its organisational description is authored here
rather than taken from a real organisation's published material. A versioned
stimulus set has to be freezable and quotable in full, which text belonging to a
third party is not, and attributing authored culture text to a real employer
would misdescribe that employer. Whether the amplification depends on the
employer being recognisable has been measured directly on a real-named level
against its matched invented twin, and the invented employer produced the larger
interaction, so recognisability is settled and those two development-only levels
are retired rather than re-run.

Context is constant within a counterfactual set, so byte-identity outside the
identity block is preserved and the normalised-hash check is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config, paths

BARE = "bare"


class ContextError(ValueError):
    """Raised when a context level is not one the design declares."""


@dataclass(frozen=True)
class Context:
    level: str
    text: str


def levels() -> tuple[str, ...]:
    return tuple(config.study()["context"]["levels"])


def realistic_levels() -> tuple[str, ...]:
    """The rich levels, each contrasted against `bare` to form an interaction."""
    return tuple(level for level in levels() if level != BARE)


def _template(level: str) -> str:
    templates = config.study()["context"]["templates"]
    if level not in templates:
        raise ContextError(
            f"unknown context level {level!r}; declared: {sorted(templates)}"
        )
    path = paths.PROMPTS / templates[level]
    if not path.exists():
        raise ContextError(f"missing context template: {path}")
    return path.read_text()


def load() -> dict[str, Context]:
    """Every declared context level, keyed by level.

    All levels are rendered in one round. Measuring them together is what makes
    the comparison between them internal to a single collection: a level
    evaluated in a later round would differ from the others by whatever else
    changed between rounds as well as by its own text.
    """
    declared = levels()
    if BARE not in declared:
        raise ContextError(
            f"the control level {BARE!r} is not among the declared levels {declared}; "
            "the estimand is an interaction and has no baseline without it"
        )
    return {level: Context(level=level, text=_template(level)) for level in declared}
