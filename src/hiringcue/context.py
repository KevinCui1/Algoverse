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

Context richness is consequently a factor with two levels, `bare` and
`realistic`, crossed with every other factor, and the primary estimand is its
interaction with the identity cue:

    [(Black - White)_realistic - (Black - White)_bare]

Estimating the interaction rather than a simple identity effect is what makes
the bare arm a within-design control: it is what allows an observed identity
effect to be attributed to context richness rather than to any property the two
arms share.

Two realistic variants are predeclared in a fixed order. Employer context alone
is tested first; employer context plus a selectivity constraint is a fallback,
evaluated only if the first fails and accepted only if it passes the same
criterion without increasing saturation. The published direct-answer results
were obtained with employer context alone; the selectivity constraint was
required only to restore the effect under a chain-of-thought response format,
which this design does not use.

The employer is fictitious and its organisational description is authored here
rather than taken from a real organisation's published material. A versioned
stimulus set has to be freezable and quotable in full, which text belonging to a
third party is not, and attributing authored culture text to a real employer
would misdescribe that employer. What D-046 manipulates is context richness, not
the identity of the employer; that the source amplification was obtained with a
recognisable employer is recorded as a generalisation limit rather than
reproduced.

Two further levels exist for the development round only and never enter the
confirmatory estimand. The fictitious employer preserves the richness
manipulation but cannot say whether the published amplification also depended
on the employer being recognisable, and if it did, a development round run on a
fictitious employer alone under-measures the interaction and could fire the
kill criterion for a reason that is about the stimulus rather than the models.
The two development levels answer that directly: one names a real organisation,
the other an invented one, and their descriptions are otherwise the same
organisation type at the same scale in the same categories at the same
granularity. Recognisability is the difference of their interactions. A single
real-named level compared against the fictitious employer of the confirmatory
arm would confound recognisability with everything else the two descriptions do
not share.

Context is constant within a counterfactual set, so byte-identity outside the
identity block is preserved and the normalised-hash check is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config, paths

BARE = "bare"
REALISTIC = "realistic"


class ContextError(ValueError):
    """Raised when a context level or variant is not one the design declares."""


@dataclass(frozen=True)
class Context:
    level: str
    variant: str
    text: str


def levels() -> tuple[str, ...]:
    """The confirmatory levels. Development-only levels are not included."""
    return tuple(config.study()["context"]["levels"])


def development_only_levels() -> tuple[str, ...]:
    return tuple(config.study()["context"].get("development_only_levels", ()))


def realistic_variants() -> tuple[str, ...]:
    """Predeclared evaluation order for the realistic level."""
    return tuple(config.study()["context"]["realistic_variants"])


def _template(variant: str) -> str:
    templates = config.study()["context"]["templates"]
    if variant not in templates:
        raise ContextError(
            f"unknown context variant {variant!r}; declared: {sorted(templates)}"
        )
    path = paths.PROMPTS / templates[variant]
    if not path.exists():
        raise ContextError(f"missing context template: {path}")
    return path.read_text()


def load(
    realistic_variant: str | None = None,
    include_development_levels: bool = False,
) -> dict[str, Context]:
    """Both context levels, with the realistic level bound to one variant.

    The variant is bound once for a whole round rather than per prompt. The
    selected template is frozen before confirmation and is never tuned per
    model: a template chosen against each model's own response would make the
    manipulation a fitted parameter instead of a fixed stimulus.
    """
    variant = realistic_variant or realistic_variants()[0]
    if variant not in realistic_variants():
        raise ContextError(
            f"{variant!r} is not a predeclared realistic variant; declared in order: "
            f"{list(realistic_variants())}"
        )
    loaded = {
        BARE: Context(level=BARE, variant=BARE, text=_template(BARE)),
        REALISTIC: Context(level=REALISTIC, variant=variant, text=_template(variant)),
    }
    if include_development_levels:
        for level in development_only_levels():
            loaded[level] = Context(level=level, variant=level, text=_template(level))
    return loaded
