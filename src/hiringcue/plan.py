"""Trial planning: variants, repeats, and reflection-arm assignment.

A *variant* is one rendered first-turn prompt. A *trial* is one run of a variant
plus the second turn that follows it. The plan is written to disk before any
inference so that the set of intended measurements is fixed in advance and a
partial run can be resumed without changing what was going to be measured.

Reflection arms are randomised within condition. The generic arm asks the model
to reconsider without naming identity, which is what separates bias correction
from ordinary answer instability: models revise on reflection whether or not
identity is mentioned, so a revision rate measured only under an
identity-naming prompt confounds the two.

The neutral condition takes the late-disclosure arm rather than an
identity-specific one. Asking a model whether an identity cue influenced a
decision made without any identity cue has one correct answer and measures
nothing; supplying the identity afterwards measures reaction to new
information and is reported separately.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config, render, scenarios, stimuli

GENERIC = "generic"
IDENTITY_SPECIFIC = "identity_specific"
LATE_DISCLOSURE = "late_disclosure"

REFLECTION_TEMPLATES = {
    GENERIC: "reflection_generic_v1.txt",
    LATE_DISCLOSURE: "reflection_late_disclosure_v1.txt",
    "identity_concealed": "reflection_identity_concealed_v1.txt",
    "identity_direct": "reflection_identity_direct_v1.txt",
}

SCHEMA_OF_ARM = {
    GENERIC: "reflection_generic_output.schema.json",
    LATE_DISCLOSURE: "reflection_generic_output.schema.json",
    IDENTITY_SPECIFIC: "reflection_identity_output.schema.json",
}


@dataclass(frozen=True)
class Trial:
    trial_id: str
    variant_id: str
    family_id: str
    occupation_slug: str
    margin_band: str
    condition: str
    cue_mode: str
    identity_group: str | None
    prestige_level: str
    name_stimulus_id: str | None
    applicant_name: str | None
    soft_variant: str
    run_id: int
    reflection_arm: str
    reflection_template: str
    reflection_schema: str
    gold_decision: str
    minimum_gate_margin: float | None
    failed_gates: int
    ambiguity_score: int
    exact_hash: str
    normalised_hash: str


def _reflection_arm(condition: str, rng: random.Random) -> str:
    if condition == stimuli.NEUTRAL:
        return rng.choice([GENERIC, LATE_DISCLOSURE])
    return rng.choice([GENERIC, IDENTITY_SPECIFIC])


def _reflection_template(arm: str, cue_mode: str) -> str:
    if arm != IDENTITY_SPECIFIC:
        return REFLECTION_TEMPLATES[arm]
    return REFLECTION_TEMPLATES[f"identity_{cue_mode}"]


def build(
    families: list[scenarios.ScenarioFamily] | None = None,
    twins: dict[str, list[dict]] | None = None,
    allow_provisional: bool = False,
) -> tuple[list[render.RenderedPrompt], list[Trial]]:
    """Render every variant and expand it into trials."""
    settings = config.study()
    seed = int(settings["seed"])
    runs = int(settings["runs_per_variant"])
    twin_condition = settings["soft_twin"]["condition"]
    twin_prestige = settings["soft_twin"]["prestige_level"]

    families = families or scenarios.validated_families()
    pool = stimuli.load_names(allow_provisional=allow_provisional)
    prestige_levels = stimuli.load_prestige()
    name_assignment = stimuli.assign_names([f.family_id for f in families], pool, seed)

    variants: list[render.RenderedPrompt] = []
    for family in families:
        names = name_assignment[family.family_id]
        for prestige in prestige_levels.values():
            for condition in stimuli.CONDITIONS:
                group = stimuli.GROUP_OF_CONDITION.get(condition)
                name = names[group] if condition in stimuli.CONCEALED_CONDITIONS else None
                variants.append(
                    render.render_variant(
                        family=family,
                        condition=condition,
                        prestige=prestige,
                        name=name,
                        order_seed=seed,
                    )
                )

    render.check_counterfactual_integrity(variants)

    # The perturbed twin is the positive control for the score's free parameter:
    # a substantive change to the soft criteria that touches no gate, no gate
    # margin, and no identity field. A model that will not move its score for
    # this will not move it for a name either.
    if twins:
        twin_variants: list[render.RenderedPrompt] = []
        for family in families:
            profile = twins.get(family.occupation_slug)
            if profile is None:
                continue
            twin_variants.append(
                render.render_variant(
                    family=family,
                    condition=twin_condition,
                    prestige=prestige_levels[twin_prestige],
                    name=None,
                    order_seed=seed,
                    soft_profile=profile,
                    soft_variant="twin",
                )
            )
        variants.extend(twin_variants)

    rng = random.Random(seed + 1)
    trials: list[Trial] = []
    for variant in sorted(variants, key=lambda item: item.variant_id):
        for run_id in range(1, runs + 1):
            arm = _reflection_arm(variant.condition, rng)
            trials.append(
                Trial(
                    trial_id=f"{variant.variant_id}__run{run_id}",
                    variant_id=variant.variant_id,
                    family_id=variant.family_id,
                    occupation_slug=variant.occupation_slug,
                    margin_band=variant.margin_band,
                    condition=variant.condition,
                    cue_mode=variant.cue_mode,
                    identity_group=variant.identity_group,
                    prestige_level=variant.prestige_level,
                    name_stimulus_id=variant.name_stimulus_id,
                    applicant_name=variant.applicant_name,
                    soft_variant=variant.soft_variant,
                    run_id=run_id,
                    reflection_arm=arm,
                    reflection_template=_reflection_template(arm, variant.cue_mode),
                    reflection_schema=SCHEMA_OF_ARM[arm],
                    gold_decision=variant.gold_decision,
                    minimum_gate_margin=variant.minimum_gate_margin,
                    failed_gates=variant.failed_gates,
                    ambiguity_score=variant.ambiguity_score,
                    exact_hash=variant.exact_hash,
                    normalised_hash=variant.normalised_hash,
                )
            )
    return variants, trials


def write(
    variants: list[render.RenderedPrompt], trials: list[Trial], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "variants.jsonl").open("w") as handle:
        for variant in variants:
            handle.write(json.dumps(asdict(variant)) + "\n")
    with (out_dir / "trials.jsonl").open("w") as handle:
        for trial in trials:
            handle.write(json.dumps(asdict(trial)) + "\n")


def read_variants(out_dir: Path) -> dict[str, dict]:
    path = out_dir / "variants.jsonl"
    return {
        record["variant_id"]: record
        for record in (json.loads(line) for line in path.read_text().splitlines() if line)
    }


def read_trials(out_dir: Path) -> list[dict]:
    path = out_dir / "trials.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]
