"""Measurement planning: which prompts exist, and which share a tensor.

A *prompt* is one rendered first-turn evaluation. The plan enumerates every
prompt the round will measure and is written to disk before any model is
loaded, so the set of intended measurements is fixed in advance and a partial
round resumes without changing what was going to be measured.

There are no repeats. The readout is a single teacher-forced forward pass, so a
repeat of a byte-identical prompt is a duplicate rather than an observation;
stability is established by the cross-batch-composition gate instead of averaged
over.

The factors crossed here are scenario family, prompt form, employer-context
richness, credential prestige, and identity condition. Identity condition expands over the
name pool rather than over one assigned pair per family: name pair is a random
factor in the analysis, and a fixed set of pairs reused across families would
put every family mean on the same draw of the name effect, where a
family-clustered interval cannot see it.

The primary estimand is restricted to families whose candidate objectively
passes the hard gates. Families that objectively fail are planned as well and
labelled as the reduced rule-following control: on a candidate who plainly
fails, a correct model answers no with high confidence, the outcome saturates,
and the cell contributes resolution rather than signal.

Perturbed soft-criteria twins are rendered at every prompt form and every
context level rather than at one nominated cell. The twin is the positive
control on the readout's free parameter, and the estimand it bounds is defined
in the rich contexts, so a control measured only in the bare condition answers a
question the design does not ask. Rendering it everywhere is what lets the
control be reported in the same cell as the effect it bounds.

Output is the rendered prompts, the batch manifest that fixes which of them
share a tensor and in what order, and a summary of what was planned.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from . import batches, config, context as context_factor, gates, render, scenarios, stimuli

PRIMARY = "primary"
RULE_CONTROL = "rule_control"
TWIN_CONTROL = "twin_control"


@dataclass(frozen=True)
class PlannedPrompt:
    prompt_id: str
    variant_id: str
    family_id: str
    occupation_slug: str
    margin_band: str
    condition: str
    cue_mode: str
    identity_group: str | None
    prompt_form: str
    context_level: str
    prestige_level: str
    name_pair_id: str | None
    name_stimulus_id: str | None
    applicant_name: str | None
    soft_variant: str
    role: str
    counterfactual_pair_id: str
    # The only span that differs within a counterfactual set. Carried through to
    # the readout so that length matching between the identity arms can be
    # applied inside it, where it leaves the normalised hash untouched.
    identity_block: str
    gold_decision: str
    minimum_gate_margin: float | None
    failed_gates: int
    ambiguity_score: int
    exact_hash: str
    normalised_hash: str
    system_prompt: str
    user_prompt: str


def _role(family: scenarios.ScenarioFamily, soft_variant: str) -> str:
    if soft_variant != "base":
        return TWIN_CONTROL
    required = config.study()["qualification"]["primary_estimand_requires_gold"]
    return PRIMARY if family.gold_decision == required else RULE_CONTROL


def _counterfactual_pair_id(
    variant: render.RenderedPrompt, name_pair_id: str | None
) -> str:
    """Identifier of the two-arm set the batcher must keep co-resident.

    Only the arms that are differenced form a pair. The neutral condition is the
    common baseline of a whole counterfactual set rather than one arm of a
    contrast, so it is placed unpaired.
    """
    if variant.condition == stimuli.NEUTRAL or variant.soft_variant != "base":
        return batches.UNPAIRED
    stem = (
        f"{variant.family_id}__{variant.prompt_form}__{variant.context_level}__"
        f"{variant.prestige_level}__{variant.cue_mode}"
    )
    return f"{stem}__{name_pair_id}" if name_pair_id else stem


def build(
    families: list[scenarios.ScenarioFamily] | None = None,
    pairs: list[stimuli.NamePair] | None = None,
    twins: dict[str, list[dict]] | None = None,
    role: str | None = None,
) -> list[PlannedPrompt]:
    """Render every planned prompt and check counterfactual integrity."""
    settings = config.study()
    seed = int(settings["seed"])

    families = families if families is not None else scenarios.validated_families()
    pairs = pairs if pairs is not None else stimuli.load_pairs(role)
    prestige_levels = stimuli.load_prestige()
    contexts = context_factor.load()
    forms = render.forms()

    variants: list[tuple[render.RenderedPrompt, stimuli.NamePair | None]] = []
    for family in families:
        for form in forms:
            for context in contexts.values():
                for prestige in prestige_levels.values():
                    for condition in stimuli.CONDITIONS:
                        if condition in stimuli.CONCEALED_CONDITIONS:
                            group = stimuli.GROUP_OF_CONDITION[condition]
                            for pair in pairs:
                                variants.append(
                                    (
                                        render.render_variant(
                                            family=family,
                                            condition=condition,
                                            prestige=prestige,
                                            name=pair.arm(group),
                                            order_seed=seed,
                                            context=context,
                                            prompt_form=form,
                                        ),
                                        pair,
                                    )
                                )
                            continue
                        variants.append(
                            (
                                render.render_variant(
                                    family=family,
                                    condition=condition,
                                    prestige=prestige,
                                    name=None,
                                    order_seed=seed,
                                    context=context,
                                    prompt_form=form,
                                ),
                                None,
                            )
                        )

    render.check_counterfactual_integrity([variant for variant, _ in variants])

    # The perturbed twin is the positive control for the readout's free
    # parameter: a substantive change to the soft criteria that touches no gate,
    # no gate margin, and no identity field. A model that will not move its
    # contrast for this will not move it for a name either.
    #
    # It is rendered at every prompt form and every context level. The control
    # bounds any cue effect in the cell it is measured in, and the estimand lives
    # in the rich contexts, so a twin held at one nominated cell would bound a
    # quantity the design does not estimate.
    if twins:
        twin_settings = settings["soft_twin"]
        for family in families:
            profile = twins.get(family.family_id)
            if profile is None:
                continue
            for form in forms:
                for context in contexts.values():
                    variants.append(
                        (
                            render.render_variant(
                                family=family,
                                condition=twin_settings["condition"],
                                prestige=prestige_levels[twin_settings["prestige_level"]],
                                name=None,
                                order_seed=seed,
                                context=context,
                                prompt_form=form,
                                soft_profile=profile,
                                soft_variant="twin",
                            ),
                            None,
                        )
                    )

    lookup = {family.family_id: family for family in families}
    planned = [
        PlannedPrompt(
            prompt_id=variant.variant_id,
            variant_id=variant.variant_id,
            family_id=variant.family_id,
            occupation_slug=variant.occupation_slug,
            margin_band=variant.margin_band,
            condition=variant.condition,
            cue_mode=variant.cue_mode,
            identity_group=variant.identity_group,
            prompt_form=variant.prompt_form,
            context_level=variant.context_level,
            prestige_level=variant.prestige_level,
            name_pair_id=pair.pair_id if pair else None,
            name_stimulus_id=variant.name_stimulus_id,
            applicant_name=variant.applicant_name,
            soft_variant=variant.soft_variant,
            role=_role(lookup[variant.family_id], variant.soft_variant),
            counterfactual_pair_id=_counterfactual_pair_id(
                variant, pair.pair_id if pair else None
            ),
            identity_block=variant.identity_block,
            gold_decision=variant.gold_decision,
            minimum_gate_margin=variant.minimum_gate_margin,
            failed_gates=variant.failed_gates,
            ambiguity_score=variant.ambiguity_score,
            exact_hash=variant.exact_hash,
            normalised_hash=variant.normalised_hash,
            system_prompt=variant.system_prompt,
            user_prompt=variant.user_prompt,
        )
        for variant, pair in variants
    ]
    planned.sort(key=lambda prompt: prompt.prompt_id)
    _check_gold_independence(planned)
    return planned


def _check_gold_independence(prompts: list[PlannedPrompt]) -> None:
    """The gold decision must not vary across anything the prompt manipulates.

    It is a pure function of the hard gates. If it moved with context, prestige
    or identity, some manipulation would be reaching the objective rule, which
    invalidates the design rather than one measurement.
    """
    decisions: dict[str, set[str]] = {}
    for prompt in prompts:
        decisions.setdefault(prompt.family_id, set()).add(prompt.gold_decision)
    unstable = {family: sorted(values) for family, values in decisions.items() if len(values) > 1}
    if unstable:
        raise render.RenderError(
            f"gold decision varies within families {sorted(unstable)}; it must be a "
            "pure function of the hard gates"
        )
    if any(prompt.gold_decision not in (gates.ADVANCE, gates.DO_NOT_ADVANCE) for prompt in prompts):
        raise render.RenderError("a planned prompt carries an unknown gold decision")


def batch_manifest(prompts: list[PlannedPrompt]) -> list[batches.Slot]:
    return batches.build(
        batches.PlannedPrompt(
            prompt_id=prompt.prompt_id,
            pair_id=prompt.counterfactual_pair_id,
            identity_arm=prompt.identity_group,
        )
        for prompt in prompts
    )


def summarise(prompts: list[PlannedPrompt], slots: list[batches.Slot]) -> dict:
    roles: dict[str, int] = {}
    for prompt in prompts:
        roles[prompt.role] = roles.get(prompt.role, 0) + 1
    return {
        "prompts": len(prompts),
        "by_role": roles,
        "families": len({prompt.family_id for prompt in prompts}),
        "primary_families": len(
            {prompt.family_id for prompt in prompts if prompt.role == PRIMARY}
        ),
        "name_pairs": len(
            {prompt.name_pair_id for prompt in prompts if prompt.name_pair_id}
        ),
        "prompt_forms": sorted({prompt.prompt_form for prompt in prompts}),
        "context_levels": sorted({prompt.context_level for prompt in prompts}),
        "cells": sorted(
            f"{form}/{level}"
            for form in {prompt.prompt_form for prompt in prompts}
            for level in {prompt.context_level for prompt in prompts}
        ),
        "batches": len({slot.batch_index for slot in slots}),
        "seed": config.study()["seed"],
    }


def write(prompts: list[PlannedPrompt], slots: list[batches.Slot], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "prompts.jsonl").open("w") as handle:
        for prompt in prompts:
            handle.write(json.dumps(asdict(prompt)) + "\n")
    batches.write(slots, out_dir / "batches.jsonl")
    (out_dir / "plan_summary.json").write_text(
        json.dumps(summarise(prompts, slots), indent=2) + "\n"
    )


def read(out_dir: Path) -> list[PlannedPrompt]:
    path = out_dir / "prompts.jsonl"
    return [
        PlannedPrompt(**json.loads(line))
        for line in path.read_text().splitlines()
        if line
    ]
