"""Stage 0: whether the instrument can be read at all, per checkpoint.

This is the cheapest decisive check in the sequence and nothing downstream is
worth building if it fails. It answers two questions for each pinned checkpoint.

*Can the answer position be located and read?* The contrast is taken at the last
token of the prompt after that model's own chat template has been applied. Which
surface forms of Yes and No occupy exactly one token at that position is a
property of the pinned tokenizer and the pinned template together, never of the
tokenizer family, so it is enumerated rather than assumed. The enumeration is
repeated across a sample of real prompts and must agree on all of them: a
variant set that depended on what preceded the boundary would mean the contrast
was taken over different token sets in different cells.

*Does the same prompt return the same number?* The readout runs each prompt
under fixed-shape layouts that vary ordering and co-resident prompts, the
properties that can vary within a collection. Batch size is pinned for a
collection and the final partial batch is padded to it, so it is not a gate
dimension. Its sensitivity is nevertheless measured permanently and disclosed
beside the gate result without an admissibility verdict.

The tokenizer half of this runs without an accelerator and without the model
weights, so it can be executed locally for open repositories and inside the job
for gated ones. The stability half needs the weights and runs on the cluster.

Output is one verdict per checkpoint plus the measured wall-clock per thousand
prompts, which Stage 3 sizing depends on and which is otherwise unknown.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

from . import batches, config, plan, readout


class Stage0Error(RuntimeError):
    """Raised when Stage 0 cannot be executed as specified."""


def model_entry(model_key: str) -> dict[str, Any]:
    for entry in config.models()["models"]:
        if entry["key"] == model_key:
            return {**config.models()["defaults"], **entry}
    known = [entry["key"] for entry in config.models()["models"]]
    raise KeyError(f"unknown model key {model_key!r}; configured: {known}")


def load_tokenizer(model_key: str, model_path: str | None = None):
    from transformers import AutoTokenizer

    entry = model_entry(model_key)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path or entry["model_id"],
        revision=None if model_path else entry["revision"],
        padding_side=config.models()["readout"]["padding_side"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def check_tokenizer(
    tokenizer: Any, prompts: Sequence[plan.PlannedPrompt]
) -> dict[str, Any]:
    """Enumerate the variant set at the boundary and confirm it does not vary."""
    if not prompts:
        raise Stage0Error("tokenizer check needs at least one planned prompt")

    enumerations = []
    for prompt in prompts:
        boundary = readout.templated_prompt(
            tokenizer, prompt.system_prompt, prompt.user_prompt
        )
        variants = readout.enumerate_variants(tokenizer, boundary)
        enumerations.append(variants)

    distinct = {
        (entry.yes_token_ids, entry.no_token_ids) for entry in enumerations
    }
    if len(distinct) != 1:
        raise Stage0Error(
            f"the admitted variant set differs across {len(distinct)} of the "
            f"{len(prompts)} sampled prompts. The contrast would be taken over "
            "different token sets in different cells, which is not one measurement."
        )

    reference = enumerations[0]
    boundary = readout.templated_prompt(
        tokenizer, prompts[0].system_prompt, prompts[0].user_prompt
    )
    return {
        "prompts_checked": len(prompts),
        "variants": reference.as_dict(),
        "boundary_tail": boundary[-120:],
        # Recorded so that a tokenizer whose pre-tokenisation has changed between
        # runs is visible even when the admitted variant ids happen not to move.
        # The token count of one fixed prompt is the cheapest quantity that sees
        # such a change.
        "boundary_token_count": len(
            tokenizer.encode(boundary, add_special_tokens=False)
        ),
        "answer_prefix": readout.answer_prefix(),
        "arm_length_gaps": readout.arm_length_gaps(tokenizer, prompts),
        "verdict": "PASS",
    }


def load_model(model_key: str, model_path: str | None = None):
    """Load one checkpoint onto one device in the recorded precision.

    The whole model is placed on a single device rather than sharded by a
    dispatcher. Every evaluated checkpoint fits one 80GB accelerator in
    bfloat16, and a dispatched placement would put different layers of the same
    forward pass on different devices, which adds a second way for one prompt's
    result to depend on something other than the prompt.
    """
    import torch
    from transformers import AutoModelForCausalLM

    entry = model_entry(model_key)
    settings = config.models()["readout"]
    kwargs: dict[str, Any] = {
        "revision": None if model_path else entry["revision"],
        "dtype": getattr(torch, settings["dtype"]),
    }
    if entry.get("attn_implementation"):
        kwargs["attn_implementation"] = entry["attn_implementation"]

    model = AutoModelForCausalLM.from_pretrained(
        model_path or entry["model_id"], **kwargs
    )
    model.to("cuda")
    model.eval()
    return model


def stability_sample(
    prompts: Sequence[plan.PlannedPrompt], size: int | None = None
) -> list[plan.PlannedPrompt]:
    """Take the sample as whole counterfactual pairs, stratified over the design.

    Both arms of a pair have to share a batch, so a sample that cuts a pair in
    half cannot be laid out at all. That much the previous sampler got right.
    What it got wrong was everything else: it walked the plan in recorded order
    and stopped at a prompt budget, and because plan order is grouped by
    occupation the resulting 200 prompts were drawn entirely from the first
    occupation and contained no `near_pass` cell at all. The gate was therefore
    measured on one sixth of the scenario set, missing the margin band with the
    most discretion - the band where an identity cue has the most room to act,
    and so the band whose stability matters most.

    The sample is now a stratified census: every scenario family, both prompt
    forms, every context level, both prestige levels and both cue modes, with a
    fixed number of complete counterfactual pairs drawn from each stratum in
    recorded plan order. The prompt form is part of the stratum because the gate
    statistic is formed per cell and a key that omitted it would let one form
    supply both draws and leave the other unmeasured. Which prompts the gate is
    read on stays independent of what they return, and the qualified cells the
    gate statistic averages over are all present rather than incidental.

    `size` is accepted and ignored; the sample size is now determined by the
    design rather than by a budget.
    """
    per_stratum = int(config.models()["readout"]["pairs_per_stratum"])

    strata: dict[tuple, dict[str, list[plan.PlannedPrompt]]] = {}
    order: list[tuple] = []
    for prompt in prompts:
        if prompt.counterfactual_pair_id == batches.UNPAIRED:
            continue
        if prompt.soft_variant != "base":
            continue
        key = (
            prompt.family_id,
            prompt.prompt_form,
            prompt.context_level,
            prompt.prestige_level,
            prompt.cue_mode,
        )
        if key not in strata:
            strata[key] = {}
            order.append(key)
        strata[key].setdefault(prompt.counterfactual_pair_id, []).append(prompt)

    selected: list[plan.PlannedPrompt] = []
    for key in order:
        complete = [
            members
            for members in strata[key].values()
            if {member.identity_group for member in members} == {"black", "white"}
        ]
        for members in complete[:per_stratum]:
            selected.extend(members)

    if not selected:
        raise Stage0Error(
            "no complete counterfactual pair survives stratification; the gate cannot "
            "be measured on a plan with no matched identity arms"
        )
    _assert_sample_covers(selected)
    return selected


def _assert_sample_covers(selected: Sequence[plan.PlannedPrompt]) -> None:
    """Refuse a sample that omits a band, a form, a context, an occupation or a cue mode.

    A gate measured on part of the design reports on part of the design. The
    previous sample silently omitted a margin band and five of six occupations,
    and nothing in the pipeline noticed; this converts that class of omission
    into a hard stop.
    """
    required_bands = set(config.study()["margin_bands"])
    required_contexts = {prompt.context_level for prompt in selected}
    seen_bands = {prompt.margin_band for prompt in selected}
    missing = required_bands - seen_bands
    if missing:
        raise Stage0Error(
            f"stability sample omits margin band(s) {sorted(missing)}; the band with the "
            "most discretion is the one whose stability matters most"
        )
    occupations = {prompt.occupation_slug for prompt in selected}
    if len(occupations) < 2:
        raise Stage0Error(
            f"stability sample covers only {sorted(occupations)}; a gate measured on one "
            "occupation reports on one occupation"
        )
    if len(required_contexts) < 2:
        raise Stage0Error("stability sample carries a single context level")
    declared_forms = set(config.study()["prompt_form"]["levels"])
    missing_forms = declared_forms - {prompt.prompt_form for prompt in selected}
    if missing_forms:
        raise Stage0Error(
            f"stability sample omits prompt form(s) {sorted(missing_forms)}; the gate "
            "statistic is formed per cell and an unmeasured form has no gate at all"
        )
    for prompt in selected:
        if prompt.identity_group not in ("black", "white"):
            raise Stage0Error(
                f"{prompt.prompt_id}: unpaired prompt reached the stability sample"
            )


def _ordered(
    prompts: Sequence[plan.PlannedPrompt], layout: dict[str, Any]
) -> list[plan.PlannedPrompt]:
    """Apply one layout's membership perturbation, before batching.

    `shuffle_seed` permutes the sample before it is laid out, which changes which
    prompts share a batch. It is a membership perturbation and not an ordering
    one; the within-batch permutation is applied after batching, in
    `_reorder_within_batches`, so the two mechanisms stay separable.
    """
    if layout.get("shuffle_seed") is None:
        return list(prompts)
    shuffled = list(prompts)
    random.Random(int(layout["shuffle_seed"])).shuffle(shuffled)
    return shuffled


def _reorder_within_batches(
    grouped: list[list[Any]], layout: dict[str, Any]
) -> list[list[Any]]:
    """Permute position inside each batch, leaving membership and shape alone.

    This is the perturbation that ought to be exactly neutral: the same
    sequences enter the same tensor at the same shape, in a different order.
    Keeping it separate from the membership and batch-size layouts is what lets
    a non-zero result name its own mechanism instead of being attributed to
    whichever perturbation was bundled with it.
    """
    seed = layout.get("within_batch_seed")
    if seed is None:
        return grouped
    generator = random.Random(int(seed))
    reordered = []
    for batch in grouped:
        copy = list(batch)
        generator.shuffle(copy)
        reordered.append(copy)
    return reordered


def measure_layout(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[plan.PlannedPrompt],
    variants: readout.VariantSet,
    layout: dict[str, Any],
) -> tuple[list[readout.Reading], float]:
    """Read every prompt under one fixed batch composition."""
    ordered = _ordered(prompts, layout)
    slots = batches.build(
        (
            batches.PlannedPrompt(
                prompt_id=prompt.prompt_id,
                pair_id=prompt.counterfactual_pair_id,
                identity_arm=prompt.identity_group,
            )
            for prompt in ordered
        ),
        batch_size=int(layout["batch_size"]),
    )
    texts = {
        prompt.prompt_id: readout.templated_prompt(
            tokenizer, prompt.system_prompt, prompt.user_prompt
        )
        for prompt in ordered
    }

    readings: list[readout.Reading] = []
    started = time.time()
    grouped = _reorder_within_batches(batches.batched(slots), layout)
    for batch in grouped:
        readings.extend(
            readout.read_batch(
                model,
                tokenizer,
                [(slot.prompt_id, texts[slot.prompt_id]) for slot in batch],
                variants,
                batch_size=int(layout["batch_size"]),
            )
        )
    return readings, time.time() - started


def saturation_breakdown(
    readings: Sequence[readout.Reading],
    prompts: Sequence[plan.PlannedPrompt],
) -> dict[str, Any]:
    """Readable-range shares overall and by every band and context level."""
    from . import diagnostics

    prompt_index = {prompt.prompt_id: prompt for prompt in prompts}
    joined = [
        (prompt_index[reading.prompt_id], reading)
        for reading in readings
        if reading.prompt_id in prompt_index
    ]
    qualified = set(config.study()["margin_bands"]) - set(
        config.study()["qualification"]["control_bands"]
    )

    def measured(subset: Sequence[tuple[Any, readout.Reading]]) -> dict[str, Any]:
        report = diagnostics.saturation(
            [
                {"implied_yes_probability": reading.implied_yes_probability}
                for _, reading in subset
            ]
        )
        if report.get("cells"):
            report["inside_share"] = 1.0 - report["outside_share"]
        return report

    qualified_rows = [entry for entry in joined if entry[0].margin_band in qualified]
    return {
        "qualified": measured(qualified_rows),
        "by_margin_band": {
            band: measured([entry for entry in joined if entry[0].margin_band == band])
            for band in config.study()["margin_bands"]
        },
        "by_context": {
            context: measured(
                [
                    entry
                    for entry in qualified_rows
                    if entry[0].context_level == context
                ]
            )
            for context in sorted({entry[0].context_level for entry in joined})
        },
    }


def analyse_readings(
    by_layout: dict[str, Sequence[readout.Reading]],
    prompts: Sequence[plan.PlannedPrompt],
) -> dict[str, Any]:
    """Apply the amended gate and mandatory sensitivity disclosure to readings."""
    from . import diagnostics

    gate_labels = [entry["label"] for entry in batches.layouts()]
    available_gate = {
        label: by_layout[label] for label in gate_labels if label in by_layout
    }
    reference = gate_labels[0]
    sensitivity_labels = [
        entry["label"] for entry in batches.batch_size_sensitivity_layouts()
    ]
    sensitivity = {
        label: by_layout[label]
        for label in [reference, *sensitivity_labels]
        if label in by_layout
    }
    return {
        "stability": diagnostics.stability(available_gate, prompts),
        "batch_size_sensitivity": diagnostics.batch_size_sensitivity(
            sensitivity, prompts
        ),
        "saturation": saturation_breakdown(by_layout[reference], prompts),
    }


def instrument_verdict(analysis: dict[str, Any]) -> tuple[str, list[str]]:
    """Require both Stage 0 gates; stability alone cannot admit a checkpoint."""
    from . import diagnostics

    failed = [
        name
        for name, verdict in (
            ("stability", analysis["stability"]["verdict"]),
            ("saturation", analysis["saturation"]["qualified"]["verdict"]),
        )
        if verdict != diagnostics.PASS
    ]
    return (diagnostics.PASS if not failed else diagnostics.FAIL), failed


def run(
    model_key: str,
    plan_dir: Path,
    out_dir: Path,
    model_path: str | None = None,
    tokenizer_only: bool = False,
) -> Path:
    """Execute Stage 0 for one checkpoint and write its verdict."""
    from . import diagnostics

    prompts = plan.read(plan_dir)
    sample = stability_sample(prompts)

    tokenizer = load_tokenizer(model_key, model_path)
    tokenizer_report = check_tokenizer(tokenizer, sample)

    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "model_key": model_key,
        "model_id": model_entry(model_key)["model_id"],
        "revision": model_entry(model_key)["revision"],
        "tokenizer": tokenizer_report,
        "stability_sample": len(sample),
    }

    if not tokenizer_only:
        variants = readout.enumerate_variants(
            tokenizer,
            readout.templated_prompt(
                tokenizer, sample[0].system_prompt, sample[0].user_prompt
            ),
        )
        model = load_model(model_key, model_path)
        by_layout: dict[str, Sequence[readout.Reading]] = {}
        elapsed: dict[str, float] = {}
        gate_layouts = batches.layouts()
        sensitivity_layouts = batches.batch_size_sensitivity_layouts()
        for layout in gate_layouts + sensitivity_layouts:
            readings, seconds = measure_layout(
                model, tokenizer, sample, variants, layout
            )
            by_layout[layout["label"]] = readings
            elapsed[layout["label"]] = seconds

        report.update(analyse_readings(by_layout, sample))
        report["seconds_per_thousand_prompts"] = {
            label: seconds / len(sample) * 1000.0 for label, seconds in elapsed.items()
        }
        report["readings"] = {
            label: [reading.as_dict() for reading in readings]
            for label, readings in by_layout.items()
        }
        report["verdict"], report["failed"] = instrument_verdict(report)
    else:
        report["verdict"] = tokenizer_report["verdict"]

    path = out_dir / f"stage0__{model_key}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path
