"""Collection: run a planned round through the readout and record it.

One record per planned prompt, holding the planned fields alongside the readout
and nothing else. There are no repeats and no second turn: the readout is a
single teacher-forced forward pass, so the record for a prompt is complete after
one pass over it.

The batch manifest is read from the plan rather than rebuilt here, so the layout
a round was measured under is the layout that was written to disk before the
model loaded. The manifest, the pinned revisions, the admitted variant sets and
the answer prefix are all recorded beside the readings, because the contrast is
only interpretable against the surface it was taken on.

Raw readings are never overwritten. A round writes under a label and refuses to
replace an existing one; a rerun takes a new label.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import batches, config, plan, readout, stage0


@dataclass
class Manifest:
    """Everything that has to be fixed for a reading to mean anything."""

    label: str
    model_key: str
    model_id: str
    model_revision: str
    started_at_utc: str
    finished_at_utc: str
    dtype: str
    padding_side: str
    maximum_length: int
    batch_size: int
    answer_prefix: str
    variants: dict[str, Any]
    prompts: int
    batches: int
    study_seed: int
    arm_order_seed: int
    scenario_set_version: str
    pool_role: str
    context_levels: list[str]
    host: str
    seconds_elapsed: float
    seconds_per_thousand_prompts: float


def run(
    model_key: str,
    plan_dir: Path,
    out_dir: Path,
    label: str,
    pool_role: str,
    model_path: str | None = None,
    limit: int | None = None,
) -> Path:
    """Measure every planned prompt and write one reading per prompt."""
    settings = config.models()["readout"]
    entry = stage0.model_entry(model_key)

    prompts = plan.read(plan_dir)
    slots = batches.read(plan_dir / "batches.jsonl")
    if limit:
        keep = {prompt.prompt_id for prompt in prompts[:limit]}
        prompts = [prompt for prompt in prompts if prompt.prompt_id in keep]
        slots = [slot for slot in slots if slot.prompt_id in keep]

    out_dir.mkdir(parents=True, exist_ok=True)
    readings_path = out_dir / f"readings__{model_key}__{label}.jsonl"
    manifest_path = out_dir / f"manifest__{model_key}__{label}.json"
    existing = [str(path) for path in (readings_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(
            f"label {label!r} would overwrite existing output: {existing}"
        )

    tokenizer = stage0.load_tokenizer(model_key, model_path)
    tokenizer_report = stage0.check_tokenizer(tokenizer, prompts)
    texts = {
        prompt.prompt_id: readout.templated_prompt(
            tokenizer, prompt.system_prompt, prompt.user_prompt
        )
        for prompt in prompts
    }
    variants = readout.enumerate_variants(tokenizer, texts[prompts[0].prompt_id])

    model = stage0.load_model(model_key, model_path)
    started = datetime.now(timezone.utc)
    clock = time.time()
    readings: dict[str, readout.Reading] = {}
    for batch in batches.batched(slots):
        for reading in readout.read_batch(
            model,
            tokenizer,
            [(slot.prompt_id, texts[slot.prompt_id]) for slot in batch],
            variants,
        ):
            readings[reading.prompt_id] = reading
    elapsed = time.time() - clock

    with readings_path.open("w") as handle:
        for prompt in prompts:
            reading = readings[prompt.prompt_id]
            handle.write(
                json.dumps(
                    {
                        **asdict(prompt),
                        **reading.as_dict(),
                        "model_key": model_key,
                        "model_id": entry["model_id"],
                        "model_revision": entry["revision"],
                        "label": label,
                    }
                )
                + "\n"
            )

    manifest = Manifest(
        label=label,
        model_key=model_key,
        model_id=entry["model_id"],
        model_revision=entry["revision"],
        started_at_utc=started.isoformat(),
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        dtype=settings["dtype"],
        padding_side=settings["padding_side"],
        maximum_length=int(settings["maximum_length"]),
        batch_size=int(settings["batch_size"]),
        answer_prefix=readout.answer_prefix(),
        variants=tokenizer_report["variants"],
        prompts=len(prompts),
        batches=len({slot.batch_index for slot in slots}),
        study_seed=int(config.study()["seed"]),
        arm_order_seed=int(settings["arm_order_seed"]),
        scenario_set_version=config.study()["sources"]["scenario_set_version"],
        pool_role=pool_role,
        context_levels=sorted({prompt.context_level for prompt in prompts}),
        host=platform.node(),
        seconds_elapsed=elapsed,
        seconds_per_thousand_prompts=elapsed / len(prompts) * 1000.0,
    )
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    return readings_path


def agreement(
    readings_path: Path, sample: int | None = None
) -> dict[str, Any]:
    """Logit-versus-greedy agreement on a fixed development-only sample.

    The sample is the first N prompts of the recorded plan order, which is fixed
    before any measurement, so which prompts the gate is read on is not a
    function of what they returned.
    """
    sample = sample or int(config.study()["development"]["agreement_sample"])
    records = [
        json.loads(line) for line in readings_path.read_text().splitlines() if line
    ][:sample]
    manifest_path = readings_path.parent / readings_path.name.replace(
        "readings__", "manifest__"
    ).replace(".jsonl", ".json")
    variants_record = json.loads(manifest_path.read_text())["variants"]
    variants = readout.VariantSet(
        yes_token_ids=tuple(variants_record["yes_token_ids"]),
        no_token_ids=tuple(variants_record["no_token_ids"]),
        yes_surfaces=tuple(variants_record["yes_surfaces"]),
        no_surfaces=tuple(variants_record["no_surfaces"]),
    )
    return readout.greedy_agreement(
        [
            readout.Reading(
                prompt_id=record["prompt_id"],
                token_log_odds=record["token_log_odds"],
                implied_yes_probability=record["implied_yes_probability"],
                off_target_mass=record["off_target_mass"],
                greedy_token_id=record["greedy_token_id"],
                greedy_is_admitted=record["greedy_is_admitted"],
                boundary_index=record["boundary_index"],
            )
            for record in records
        ],
        variants,
    )
