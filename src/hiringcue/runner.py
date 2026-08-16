"""Batched two-turn inference against a locally served open-weight model.

The run is a single process holding one model, issuing every prompt in the plan
as one batch and letting the engine schedule them. Two properties matter for
cost:

*Prefix reuse.* Repeated runs of a variant are byte-identical, and every
second-turn prompt contains its own first turn verbatim. With prefix caching on,
the shared portion is prefilled once rather than once per request. On this plan
that removes most of the prefill work in the second pass.

*Constrained decoding.* Each response is decoded under its JSON schema, so the
engine cannot emit an unparseable structure. This is not only convenience: a
parse failure would drop a cell, and the conditions most likely to produce one -
a direct identity statement drawing a refusal or a hedge - are exactly the
conditions whose comparison the missing cell would distort.

Turn two must follow turn one in the same conversation, so the passes are
sequential. Within each pass everything is submitted at once.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config, paths, plan


@dataclass
class RunManifest:
    run_label: str
    model_key: str
    model_id: str
    model_revision: str
    started_at_utc: str
    finished_at_utc: str | None
    temperature: float
    top_p: float
    max_tokens: int
    seed_base: int
    tensor_parallel_size: int
    accelerator_model: str
    accelerator_memory_gib: int
    dtype: str
    max_model_len: int
    structured_output: bool
    prefix_caching: bool
    study_seed: int
    runs_per_variant: int
    trial_count: int
    scenario_set_version: str
    name_pool: str
    name_pool_provisional: bool
    host: str
    engine_version: str
    notes: str = ""


def _model_entry(model_key: str) -> dict[str, Any]:
    for entry in config.models()["models"]:
        if entry["key"] == model_key:
            return entry
    known = [entry["key"] for entry in config.models()["models"]]
    raise KeyError(f"unknown model key {model_key!r}; configured: {known}")


def _reflection_prompt(trial: dict, initial_raw: str) -> str:
    template = (paths.PROMPTS / trial["reflection_template"]).read_text()
    replacements = {
        "APPLICANT_NAME": trial.get("applicant_name") or "",
        "PERCEIVED_IDENTITY_LABEL": trial.get("identity_label") or "",
        "DIRECT_IDENTITY_LABEL": trial.get("identity_label") or "",
    }
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    if "{{" in template:
        raise ValueError(f"{trial['trial_id']}: unfilled reflection placeholder")
    return template


def _identity_label(condition: str) -> str | None:
    from . import stimuli

    return stimuli.reflection_label(condition)


def run(
    model_key: str,
    plan_dir: Path,
    out_dir: Path,
    run_label: str,
    limit: int | None = None,
    model_path: str | None = None,
) -> Path:
    """Execute both turns for every planned trial and write one JSONL record each."""
    entry = _model_entry(model_key)
    defaults = config.models()["defaults"]
    sampling_cfg = config.models()["sampling"]
    study = config.study()
    revision = entry["revision"]

    load_target = model_path or entry["model_id"]
    if model_path and Path(model_path).resolve().name != revision:
        raise ValueError(
            "local model snapshot directory must be named with the configured revision "
            f"{revision}; got {Path(model_path).resolve().name}"
        )

    variants = plan.read_variants(plan_dir)
    trials = plan.read_trials(plan_dir)
    if limit:
        trials = trials[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / f"responses__{model_key}__{run_label}.jsonl"
    manifest_path = out_dir / f"manifest__{model_key}__{run_label}.json"
    existing = [str(path) for path in (responses_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(
            f"run label {run_label!r} would overwrite existing output: {existing}"
        )

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    started = datetime.now(timezone.utc).isoformat()

    llm = LLM(
        model=load_target,
        revision=None if model_path else revision,
        tensor_parallel_size=entry["tensor_parallel_size"],
        dtype=defaults["dtype"],
        max_model_len=defaults["max_model_len"],
        gpu_memory_utilization=defaults["gpu_memory_utilization"],
        enable_prefix_caching=defaults["enable_prefix_caching"],
        trust_remote_code=True,
    )

    chat_kwargs = entry.get("chat_template_kwargs") or {}

    def sampling_for(schema_name: str, seed: int) -> "SamplingParams":
        schema = json.loads((paths.PROMPTS / schema_name).read_text())
        params: dict[str, Any] = dict(
            temperature=sampling_cfg["temperature"],
            top_p=sampling_cfg["top_p"],
            max_tokens=sampling_cfg["max_tokens"],
            seed=seed,
        )
        if not defaults["structured_output"]:
            return SamplingParams(**params)
        return SamplingParams(**params, **_structured_kwargs(schema))

    # First turn.
    first_messages = []
    for trial in trials:
        variant = variants[trial["variant_id"]]
        first_messages.append(
            [
                {"role": "system", "content": variant["system_prompt"]},
                {"role": "user", "content": variant["user_prompt"]},
            ]
        )

    first_params = [
        sampling_for("initial_output.schema.json", sampling_cfg["seed_base"] + index)
        for index in range(len(trials))
    ]
    first_start = time.time()
    first_outputs = llm.chat(first_messages, first_params, chat_template_kwargs=chat_kwargs)
    first_elapsed = time.time() - first_start

    # Second turn, in the same conversation so the model can reference its own answer.
    second_messages = []
    second_params = []
    for index, trial in enumerate(trials):
        initial_raw = first_outputs[index].outputs[0].text
        trial = dict(trial)
        trial["identity_label"] = _identity_label(trial["condition"])
        second_messages.append(
            first_messages[index]
            + [
                {"role": "assistant", "content": initial_raw},
                {"role": "user", "content": _reflection_prompt(trial, initial_raw)},
            ]
        )
        second_params.append(
            sampling_for(
                trial["reflection_schema"], sampling_cfg["seed_base"] + 100_000 + index
            )
        )

    second_start = time.time()
    second_outputs = llm.chat(second_messages, second_params, chat_template_kwargs=chat_kwargs)
    second_elapsed = time.time() - second_start

    with responses_path.open("w") as handle:
        for index, trial in enumerate(trials):
            handle.write(
                json.dumps(
                    {
                        **trial,
                        "model_key": model_key,
                        "model_id": entry["model_id"],
                        "model_revision": revision,
                        "run_label": run_label,
                        "initial_raw": first_outputs[index].outputs[0].text,
                        "reflection_raw": second_outputs[index].outputs[0].text,
                        "reflection_prompt": second_messages[index][-1]["content"],
                    }
                )
                + "\n"
            )

    from . import stimuli

    pool_path = stimuli.name_pool_path(allow_provisional=True)
    manifest = RunManifest(
        run_label=run_label,
        model_key=model_key,
        model_id=entry["model_id"],
        model_revision=revision,
        started_at_utc=started,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        temperature=sampling_cfg["temperature"],
        top_p=sampling_cfg["top_p"],
        max_tokens=sampling_cfg["max_tokens"],
        seed_base=sampling_cfg["seed_base"],
        tensor_parallel_size=entry["tensor_parallel_size"],
        accelerator_model=torch.cuda.get_device_name(0),
        accelerator_memory_gib=int(entry["gpu_memory_gib"]),
        dtype=defaults["dtype"],
        max_model_len=defaults["max_model_len"],
        structured_output=defaults["structured_output"],
        prefix_caching=defaults["enable_prefix_caching"],
        study_seed=int(study["seed"]),
        runs_per_variant=int(study["runs_per_variant"]),
        trial_count=len(trials),
        scenario_set_version=study["sources"]["scenario_set_version"],
        name_pool=pool_path.name,
        name_pool_provisional=stimuli.is_provisional(pool_path),
        host=platform.node(),
        engine_version=vllm.__version__,
        notes=(
            f"first turn {first_elapsed:.1f}s, second turn {second_elapsed:.1f}s"
        ),
    )
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2) + "\n"
    )
    return responses_path


def _structured_kwargs(schema: dict[str, Any]) -> dict[str, Any]:
    """Build the schema-constrained decoding argument for the installed engine.

    The argument name for JSON-schema decoding changed between engine releases;
    both spellings are attempted so a version bump does not silently fall back
    to unconstrained generation.
    """
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(json=schema)}
    except ImportError as exc:
        raise RuntimeError(
            "installed engine exposes neither StructuredOutputsParams nor "
            "GuidedDecodingParams; set structured_output to false in configs/models.yaml "
            "only if unconstrained decoding is acceptable for this run"
        ) from exc
