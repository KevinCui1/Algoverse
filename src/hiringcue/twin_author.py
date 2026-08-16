"""Authoring pass that produces the perturbed soft-criteria twins.

Runs against the same locally served model used for evaluation, one call per
occupation. Output is validated against the twin contract before it is written,
and a profile that fails validation is retried once at a higher temperature
before the occupation is reported as unauthored - a silently missing twin would
disable the free-parameter diagnostic for every scenario in that occupation.
"""

from __future__ import annotations

import json
from typing import Any

from . import config, parse, paths, scenarios, twins

MAX_ATTEMPTS = 2


def _one_per_occupation(
    families: list[scenarios.ScenarioFamily],
) -> dict[str, scenarios.ScenarioFamily]:
    """One representative family per occupation.

    The soft profile does not vary across margin bands within an occupation in
    the current scenario set, so a single twin serves all of its bands. This is
    asserted rather than assumed.
    """
    chosen: dict[str, scenarios.ScenarioFamily] = {}
    for family in families:
        existing = chosen.get(family.occupation_slug)
        if existing is None:
            chosen[family.occupation_slug] = family
            continue
        if existing.soft_profile != family.soft_profile:
            raise twins.TwinError(
                f"{family.occupation_slug}: soft profile varies across margin bands, "
                "so one twin per occupation is not sufficient"
            )
    return chosen


def author_all(
    families: list[scenarios.ScenarioFamily], model_key: str
) -> dict[str, list[dict[str, Any]]]:
    from vllm import LLM, SamplingParams

    from .runner import _model_entry, _structured_kwargs

    entry = _model_entry(model_key)
    defaults = config.models()["defaults"]
    schema = json.loads((paths.PROMPTS / "soft_twin_author_output.schema.json").read_text())

    representatives = _one_per_occupation(families)
    slugs = sorted(representatives)

    llm = LLM(
        model=entry["model_id"],
        tensor_parallel_size=entry["tensor_parallel_size"],
        dtype=defaults["dtype"],
        max_model_len=defaults["max_model_len"],
        gpu_memory_utilization=defaults["gpu_memory_utilization"],
        enable_prefix_caching=defaults["enable_prefix_caching"],
        trust_remote_code=True,
    )
    chat_kwargs = entry.get("chat_template_kwargs") or {}

    profiles: dict[str, list[dict[str, Any]]] = {}
    pending = list(slugs)
    for attempt in range(MAX_ATTEMPTS):
        if not pending:
            break
        messages = []
        for slug in pending:
            system, user = twins.build_prompt(representatives[slug])
            messages.append(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
        params = SamplingParams(
            temperature=0.4 + 0.4 * attempt,
            top_p=0.95,
            max_tokens=900,
            seed=config.models()["sampling"]["seed_base"] + attempt,
            **_structured_kwargs(schema),
        )
        outputs = llm.chat(messages, params, chat_template_kwargs=chat_kwargs)

        still_pending = []
        for slug, output in zip(pending, outputs):
            result = parse.parse(
                output.outputs[0].text, "soft_twin_author_output.schema.json"
            )
            if not result.valid:
                still_pending.append(slug)
                continue
            profile = result.parsed["soft_profile"]
            try:
                twins.validate(representatives[slug], profile)
            except twins.TwinError:
                still_pending.append(slug)
                continue
            profiles[slug] = profile
        pending = still_pending

    if pending:
        raise twins.TwinError(
            f"unauthored twins after {MAX_ATTEMPTS} attempts: {pending}"
        )
    return profiles
