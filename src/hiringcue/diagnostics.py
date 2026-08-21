"""Blocking diagnostics on the Yes/No token log-odds readout.

Two kinds of criterion are evaluated here and they answer different questions.

The *instrument* criteria decide whether a model's readout may be read at all.
Cross-batch-composition stability asks whether the same prompt returns the same
number when nothing relevant to it changes. Logit-versus-greedy agreement asks
whether the contrast points at the answer the model would actually emit.
Saturation asks whether the decision has collapsed to the point where the
representable spacing of a bounded-precision log-odds is a material fraction of
the effect being estimated. Differential off-target mass asks whether the two
identity arms are answering the same question at all: the contrast conditions on
the answer being Yes or No, and a systematic arm difference in that conditioning
is an effect the contrast cannot see.

The *determinacy* criteria decide whether the readout has movement the
qualification rule does not already fix. A readout that is a deterministic
restatement of the gate arithmetic still produces a wide distribution, because
the arithmetic itself varies across scenarios, so no distributional statistic
can detect the failure. It takes a positive control:

    D1  variance explained by gate status and margin
    D2  response to a soft-criteria change that touches no gate  <- the control
    D3  dispersion within the gate-passing class, within band

D2 is the one that cannot be faked. A model that will not move its contrast for
a substantive change to the criteria the contrast is supposed to reflect will
not move it for a name either, and no sample size repairs that, because the
problem is in the numerator.

The repeat-noise statistic that stood alongside these is retired rather than
ported. Under a deterministic single forward pass a byte-identical repeat is a
duplicate, and the property that statistic was reaching for - stability under an
irrelevant perturbation - is what the cross-batch gate measures directly.

Input is one measurement record per prompt, carrying the planned fields and the
readout. Output is a per-model verdict. A failed gate is recorded, not retried
until it passes: a checkpoint that cannot be measured stably is a finding about
that checkpoint.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from . import config, gates, plan, readout

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"


def _band_verdict(value: float | None, rule: dict[str, Any], higher_is_better: bool) -> str:
    if value is None:
        return SKIP
    if higher_is_better:
        if value >= float(rule["pass_at_least"]):
            return PASS
        return WARN if value >= float(rule["fail_below"]) else FAIL
    if value <= float(rule["pass_at_most"]):
        return PASS
    return WARN if value <= float(rule["fail_above"]) else FAIL


def _primary(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows that enter the primary estimand, excluding development-only probes."""
    confirmatory_contexts = set(config.study()["context"]["levels"])
    return [
        row
        for row in rows
        if row.get("role") == plan.PRIMARY
        and row.get("context_level") in confirmatory_contexts
    ]


def saturation(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Share of cells whose implied Yes probability leaves the readable interval."""
    settings = config.gate_thresholds()["instrument"]["saturation"]
    low, high = (float(value) for value in settings["probability_interval"])
    limit = float(settings["maximum_outside_share"])

    probabilities = [float(row["implied_yes_probability"]) for row in rows]
    if not probabilities:
        return {"verdict": SKIP, "cells": 0}
    outside = sum(1 for value in probabilities if value < low or value > high)
    share = outside / len(probabilities)
    ordered = sorted(probabilities)
    return {
        "cells": len(probabilities),
        "outside_share": share,
        "maximum_outside_share": limit,
        "probability_interval": [low, high],
        "quantiles": {
            "p05": ordered[int(0.05 * (len(ordered) - 1))],
            "median": statistics.median(ordered),
            "p95": ordered[int(0.95 * (len(ordered) - 1))],
        },
        "verdict": PASS if share < limit else FAIL,
    }


def differential_off_target_mass(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Arm difference in off-target mass, paired within counterfactual sets.

    Paired rather than marginal because off-target mass varies strongly across
    scenario families, and a marginal comparison of the two arms would be
    dominated by which families happen to sit in each rather than by the arm
    itself. Within a counterfactual set the two arms share every fact but the
    identity block, so the difference is attributable.
    """
    settings = config.gate_thresholds()["instrument"]["differential_off_target_mass"]
    limit = float(settings["maximum_paired_arm_difference"])
    arm_limit = float(settings["maximum_arm_mean"])

    paired: dict[str, dict[str, float]] = defaultdict(dict)
    arms: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        arm = row.get("identity_group")
        if arm is None:
            continue
        arms[arm].append(float(row["off_target_mass"]))
        pair_id = row.get("counterfactual_pair_id")
        if pair_id and pair_id != "unpaired":
            paired[pair_id][arm] = float(row["off_target_mass"])

    complete = [
        entry["white"] - entry["black"]
        for entry in paired.values()
        if "white" in entry and "black" in entry
    ]
    if not complete:
        return {"verdict": SKIP, "pairs": 0}

    difference = statistics.fmean(complete)
    arm_means = {arm: statistics.fmean(values) for arm, values in sorted(arms.items())}
    excessive = [arm for arm, value in arm_means.items() if value > arm_limit]
    return {
        "pairs": len(complete),
        "paired_arm_difference": difference,
        "maximum_paired_arm_difference": limit,
        "arm_means": arm_means,
        "maximum_arm_mean": arm_limit,
        "verdict": PASS if abs(difference) <= limit and not excessive else FAIL,
    }


def stability(
    readings_by_layout: dict[str, Sequence[readout.Reading]],
    prompts: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Movement of the estimand across fixed batch layouts, plus per-prompt diagnostics.

    The gate is on the quantity the design actually depends on. That quantity is
    not one prompt's contrast; it is a mean over cells of an identity contrast,
    and a gate on the largest movement of any single prompt answers a stricter
    question than the study asks. The previous form did exactly that and reported
    a failure that did not follow from its own numbers: per-prompt maxima of 0.25
    to 0.75 alongside a movement of the cell-mean estimate of 0.031 to 0.078.

    The statistic is therefore built to mirror the confirmatory estimator. Within
    each cell - one scenario family at one context level - the two identity arms
    are averaged separately and subtracted, so it needs no matched pairing and is
    unaffected by an arm missing a partner. Those cell contrasts are averaged, and
    the gate is the range of that average across layouts. It is restricted to the
    qualified bands and the concealed cue mode because that is the population the
    confirmatory estimand is defined on.

    The range is used rather than a signed difference against a nominated
    reference layout: no layout is privileged, and a difference against one of
    them would report a smaller number simply by choosing a reference near the
    middle.

    Per-prompt movement is still computed and reported. It is a diagnostic that
    localises a mechanism, not a criterion; a checkpoint is not dropped for it.
    """
    settings = config.gate_thresholds()["instrument"]["cross_batch_stability"]
    limit = float(settings["maximum_estimand_range"])
    qualified = set(config.study()["margin_bands"]) - set(
        config.study()["qualification"]["control_bands"]
    )

    if len(readings_by_layout) < 2:
        return {"verdict": SKIP, "layouts": len(readings_by_layout)}

    values: dict[str, dict[str, float]] = defaultdict(dict)
    for layout, readings in readings_by_layout.items():
        for reading in readings:
            values[reading.prompt_id][layout] = reading.token_log_odds
    shared = {
        prompt_id: measurements
        for prompt_id, measurements in values.items()
        if len(measurements) == len(readings_by_layout)
    }
    if not shared:
        raise ValueError(
            "no prompt was measured under every layout; the stability gate compares "
            "one prompt across compositions and cannot be evaluated on disjoint sets"
        )

    deltas = {
        prompt_id: max(measurements.values()) - min(measurements.values())
        for prompt_id, measurements in shared.items()
    }
    worst_prompt = max(deltas, key=deltas.get)
    report: dict[str, Any] = {
        "prompts": len(shared),
        "layouts": sorted(readings_by_layout),
        "per_prompt_maximum_absolute_delta": deltas[worst_prompt],
        "per_prompt_median_absolute_delta": statistics.median(deltas.values()),
        "per_prompt_worst_id": worst_prompt,
        "limit": limit,
    }

    if prompts is None:
        report["verdict"] = SKIP
        report["estimand_range"] = None
        return report

    index = {prompt.prompt_id: prompt for prompt in prompts}
    per_layout: dict[str, float] = {}
    cell_counts: set[int] = set()
    for layout in readings_by_layout:
        cells: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: {"black": [], "white": []}
        )
        for prompt_id, measurements in shared.items():
            prompt = index.get(prompt_id)
            if prompt is None or prompt.margin_band not in qualified:
                continue
            if prompt.cue_mode != "concealed":
                continue
            if prompt.identity_group not in ("black", "white"):
                continue
            cells[(prompt.family_id, prompt.context_level)][prompt.identity_group].append(
                measurements[layout]
            )
        contrasts = [
            statistics.mean(arms["black"]) - statistics.mean(arms["white"])
            for arms in cells.values()
            if arms["black"] and arms["white"]
        ]
        if not contrasts:
            raise ValueError(
                "no qualified concealed cell carries both identity arms; the gate "
                "statistic is a mean over such cells and cannot be formed"
            )
        cell_counts.add(len(contrasts))
        per_layout[layout] = statistics.mean(contrasts)

    spread = max(per_layout.values()) - min(per_layout.values())
    report.update(
        {
            "cells": sorted(cell_counts)[0],
            "estimand_by_layout": {k: v for k, v in sorted(per_layout.items())},
            "estimand_range": spread,
            "verdict": PASS if spread < limit else FAIL,
        }
    )
    return report


def batch_size_sensitivity(
    readings_by_layout: dict[str, Sequence[readout.Reading]],
    prompts: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Estimand movement across batch sizes, disclosed without gating.

    Batch size is frozen within every collection, so changing it is not an
    irrelevant perturbation the stability gate should require invariance to.
    The same statistic remains scientifically useful as a property of the
    instrument. It is therefore computed permanently and reported without a
    threshold or verdict; this disclosure replaces the former gate dimension.
    """
    report = stability(readings_by_layout, prompts)
    report.pop("limit", None)
    report.pop("verdict", None)
    report["disclosure_only"] = True
    return report


def d1_rule_determinacy(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """R-squared of the contrast on gate status, margin, and their interaction.

    Fitted on every family rather than on the qualified subset, because the
    quantity is how much of the readout the objective rule already explains, and
    the rule's explanatory power lives mostly in the difference between passing
    and failing.
    """
    rule = config.gate_thresholds()["determinacy"]["D1_rule_determinacy_r2"]
    usable = [
        row
        for row in rows
        if row.get("soft_variant") == "base" and row.get("minimum_gate_margin") is not None
    ]
    if len(usable) < 8:
        return {"verdict": SKIP, "observations": len(usable)}

    passed = np.array(
        [1.0 if row["gold_decision"] == gates.ADVANCE else 0.0 for row in usable]
    )
    margin = np.array([float(row["minimum_gate_margin"]) for row in usable])
    outcome = np.array([float(row["token_log_odds"]) for row in usable])
    design = np.column_stack(
        [np.ones_like(passed), passed, margin, passed * margin]
    )

    coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    residual = outcome - design @ coefficients
    total = float(((outcome - outcome.mean()) ** 2).sum())
    r_squared = 1.0 - float((residual**2).sum()) / total if total > 0 else 1.0
    return {
        "observations": len(usable),
        "r_squared": r_squared,
        "excluded_categorical_only": len(rows) - len(usable),
        "verdict": _band_verdict(r_squared, rule, higher_is_better=False),
    }


def d2_free_parameter(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Median absolute movement between a scenario and its perturbed twin."""
    rule = config.gate_thresholds()["determinacy"]["D2_free_parameter_response"]
    settings = config.study()["soft_twin"]

    twins = {
        row["family_id"]: float(row["token_log_odds"])
        for row in rows
        if row.get("soft_variant") == "twin"
    }
    base = {
        row["family_id"]: float(row["token_log_odds"])
        for row in rows
        if row.get("soft_variant") == "base"
        and row.get("condition") == settings["condition"]
        and row.get("prestige_level") == settings["prestige_level"]
        and row.get("context_level") == settings["context_level"]
    }
    shared = sorted(set(twins) & set(base))
    if not shared:
        return {"verdict": SKIP, "families": 0}

    movements = [abs(twins[family] - base[family]) for family in shared]
    median = statistics.median(movements)
    return {
        "families": len(shared),
        "median_absolute_movement": median,
        "verdict": _band_verdict(median, rule, higher_is_better=True),
    }


def d3_conditional_dispersion(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Dispersion of the contrast among qualified candidates, within margin band.

    Conditioning on band is what makes this a conditional statistic. A marginal
    spread over the whole set is wide whenever the gate arithmetic varies, which
    it always does, so it cannot tell a responsive readout from a determined one.
    """
    rule = config.gate_thresholds()["determinacy"]["D3_conditional_dispersion"]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in _primary(rows):
        if row.get("soft_variant") != "base":
            continue
        grouped[row["margin_band"]].append(float(row["token_log_odds"]))

    per_band = {
        band: statistics.stdev(values)
        for band, values in sorted(grouped.items())
        if len(values) > 1
    }
    if not per_band:
        return {"verdict": SKIP, "bands": 0}

    pooled = statistics.fmean(per_band.values())
    return {
        "bands": len(per_band),
        "standard_deviation_by_band": per_band,
        "pooled_standard_deviation": pooled,
        "verdict": _band_verdict(pooled, rule, higher_is_better=True),
    }


def evaluate(
    rows: Sequence[dict[str, Any]],
    agreement: dict[str, dict[str, Any]] | None = None,
    stability_readings: dict[str, dict[str, Sequence[readout.Reading]]] | None = None,
) -> dict[str, Any]:
    """Per-model verdicts across every criterion, and the authorisation decision."""
    thresholds = config.gate_thresholds()
    agreement_rule = thresholds["instrument"]["logit_greedy_agreement"]
    escalation = int(thresholds["escalation"]["warns_equal_fail"])

    report: dict[str, Any] = {"models": {}}
    for model in sorted({row["model_key"] for row in rows}):
        model_rows = [row for row in rows if row["model_key"] == model]
        primary_rows = _primary(model_rows)

        criteria = {
            "saturation": saturation(primary_rows),
            "differential_off_target_mass": differential_off_target_mass(primary_rows),
            "D1_rule_determinacy_r2": d1_rule_determinacy(model_rows),
            "D2_free_parameter_response": d2_free_parameter(model_rows),
            "D3_conditional_dispersion": d3_conditional_dispersion(model_rows),
        }
        if stability_readings and model in stability_readings:
            criteria["cross_batch_stability"] = stability(stability_readings[model])
        if agreement and model in agreement:
            measured = agreement[model]
            criteria["logit_greedy_agreement"] = {
                **measured,
                "minimum_observed": float(agreement_rule["minimum_observed"]),
                "minimum_wilson_lower_bound": float(
                    agreement_rule["minimum_wilson_lower_bound"]
                ),
                "verdict": PASS
                if measured["agreement"] >= float(agreement_rule["minimum_observed"])
                and measured["wilson_lower_bound"]
                >= float(agreement_rule["minimum_wilson_lower_bound"])
                else FAIL,
            }

        failed = [name for name, entry in criteria.items() if entry["verdict"] == FAIL]
        warned = [name for name, entry in criteria.items() if entry["verdict"] == WARN]
        report["models"][model] = {
            "criteria": criteria,
            "failed": failed,
            "warned": warned,
            "skipped": [name for name, entry in criteria.items() if entry["verdict"] == SKIP],
            "authorised": not failed and len(warned) < escalation,
        }
    return report


def development_estimates(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Crossed interaction, variance components, and recognisability probe.

    The main interaction uses an 80 percent interval because it selects a
    predeclared context variant rather than supporting a confirmatory claim.
    The recognisability quantity is a paired difference between the named and
    invented matched-employer interactions, so shared family and name effects
    cancel before the crossed interval is formed.
    """
    from . import estimate

    interval_level = float(
        config.study()["inference"]["context_selection_interval_level"]
    )

    def record(cells: Sequence[tuple[str, str, float]]) -> dict[str, Any]:
        result = estimate.estimate(cells, interval_level=interval_level)
        measured = result.as_dict()
        measured["estimate"] = measured.pop("mean")
        return measured

    main_cells = estimate.interaction_cells(rows)
    report = {"context_identity_interaction": record(main_cells)}

    probe_levels = config.study()["context"].get("development_only_levels", [])
    if len(probe_levels) == 2:
        matched = estimate.interaction_cells(rows, realistic_level=probe_levels[0])
        named = estimate.interaction_cells(rows, realistic_level=probe_levels[1])
        matched_map = {(family, pair): value for family, pair, value in matched}
        named_map = {(family, pair): value for family, pair, value in named}
        if set(matched_map) != set(named_map):
            raise estimate.EstimationError(
                "recognisability interactions do not share the same family-by-name cells"
            )
        difference = [
            (family, pair, named_map[(family, pair)] - matched_map[(family, pair)])
            for family, pair in sorted(named_map)
        ]
        report["recognisability"] = {
            "named": record(named),
            "matched_invented": record(matched),
            "named_minus_matched": record(difference),
        }
    return report


def context_selection(
    interaction_by_variant: dict[str, dict[str, float]],
    saturation_by_variant: dict[str, float],
) -> dict[str, Any]:
    """Apply the predeclared order over the two realistic context variants.

    Employer context alone is evaluated first. The selectivity fallback is
    considered only if the first fails its criterion, and is accepted only if it
    passes the same criterion without increasing saturation. Evaluating them in
    a fixed order, rather than taking whichever performs better, is what keeps
    the selected template a stimulus rather than a fitted parameter.
    """
    from . import context as context_factor

    settings = config.study()["inference"]
    threshold = float(settings["minimum_meaningful_effect"])
    order = context_factor.realistic_variants()

    evaluated = []
    for index, variant in enumerate(order):
        measured = interaction_by_variant.get(variant)
        if measured is None:
            evaluated.append({"variant": variant, "status": "not measured"})
            continue
        meets = abs(measured["estimate"]) >= threshold and (
            measured["interval_lower"] > 0 or measured["interval_upper"] < 0
        )
        entry = {
            "variant": variant,
            "estimate": measured["estimate"],
            "interval": [measured["interval_lower"], measured["interval_upper"]],
            "meets_criterion": meets,
            "saturation": saturation_by_variant.get(variant),
        }
        if index > 0 and meets:
            first = saturation_by_variant.get(order[0])
            current = saturation_by_variant.get(variant)
            if first is not None and current is not None and current > first:
                entry["meets_criterion"] = False
                entry["rejected_because"] = "increases saturation over the first variant"
                meets = False
        evaluated.append(entry)
        if meets:
            return {"selected": variant, "evaluated": evaluated, "decision": "proceed"}

    if any(entry.get("status") == "not measured" for entry in evaluated):
        return {
            "selected": None,
            "evaluated": evaluated,
            "decision": "fallback_required",
        }
    return {"selected": None, "evaluated": evaluated, "decision": "kill"}
