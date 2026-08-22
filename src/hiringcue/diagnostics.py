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

Every criterion is evaluated per design cell - one prompt form at one context
level - and never pooled across cells. The cells differ in exactly the property
under test, so a pooled figure would average over the manipulation and describe
no condition the study ran.

The confirmatory candidate among the admissible cells is chosen by the highest
free-parameter response, ties broken by whichever readout sits closest to the
middle of its range. Neither quantity is the identity effect: the cell is
selected for how much room the readout has, not for what it returned.

Input is one measurement record per prompt, carrying the planned fields and the
readout. Output is a per-model, per-cell verdict. A failed gate is recorded, not
retried until it passes: a checkpoint that cannot be measured stably is a
finding about that checkpoint.
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
    """Rows that enter the primary estimand."""
    return [row for row in rows if row.get("role") == plan.PRIMARY]


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


def cross_batch_stability(
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
    each cell - one scenario family at one prompt form and one context level -
    the two identity arms are averaged separately and subtracted, so it needs no
    matched pairing and is unaffected by an arm missing a partner. The prompt
    form is part of the cell key for the same reason the context level is: the
    two forms are the manipulation under test, and merging them would average the
    gate statistic across it and halve the number of cells the mean is formed
    over. Those cell contrasts are averaged, and
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
        cells: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
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
            cells[
                (prompt.family_id, prompt.prompt_form, prompt.context_level)
            ][prompt.identity_group].append(measurements[layout])
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


# The measuring form keeps its original name. Stage 0 runs the layouts and calls
# it there; `evaluate` accepts the verdict it produced rather than re-reading the
# layouts, which are not carried alongside the collection.
stability = cross_batch_stability


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
    report = cross_batch_stability(readings_by_layout, prompts)
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
    """Median absolute movement between a scenario and its perturbed twin.

    Keyed on the design cell as well as the family. Twins now exist at every
    prompt form and every context level, and a key that carried only the family
    would collapse several twins onto one entry and silently report whichever
    the iteration order happened to leave last - a number attributed to a cell
    it was not measured in.
    """
    rule = config.gate_thresholds()["determinacy"]["D2_free_parameter_response"]
    settings = config.study()["soft_twin"]

    def cell(row: dict[str, Any]) -> tuple[str, str, str]:
        return (row["family_id"], row["prompt_form"], row["context_level"])

    twins = {
        cell(row): float(row["token_log_odds"])
        for row in rows
        if row.get("soft_variant") == "twin"
    }
    base = {
        cell(row): float(row["token_log_odds"])
        for row in rows
        if row.get("soft_variant") == "base"
        and row.get("condition") == settings["condition"]
        and row.get("prestige_level") == settings["prestige_level"]
    }
    shared = sorted(set(twins) & set(base))
    if not shared:
        return {"verdict": SKIP, "families": 0}

    movements = [abs(twins[key] - base[key]) for key in shared]
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


def arm_length_gap(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Templated token-length difference between the identity arms.

    Reported, not gated. The two arms differ only inside the identity block, so
    any length difference is the length of the names themselves. It is disclosed
    because a systematic gap places the answer boundary at a different absolute
    position in one arm than the other, which is a property of the surface the
    contrast is read on rather than of the judgement being read.
    """
    paired: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        arm = row.get("identity_group")
        pair_id = row.get("counterfactual_pair_id")
        length = row.get("boundary_index")
        if arm is None or not pair_id or pair_id == "unpaired" or length is None:
            continue
        paired[pair_id][arm] = int(length)

    gaps = [
        entry["black"] - entry["white"]
        for entry in paired.values()
        if "black" in entry and "white" in entry
    ]
    if not gaps:
        return {"pairs": 0}
    return {
        "pairs": len(gaps),
        "mean_black_minus_white_tokens": statistics.fmean(gaps),
        "maximum_absolute_gap": max(abs(gap) for gap in gaps),
        "share_exactly_equal": sum(1 for gap in gaps if gap == 0) / len(gaps),
    }


def _cell_key(prompt_form: str, context_level: str) -> str:
    return f"{prompt_form}/{context_level}"


def cells(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every diagnostic, evaluated separately in each design cell.

    A cell is one prompt form at one context level. Nothing here is pooled
    across cells: the cells differ in exactly the property under test - whether
    the decision rule is supplied, and how much organisational context surrounds
    it - so a pooled figure would average over the manipulation and report a
    number belonging to no condition the study ran.
    """
    from . import estimate as estimator

    interval_level = float(
        config.study()["inference"]["development_interval_level"]
    )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["prompt_form"], row["context_level"])].append(row)

    def measured(build, crossed: bool) -> dict[str, Any]:
        """Report an estimate, or why the cell could not supply one.

        Both the contrast construction and the fit are guarded. An incomplete
        cell is a fact about that cell and must not abort the diagnosis of every
        other one.
        """
        try:
            contrasts = build()
            result = (
                estimator.estimate(contrasts, interval_level=interval_level)
                if crossed
                else estimator.clustered(contrasts, interval_level=interval_level)
            )
        except estimator.EstimationError as exc:
            return {"unavailable": str(exc)}
        record = result.as_dict()
        record["estimate"] = record.pop("mean")
        return record

    report: dict[str, dict[str, Any]] = {}
    for (form, level), cell_rows in sorted(grouped.items()):
        primary_rows = _primary(cell_rows)
        criteria = {
            "saturation": saturation(primary_rows),
            "differential_off_target_mass": differential_off_target_mass(primary_rows),
            "D1_rule_determinacy_r2": d1_rule_determinacy(cell_rows),
            "D2_free_parameter_response": d2_free_parameter(cell_rows),
            "D3_conditional_dispersion": d3_conditional_dispersion(cell_rows),
        }
        entry: dict[str, Any] = {
            "prompt_form": form,
            "context_level": level,
            "criteria": criteria,
            "arm_length_gap": arm_length_gap(primary_rows),
            "identity_effect": {
                "concealed": measured(
                    lambda: estimator.identity_cells(rows, level, "concealed", form),
                    True,
                ),
                "direct": measured(
                    lambda: estimator.identity_cells(rows, level, "direct", form),
                    False,
                ),
            },
            "failed": [
                name for name, value in criteria.items() if value["verdict"] == FAIL
            ],
            "warned": [
                name for name, value in criteria.items() if value["verdict"] == WARN
            ],
        }
        if level != "bare":
            entry["interaction_vs_bare"] = {
                "concealed": measured(
                    lambda: estimator.interaction_cells(rows, "concealed", level, form),
                    True,
                ),
                "direct": measured(
                    lambda: estimator.interaction_cells(rows, "direct", level, form),
                    False,
                ),
            }
        report[_cell_key(form, level)] = entry
    return report


def select_cell(
    cell_report: dict[str, dict[str, Any]], instrument_passes: bool
) -> dict[str, Any]:
    """Choose the confirmatory candidate cell, blind to the identity effect.

    Admissibility is every instrument criterion and every determinacy criterion
    at the thresholds already registered; no threshold moves for this selection.
    Among admissible cells the candidate is the one with the highest D2, ties
    broken by whichever median implied Yes probability sits closest to 0.5.

    Neither input mentions the identity effect, its sign or its interval. The
    rule is stated this way so that the cell is chosen for how much room the
    readout has, which is a property of the instrument in that condition, rather
    than for how large an effect it happened to return - which would make the
    reported effect a selection.
    """
    admissible = []
    for key, entry in sorted(cell_report.items()):
        ok = instrument_passes and not entry["failed"] and not entry["warned"]
        entry["admissible"] = ok
        if ok:
            admissible.append((key, entry))

    if not admissible:
        return {
            "selected": None,
            "admissible": [],
            "decision": "no_admissible_cell",
        }

    def rank(item: tuple[str, dict[str, Any]]) -> tuple[float, float]:
        _, entry = item
        d2 = float(entry["criteria"]["D2_free_parameter_response"]["median_absolute_movement"])
        median = float(entry["criteria"]["saturation"]["quantiles"]["median"])
        return (-d2, abs(median - 0.5))

    selected, entry = min(admissible, key=rank)
    return {
        "selected": selected,
        "admissible": [key for key, _ in admissible],
        "inputs": {
            key: {
                "D2": value["criteria"]["D2_free_parameter_response"][
                    "median_absolute_movement"
                ],
                "median_implied_yes_probability": value["criteria"]["saturation"][
                    "quantiles"
                ]["median"],
            }
            for key, value in admissible
        },
        "decision": "proceed",
    }


def kill_criterion(
    cell_report: dict[str, dict[str, Any]], selected: str | None
) -> dict[str, Any]:
    """Whether any rich context in the selected cell reaches the registered effect.

    The criterion is a magnitude and an interval, not a sign: the published
    interaction this design is anchored on favours the Black-associated arm, so
    a test written on the sign alone would score a same-sized opposite effect as
    a failure to replicate rather than as the different finding it is. The sign
    is reported beside the magnitude instead.
    """
    settings = config.study()["inference"]
    threshold = float(settings["minimum_meaningful_effect"])
    if selected is None:
        # Same shape as the evaluated branch. A report whose keys depend on the
        # outcome forces the reader of the result to special-case the failure.
        return {
            "minimum_meaningful_effect": threshold,
            "selected_cell": None,
            "evaluated": [],
            "verdict": SKIP,
            "reason": "no admissible cell was selected",
        }

    form = cell_report[selected]["prompt_form"]
    evaluated = []
    for key, entry in sorted(cell_report.items()):
        measured = entry.get("interaction_vs_bare", {}).get("concealed")
        if entry["prompt_form"] != form or measured is None or "estimate" not in measured:
            continue
        excludes_zero = (
            measured["interval_lower"] > 0.0 or measured["interval_upper"] < 0.0
        )
        evaluated.append(
            {
                "cell": key,
                "estimate": measured["estimate"],
                "interval": [measured["interval_lower"], measured["interval_upper"]],
                "reaches_threshold": abs(measured["estimate"]) >= threshold
                and excludes_zero,
            }
        )
    survives = [entry for entry in evaluated if entry["reaches_threshold"]]
    return {
        "minimum_meaningful_effect": threshold,
        "selected_cell": selected,
        "evaluated": evaluated,
        "verdict": "replicates" if survives else "kill",
    }


def evaluate(
    rows: Sequence[dict[str, Any]],
    agreement: dict[str, dict[str, Any]] | None = None,
    stability_readings: dict[str, dict[str, Sequence[readout.Reading]]] | None = None,
    stability: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Per-model, per-cell verdicts and the blind confirmatory-cell selection.

    The instrument criteria that are properties of the checkpoint rather than of
    a cell - cross-batch stability and logit-versus-greedy agreement - are
    evaluated once per model and gate every cell of it equally.
    """
    thresholds = config.gate_thresholds()
    agreement_rule = thresholds["instrument"]["logit_greedy_agreement"]

    report: dict[str, Any] = {"models": {}}
    for model in sorted({row["model_key"] for row in rows}):
        model_rows = [row for row in rows if row["model_key"] == model]
        instrument: dict[str, Any] = {}
        # Either measured here from layout readings, or carried in already
        # evaluated from the Stage 0 report, which is where the layouts are run.
        if stability_readings and model in stability_readings:
            instrument["cross_batch_stability"] = cross_batch_stability(
                stability_readings[model]
            )
        elif stability and model in stability:
            instrument["cross_batch_stability"] = stability[model]
        if agreement and model in agreement:
            measured = agreement[model]
            instrument["logit_greedy_agreement"] = {
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
        instrument_passes = all(
            entry.get("verdict") == PASS for entry in instrument.values()
        )

        cell_report = cells(model_rows)
        selection = select_cell(cell_report, instrument_passes)
        report["models"][model] = {
            "instrument": instrument,
            "instrument_passes": instrument_passes,
            "cells": cell_report,
            "selection": selection,
            "kill_criterion": kill_criterion(cell_report, selection["selected"]),
            "authorised": selection["selected"] is not None,
        }
    return report


def variance_components(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Blinded variance components per cell, for the variance-only sizing rule.

    Only the components are returned. The observed interaction is deliberately
    absent: sizing a confirmatory design against the effect the development
    round happened to return would carry that round's noise into the sample size
    and inflate the apparent precision of the collection it justifies.
    """
    from . import estimate as estimator

    report: dict[str, Any] = {}
    for level in config.study()["context"]["levels"]:
        if level == "bare":
            continue
        for form in config.study()["prompt_form"]["levels"]:
            try:
                contrasts = estimator.interaction_cells(rows, "concealed", level, form)
                values, _, _ = estimator.cell_matrix(contrasts)
                parts = estimator.components(values)
            except estimator.EstimationError as exc:
                report[_cell_key(form, level)] = {"unavailable": str(exc)}
                continue
            report[_cell_key(form, level)] = {
                "components": parts.as_dict(),
                "standard_deviations": {
                    "family": parts.family**0.5,
                    "name": parts.name**0.5,
                    "residual": parts.residual**0.5,
                },
                "sizing": estimator.select_design(
                    parts,
                    name_grid=[
                        int(value)
                        for value in config.study()["sizing"]["available_name_grid"]
                    ],
                ),
            }
    return report
