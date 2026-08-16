"""Descriptive pilot outputs and confirmatory family-count sizing.

The pilot is not an efficacy analysis. Its job is to expose instrument
behaviour, missing or guarded responses, preliminary heterogeneity, execution
cost, and the variance term needed to size a later family-clustered study. All
substantive summaries exclude the perturbed soft-criteria control; that control
exists only to test whether the score has a legitimate free parameter.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Callable

from . import config, derive, diagnostics


def _rate(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float | None:
    return sum(1 for row in rows if predicate(row)) / len(rows) if rows else None


def _group_summary(
    rows: list[dict[str, Any]], key: str, value: str
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(key) is not None and row.get(value) is not None:
            grouped[str(row[key])].append(float(row[value]))
    means = {group: statistics.fmean(values) for group, values in sorted(grouped.items())}
    return {
        "groups": len(means),
        "sd_of_group_means": statistics.stdev(means.values()) if len(means) > 1 else None,
        "group_means": means,
    }


def _integrity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["family_id"], row["prestige_level"], row["soft_variant"])].append(row)

    failures = []
    for key, entries in sorted(groups.items()):
        normalised = {row["normalised_hash"] for row in entries}
        exact_by_condition: dict[str, set[str]] = defaultdict(set)
        for row in entries:
            exact_by_condition[row["condition"]].add(row["exact_hash"])
        exact = {next(iter(values)) for values in exact_by_condition.values()}
        if len(normalised) != 1 or len(exact) != len(exact_by_condition):
            failures.append({"block": key, "normalised_hashes": len(normalised)})
    return {"blocks": len(groups), "failures": failures, "pass": not failures}


def _performance(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    report = {}
    for manifest in manifests:
        start = datetime.fromisoformat(manifest["started_at_utc"])
        finish = datetime.fromisoformat(manifest["finished_at_utc"])
        seconds = (finish - start).total_seconds()
        trials = int(manifest["trial_count"])
        accelerators = int(manifest["tensor_parallel_size"])
        report[manifest["model_key"]] = {
            "trials": trials,
            "wall_clock_seconds": seconds,
            "wall_clock_seconds_per_trial": seconds / trials,
            "accelerator_hours": seconds * accelerators / 3600,
            "accelerator_hours_per_trial": seconds * accelerators / 3600 / trials,
        }
    return report


def summary(
    rows: list[dict[str, Any]],
    differences: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return every descriptive quantity required from the pilot."""
    main = [row for row in rows if row.get("soft_variant") != "twin"]
    valid = [row for row in main if row.get("initial_valid")]
    main_differences = [row for row in differences if row.get("soft_variant") != "twin"]
    near_bands = set(config.study()["near_threshold_bands"])
    models = sorted({row["model_key"] for row in rows})

    report: dict[str, Any] = {
        "counterfactual_integrity": _integrity(rows),
        "execution": _performance(manifests),
        "models": {},
    }
    for model in models:
        model_all = [row for row in rows if row["model_key"] == model]
        model_main = [row for row in main if row["model_key"] == model]
        model_valid = [row for row in valid if row["model_key"] == model]
        model_diff = [row for row in main_differences if row["model_key"] == model]

        by_condition = {}
        for condition in sorted({row["condition"] for row in model_all}):
            subset = [row for row in model_all if row["condition"] == condition]
            by_condition[condition] = {
                "n": len(subset),
                "initial_schema_failure_rate": _rate(subset, lambda row: not row["initial_valid"]),
                "reflection_schema_failure_rate": _rate(subset, lambda row: not row["reflection_valid"]),
                "initial_refusal_rate": _rate(
                    subset, lambda row: bool(row["flags"]["initial_refusal"])
                ),
                "reflection_refusal_rate": _rate(
                    subset, lambda row: bool(row["flags"]["reflection_refusal"])
                ),
            }

        accuracy = {}
        for band in config.study()["margin_bands"]:
            subset = [row for row in model_valid if row["margin_band"] == band]
            accuracy[band] = {
                "n": len(subset),
                "accuracy": _rate(subset, lambda row: bool(row["initial_correct"])),
            }

        fairness = {}
        for cue_mode in ("neutral", "concealed", "direct"):
            subset = [row for row in model_main if row["cue_mode"] == cue_mode]
            fairness[cue_mode] = {
                "n": len(subset),
                "initial_fairness_commentary_rate": _rate(
                    subset, lambda row: bool(row["flags"]["initial_fairness_commentary"])
                ),
            }

        near = [row for row in model_diff if row["margin_band"] in near_bands]
        histogram = Counter(int(row["suitability_score"]) for row in model_valid)

        name_rows = []
        for row in model_diff:
            for group in ("white", "black"):
                shift = f"neutral_shift_concealed_{group}"
                name = f"name_concealed_{group}"
                if row.get(shift) is not None and row.get(name) is not None:
                    name_rows.append({"name": row[name], "shift": row[shift]})

        report["models"][model] = {
            "response_quality_by_condition": by_condition,
            "initial_accuracy_by_margin_band": accuracy,
            "unprompted_fairness_commentary_by_cue_mode": fairness,
            "near_threshold_binary_flip_rate": {
                mode: {
                    "n": sum(1 for row in near if f"decision_shift_{mode}" in row),
                    "rate": _rate(
                        [row for row in near if f"decision_shift_{mode}" in row],
                        lambda row, field=f"decision_shift_{mode}": row[field] != 0,
                    ),
                }
                for mode in ("concealed", "direct")
            },
            "score_distribution": {
                "n": len(model_valid),
                "histogram": {str(score): count for score, count in sorted(histogram.items())},
                **diagnostics.granularity(model_valid).get(model, {}),
            },
            "paired_score_difference": {
                field: derive.paired_difference_sd(model_diff, field).get(model)
                for field in (
                    "score_shift_concealed",
                    "score_shift_direct",
                    "cue_mode_interaction",
                )
            },
            "preliminary_name_variability": _group_summary(name_rows, "name", "shift"),
            "preliminary_occupation_variability": _group_summary(
                model_diff, "occupation_slug", "score_shift_concealed"
            ),
        }
    return report


def confirmatory_sizing(
    differences: list[dict[str, Any]],
    authorised_models: set[str] | None = None,
) -> dict[str, Any]:
    """Size family clusters from the concealed paired-difference variance."""
    settings = config.study()["confirmatory"]
    delta = float(settings["minimum_meaningful_score_difference"])
    alpha = float(settings["two_sided_alpha"])
    power = float(settings["power"])
    attrition = float(settings["family_attrition_fraction"])
    block = int(settings["margin_bands_per_occupation"])
    floor = int(settings["minimum_occupations"]) * block

    normal = statistics.NormalDist()
    critical = normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in differences:
        if row.get("soft_variant") != "twin" and row.get("score_shift_concealed") is not None:
            grouped[(row["model_key"], row["family_id"])].append(
                float(row["score_shift_concealed"])
            )

    by_model: dict[str, Any] = {}
    models = sorted({key[0] for key in grouped})
    for model in models:
        family_means = [
            statistics.fmean(values)
            for (model_key, _family), values in grouped.items()
            if model_key == model
        ]
        sd = statistics.stdev(family_means) if len(family_means) > 1 else None
        raw = math.ceil((critical * sd / delta) ** 2) if sd is not None else None
        attrition_adjusted = math.ceil(raw / (1 - attrition)) if raw is not None else None
        target = max(floor, attrition_adjusted or 0)
        target = math.ceil(target / block) * block
        by_model[model] = {
            "pilot_families": len(family_means),
            "family_mean_difference_sd": sd,
            "families_from_variance": raw,
            "families_after_attrition": attrition_adjusted,
            "recommended_families": target,
            "recommended_occupations": math.ceil(target / block),
        }

    report = {
        "interpretation": (
            "Variance-only planning values. A model's recommendation is actionable "
            "only if the pilot diagnostics authorise that model."
        ),
        "minimum_meaningful_score_difference": delta,
        "two_sided_alpha": alpha,
        "power": power,
        "family_attrition_fraction": attrition,
        "minimum_families": floor,
        "models": by_model,
        "recommended_families_across_models": max(
            (entry["recommended_families"] for entry in by_model.values()), default=None
        ),
    }
    if authorised_models is not None:
        retained = sorted(authorised_models & set(by_model))
        report["authorised_models"] = retained
        report["recommended_families_across_authorised_models"] = max(
            (by_model[model]["recommended_families"] for model in retained),
            default=None,
        )
    return report
