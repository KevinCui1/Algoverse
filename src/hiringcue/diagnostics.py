"""Pilot diagnostics on the suitability score.

The score carries the study's statistical power, because a binary decision flip
discards every movement that does not cross the threshold and published
demographic flip rates are small enough that a binary-only design would need far
more observations than is available here.

That only holds if the score has room to move that the qualification rule does
not already fix. A score that is a deterministic restatement of the gate
arithmetic still produces a wide, many-valued histogram, because the arithmetic
itself varies across scenarios - so no distributional statistic can detect the
failure. Determinacy is a property of the score's variance *conditional on* the
rule, and it takes a positive control to see it:

    D1  variance explained by gate status and margin
    D2  response to a soft-criteria change that touches no gate  <- the control
    D3  dispersion within the gate-passing class, within band
    D4  variation across identical repeated prompts

D2 is the one that cannot be faked. A model that will not move its score for a
substantive change to the criteria the score is supposed to reflect will not
move it for a name either, and no sample size repairs that, because the problem
is in the numerator.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

from . import config

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"


def _ols_r2(response: list[float], design: list[list[float]]) -> float:
    """R-squared from a small ordinary least-squares fit by normal equations."""
    n = len(response)
    if n < 3:
        return float("nan")
    columns = len(design[0])
    xtx = [[sum(design[i][a] * design[i][b] for i in range(n)) for b in range(columns)]
           for a in range(columns)]
    xty = [sum(design[i][a] * response[i] for i in range(n)) for a in range(columns)]

    # Gaussian elimination with partial pivoting and a ridge floor for stability.
    for index in range(columns):
        xtx[index][index] += 1e-8
    matrix = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for col in range(columns):
        pivot = max(range(col, columns), key=lambda r: abs(matrix[r][col]))
        if abs(matrix[pivot][col]) < 1e-12:
            return float("nan")
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        for row in range(columns):
            if row == col:
                continue
            factor = matrix[row][col] / matrix[col][col]
            for k in range(col, columns + 1):
                matrix[row][k] -= factor * matrix[col][k]
    beta = [matrix[i][columns] / matrix[i][i] for i in range(columns)]

    mean = statistics.fmean(response)
    total = sum((value - mean) ** 2 for value in response)
    residual = sum(
        (response[i] - sum(beta[a] * design[i][a] for a in range(columns))) ** 2
        for i in range(n)
    )
    return 1.0 - residual / total if total > 0 else float("nan")


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _band(value: float, spec: dict[str, Any]) -> str:
    if "fail_above" in spec:
        if value > spec["fail_above"]:
            return FAIL
        return PASS if value <= spec["pass_at_most"] else WARN
    if "fail_below" in spec:
        if value < spec["fail_below"]:
            return FAIL
        return PASS if value >= spec["pass_at_least"] else WARN
    low, high = spec["pass_range"]
    if low <= value <= high:
        return PASS
    warn_low, warn_high = spec["warn_range"]
    if warn_low <= value <= warn_high:
        return WARN
    return FAIL


def d1_rule_determinacy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Score regressed on gate status, minimum gate margin, and their interaction."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("initial_valid")
            and row.get("soft_variant") != "twin"
            and row.get("minimum_gate_margin") is not None
        ):
            grouped[row["model_key"]].append(row)

    results = {}
    for model, model_rows in grouped.items():
        response = [float(row["suitability_score"]) for row in model_rows]
        design = []
        for row in model_rows:
            passed = 1.0 if row["gold_decision"] == "advance" else 0.0
            margin = float(row["minimum_gate_margin"])
            design.append([1.0, passed, margin, passed * margin])
        results[model] = _ols_r2(response, design)
    return results


def d2_free_parameter(rows: list[dict[str, Any]], detail: bool = False):
    """Score response to a soft-criteria change, net of the model's own noise.

    A twin comparison is between two cell means, so some of the observed
    difference is sampling variation rather than response to the changed
    criteria. A model that moves its score at random by a few points on repeated
    identical prompts would otherwise register a free-parameter response it does
    not have. The expected absolute difference of two means of `n` draws from a
    distribution with the model's own within-variant standard deviation is
    subtracted before the gate is applied.
    """
    base: dict[tuple, list[float]] = defaultdict(list)
    twin: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        if not row.get("initial_valid"):
            continue
        key = (row["model_key"], row["family_id"], row["prestige_level"], row["condition"])
        target = twin if row["soft_variant"] == "twin" else base
        target[key].append(float(row["suitability_score"]))

    noise = d4_run_to_run_sd(rows)
    observed: dict[str, list[float]] = defaultdict(list)
    runs: dict[str, list[int]] = defaultdict(list)
    for key, twin_scores in twin.items():
        base_scores = base.get(key)
        if not base_scores:
            continue
        observed[key[0]].append(
            abs(statistics.fmean(twin_scores) - statistics.fmean(base_scores))
        )
        runs[key[0]].append(min(len(base_scores), len(twin_scores)))

    report: dict[str, dict[str, float]] = {}
    for model, values in observed.items():
        if not values:
            continue
        n = statistics.median(runs[model]) or 1
        sigma = noise.get(model)
        # Half-normal mean: expected |difference of two independent cell means|.
        expected = (
            sigma * math.sqrt(2.0 / n) * math.sqrt(2.0 / math.pi)
            if sigma and sigma == sigma
            else 0.0
        )
        raw = statistics.median(values)
        report[model] = {
            "raw": raw,
            "noise_reference": expected,
            "adjusted": max(0.0, raw - expected),
        }

    if detail:
        return report
    return {model: entry["adjusted"] for model, entry in report.items()}


def d3_conditional_dispersion(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Median across margin bands of the score IQR within the gate-passing class."""
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row.get("initial_valid")
            and row.get("soft_variant") != "twin"
            and row["gold_decision"] == "advance"
        ):
            grouped[(row["model_key"], row["margin_band"])].append(
                float(row["suitability_score"])
            )

    by_model: dict[str, list[float]] = defaultdict(list)
    for (model, _band_name), values in grouped.items():
        if len(values) >= 4:
            by_model[model].append(_quantile(values, 0.75) - _quantile(values, 0.25))
    return {
        model: statistics.median(values) if values else float("nan")
        for model, values in by_model.items()
    }


def d4_run_to_run_sd(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Median score SD across independent runs of a byte-identical prompt."""
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("initial_valid"):
            grouped[(row["model_key"], row["variant_id"])].append(
                float(row["suitability_score"])
            )

    by_model: dict[str, list[float]] = defaultdict(list)
    for (model, _variant), values in grouped.items():
        if len(values) > 1:
            by_model[model].append(statistics.stdev(values))
    return {
        model: statistics.median(values) if values else float("nan")
        for model, values in by_model.items()
    }


def granularity(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Effective distinct score values and reliance on multiples of five."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row.get("initial_valid") and row.get("soft_variant") != "twin":
            grouped[row["model_key"]].append(int(row["suitability_score"]))

    results = {}
    for model, values in grouped.items():
        counts = Counter(values)
        total = sum(counts.values())
        entropy = -sum(
            (count / total) * math.log(count / total) for count in counts.values()
        )
        results[model] = {
            "effective_distinct_values": math.exp(entropy),
            "distinct_values": len(counts),
            "multiple_of_five_share": sum(1 for v in values if v % 5 == 0) / total,
        }
    return results


def evaluate(rows: list[dict[str, Any]], temperature: float) -> dict[str, Any]:
    """Run every gate and return a per-model verdict."""
    thresholds = config.gate_thresholds()
    determinacy = thresholds["determinacy"]
    gran_spec = thresholds["granularity"]
    escalation = thresholds["escalation"]

    measured = {
        "D1_rule_determinacy_r2": d1_rule_determinacy(rows),
        "D2_free_parameter_response": d2_free_parameter(rows),
        "D3_conditional_dispersion": d3_conditional_dispersion(rows),
        "D4_run_to_run_sd": d4_run_to_run_sd(rows),
    }
    gran = granularity(rows)
    models = sorted({row["model_key"] for row in rows})

    d2_detail = d2_free_parameter(rows, detail=True)
    report: dict[str, Any] = {"temperature": temperature, "models": {}}
    for model in models:
        outcomes: dict[str, dict[str, Any]] = {}
        for gate_name, values in measured.items():
            spec = determinacy[gate_name]
            value = values.get(model, float("nan"))
            if (
                gate_name == "D4_run_to_run_sd"
                and temperature == 0
                and spec.get("skip_if_temperature_zero")
            ):
                outcomes[gate_name] = {"value": value, "verdict": SKIP}
                continue
            if value != value:
                outcomes[gate_name] = {"value": None, "verdict": SKIP}
                continue
            entry = {"value": value, "verdict": _band(value, spec)}
            if gate_name == "D2_free_parameter_response" and model in d2_detail:
                entry |= {
                    "raw": d2_detail[model]["raw"],
                    "noise_reference": d2_detail[model]["noise_reference"],
                }
            outcomes[gate_name] = entry

        stats = gran.get(model, {})
        granularity_verdict = PASS
        if stats:
            if stats["effective_distinct_values"] < gran_spec["minimum_effective_distinct_values"]:
                granularity_verdict = FAIL
            elif stats["multiple_of_five_share"] > gran_spec["maximum_multiple_of_five_share"]:
                granularity_verdict = WARN

        warns = sum(1 for entry in outcomes.values() if entry["verdict"] == WARN)
        fails = sum(1 for entry in outcomes.values() if entry["verdict"] == FAIL)
        if fails or warns >= escalation["warns_equal_fail"]:
            verdict = FAIL
        elif warns:
            verdict = WARN
        else:
            verdict = PASS

        report["models"][model] = {
            "gates": outcomes,
            "granularity": stats | {"verdict": granularity_verdict},
            "verdict": verdict,
            "confirmatory_run_authorised": verdict != FAIL,
        }
    return report
