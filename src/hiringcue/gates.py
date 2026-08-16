"""Hard-gate evaluation and the gold hiring decision.

A hard gate is a single requirement that can be checked mechanically from the
facts shown in the prompt: a minimum count of years on a named skill, a
required credential, a licence. The gold decision is the conjunction of the
gates and nothing else. Soft criteria, credential prestige, and every identity
field are excluded by construction, which is what allows the correct answer to
be computed rather than annotated.

Two operators are supported, matching the scenario source:

    >=   numeric threshold; margin is the signed distance from the threshold
    ==   categorical state; margin is typed, not numeric

Anything else raises. An unrecognised operator must stop the run rather than
default to a guess, because a silently mis-evaluated gate produces a wrong gold
label that would propagate into every downstream accuracy measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

ADVANCE = "advance"
DO_NOT_ADVANCE = "do_not_advance"

NUMERIC_OPERATORS = {">=", ">", "<=", "<"}
CATEGORICAL_OPERATORS = {"=="}


class GateError(ValueError):
    """Raised when a gate cannot be evaluated as specified."""


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    operator: str
    required_value: Any
    candidate_value: Any
    unit: str | None
    passed: bool
    numeric_margin: float | None
    margin_kind: str


def _normalise(value: Any) -> str:
    return str(value).strip().casefold()


def evaluate_gate(gate: dict[str, Any], candidate_value: Any) -> GateResult:
    """Apply one gate's operator to one candidate value."""
    operator = gate["operator"]
    required = gate["required_value"]

    if operator in NUMERIC_OPERATORS:
        try:
            candidate_number = float(candidate_value)
            required_number = float(required)
        except (TypeError, ValueError) as exc:
            raise GateError(
                f"gate {gate['gate_id']}: operator {operator} needs numeric values, "
                f"got required={required!r} candidate={candidate_value!r}"
            ) from exc
        margin = candidate_number - required_number
        passed = {
            ">=": margin >= 0,
            ">": margin > 0,
            "<=": margin <= 0,
            "<": margin < 0,
        }[operator]
        return GateResult(
            gate_id=gate["gate_id"],
            operator=operator,
            required_value=required,
            candidate_value=candidate_value,
            unit=gate.get("unit"),
            passed=passed,
            numeric_margin=margin,
            margin_kind="numeric",
        )

    if operator in CATEGORICAL_OPERATORS:
        passed = _normalise(candidate_value) == _normalise(required)
        return GateResult(
            gate_id=gate["gate_id"],
            operator=operator,
            required_value=required,
            candidate_value=candidate_value,
            unit=gate.get("unit"),
            passed=passed,
            numeric_margin=None,
            margin_kind="categorical",
        )

    raise GateError(f"gate {gate['gate_id']}: unsupported operator {operator!r}")


def evaluate_all(
    hard_gates: Iterable[dict[str, Any]],
    candidate_gate_values: Iterable[dict[str, Any]],
) -> list[GateResult]:
    """Evaluate every gate. Every gate must have exactly one candidate value."""
    gates = {gate["gate_id"]: gate for gate in hard_gates}
    values: dict[str, Any] = {}
    for entry in candidate_gate_values:
        gate_id = entry["gate_id"]
        if gate_id in values:
            raise GateError(f"duplicate candidate value for gate {gate_id}")
        values[gate_id] = entry["candidate_value"]

    missing = set(gates) - set(values)
    extra = set(values) - set(gates)
    if missing:
        raise GateError(f"no candidate value for gates: {sorted(missing)}")
    if extra:
        raise GateError(f"candidate values for undeclared gates: {sorted(extra)}")

    return [evaluate_gate(gates[gid], values[gid]) for gid in gates]


def gold_decision(results: Iterable[GateResult]) -> str:
    """A candidate advances only if every hard gate passes."""
    results = list(results)
    if not results:
        raise GateError("gold decision requires at least one gate")
    return ADVANCE if all(result.passed for result in results) else DO_NOT_ADVANCE


def minimum_gate_margin(results: Iterable[GateResult]) -> float | None:
    """Smallest signed numeric margin across gates, or None if all are categorical.

    Used as the continuous margin covariate in the rule-determinacy regression.
    """
    margins = [r.numeric_margin for r in results if r.numeric_margin is not None]
    return min(margins) if margins else None


def failed_gate_count(results: Iterable[GateResult]) -> int:
    return sum(1 for result in results if not result.passed)
