"""Scenario families: loading, structural validation, and gold labelling.

A *scenario family* is one occupation paired with one qualification margin band.
It holds the facts that stay fixed while identity and credential prestige are
varied: the job summary, the hard gates, the soft criteria, the candidate's gate
values, and the candidate's soft profile.

The structural checks in `validate_family` are the emit-time contract for the
scenario set. They are run before any prompt is rendered, because a scenario
that violates them produces measurements that cannot be interpreted:

    S1  the gold decision is a pure function of the hard gates
    S2  the soft profile is mixed - at least one above and one below
    S3  soft criteria span at least three distinct O*NET dimension families
    S4  every hard gate has a declared minimum margin unit
    S5  near-threshold candidates sit exactly one minimum unit from the bar
    S6  no soft criterion appears in the gate set

S1 and S6 protect the gold label. S2 and S3 are what leave the suitability
score room to move for reasons the qualification rule does not already fix: a
uniformly strong or uniformly weak profile can be collapsed to a single scalar,
and criteria sharing one unit can be added up. S5 keeps the near bands where a
cue could plausibly flip a decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

from . import config, gates, paths

MARGIN_BANDS = ("clear_fail", "near_fail", "near_pass", "clear_pass")
NEAR_BANDS = ("near_fail", "near_pass")
MIN_DIMENSION_FAMILIES = 3


class ScenarioError(ValueError):
    """Raised when a scenario family violates the structural contract."""


@dataclass(frozen=True)
class ScenarioFamily:
    family_id: str
    occupation_slug: str
    occupation: str
    job_summary: str
    margin_band: str
    hard_gates: list[dict[str, Any]]
    soft_criteria: list[dict[str, Any]]
    candidate_gate_values: list[dict[str, Any]]
    soft_profile: list[dict[str, Any]]
    candidate_summary: str
    gate_results: list[gates.GateResult] = field(repr=False, default_factory=list)

    @property
    def gold_decision(self) -> str:
        return gates.gold_decision(self.gate_results)

    @property
    def minimum_gate_margin(self) -> float | None:
        return gates.minimum_gate_margin(self.gate_results)

    @property
    def failed_gates(self) -> int:
        return gates.failed_gate_count(self.gate_results)

    @property
    def ambiguity_score(self) -> int:
        """Count of soft criteria on which the candidate sits close to expectation.

        Used as the ambiguity floor check for near-threshold families, where the
        gate arithmetic is most conspicuous and the soft layer has to carry the
        score's free movement.
        """
        return sum(1 for entry in self.soft_profile if entry["position"] == "close")

    def soft_criterion(self, criterion_id: str) -> dict[str, Any]:
        for criterion in self.soft_criteria:
            if criterion["criterion_id"] == criterion_id:
                return criterion
        raise ScenarioError(f"{self.family_id}: unknown criterion {criterion_id}")


def _validate_structure(family: ScenarioFamily) -> None:
    positions = [entry["position"] for entry in family.soft_profile]
    if "above" not in positions or "below" not in positions:
        raise ScenarioError(
            f"{family.family_id}: soft profile is not mixed ({sorted(set(positions))}); "
            "a uniform profile is scalarisable and leaves the score no free parameter"
        )

    dimension_families = {
        family.soft_criterion(entry["criterion_id"])["onet_dimension_family"]
        for entry in family.soft_profile
    }
    if len(dimension_families) < MIN_DIMENSION_FAMILIES:
        raise ScenarioError(
            f"{family.family_id}: soft criteria span {len(dimension_families)} O*NET "
            f"dimension families, need {MIN_DIMENSION_FAMILIES}"
        )

    if not 2 <= len(family.hard_gates) <= 4:
        raise ScenarioError(
            f"{family.family_id}: {len(family.hard_gates)} hard gates, expected 2-4"
        )
    if not 3 <= len(family.soft_profile) <= 5:
        raise ScenarioError(
            f"{family.family_id}: {len(family.soft_profile)} soft criteria, expected 3-5"
        )

    for gate in family.hard_gates:
        if gate.get("minimum_margin_unit") in (None, ""):
            raise ScenarioError(
                f"{family.family_id}: gate {gate['gate_id']} has no minimum_margin_unit"
            )

    gate_ids = {gate["gate_id"] for gate in family.hard_gates}
    criterion_ids = {entry["criterion_id"] for entry in family.soft_profile}
    if gate_ids & criterion_ids:
        raise ScenarioError(
            f"{family.family_id}: identifiers shared between gates and soft criteria"
        )


def _validate_margin_band(family: ScenarioFamily) -> None:
    expected_advance = family.margin_band in ("near_pass", "clear_pass")
    actual_advance = family.gold_decision == gates.ADVANCE
    if expected_advance != actual_advance:
        raise ScenarioError(
            f"{family.family_id}: margin band {family.margin_band} implies "
            f"{'advance' if expected_advance else 'do_not_advance'} but the gates give "
            f"{family.gold_decision}"
        )

    if family.margin_band not in NEAR_BANDS:
        return

    numeric = [
        (result, gate)
        for result in family.gate_results
        for gate in family.hard_gates
        if gate["gate_id"] == result.gate_id and result.numeric_margin is not None
    ]
    if not numeric:
        return

    tightest = min(numeric, key=lambda pair: abs(pair[0].numeric_margin))
    result, gate = tightest
    unit = gate["minimum_margin_unit"]
    if not isinstance(unit, (int, float)):
        return
    if abs(result.numeric_margin) > abs(float(unit)):
        raise ScenarioError(
            f"{family.family_id}: near-threshold band but the tightest gate sits "
            f"{result.numeric_margin} from the bar, more than one unit ({unit})"
        )


def _validate_gold_independence(family: ScenarioFamily) -> None:
    """The gold decision must not move when the soft layer is permuted or dropped."""
    baseline = family.gold_decision
    permuted = list(reversed(family.soft_profile))
    probe = ScenarioFamily(
        family_id=family.family_id,
        occupation_slug=family.occupation_slug,
        occupation=family.occupation,
        job_summary=family.job_summary,
        margin_band=family.margin_band,
        hard_gates=family.hard_gates,
        soft_criteria=family.soft_criteria,
        candidate_gate_values=family.candidate_gate_values,
        soft_profile=permuted,
        candidate_summary=family.candidate_summary,
        gate_results=gates.evaluate_all(family.hard_gates, family.candidate_gate_values),
    )
    if probe.gold_decision != baseline:
        raise ScenarioError(f"{family.family_id}: gold decision depends on the soft layer")


def validate_family(family: ScenarioFamily) -> None:
    _validate_structure(family)
    _validate_margin_band(family)
    _validate_gold_independence(family)


def excluded_occupations() -> dict[str, str]:
    """Occupations held out of the scenario set, mapped to the recorded reason."""
    record = config.load("scenario_exclusions").get("excluded_occupations") or {}
    return {slug: entry["reason"] for slug, entry in record.items()}


def load_families(source_dir=None, apply_exclusions: bool = True) -> list[ScenarioFamily]:
    """Load every scenario family from the generated-scenario directory."""
    source_dir = source_dir or paths.SCENARIO_SOURCE
    excluded = excluded_occupations() if apply_exclusions else {}
    families: list[ScenarioFamily] = []
    for output_path in sorted(source_dir.glob("*/output.json")):
        slug = output_path.parent.name
        if slug in excluded:
            continue
        record = json.loads(output_path.read_text())
        for candidate in record["candidate_scenarios"]:
            band = candidate["margin_band"]
            if band not in MARGIN_BANDS:
                raise ScenarioError(f"{slug}: unknown margin band {band!r}")
            gate_results = gates.evaluate_all(
                record["hard_gates"], candidate["candidate_gate_values"]
            )
            families.append(
                ScenarioFamily(
                    family_id=f"{slug}__{band}",
                    occupation_slug=slug,
                    occupation=record["occupation"],
                    job_summary=record["job_summary"],
                    margin_band=band,
                    hard_gates=record["hard_gates"],
                    soft_criteria=record["soft_criteria"],
                    candidate_gate_values=candidate["candidate_gate_values"],
                    soft_profile=candidate["soft_profile"],
                    candidate_summary=candidate["candidate_summary"],
                    gate_results=gate_results,
                )
            )
    if not families:
        raise ScenarioError(f"no scenario families found under {source_dir}")
    return families


def validated_families(source_dir=None, apply_exclusions: bool = True) -> list[ScenarioFamily]:
    families = load_families(source_dir, apply_exclusions=apply_exclusions)
    for family in families:
        validate_family(family)
    _validate_ambiguity_floor(families)
    return families


def _validate_ambiguity_floor(families: list[ScenarioFamily]) -> None:
    """Near-threshold families must not also be thin on soft criteria.

    Near the bar the gate arithmetic is most visible, so a family that is both
    near-threshold and short of evaluative ambiguity is the worst case in the
    set: the score has nowhere to move that the rule has not already fixed.
    """
    scores = sorted(family.ambiguity_score for family in families)
    median = scores[len(scores) // 2]
    thin = [
        family.family_id
        for family in families
        if family.margin_band in NEAR_BANDS and family.ambiguity_score < median
    ]
    if thin:
        raise ScenarioError(
            "near-threshold families below the set median ambiguity score "
            f"({median}): {thin}"
        )


def iter_by_occupation(families: list[ScenarioFamily]) -> Iterator[tuple[str, list[ScenarioFamily]]]:
    grouped: dict[str, list[ScenarioFamily]] = {}
    for family in families:
        grouped.setdefault(family.occupation_slug, []).append(family)
    yield from grouped.items()
