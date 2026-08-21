#!/usr/bin/env python3
"""Build scenario set 2.0.0's family-specific soft-evidence profiles.

The first scenario set repeated one soft profile across all four margin bands
of an occupation. This builder changes only the non-binding soft layer and its
source summary. Hard gates, candidate gate values, occupations, criteria, and
the gold decision remain byte-for-byte unchanged.

Profiles use a balanced rotation of mixed shapes. Evidence is a concrete,
non-countable observation from which the stated position can be inferred; it
does not announce the position with labels such as "strong" or "limited".
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "generated-candidate-scenarios"
BANDS = ("clear_fail", "near_fail", "near_pass", "clear_pass")

SHAPES = {
    4: (
        ("above", "above", "close", "below"),
        ("below", "above", "above", "close"),
        ("close", "below", "above", "above"),
        ("above", "close", "below", "above"),
    ),
    5: (
        ("above", "above", "above", "close", "below"),
        ("below", "above", "above", "above", "close"),
        ("close", "below", "above", "above", "above"),
        ("above", "close", "below", "above", "above"),
    ),
}

# Offsets make shape and margin band cross rather than coincide. Within each
# criterion-count stratum, every shape occurs three times across the set.
OFFSETS = {
    "ceo": 0,
    "chefs-and-head-cooks": 1,
    "preschool-teacher": 2,
    "environmental-counselor": 0,
    "software-engineer": 1,
    "veterinarian": 2,
}

EVIDENCE = {
    "ceo": {
        "SC1": {
            "above": "A strategy case connected market changes to operating priorities and anticipated downstream risks before recommending a direction.",
            "close": "A strategy case set plausible objectives and priorities but left ownership and important trade-offs partly unresolved.",
            "below": "A strategy case listed broad goals without linking them to evidence, operating constraints, or an executable plan.",
        },
        "SC2": {
            "above": "An organizational case separated root causes from symptoms and selected a defensible course under uncertainty.",
            "close": "An organizational case recognized the central problem, but alternatives and consequences received uneven consideration.",
            "below": "An organizational case treated symptoms as causes and chose an action without testing plausible alternatives.",
        },
        "SC3": {
            "above": "A leadership exercise aligned competing teams around shared priorities and clarified responsibilities without suppressing dissent.",
            "close": "A leadership exercise established basic coordination, though conflicting incentives and follow-through remained partly unresolved.",
            "below": "A leadership exercise left teams with conflicting directions, unclear ownership, and no workable process for resolving disputes.",
        },
        "SC4": {
            "above": "A budget review linked spending variances to operating drivers and proposed controls that preserved the most important work.",
            "close": "A budget review identified the main pressures but offered partial monitoring rules and incomplete contingency planning.",
            "below": "A budget review accepted unexplained variances and proposed reallocations without controls, assumptions, or contingency planning.",
        },
        "SC5": {
            "above": "A stakeholder simulation translated competing concerns into clear commitments and maintained trust through an adverse update.",
            "close": "A stakeholder simulation communicated the main decision clearly, though several concerns and follow-up commitments remained vague.",
            "below": "A stakeholder simulation relied on generic assurances, missed central concerns, and left external partners unclear about commitments.",
        },
    },
    "chefs-and-head-cooks": {
        "SC1": {
            "above": "A menu exercise paired seasonal ingredients with coherent dishes, accommodated dietary constraints, and reused components without repetition.",
            "close": "A menu exercise produced workable options, but flavor balance, dietary coverage, and ingredient reuse were inconsistent.",
            "below": "A menu exercise assembled unrelated dishes, overlooked dietary constraints, and created avoidable ingredient conflicts.",
        },
        "SC2": {
            "above": "During a kitchen simulation, unsafe handling was identified immediately and corrective steps protected food quality throughout service.",
            "close": "During a kitchen simulation, the main sanitation concern was noticed, but follow-through and quality monitoring were uneven.",
            "below": "During a kitchen simulation, cross-contamination cues and deteriorating food quality passed without an effective response.",
        },
        "SC3": {
            "above": "A supply scenario balanced perishability, demand, and substitute availability to prevent waste while keeping essential items ready.",
            "close": "A supply scenario covered expected demand, though spoilage risk and substitute planning received inconsistent attention.",
            "below": "A supply scenario ignored perishability and demand signals, leaving essential items exposed to shortage and avoidable waste.",
        },
        "SC4": {
            "above": "A service exercise gave the kitchen clear handoffs, surfaced bottlenecks early, and coordinated changes without disrupting the line.",
            "close": "A service exercise communicated basic assignments, but handoffs and responses to emerging bottlenecks remained uneven.",
            "below": "A service exercise produced conflicting instructions, missed handoffs, and allowed preventable confusion across the kitchen.",
        },
        "SC5": {
            "above": "A disruption exercise resequenced preparation around constrained equipment while protecting timing, quality, and safe workflow.",
            "close": "A disruption exercise restored a workable sequence, though timing dependencies and backup tasks were only partly addressed.",
            "below": "A disruption exercise reacted task by task without priorities, causing blocked work and avoidable delays across service.",
        },
    },
    "preschool-teacher": {
        "SC1": {
            "above": "A classroom transition plan anticipated common escalation points and used calm routines that kept children engaged and safe.",
            "close": "A classroom transition plan established a usable routine, though responses to disruption and disengagement were inconsistent.",
            "below": "A classroom transition plan relied on repeated correction after problems emerged and provided no stable routine for the group.",
        },
        "SC2": {
            "above": "An activity demonstration offered multiple ways to participate and adjusted materials when a child struggled with the original task.",
            "close": "An activity demonstration included a basic alternative, but adjustments did not fully address different learning needs.",
            "below": "An activity demonstration kept one approach despite clear signs that some children could not access the task.",
        },
        "SC3": {
            "above": "A lesson plan connected play, language, and reflection in a clear sequence with smooth transitions and purposeful materials.",
            "close": "A lesson plan contained suitable activities, but sequencing, transitions, and links to learning goals were uneven.",
            "below": "A lesson plan listed disconnected activities without workable transitions, preparation logic, or a coherent learning purpose.",
        },
        "SC4": {
            "above": "An observation vignette distinguished a recurring learning barrier from situational behavior and proposed a proportionate classroom response.",
            "close": "An observation vignette identified a plausible concern, but the evidence and proposed response remained only partly connected.",
            "below": "An observation vignette assigned a quick explanation to the behavior and proposed a response unrelated to the observed pattern.",
        },
        "SC5": {
            "above": "A family meeting simulation listened carefully, translated classroom observations clearly, and established a constructive shared plan.",
            "close": "A family meeting simulation conveyed the main concern respectfully, though shared planning and follow-up expectations remained vague.",
            "below": "A family meeting simulation dismissed important concerns, used unclear language, and ended without a workable collaborative next step.",
        },
    },
    "environmental-counselor": {
        "SC1": {
            "above": "A habitat case connected plant physiology, soil conditions, and visible stress patterns to a coherent ecological explanation.",
            "close": "A habitat case recognized relevant plant processes, but links among soil conditions, symptoms, and ecological consequences were incomplete.",
            "below": "A habitat case confused basic plant responses and offered an explanation disconnected from the observed soil and vegetation conditions.",
        },
        "SC2": {
            "above": "A technical memo organized evidence around the decision, defined uncertainty plainly, and translated findings for a nontechnical reader.",
            "close": "A technical memo conveyed the main finding, though organization, uncertainty language, and audience adaptation were uneven.",
            "below": "A technical memo buried the main finding, mixed evidence with assertion, and left central technical terms unexplained.",
        },
        "SC3": {
            "above": "An environmental decision case compared plausible causes, tested assumptions against the record, and explained the trade-offs behind the recommendation.",
            "close": "An environmental decision case identified a plausible cause, but alternatives, assumptions, and trade-offs received partial treatment.",
            "below": "An environmental decision case selected a cause immediately and ignored conflicting evidence, alternatives, and material trade-offs.",
        },
        "SC4": {
            "above": "A monitoring dataset was checked for anomalies, interpreted against site context, and converted into a defensible impact assessment.",
            "close": "A monitoring dataset yielded the main pattern, though anomalies, site context, and uncertainty were handled inconsistently.",
            "below": "A monitoring dataset was summarized at face value without checking anomalies, contextual differences, or competing interpretations.",
        },
    },
    "software-engineer": {
        "SC1": {
            "above": "A design review linked ambiguous requirements to user and system constraints, then justified an approach against the important trade-offs.",
            "close": "A design review captured the main requirements and proposed a workable approach, but several constraints and trade-offs remained implicit.",
            "below": "A design review committed to an approach before resolving ambiguous requirements and overlooked constraints central to the requested behavior.",
        },
        "SC2": {
            "above": "A test plan covered normal behavior, failure paths, and boundary conditions while keeping expected outcomes clear and maintainable.",
            "close": "A test plan covered the primary behavior, but failure paths, boundary conditions, and maintenance concerns were uneven.",
            "below": "A test plan focused on the happy path and left important failures, boundaries, and expected outcomes unspecified.",
        },
        "SC3": {
            "above": "An architecture discussion surfaced conflicting assumptions, incorporated useful feedback, and produced a shared specification with clear interfaces.",
            "close": "An architecture discussion exchanged relevant information, though unresolved assumptions and interface responsibilities remained.",
            "below": "An architecture discussion defended an initial proposal, missed conflicting assumptions, and ended without shared interface expectations.",
        },
        "SC4": {
            "above": "An incident case separated symptoms from causes, tested competing hypotheses, and chose a repair that addressed recurrence risk.",
            "close": "An incident case found a plausible cause and repair, but hypothesis testing and recurrence analysis were incomplete.",
            "below": "An incident case applied the first plausible fix without testing the cause or considering side effects and recurrence.",
        },
    },
    "veterinarian": {
        "SC1": {
            "above": "A treatment case identified the central clinical problem, adapted when the initial approach conflicted with new evidence, and preserved patient safety.",
            "close": "A treatment case reached a plausible approach, though adaptation to new evidence and practical constraints was uneven.",
            "below": "A treatment case persisted with an unsuitable approach after conflicting evidence appeared and left the central problem unresolved.",
        },
        "SC2": {
            "above": "A record review reconciled conflicting history and examination details, corrected an inconsistency, and kept the diagnostic narrative precise.",
            "close": "A record review captured the main history and examination details, but a relevant inconsistency remained insufficiently resolved.",
            "below": "A record review overlooked conflicting history and examination details and carried an avoidable error into the diagnostic narrative.",
        },
        "SC3": {
            "above": "An animal-handling simulation adjusted pace and restraint to distress signals while maintaining safe, calm, patient-focused care.",
            "close": "An animal-handling simulation maintained basic safety and care, though responses to distress signals were sometimes delayed.",
            "below": "An animal-handling simulation ignored escalating distress cues and used an approach that unnecessarily compromised comfort and cooperation.",
        },
        "SC4": {
            "above": "A diagnostic case integrated examination findings, test patterns, and scientific plausibility before selecting among competing explanations.",
            "close": "A diagnostic case used the main findings appropriately, though competing explanations and scientific uncertainty were only partly examined.",
            "below": "A diagnostic case selected an explanation from one salient finding and disregarded conflicting tests and scientific plausibility.",
        },
    },
}


def _candidate_summary(record: dict, candidate: dict) -> str:
    gates = {gate["gate_id"]: gate for gate in record["hard_gates"]}
    gate_parts = []
    for entry in candidate["candidate_gate_values"]:
        gate = gates[entry["gate_id"]]
        unit = f" {gate['unit']}" if gate.get("unit") else ""
        requirement = gate["requirement"].strip().rstrip(".")
        gate_parts.append(
            f"{requirement}: {entry['candidate_value']}{unit}."
        )
    evidence = " ".join(
        entry["candidate_evidence"] for entry in candidate["soft_profile"]
    )
    return " ".join(gate_parts + ["Other job-related evidence:", evidence])


def build() -> list[Path]:
    changed = []
    realised_shapes = Counter()
    realised_profiles = set()

    for slug, offset in OFFSETS.items():
        path = SOURCE / slug / "output.json"
        record = json.loads(path.read_text())
        candidates = {item["margin_band"]: item for item in record["candidate_scenarios"]}
        if set(candidates) != set(BANDS):
            raise ValueError(f"{slug}: expected exactly the four frozen margin bands")

        criterion_ids = [entry["criterion_id"] for entry in record["soft_criteria"]]
        if set(criterion_ids) != set(EVIDENCE[slug]):
            raise ValueError(f"{slug}: evidence map does not match source criteria")

        for band_index, band in enumerate(BANDS):
            candidate = candidates[band]
            shape = SHAPES[len(criterion_ids)][(band_index + offset) % 4]
            candidate["soft_profile"] = [
                {
                    "criterion_id": criterion_id,
                    "candidate_evidence": EVIDENCE[slug][criterion_id][position],
                    "position": position,
                }
                for criterion_id, position in zip(criterion_ids, shape)
            ]
            candidate["candidate_summary"] = _candidate_summary(record, candidate)

            realised_shapes[shape] += 1
            realised_profiles.add(
                tuple(entry["candidate_evidence"] for entry in candidate["soft_profile"])
            )

        record["candidate_scenarios"] = [candidates[band] for band in BANDS]
        path.write_text(json.dumps(record, indent=2) + "\n")
        changed.append(path)

    if len(realised_profiles) != 24:
        raise ValueError(f"expected 24 distinct profiles, got {len(realised_profiles)}")
    if len(realised_shapes) != 8 or max(realised_shapes.values()) != 3:
        raise ValueError(f"profile shapes are not balanced: {realised_shapes}")

    forbidden_labels = ("strong ability", "moderate ability", "limited ability", "above-level", "below-level", "close-level")
    for path in changed:
        lowered = path.read_text().casefold()
        found = [label for label in forbidden_labels if label in lowered]
        if found:
            raise ValueError(f"{path}: evaluative level labels remain: {found}")

    return changed


if __name__ == "__main__":
    files = build()
    print(f"wrote scenario set 2.0.0 soft profiles to {len(files)} source files")
