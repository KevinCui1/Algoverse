"""Prompt rendering and counterfactual integrity.

Every prompt is rendered from structured fields, never edited by hand. Within a
counterfactual set - the five identity conditions of one scenario family at one
prompt form, one prestige level and one context level - the rendered text must
differ in the identity block and nowhere else. That is enforced two ways:

    exact hash        SHA-256 of the rendered prompt, expected to differ
    normalised hash   identity block replaced by a fixed token, expected to
                      match across all five conditions

If the normalised hashes of a set do not all match, something other than
identity varies between conditions and the comparison is not a counterfactual.
The check runs before inference, because after inference the money is spent.

Soft-criterion order is randomised once per family and then held fixed across
conditions and prestige levels. Randomising per family prevents a stable
presentation order from implying a ranking; holding it fixed within the family
keeps the evaluative ambiguity identical inside each counterfactual pair, so it
cancels in the paired difference instead of inflating it.

Two prompt forms are rendered from the same structured family. `gated` states
the hard requirements as a numbered checklist and supplies the eligibility rule.
`holistic` supplies neither: the requirement block and the eligibility sentence
are absent, and the applicant's own gate facts appear inside the applicant
evidence as ordinary prose in the register the rest of the evidence uses. The
model is then asked to form a judgement rather than to execute a rule.

The prose is derived from the structured gate fields by a fixed transformation
rather than authored per family. Hand-writing twenty-four families twice would
let a register difference between the forms - sentence length, vocabulary,
enthusiasm - be read as an effect of removing the rule, which is exactly the
quantity the two forms are there to contrast.

Only the applicant's value crosses into `holistic`; the threshold does not. A
prompt that still states the bar has not removed the rule, only its formatting.
The gold decision is unaffected either way, because it is computed from the
structured gate fields and never reads prompt text.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any

from . import context as context_factor
from . import config, gates, paths, scenarios, stimuli

GATED = "gated"
HOLISTIC = "holistic"

IDENTITY_TOKEN = "{{IDENTITY}}"
PLACEHOLDER_PATTERN = re.compile(r"\{\{[A-Z_]+\}\}")


class RenderError(ValueError):
    """Raised when a prompt cannot be rendered or fails an integrity check."""


@dataclass(frozen=True)
class RenderedPrompt:
    variant_id: str
    family_id: str
    occupation_slug: str
    margin_band: str
    condition: str
    cue_mode: str
    identity_group: str | None
    prestige_level: str
    context_level: str
    prompt_form: str
    name_stimulus_id: str | None
    applicant_name: str | None
    system_prompt: str
    user_prompt: str
    identity_block: str
    exact_hash: str
    normalised_hash: str
    gold_decision: str
    minimum_gate_margin: float | None
    failed_gates: int
    ambiguity_score: int
    soft_variant: str


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _template(name: str) -> str:
    path = paths.PROMPTS / name
    if not path.exists():
        raise RenderError(f"missing prompt template: {path}")
    return path.read_text()


def forms() -> tuple[str, ...]:
    return tuple(config.study()["prompt_form"]["levels"])


def _form_template(form: str) -> str:
    templates = config.study()["prompt_form"]["templates"]
    if form not in templates:
        raise RenderError(
            f"unknown prompt form {form!r}; declared: {sorted(templates)}"
        )
    return _template(templates[form])


def gate_requirement_text(gate: dict[str, Any]) -> str:
    """Requirement wording that states its own threshold.

    Some source records phrase a numeric requirement without the number
    ("Experience as a Chef or Head Cook" for a two-year minimum). A reader of
    the prompt then cannot tell whether the applicant's reported value clears
    the bar, which removes the property the whole design rests on: that the
    correct decision is determinable from the prompt without judgement. An
    undeterminable gate is also the place a cue can move a decision while
    looking like bias, so the threshold is restored from the structured value
    whenever the wording omits it.
    """
    requirement = str(gate["requirement"]).strip().rstrip(".")
    operator = gate["operator"]
    required = gate["required_value"]

    if operator in ("<", "<=", ">", ">="):
        if re.search(rf"\b{re.escape(str(required))}\b", requirement):
            return requirement + "."
        unit = gate.get("unit") or "units"
        if float(required) == 1 and unit.endswith("s"):
            unit = unit[:-1]
        comparator = {
            ">=": "at least",
            ">": "more than",
            "<=": "at most",
            "<": "fewer than",
        }[operator]
        return f"{requirement} — {comparator} {required} {unit}."

    if requirement.casefold().find(str(required).casefold()) >= 0:
        return requirement + "."
    return f"{requirement} — required value: {required}."


def _numbered_gates(family: scenarios.ScenarioFamily) -> str:
    return "\n".join(
        f"{index}. {gate_requirement_text(gate)}"
        for index, gate in enumerate(family.hard_gates, start=1)
    )


def _soft_criteria_block(
    family: scenarios.ScenarioFamily, order: list[str]
) -> str:
    return "\n".join(
        f"- {family.soft_criterion(criterion_id)['criterion']}" for criterion_id in order
    )


# Spelled out to the largest candidate value the scenario set holds, so a gate
# fact reads as prose rather than as a figure lifted out of a form.
_NUMBER_WORDS = (
    "No",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Eleven",
    "Twelve",
)

# Comparator wording a requirement may carry, optionally followed by the
# threshold itself. Stripping it is what keeps the bar out of the holistic form:
# a prompt that still says "at least 3 years" has reformatted the rule, not
# removed it.
_COMPARATOR_PREFIX = re.compile(
    r"^(?:at least|at most|minimum(?:\s+of)?|maximum(?:\s+of)?|no fewer than"
    r"|no more than|fewer than|more than)\s+"
    r"(?:\d+(?:\.\d+)?\s+\S+\s+of\s+)?",
    re.IGNORECASE,
)


def _lower_initial(text: str) -> str:
    """Lowercase a leading capital that is only there because it began a sentence.

    A first token that is entirely uppercase is an acronym and is left alone.
    """
    head = text.split(" ", 1)[0]
    if head.isupper():
        return text
    return text[:1].lower() + text[1:]


def _gate_subject(gate: dict[str, Any]) -> str:
    """The requirement stripped of its threshold, as a noun phrase.

    What remains names the thing being evidenced - "experience as a Chef or Head
    Cook" - without stating how much of it the role demands.
    """
    subject = str(gate["requirement"]).strip().rstrip(".")
    if gate["operator"] in gates.NUMERIC_OPERATORS:
        subject = _COMPARATOR_PREFIX.sub("", subject, count=1)
    else:
        required = str(gate["required_value"]).strip()
        subject = re.sub(
            rf"^{re.escape(required)}\s+", "", subject, count=1, flags=re.IGNORECASE
        )
    subject = subject.strip()
    if not subject:
        raise RenderError(
            f"gate {gate['gate_id']}: nothing remains of the requirement once its "
            "threshold is removed, so it cannot be stated as applicant evidence"
        )
    return _lower_initial(subject)


def gate_evidence_text(gate: dict[str, Any], candidate_value: Any) -> str:
    """One applicant gate fact as prose, carrying the value but not the bar.

    Derived from the structured fields so that both prompt forms describe the
    same applicant. Nothing here consults the gold decision; a categorical gate
    is rendered from the comparison its own operator defines, which is the same
    fact the checklist form states as a reported value.
    """
    subject = _gate_subject(gate)

    if gate["operator"] in gates.NUMERIC_OPERATORS:
        try:
            amount = float(candidate_value)
        except (TypeError, ValueError) as exc:
            raise RenderError(
                f"gate {gate['gate_id']}: numeric operator with non-numeric "
                f"candidate value {candidate_value!r}"
            ) from exc
        if amount == 0:
            return f"No {subject}."
        unit = gate.get("unit") or "units"
        if amount == 1 and unit.endswith("s"):
            unit = unit[:-1]
        quantity = (
            _NUMBER_WORDS[int(amount)]
            if amount == int(amount) and 0 <= int(amount) < len(_NUMBER_WORDS)
            else str(candidate_value)
        )
        return f"{quantity} {unit} of {subject}."

    satisfied = gates.evaluate_gate(gate, candidate_value).passed
    prefix = "Evidence of" if satisfied else "No evidence of"
    return f"{prefix} {subject} is on record."


def _applicant_evidence(
    family: scenarios.ScenarioFamily,
    order: list[str],
    prestige: stimuli.Prestige,
    soft_profile: list[dict[str, Any]],
    form: str,
) -> str:
    """The applicant's facts, laid out as the prompt form requires.

    `gated` keeps the gate facts under their own heading, each restating the
    requirement it answers, because that form's job is to make the rule
    executable. `holistic` merges them into one undifferentiated list in the
    evidence register, because a separate heading would re-flag them as the gate
    layer and reinstate the two-tier reading the form exists to remove.
    """
    profile = {entry["criterion_id"]: entry for entry in soft_profile}
    gate_lookup = {gate["gate_id"]: gate for gate in family.hard_gates}
    soft_lines = [f"- {profile[criterion_id]['candidate_evidence']}" for criterion_id in order]

    if form == HOLISTIC:
        gate_lines = [
            f"- {gate_evidence_text(gate_lookup[entry['gate_id']], entry['candidate_value'])}"
            for entry in family.candidate_gate_values
        ]
        return "\n".join([prestige.text, "", *gate_lines, *soft_lines])

    lines = [prestige.text, "", "Reported qualifications:"]
    for entry in family.candidate_gate_values:
        gate = gate_lookup[entry["gate_id"]]
        unit = f" {gate['unit']}" if gate.get("unit") else ""
        lines.append(
            f"- {gate_requirement_text(gate)} "
            f"Applicant reports: {entry['candidate_value']}{unit}."
        )
    return "\n".join([*lines, "", "Other job-related evidence:", *soft_lines])


def soft_order(family: scenarios.ScenarioFamily, seed: int) -> list[str]:
    """Presentation order of the soft criteria, fixed for the whole family."""
    order = [entry["criterion_id"] for entry in family.soft_profile]
    rng = random.Random(f"{seed}:{family.family_id}")
    rng.shuffle(order)
    return order


def render_variant(
    family: scenarios.ScenarioFamily,
    condition: str,
    prestige: stimuli.Prestige,
    name: stimuli.Name | None,
    order_seed: int,
    context: context_factor.Context,
    prompt_form: str = GATED,
    soft_profile: list[dict[str, Any]] | None = None,
    soft_variant: str = "base",
) -> RenderedPrompt:
    """Render one identity condition at one cell of the design."""
    if prompt_form not in forms():
        raise RenderError(
            f"unknown prompt form {prompt_form!r}; declared: {list(forms())}"
        )
    profile = soft_profile if soft_profile is not None else family.soft_profile
    order = soft_order(family, order_seed)
    if {entry["criterion_id"] for entry in profile} != set(order):
        raise RenderError(
            f"{family.family_id}: soft profile criterion ids do not match the family"
        )

    block = stimuli.identity_block(condition, name)
    system_prompt = _template("decision_system_v1.txt").strip()
    user_template = _form_template(prompt_form)

    fields = {
        "IDENTITY_BLOCK": block,
        "CONTEXT_BLOCK": context.text,
        "OCCUPATION_TITLE": family.occupation,
        "JOB_SUMMARY": family.job_summary,
        "NUMBERED_HARD_GATES": _numbered_gates(family),
        "RANDOMIZED_SOFT_CRITERIA": _soft_criteria_block(family, order),
        "CANDIDATE_FACTS_WITH_PRESTIGE_STIMULUS": _applicant_evidence(
            family, order, prestige, profile, prompt_form
        ),
    }

    user_prompt = user_template
    for key, value in fields.items():
        user_prompt = user_prompt.replace("{{" + key + "}}", value)

    leftover = PLACEHOLDER_PATTERN.findall(user_prompt)
    if leftover:
        raise RenderError(f"{family.family_id}: unfilled placeholders {sorted(set(leftover))}")

    if user_prompt.count(block) != 1:
        raise RenderError(
            f"{family.family_id}: identity block appears "
            f"{user_prompt.count(block)} times in the rendered prompt, expected once"
        )
    normalised = user_prompt.replace(block, IDENTITY_TOKEN, 1)

    stimulus = f"__{name.stimulus_id}" if name else ""
    suffix = "" if soft_variant == "base" else f"__{soft_variant}"
    variant_id = (
        f"{family.family_id}__{prompt_form}__{context.level}__{prestige.level}__"
        f"{condition}{stimulus}{suffix}"
    )

    return RenderedPrompt(
        variant_id=variant_id,
        family_id=family.family_id,
        occupation_slug=family.occupation_slug,
        margin_band=family.margin_band,
        condition=condition,
        cue_mode=stimuli.CUE_MODE_OF_CONDITION[condition],
        identity_group=stimuli.GROUP_OF_CONDITION.get(condition),
        prestige_level=prestige.level,
        context_level=context.level,
        prompt_form=prompt_form,
        name_stimulus_id=name.stimulus_id if name else None,
        applicant_name=name.full_name if name else None,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        identity_block=block,
        exact_hash=_sha256(system_prompt + "\n\n" + user_prompt),
        normalised_hash=_sha256(system_prompt + "\n\n" + normalised),
        gold_decision=family.gold_decision,
        minimum_gate_margin=family.minimum_gate_margin,
        failed_gates=family.failed_gates,
        ambiguity_score=family.ambiguity_score,
        soft_variant=soft_variant,
    )


def check_counterfactual_integrity(variants: list[RenderedPrompt]) -> None:
    """Every counterfactual set must share one normalised hash and no exact hash.

    Also screens the neutral condition for residual identity signal: a leftover
    name or race term there would contaminate the baseline that every cue
    effect is measured against.
    """
    grouped: dict[tuple[str, str, str, str, str], list[RenderedPrompt]] = {}
    for variant in variants:
        key = (
            variant.family_id,
            variant.prompt_form,
            variant.context_level,
            variant.prestige_level,
            variant.soft_variant,
        )
        grouped.setdefault(key, []).append(variant)

    for key, group in grouped.items():
        normalised = {variant.normalised_hash for variant in group}
        if len(normalised) != 1:
            differing = sorted(variant.condition for variant in group)
            raise RenderError(
                f"{key}: conditions {differing} differ outside the identity block "
                f"({len(normalised)} distinct normalised hashes)"
            )
        exact = {variant.exact_hash for variant in group}
        if len(exact) != len(group):
            raise RenderError(f"{key}: two conditions rendered byte-identical prompts")

    _screen_neutral(variants)


def _screen_neutral(variants: list[RenderedPrompt]) -> None:
    race_terms = ("white", "black", "african american", "caucasian", "race", "ethnicity")
    names = {
        variant.applicant_name
        for variant in variants
        if variant.applicant_name is not None
    }
    for variant in variants:
        if variant.condition != stimuli.NEUTRAL:
            continue
        lowered = variant.user_prompt.casefold()
        for term in race_terms:
            if term in lowered:
                raise RenderError(
                    f"{variant.variant_id}: neutral prompt contains the term {term!r}"
                )
        for name in names:
            if name.casefold() in lowered:
                raise RenderError(
                    f"{variant.variant_id}: neutral prompt contains the name {name!r}"
                )

    for variant in variants:
        if variant.cue_mode != "concealed":
            continue
        lowered = variant.user_prompt.casefold()
        if "self-identifies as" in lowered:
            raise RenderError(
                f"{variant.variant_id}: concealed prompt carries a direct identity statement"
            )
