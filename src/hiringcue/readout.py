"""The primary readout: a Yes/No token log-odds contrast from one forward pass.

The measured quantity for one prompt is

    logsumexp(admitted Yes-variant logits) - logsumexp(admitted No-variant logits)

taken in float32 from the full-vocabulary next-token logit vector at the
answer-position boundary of a single teacher-forced forward pass. There is no
decoding, no key-value cache and no serving engine anywhere in this path.

**Why the outcome is read rather than sampled.** A sampled binary decision
carries binomial variance a read logit does not; matching the precision of one
logit read takes tens of sampled repeats per cell. Reading also removes every
autoregressive step at which numerical divergence could accumulate.

**Why not a serving engine.** Continuous batching, prefix caching and chunked
prefill exist to make many-token generation efficient, and each introduces a
path by which a sequence's logits depend on what else is resident in the batch.
The hazard is specific to this design: counterfactual variants are byte-
identical outside the identity block, so under prefix caching one arm of a
matched pair is a cache miss and the other a hit, and the two arms are computed
along different numerical paths. That difference lies exactly on the contrast
being estimated.

**The answer-position boundary.** The boundary is the last token of the prompt
after that model's own chat template has been applied with a generation prompt
and the registered answer prefix appended. Padding is applied on the right and
the boundary is located per sequence from its unpadded length, rather than on
the left with a shared final index: a raw forward pass numbers positions from
the start of the tensor, so left padding would place every sequence at an offset
its template never produced.

**One tensor shape for the whole study.** Every sequence is padded to a fixed
length and every batch carries a fixed number of sequences, so the tensor
entering the forward pass has the same shape on every call. Padding instead to
the longest sequence in each batch - the ordinary default - makes that shape a
function of which prompts happen to be co-resident, and a matrix multiply's
reduction order follows the shape of the tensor it runs over, so in bounded
precision the low bits of one prompt's logits become a function of its
neighbours. That was measured on this instrument before the shape was fixed:
across five checkpoints every prompt whose reading moved under a re-batching
had had its padded length changed, and no prompt moved at an unchanged padded
length. Fixing the shape removes the mechanism instead of bounding it, and a
short final batch is filled to size rather than run narrow.

**Variant admission.** Yes and No surface forms are admitted only if appending
one to the templated prompt re-tokenises to the prompt's own token sequence plus
exactly one further token. Tokenising a surface form in isolation answers a
different question, because merge behaviour at a boundary depends on what
precedes it, and a variant that splits there would contribute the logit of a
word fragment rather than of an answer. Variant sets are enumerated per pinned
tokenizer and are never inferred from tokenizer family.

Inputs are a tokenizer, a model, and rendered prompts. Outputs are one contrast
per prompt with the diagnostic quantities that decide whether it can be read:
the implied Yes probability, the probability mass falling outside the admitted
variants, and the greedy next token. The module raises rather than returning a
contrast it cannot compute from an admitted variant set on both sides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from . import config


class ReadoutError(RuntimeError):
    """Raised when the answer boundary or the variant sets cannot be established."""


class Tokenizer(Protocol):
    """The tokenizer surface this module uses, stated so it can be substituted."""

    def apply_chat_template(
        self, conversation: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


@dataclass(frozen=True)
class VariantSet:
    """Admitted single-token answer variants for one model and one boundary."""

    yes_token_ids: tuple[int, ...]
    no_token_ids: tuple[int, ...]
    yes_surfaces: tuple[str, ...]
    no_surfaces: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "yes_token_ids": list(self.yes_token_ids),
            "no_token_ids": list(self.no_token_ids),
            "yes_surfaces": list(self.yes_surfaces),
            "no_surfaces": list(self.no_surfaces),
            "rejected": [list(entry) for entry in self.rejected],
        }


@dataclass(frozen=True)
class Reading:
    """One prompt's measurement and the quantities that qualify it."""

    prompt_id: str
    token_log_odds: float
    implied_yes_probability: float
    off_target_mass: float
    greedy_token_id: int
    greedy_is_admitted: bool
    boundary_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "token_log_odds": self.token_log_odds,
            "implied_yes_probability": self.implied_yes_probability,
            "off_target_mass": self.off_target_mass,
            "greedy_token_id": self.greedy_token_id,
            "greedy_is_admitted": self.greedy_is_admitted,
            "boundary_index": self.boundary_index,
        }


def answer_prefix() -> str:
    return str(config.models()["readout"]["answer_prefix"])


def templated_prompt(
    tokenizer: Tokenizer, system_prompt: str, user_prompt: str
) -> str:
    """Apply the model's own chat template and append the registered answer prefix.

    The instruction text is always folded into a single user turn rather than
    sent as a system turn. Several of the evaluated templates have no system
    role, and routing the same study text through a system turn on some models
    and a user turn on others would mean the checkpoints were not measured on
    the same stimulus.
    """
    conversation = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
    return (
        tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        + answer_prefix()
    )


def enumerate_variants(
    tokenizer: Tokenizer,
    boundary_prompt: str,
    yes_surfaces: Sequence[str] | None = None,
    no_surfaces: Sequence[str] | None = None,
) -> VariantSet:
    """Admit the surface forms that are a single token at this exact boundary."""
    settings = config.models()["readout"]
    yes_surfaces = list(yes_surfaces if yes_surfaces is not None else settings["yes_surfaces"])
    no_surfaces = list(no_surfaces if no_surfaces is not None else settings["no_surfaces"])

    prefix_ids = tokenizer.encode(boundary_prompt, add_special_tokens=False)
    admitted: dict[str, list[int]] = {"yes": [], "no": []}
    surfaces: dict[str, list[str]] = {"yes": [], "no": []}
    rejected: list[tuple[str, str]] = []

    for label, candidates in (("yes", yes_surfaces), ("no", no_surfaces)):
        for surface in candidates:
            extended = tokenizer.encode(
                boundary_prompt + surface, add_special_tokens=False
            )
            if extended[: len(prefix_ids)] != prefix_ids:
                rejected.append((surface, "re-tokenises the prompt at the boundary"))
                continue
            if len(extended) != len(prefix_ids) + 1:
                rejected.append(
                    (surface, f"spans {len(extended) - len(prefix_ids)} tokens at the boundary")
                )
                continue
            token_id = extended[-1]
            if token_id in admitted[label]:
                continue
            admitted[label].append(token_id)
            surfaces[label].append(surface)

    overlap = set(admitted["yes"]) & set(admitted["no"])
    if overlap:
        raise ReadoutError(
            f"token ids {sorted(overlap)} are admitted on both sides of the contrast; "
            "the two arms must be disjoint or the difference is not a contrast"
        )
    for label in ("yes", "no"):
        if not admitted[label]:
            raise ReadoutError(
                f"no {label.upper()} surface form is a single token at this boundary "
                f"(tried {yes_surfaces if label == 'yes' else no_surfaces}). This "
                "checkpoint cannot be measured on this readout and is dropped rather "
                "than measured on a fragment."
            )

    return VariantSet(
        yes_token_ids=tuple(admitted["yes"]),
        no_token_ids=tuple(admitted["no"]),
        yes_surfaces=tuple(surfaces["yes"]),
        no_surfaces=tuple(surfaces["no"]),
        rejected=tuple(rejected),
    )


def _logsumexp(values: Sequence[float]) -> float:
    largest = max(values)
    return largest + math.log(sum(math.exp(value - largest) for value in values))


def contrast(logits: Sequence[float], variants: VariantSet) -> tuple[float, float, float]:
    """Contrast, implied Yes probability, and off-target mass from one logit row.

    The logits are expected already in float32. Off-target mass is reported
    because the contrast conditions on the answer being one of the admitted
    variants: if most of the distribution sits elsewhere, the ratio is still
    well defined but describes a decision the model is not making, and a
    difference in that mass across identity arms would be an effect the contrast
    cannot see.
    """
    yes = _logsumexp([logits[token] for token in variants.yes_token_ids])
    no = _logsumexp([logits[token] for token in variants.no_token_ids])
    total = _logsumexp(list(logits))
    admitted = _logsumexp([yes, no])
    return yes - no, 1.0 / (1.0 + math.exp(-(yes - no))), 1.0 - math.exp(admitted - total)


def arm_length_gaps(tokenizer, prompts) -> dict[str, Any]:
    """Measure, per tokenizer, how far the two identity arms differ in token count.

    The rosters are not matched on token length, so the Black-associated arm
    templates slightly longer than its partner. That was worth taking seriously
    while the tensor shape depended on the batch, because sequence length then
    reached the arithmetic and did so in correspondence with the treatment.
    Under a fixed padded length and a fixed batch size it no longer does: both
    arms enter an identically shaped tensor, the reduction order is the same for
    both, and what differs is the attention mask and the boundary position - a
    property of the stimulus rather than of the numerics.

    Equalising the counts by padding the shorter arm was implemented and then
    rejected on measurement, which `D-063` records. No whitespace pad adds tokens
    reliably: on all five pinned tokenizers repeated spaces and tabs merge and
    saturate after one token. The pads that do increment reliably insert visible
    characters, and they would be appended only to the shorter arm - which is
    systematically the White-associated one - replacing a length asymmetry with a
    visible content asymmetry lying along the same contrast. That is the worse
    of the two.

    The gap is therefore measured and recorded per checkpoint rather than
    removed, so that a later reader can see its size instead of taking a claim
    that it is small.
    """
    paired: dict[str, dict[str, int]] = {}
    for prompt in prompts:
        if prompt.identity_group not in ("black", "white"):
            continue
        templated = templated_prompt(tokenizer, prompt.system_prompt, prompt.user_prompt)
        paired.setdefault(prompt.counterfactual_pair_id, {})[prompt.identity_group] = len(
            tokenizer.encode(templated, add_special_tokens=False)
        )
    gaps = [
        arms["black"] - arms["white"]
        for arms in paired.values()
        if "black" in arms and "white" in arms
    ]
    if not gaps:
        return {"pairs": 0}
    return {
        "pairs": len(gaps),
        "mean_black_minus_white_tokens": sum(gaps) / len(gaps),
        "maximum_absolute_gap": max(abs(value) for value in gaps),
        "share_exactly_equal": sum(1 for value in gaps if value == 0) / len(gaps),
    }


def read_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[tuple[str, str]],
    variants: VariantSet,
    maximum_length: int | None = None,
    batch_size: int | None = None,
) -> list[Reading]:
    """Run one fixed padded batch and read the boundary logits.

    The batch is fixed and padded on the right, evaluation mode is asserted,
    gradients are disabled and the cache is off, so the only thing that varies
    between two runs of the same prompt is which other prompts share the tensor.
    That is exactly the dependence the stability gate measures.
    """
    import torch

    settings = config.models()["readout"]
    maximum_length = maximum_length or int(settings["maximum_length"])

    padded_length = int(settings["padded_length"])
    # The sequence dimension is fixed for the whole study. The batch dimension is
    # fixed within a run, and the caller supplies it because the stability gate
    # deliberately compares runs at different batch sizes; every batch inside one
    # of those runs still enters at one shape.
    batch_size = batch_size or int(settings["batch_size"])

    texts = [text for _, text in prompts]
    if len(texts) > batch_size:
        raise ReadoutError(
            f"batch of {len(texts)} exceeds the fixed batch size {batch_size}; the "
            "shape entering the forward pass is fixed for the whole study"
        )
    # A short final batch would enter the forward pass at a different shape from
    # every other batch, which is the dependence this readout exists to remove.
    # It is filled to size with copies of its own first prompt and the fillers'
    # readings are discarded.
    filled = list(texts)
    while len(filled) < batch_size:
        filled.append(texts[0])

    encoded = tokenizer(
        filled,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=padded_length,
        add_special_tokens=False,
    )
    lengths = encoded["attention_mask"].sum(dim=1)
    if int(lengths.max()) >= padded_length:
        raise ReadoutError(
            f"a prompt reaches the {padded_length}-token fixed padded length, so its "
            "answer boundary was truncated away. Raise padded_length rather than "
            "measuring a truncated prompt."
        )
    if int(encoded["input_ids"].shape[1]) != padded_length:
        raise ReadoutError(
            f"tokenizer returned width {int(encoded['input_ids'].shape[1])} rather than "
            f"the fixed {padded_length}; the forward pass would not run at one shape"
        )

    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    model.eval()
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits

    readings = []
    for index, (prompt_id, _) in enumerate(prompts):
        boundary = int(lengths[index]) - 1
        row = logits[index, boundary, :].float().tolist()
        value, probability, off_target = contrast(row, variants)
        greedy = int(max(range(len(row)), key=row.__getitem__))
        readings.append(
            Reading(
                prompt_id=prompt_id,
                token_log_odds=value,
                implied_yes_probability=probability,
                off_target_mass=off_target,
                greedy_token_id=greedy,
                greedy_is_admitted=greedy
                in set(variants.yes_token_ids) | set(variants.no_token_ids),
                boundary_index=boundary,
            )
        )
    return readings


def greedy_agreement(readings: Sequence[Reading], variants: VariantSet) -> dict[str, Any]:
    """Share of readings whose greedy next token agrees with the contrast's sign.

    This is the instrument-validity check for the readout: the contrast is only
    a measurement of the model's answer if the answer the model would actually
    emit is the one the contrast points at. A model can be confidently on one
    side of the Yes/No contrast while emitting something else entirely, and the
    off-target mass alone would not reveal a systematic disagreement.
    """
    yes = set(variants.yes_token_ids)
    no = set(variants.no_token_ids)
    agreeing = 0
    evaluated = 0
    for reading in readings:
        if not reading.greedy_is_admitted:
            evaluated += 1
            continue
        evaluated += 1
        predicted = yes if reading.token_log_odds > 0 else no
        agreeing += int(reading.greedy_token_id in predicted)
    if not evaluated:
        raise ReadoutError("greedy agreement requires at least one reading")
    rate = agreeing / evaluated
    return {
        "sample": evaluated,
        "agreeing": agreeing,
        "agreement": rate,
        "wilson_lower_bound": wilson_lower_bound(agreeing, evaluated),
    }


def wilson_lower_bound(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    """Lower end of the Wilson score interval.

    Wilson rather than the normal approximation because the quantity is a
    proportion near one, where the normal interval extends past the boundary and
    understates the lower end exactly where the gate is read.
    """
    if trials <= 0:
        raise ReadoutError("Wilson interval requires at least one trial")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2 * trials)
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4 * trials * trials)
    )
    return (centre - spread) / denominator
