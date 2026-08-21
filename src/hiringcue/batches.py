"""Deterministic batch manifests for the forward-pass readout.

A batch manifest fixes which prompts share a tensor and in what order, and is
written to disk before any measurement. It exists because the readout's value
for one prompt can depend on what else is resident in the batch: the reduction
order of a matrix multiply changes with the shape of the tensor it runs over,
and in bounded precision that changes the low bits of the result. The stability
gate measures how large that dependence is; the manifest is what makes it a
recorded property of a run rather than an accident of scheduling.

Two rules shape the layout, and both exist to keep that dependence off the
contrast being estimated.

*Both arms of a counterfactual pair are placed in the same batch.* Arms split
across batches would sit in systematically different tensors, and the difference
between those tensors would lie exactly along the identity contrast. Co-residence
makes whatever batch-level perturbation exists common to the pair.

*Which arm is placed first is randomised against a recorded seed.* Position
within a batch is not perfectly neutral, so any residual asymmetry is made
orthogonal to the identity condition rather than aligned with it.

Input is the planned prompts with their pair membership. Output is an ordered
manifest that reproduces exactly from the same plan and seed.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from . import config

UNPAIRED = "unpaired"


class BatchError(ValueError):
    """Raised when a batch layout cannot be built as specified."""


@dataclass(frozen=True)
class Slot:
    batch_index: int
    position: int
    prompt_id: str
    pair_id: str
    identity_arm: str | None


@dataclass(frozen=True)
class PlannedPrompt:
    """A prompt as the batcher needs it: an identity, a pair, and an arm."""

    prompt_id: str
    pair_id: str
    identity_arm: str | None


def _group(prompts: Sequence[PlannedPrompt]) -> list[list[PlannedPrompt]]:
    groups: dict[str, list[PlannedPrompt]] = {}
    singles: list[list[PlannedPrompt]] = []
    for prompt in prompts:
        if prompt.pair_id == UNPAIRED:
            singles.append([prompt])
            continue
        groups.setdefault(prompt.pair_id, []).append(prompt)

    for pair_id, members in groups.items():
        if len(members) != 2:
            raise BatchError(
                f"counterfactual pair {pair_id!r} has {len(members)} members; a pair "
                "the readout can difference must have exactly two arms"
            )
    ordered = [groups[pair_id] for pair_id in sorted(groups)] + singles
    return ordered


def build(
    prompts: Iterable[PlannedPrompt],
    batch_size: int | None = None,
    seed: int | None = None,
) -> list[Slot]:
    """Lay every prompt out into fixed batches, keeping pairs co-resident."""
    settings = config.models()["readout"]
    batch_size = batch_size or int(settings["batch_size"])
    seed = seed if seed is not None else int(settings["arm_order_seed"])
    if batch_size < 2:
        raise BatchError(
            f"batch size {batch_size} cannot hold a counterfactual pair; the two arms "
            "must share a tensor for the batch-level perturbation to be common to them"
        )

    units = _group(list(prompts))
    generator = random.Random(seed)
    for unit in units:
        if len(unit) == 2 and generator.random() < 0.5:
            unit.reverse()

    slots: list[Slot] = []
    batch_index = 0
    position = 0
    for unit in units:
        if position + len(unit) > batch_size:
            batch_index += 1
            position = 0
        for prompt in unit:
            slots.append(
                Slot(
                    batch_index=batch_index,
                    position=position,
                    prompt_id=prompt.prompt_id,
                    pair_id=prompt.pair_id,
                    identity_arm=prompt.identity_arm,
                )
            )
            position += 1
    return slots


def layouts() -> list[dict]:
    """The fixed-batch-size compositions the stability gate compares.

    The gate is a claim about invariance under irrelevant perturbation, so the
    layouts differ in ordering and membership, properties that vary within a
    collection. Batch size is a frozen instrument parameter and is measured
    separately as a disclosed sensitivity rather than varied by this gate.
    """
    return [dict(entry) for entry in config.models()["readout"]["stability_layouts"]]


def batch_size_sensitivity_layouts() -> list[dict]:
    """Departures from the frozen batch size used only for sensitivity disclosure."""
    return [
        dict(entry)
        for entry in config.models()["readout"]["batch_size_sensitivity_layouts"]
    ]


def write(slots: Sequence[Slot], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for slot in slots:
            handle.write(json.dumps(asdict(slot)) + "\n")


def read(path: Path) -> list[Slot]:
    return [
        Slot(**json.loads(line)) for line in path.read_text().splitlines() if line
    ]


def batched(slots: Sequence[Slot]) -> list[list[Slot]]:
    """Regroup a manifest into the batches it describes, in recorded order."""
    grouped: dict[int, list[Slot]] = {}
    for slot in slots:
        grouped.setdefault(slot.batch_index, []).append(slot)
    return [
        sorted(grouped[index], key=lambda slot: slot.position)
        for index in sorted(grouped)
    ]
