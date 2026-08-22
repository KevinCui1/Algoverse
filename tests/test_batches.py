"""Checks on the batch manifest.

Two properties of the layout keep batch-level numerical dependence off the
contrast, and neither is visible in any downstream number if it breaks: the two
arms of a counterfactual pair must share a tensor, and which of them comes first
must not be a function of the identity condition.
"""

import pytest

from hiringcue import batches


def _pair(index):
    return [
        batches.PlannedPrompt(f"p{index}_white", f"pair{index}", "white"),
        batches.PlannedPrompt(f"p{index}_black", f"pair{index}", "black"),
    ]


def _prompts(pairs=6, unpaired=0):
    planned = [prompt for index in range(pairs) for prompt in _pair(index)]
    planned += [
        batches.PlannedPrompt(f"n{index}", batches.UNPAIRED, None)
        for index in range(unpaired)
    ]
    return planned


def test_both_arms_of_a_pair_share_a_batch():
    slots = batches.build(_prompts(pairs=10), batch_size=4, seed=1)
    by_pair = {}
    for slot in slots:
        by_pair.setdefault(slot.pair_id, set()).add(slot.batch_index)
    assert all(len(indices) == 1 for indices in by_pair.values())


def test_arm_order_is_randomised_rather_than_fixed():
    """A layout that always placed one arm first would align any residual
    position effect with the identity condition being estimated."""
    slots = batches.build(_prompts(pairs=40), batch_size=8, seed=7)
    first_arms = {}
    for slot in slots:
        if slot.pair_id != batches.UNPAIRED:
            first_arms.setdefault(slot.pair_id, slot.identity_arm)
    leading = list(first_arms.values())
    assert set(leading) == {"white", "black"}
    assert 0.2 < leading.count("white") / len(leading) < 0.8


def test_the_manifest_reproduces_from_the_same_seed():
    left = batches.build(_prompts(pairs=12), batch_size=8, seed=3)
    right = batches.build(_prompts(pairs=12), batch_size=8, seed=3)
    assert left == right


def test_a_batch_never_exceeds_the_declared_size():
    slots = batches.build(_prompts(pairs=9, unpaired=5), batch_size=4, seed=2)
    for batch in batches.batched(slots):
        assert len(batch) <= 4


def test_a_batch_too_small_to_hold_a_pair_is_refused():
    with pytest.raises(batches.BatchError, match="counterfactual pair"):
        batches.build(_prompts(pairs=2), batch_size=1, seed=1)


def test_an_incomplete_pair_is_refused():
    """A pair missing an arm would be differenced against nothing."""
    planned = _prompts(pairs=2)[:-1]
    with pytest.raises(batches.BatchError, match="exactly two arms"):
        batches.build(planned, batch_size=4, seed=1)


def test_the_manifest_round_trips_through_disk(tmp_path):
    slots = batches.build(_prompts(pairs=5), batch_size=4, seed=1)
    path = tmp_path / "batches.jsonl"
    batches.write(slots, path)
    assert batches.read(path) == slots


def test_the_declared_layouts_differ_in_composition():
    """Comparing a layout with a duplicate of itself would pass by construction."""
    declared = batches.layouts()
    assert len(declared) >= 3
    signatures = {
        (
            entry["batch_size"],
            entry["shuffle_seed"],
            entry["within_batch_seed"],
            entry["fresh_process"],
        )
        for entry in declared
    }
    assert len(signatures) == len(declared)


def test_the_gate_layouts_separate_ordering_from_membership_at_frozen_batch_size():
    """Each gated mechanism must be varied by a layout that varies nothing else.

    The retired `shuffled` layout permuted the sample before batching, so it
    changed which prompts shared a tensor and - under longest-in-batch padding -
    the padded length too, while being labelled an ordering perturbation. A
    non-zero result could not then be attributed to a mechanism.
    """
    declared = {entry["label"]: entry for entry in batches.layouts()}
    reference = declared["reference"]

    ordering = declared["reordered"]
    assert ordering["batch_size"] == reference["batch_size"]
    assert ordering["shuffle_seed"] is None
    assert ordering["within_batch_seed"] is not None

    membership = declared["regrouped"]
    assert membership["batch_size"] == reference["batch_size"]
    assert membership["shuffle_seed"] is not None
    assert membership["within_batch_seed"] is None

    warm = declared["warm_reference"]
    assert warm["batch_size"] == reference["batch_size"]
    assert warm["shuffle_seed"] is None
    assert warm["within_batch_seed"] is None
    assert warm["fresh_process"] is False

    assert {entry["batch_size"] for entry in declared.values()} == {
        reference["batch_size"]
    }
    assert "small_batch" not in declared
    assert "warm_large" not in declared
    assert "shuffled" not in declared


def test_batch_size_departures_are_disclosures_not_gate_layouts():
    gate_labels = {entry["label"] for entry in batches.layouts()}
    disclosed = batches.batch_size_sensitivity_layouts()
    reference_size = batches.layouts()[0]["batch_size"]
    assert {entry["label"] for entry in disclosed}.isdisjoint(gate_labels)
    assert all(entry["batch_size"] != reference_size for entry in disclosed)


def test_the_stability_sample_never_cuts_a_pair_in_half():
    """Both arms of a pair must share a batch, so a half pair cannot be laid out."""
    from hiringcue import plan, stage0, stimuli

    prompts = plan.build(pairs=stimuli.load_pairs(stimuli.DEVELOPMENT))
    sample = stage0.stability_sample(prompts)
    counts: dict[str, int] = {}
    for prompt in sample:
        counts[prompt.counterfactual_pair_id] = (
            counts.get(prompt.counterfactual_pair_id, 0) + 1
        )
    assert all(count == 2 for count in counts.values())
    batches.build(
        batches.PlannedPrompt(
            prompt.prompt_id, prompt.counterfactual_pair_id, prompt.identity_group
        )
        for prompt in sample
    )


def test_the_stability_sample_is_a_stratified_census():
    """A sample drawn in plan order covered one occupation and omitted a band.

    Plan order is grouped by occupation, so a flat prefix of it took every one
    of its 200 prompts from the first occupation and contained no `near_pass`
    cell - the band with the most discretion, and so the one whose stability
    matters most. The sampler now has to span the design.
    """
    from hiringcue import config, plan, stage0, stimuli

    prompts = plan.build(pairs=stimuli.load_pairs(stimuli.DEVELOPMENT))
    sample = stage0.stability_sample(prompts)

    assert {p.margin_band for p in sample} == set(config.study()["margin_bands"])
    assert {p.context_level for p in sample} == set(config.study()["context"]["levels"])
    assert {p.prompt_form for p in sample} == set(
        config.study()["prompt_form"]["levels"]
    )
    assert {p.prestige_level for p in sample} == {
        p.prestige_level for p in prompts if p.soft_variant == "base"
    }
    assert {p.occupation_slug for p in sample} == {
        p.occupation_slug for p in prompts
    }
    assert len({p.occupation_slug for p in sample}) == 6

    qualified = set(config.study()["margin_bands"]) - set(
        config.study()["qualification"]["control_bands"]
    )
    cells = {
        (p.family_id, p.prompt_form, p.context_level)
        for p in sample
        if p.margin_band in qualified and p.cue_mode == "concealed"
    }
    # Twelve qualified families at three context levels in each of two prompt
    # forms. The gate statistic is a mean over these, so a missing cell is a
    # silently narrowed gate.
    assert len(cells) == 72


def test_stage0_requires_both_stability_and_saturation():
    from hiringcue import stage0

    verdict, failed = stage0.instrument_verdict(
        {
            "stability": {"verdict": "PASS"},
            "saturation": {"qualified": {"verdict": "FAIL"}},
        }
    )
    assert verdict == "FAIL"
    assert failed == ["saturation"]
