"""Contract checks for the perturbed soft-criteria positive control."""

from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

import pytest

from hiringcue import runner, scenarios, twin_author, twins


@pytest.fixture
def family():
    return scenarios.validated_families()[0]


@pytest.fixture
def valid_profile(family):
    return [
        {
            "criterion_id": entry["criterion_id"],
            "candidate_evidence": entry["candidate_evidence"],
            "position": twins.POSITION_TARGET[entry["position"]],
        }
        for entry in family.soft_profile
    ]


def test_valid_profile_matches_every_requested_direction(family, valid_profile):
    twins.validate(family, valid_profile)


def test_reordered_criteria_are_rejected(family, valid_profile):
    reordered = list(reversed(valid_profile))
    with pytest.raises(twins.TwinError, match="reorder"):
        twins.validate(family, reordered)


def test_wrong_position_direction_is_rejected(family, valid_profile):
    wrong = copy.deepcopy(valid_profile)
    wrong[0]["position"] = family.soft_profile[0]["position"]
    with pytest.raises(twins.TwinError, match="requested directions"):
        twins.validate(family, wrong)


def test_personal_pronouns_are_rejected(family, valid_profile):
    with_pronoun = copy.deepcopy(valid_profile)
    words = family.soft_profile[0]["candidate_evidence"].split()
    with_pronoun[0]["candidate_evidence"] = " ".join(["They", *words[1:]])
    with pytest.raises(twins.TwinError, match="personal pronouns"):
        twins.validate(family, with_pronoun)


def test_large_length_change_is_rejected(family, valid_profile):
    too_long = copy.deepcopy(valid_profile)
    base_words = len(family.soft_profile[0]["candidate_evidence"].split())
    too_long[0]["candidate_evidence"] = " ".join(["relevant"] * (base_words * 2))
    with pytest.raises(twins.TwinError, match="word-count ratio"):
        twins.validate(family, too_long)


def test_authoring_fails_instead_of_returning_a_partial_pool(
    family, monkeypatch
):
    class InvalidOutputModel:
        def __init__(self, **kwargs):
            pass

        def chat(self, messages, params, chat_template_kwargs):
            output = SimpleNamespace(outputs=[SimpleNamespace(text="{}")])
            return [output for _ in messages]

    fake_vllm = SimpleNamespace(
        LLM=InvalidOutputModel,
        SamplingParams=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(runner, "_structured_kwargs", lambda schema: {})

    with pytest.raises(twins.TwinError, match="unauthored twins"):
        twin_author.author_all([family], model_key="qwen3-32b")
