"""The employer-context factor.

Three levels are measured together: `bare`, organisational context alone, and
organisational context under a selectivity constraint. The tests here fix the
properties the comparison between them depends on - that the control level is
present, that each rich level is renderable and distinct, and that no level
carries an identity term into a first-turn prompt.
"""

from __future__ import annotations

import pytest

from hiringcue import context


def test_the_declared_levels_include_the_control_and_both_rich_levels():
    assert context.levels() == ("bare", "employer", "employer_selectivity")
    assert context.realistic_levels() == ("employer", "employer_selectivity")


def test_every_declared_level_is_loaded_in_one_round():
    """The levels are contrasted with each other, so they share one collection.

    A level measured in a separate round would differ from the others by
    whatever else changed between rounds as well as by its own text.
    """
    loaded = context.load()
    assert sorted(loaded) == ["bare", "employer", "employer_selectivity"]
    assert loaded["bare"].text.strip() == ""


def test_the_selectivity_level_adds_a_constraint_to_the_same_employer():
    """Selectivity is the manipulation, so the organisation must not move with it.

    If the two rich levels described different employers, the difference of
    their interactions would absorb everything else about the two descriptions
    and be reported as the effect of the constraint.
    """
    loaded = context.load()
    plain = loaded["employer"].text.strip()
    selective = loaded["employer_selectivity"].text.strip()

    assert selective.startswith(plain)
    added = selective[len(plain) :].strip()
    assert "312 applications" in added
    assert "only the strongest applicants" in added


def test_no_context_text_carries_an_identity_term():
    """A first-turn prompt naming race would prime the behaviour being observed."""
    banned = ("white", "black", "african american", "caucasian", "race", "ethnicity")
    for level, loaded in context.load().items():
        lowered = loaded.text.casefold()
        for term in banned:
            assert term not in lowered, f"{level} contains {term!r}"


def test_an_undeclared_level_is_refused():
    with pytest.raises(context.ContextError):
        context._template("employer_recognisable")
