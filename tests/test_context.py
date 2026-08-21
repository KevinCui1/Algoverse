"""The employer-context factor, including the development-only recognisability probe.

The confirmatory estimand crosses `bare` with `realistic` alone. Two further
levels exist to answer a question the fictitious employer cannot answer about
itself - whether the published amplification depends on the employer being
recognisable - and the tests here fix the two properties that make that answer
readable: the two probe levels must differ only in the identity of the
organisation, and they must stay out of the confirmatory design.
"""

from __future__ import annotations

import difflib

import pytest

from hiringcue import context


def test_confirmatory_levels_exclude_the_probe():
    assert context.levels() == ("bare", "realistic")
    assert set(context.development_only_levels()).isdisjoint(context.levels())


def test_probe_levels_are_absent_unless_asked_for():
    assert sorted(context.load()) == ["bare", "realistic"]
    assert sorted(context.load(include_development_levels=True)) == [
        "bare",
        "realistic",
        "realistic_matched",
        "realistic_named",
    ]


def test_probe_arms_differ_only_in_who_the_employer_is():
    """Recognisability is the manipulation, so nothing else may vary with it.

    The estimate is a difference of two interactions. Any other difference
    between the two descriptions - length, scale, sector, how much detail is
    given - would be absorbed into it and reported as recognisability.
    """
    loaded = context.load(include_development_levels=True)
    named = loaded["realistic_named"].text.split()
    matched = loaded["realistic_matched"].text.split()

    changed = [
        line
        for line in difflib.unified_diff(matched, named, lineterm="", n=0)
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    # Both descriptions name an organisation and a place; those are the only
    # tokens permitted to move.
    identity_tokens = {
        "Halverston", "State", "University", "The", "Ohio", "Columbus,", "Ohio",
        "at", "in", "the", "upper", "Midwest", "main", "campus", "Columbus",
        "and", "additional", "campuses", "elsewhere", "with", "its", "on",
    }
    for line in changed:
        assert line[1:] in identity_tokens, f"probe arms differ outside the employer identity: {line}"

    assert len(named) == pytest.approx(len(matched), abs=6)


def test_probe_text_carries_no_identity_term():
    """A first-turn prompt naming race would prime the behaviour being observed."""
    banned = ("white", "black", "african american", "caucasian", "race", "ethnicity")
    loaded = context.load(include_development_levels=True)
    for level in context.development_only_levels():
        lowered = loaded[level].text.casefold()
        for term in banned:
            assert term not in lowered, f"{level} contains {term!r}"


def test_unknown_variant_is_refused():
    with pytest.raises(context.ContextError):
        context.load("employer_recognisable")
