"""Checks on the crossed estimator and the variance-only sizing rule.

The central test here reproduces the requirement the sizing rule is accepted
under rather than asserting it: at every registered combination of family and
name counts, and at every registered variance structure, the estimator must
reject a true null between 0.045 and 0.055. That requirement exists because the
alternative estimator - clustering on scenario family alone while reusing one
set of name pairs across every family - rejects a true null a quarter of the
time. An interval computed that way is not conservative or approximate; it is
wrong, and it gets worse as families are added.

Every simulation is fixed-seed, so these are deterministic checks on the
estimator rather than draws that might happen to fall inside the window.
"""

import numpy as np
import pytest

from hiringcue import config, estimate

SIZING = config.study()["sizing"]
INFERENCE = config.study()["inference"]
FAMILY_GRID = [int(value) for value in SIZING["family_grid"]]
NAME_GRID = [int(value) for value in SIZING["name_grid"]]
LOW, HIGH = (float(value) for value in SIZING["calibration_interval"])


def _components(family_sd, name_sd, residual_sd):
    return estimate.VarianceComponents(family_sd**2, name_sd**2, residual_sd**2)


def _cells(values):
    return [
        (f"family{row}", f"pair{column}", float(value))
        for row, line in enumerate(values)
        for column, value in enumerate(line)
    ]


def _draw(families, names, effect, parts, seed):
    generator = np.random.default_rng(seed)
    return (
        effect
        + generator.normal(0.0, np.sqrt(parts.family), (families, 1))
        + generator.normal(0.0, np.sqrt(parts.name), (1, names))
        + generator.normal(0.0, np.sqrt(parts.residual), (families, names))
    )


@pytest.mark.parametrize("scenario", SIZING["calibration_scenarios"], ids=lambda s: s["label"])
def test_every_registered_grid_cell_is_calibrated_under_the_null(scenario):
    parts = _components(
        float(scenario["family_sd"]),
        float(scenario["name_sd"]),
        float(scenario["residual_sd"]),
    )
    rates = estimate.calibration(parts)
    assert set(rates) == {
        (families, names) for families in FAMILY_GRID for names in NAME_GRID
    }
    outside = {cell: rate for cell, rate in rates.items() if not LOW <= rate <= HIGH}
    assert not outside, outside


def test_ignoring_the_name_factor_is_anticonservative():
    """The failure the crossed estimator exists to prevent, measured directly.

    One draw of the name effect is shared by every family, so it is confounded
    with the grand mean and a family-clustered interval cannot see it.
    """
    from scipy.stats import t as student

    parts = _components(0.30, 0.25, 0.50)
    families, names, replications = 24, 8, 4000
    generator = np.random.default_rng(11)
    rejected = 0
    for _ in range(replications):
        family = generator.normal(0.0, np.sqrt(parts.family), (families, 1))
        name = generator.normal(0.0, np.sqrt(parts.name), (1, names))
        residual = generator.normal(0.0, np.sqrt(parts.residual), (families, names))
        family_means = (family + name + residual).mean(axis=1)
        statistic = family_means.mean() / (
            family_means.std(ddof=1) / np.sqrt(families)
        )
        rejected += abs(statistic) > student.ppf(0.975, families - 1)
    assert rejected / replications > 0.15


def test_the_variance_components_recover_what_generated_them():
    parts = _components(0.45, 0.25, 0.50)
    values = _draw(200, 200, 0.0, parts, seed=5)
    recovered = estimate.components(values)
    assert recovered.family == pytest.approx(parts.family, rel=0.25)
    assert recovered.name == pytest.approx(parts.name, rel=0.25)
    assert recovered.residual == pytest.approx(parts.residual, rel=0.05)


def test_the_standard_error_carries_all_three_components():
    parts = _components(0.45, 0.25, 0.50)
    result = estimate.estimate(_cells(_draw(24, 24, 0.30, parts, seed=9)))
    family_only = np.sqrt(result.components.family / result.families)
    assert result.standard_error > family_only
    assert result.degrees_freedom < result.families * result.names


def test_an_incomplete_cell_table_is_refused():
    """A hole in the table breaks the decomposition silently, so it stops here."""
    cells = _cells(_draw(4, 4, 0.0, _components(0.4, 0.2, 0.5), seed=1))
    with pytest.raises(estimate.EstimationError, match="cells are empty"):
        estimate.estimate(cells[:-1])


def test_a_duplicated_cell_is_refused():
    cells = _cells(_draw(4, 4, 0.0, _components(0.4, 0.2, 0.5), seed=1))
    with pytest.raises(estimate.EstimationError, match="duplicate contrast"):
        estimate.estimate(cells + [cells[0]])


def test_sizing_selects_the_smallest_adequate_design():
    parts = _components(0.45, 0.25, 0.50)
    report = estimate.select_design(parts, replications=4000)
    selected = report["selected"]
    assert selected is not None
    adequate = [
        entry for entry in report["grid"] if entry["power"] >= report["target_power"]
    ]
    assert selected["cells"] == min(entry["cells"] for entry in adequate)


def test_sizing_declines_to_collect_when_no_registered_design_suffices():
    """Not reaching the target is a stopping rule, not a prompt to enlarge the grid."""
    parts = _components(2.0, 2.0, 2.0)
    report = estimate.select_design(parts, replications=2000)
    assert report["selected"] is None
    assert report["decision"] == "do_not_collect"


def test_sizing_does_not_read_the_observed_effect():
    """The rule is variance-only; an observed interaction must not size the study."""
    parts = _components(0.45, 0.25, 0.50)
    first = estimate.select_design(parts, replications=2000)
    second = estimate.select_design(parts, replications=2000)
    assert first["selected"] == second["selected"]
    assert "observed" not in report_keys(first)


def report_keys(report):
    return set(report) | {key for entry in report["grid"] for key in entry}


def test_the_pool_constrains_the_name_grid_the_rule_may_select_from():
    """The registered grid is what calibration is verified on; the available grid
    is what the matched name pool can supply, and sizing runs against the latter."""
    available = [int(value) for value in SIZING["available_name_grid"]]
    assert available
    assert set(available) <= set(NAME_GRID)


def test_the_interaction_cell_is_the_declared_contrast():
    rows = []
    values = {
        ("bare", "white"): 0.1,
        ("bare", "black"): 0.3,
        ("realistic", "white"): 0.5,
        ("realistic", "black"): 1.4,
    }
    for (level, arm), value in values.items():
        rows.append(
            {
                "role": "primary",
                "cue_mode": "concealed",
                "identity_group": arm,
                "name_pair_id": "pair_000",
                "family_id": "family_a",
                "context_level": level,
                "token_log_odds": value,
            }
        )
    cells = estimate.interaction_cells(rows)
    assert cells == [("family_a", "pair_000", pytest.approx(0.7))]


def test_rule_control_rows_never_enter_the_estimand():
    """On a candidate who objectively fails, the outcome saturates and the cell
    contributes resolution rather than signal."""
    rows = [
        {
            "role": "rule_control",
            "cue_mode": "concealed",
            "identity_group": arm,
            "name_pair_id": "pair_000",
            "family_id": "family_a",
            "context_level": level,
            "token_log_odds": 1.0,
        }
        for level in ("bare", "realistic")
        for arm in ("white", "black")
    ]
    assert estimate.interaction_cells(rows) == []


def test_a_cell_missing_an_arm_stops_the_estimate():
    rows = [
        {
            "role": "primary",
            "cue_mode": "concealed",
            "identity_group": "white",
            "name_pair_id": "pair_000",
            "family_id": "family_a",
            "context_level": level,
            "token_log_odds": 1.0,
        }
        for level in ("bare", "realistic")
    ]
    with pytest.raises(estimate.EstimationError, match="both identity arms"):
        estimate.interaction_cells(rows)


def test_development_estimates_report_variance_and_paired_recognisability():
    from hiringcue import diagnostics

    rows = []
    for family_index in range(2):
        for pair_index in range(2):
            for level in ("bare", "realistic", "realistic_matched", "realistic_named"):
                for arm in ("white", "black"):
                    arm_effect = 1.0 if arm == "black" else 0.0
                    multiplier = {
                        "bare": 0.0,
                        "realistic": 0.4,
                        "realistic_matched": 0.2,
                        "realistic_named": 0.5,
                    }[level]
                    rows.append(
                        {
                            "role": "primary",
                            "cue_mode": "concealed",
                            "identity_group": arm,
                            "name_pair_id": f"pair_{pair_index}",
                            "family_id": f"family_{family_index}",
                            "context_level": level,
                            "token_log_odds": arm_effect * multiplier,
                        }
                    )
    report = diagnostics.development_estimates(rows)
    assert report["context_identity_interaction"]["estimate"] == pytest.approx(0.4)
    assert "components" in report["context_identity_interaction"]
    assert report["recognisability"]["named_minus_matched"]["estimate"] == pytest.approx(0.3)
