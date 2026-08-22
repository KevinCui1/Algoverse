"""Crossed variance-component inference and variance-only sizing.

The quantity estimated is one contrast per crossed cell of scenario family and
name pair:

    [(Black - White)_realistic - (Black - White)_bare]

in the token log-odds units of the primary readout. Both factors are random:
the study generalises over hiring scenarios and over names, not over the
particular lists of them that were authored.

**Why the second random factor is not a refinement.** If one fixed set of name
pairs appears in every family, every family mean carries the same draw of the
name effect. That draw is confounded with the grand mean and a family-clustered
standard error cannot see it, because it has no variation across the clusters
being resampled. The interval is then too narrow by whatever the name variance
contributes, and the error grows with the number of families: the family term
shrinks as families are added while the name term does not. This is the
language-as-fixed-effect fallacy (Clark, 1973), and it invalidates any interval
computed the other way round.

The estimator is therefore a crossed two-way random-effects ANOVA. Variance
components come from the mean squares,

    Var(mean) = tau^2/F + sigma_name^2/J + MSE/(F*J)

and the reference distribution uses Satterthwaite degrees of freedom formed from
the same three terms. Components are truncated at zero, because a negative
variance estimate is a sampling artefact and carrying it forward would shrink
the standard error below the family term alone.

**Sizing is variance-only.** After the development round, blinded variance
estimates are placed into a fixed grid of family and name counts and the
smallest combination reaching the required power at the minimum meaningful
effect is selected, ties broken by lower family count then lower name count. The
observed interaction may not enter sizing: choosing the size that makes the
observed effect significant is what a registered rule exists to prevent.

Inputs are per-cell contrasts keyed by family and name pair, or variance
components directly when sizing. Outputs are an estimate with an interval, or a
selected design. Every entry point raises rather than returning a result
computed on an incomplete grid: an unbalanced cell table makes the mean squares
below the wrong decomposition, and the failure is silent.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import nct, t

from . import config


class EstimationError(ValueError):
    """Raised when the cell table cannot support the crossed decomposition."""


@dataclass(frozen=True)
class VarianceComponents:
    family: float
    name: float
    residual: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Estimate:
    mean: float
    standard_error: float
    degrees_freedom: float
    statistic: float
    p_value: float
    interval_lower: float
    interval_upper: float
    interval_level: float
    families: int
    names: int
    components: VarianceComponents

    def as_dict(self) -> dict[str, float | int | dict[str, float]]:
        record = asdict(self)
        record["components"] = self.components.as_dict()
        return record


def cell_matrix(
    contrasts: Iterable[tuple[str, str, float]]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Arrange one contrast per family-by-name cell into a complete matrix.

    A missing or duplicated cell is a hard stop rather than an imputation. The
    mean squares below assume one observation per crossed cell; with a hole in
    the table the family and name sums of squares no longer decompose the total,
    and the resulting interval is wrong in a direction nothing downstream can
    detect.
    """
    records = list(contrasts)
    if not records:
        raise EstimationError("no contrasts supplied")

    families = sorted({family for family, _, _ in records})
    names = sorted({name for _, name, _ in records})
    if len(families) < 2 or len(names) < 2:
        raise EstimationError(
            f"crossed inference needs at least two levels of each factor; got "
            f"{len(families)} families and {len(names)} name pairs"
        )

    family_index = {family: index for index, family in enumerate(families)}
    name_index = {name: index for index, name in enumerate(names)}
    values = np.full((len(families), len(names)), np.nan)
    for family, name, value in records:
        row, column = family_index[family], name_index[name]
        if not np.isnan(values[row, column]):
            raise EstimationError(f"duplicate contrast for cell ({family}, {name})")
        values[row, column] = value

    missing = np.argwhere(np.isnan(values))
    if missing.size:
        first = missing[0]
        raise EstimationError(
            f"{len(missing)} of {values.size} family-by-name cells are empty "
            f"(e.g. {families[first[0]]!r} x {names[first[1]]!r}); the crossed "
            "decomposition requires a complete table"
        )
    return values, families, names


def components(values: np.ndarray) -> VarianceComponents:
    """ANOVA variance components for a complete family-by-name contrast table."""
    families, names = values.shape
    df_family = families - 1
    df_name = names - 1
    df_residual = df_family * df_name

    grand = values.mean()
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ms_family = names * ((row_means - grand) ** 2).sum() / df_family
    ms_name = families * ((column_means - grand) ** 2).sum() / df_name
    centred = values - row_means[:, None] - column_means[None, :] + grand
    ms_residual = (centred * centred).sum() / df_residual

    return VarianceComponents(
        family=max((ms_family - ms_residual) / names, 0.0),
        name=max((ms_name - ms_residual) / families, 0.0),
        residual=ms_residual,
    )


def _satterthwaite(
    component_variances: Sequence[float], component_df: Sequence[int]
) -> float:
    total = float(sum(component_variances))
    denominator = sum(
        value**2 / degrees
        for value, degrees in zip(component_variances, component_df)
        if degrees > 0
    )
    if denominator <= 0.0:
        return float(min(component_df))
    return total**2 / denominator


def estimate(
    contrasts: Iterable[tuple[str, str, float]],
    alpha: float | None = None,
    interval_level: float | None = None,
) -> Estimate:
    """Grand mean of the crossed contrast with a Satterthwaite interval."""
    settings = config.study()["inference"]
    alpha = float(settings["two_sided_alpha"]) if alpha is None else alpha
    interval_level = (
        float(settings["interval_level"]) if interval_level is None else interval_level
    )

    values, families, names = cell_matrix(contrasts)
    parts = components(values)
    family_count, name_count = values.shape

    variances = (
        parts.family / family_count,
        parts.name / name_count,
        parts.residual / (family_count * name_count),
    )
    degrees = (family_count - 1, name_count - 1, (family_count - 1) * (name_count - 1))
    variance = float(sum(variances))
    standard_error = float(np.sqrt(variance))
    degrees_freedom = _satterthwaite(variances, degrees)

    mean = float(values.mean())
    statistic = mean / standard_error if standard_error > 0 else float("inf")
    p_value = float(2.0 * t.sf(abs(statistic), degrees_freedom))
    half_width = float(t.ppf(1.0 - (1.0 - interval_level) / 2.0, degrees_freedom)) * standard_error

    return Estimate(
        mean=mean,
        standard_error=standard_error,
        degrees_freedom=float(degrees_freedom),
        statistic=float(statistic),
        p_value=p_value,
        interval_lower=mean - half_width,
        interval_upper=mean + half_width,
        interval_level=interval_level,
        families=len(families),
        names=len(names),
        components=parts,
    )


def analytic_power(
    *,
    families: int,
    names: int,
    effect: float,
    parts: VarianceComponents,
    alpha: float,
) -> float:
    """Power of the Satterthwaite test at a design and a set of true components."""
    variances = np.array(
        [
            parts.family / families,
            parts.name / names,
            parts.residual / (families * names),
        ]
    )
    degrees = np.array([families - 1, names - 1, (families - 1) * (names - 1)])
    variance = float(variances.sum())
    if variance <= 0.0:
        return 1.0
    degrees_freedom = _satterthwaite(variances, degrees)
    critical = float(t.ppf(1.0 - alpha / 2.0, degrees_freedom))
    noncentrality = effect / np.sqrt(variance)
    return float(
        nct.cdf(-critical, degrees_freedom, noncentrality)
        + nct.sf(critical, degrees_freedom, noncentrality)
    )


def rejection_rate(
    *,
    families: int,
    names: int,
    effect: float,
    parts: VarianceComponents,
    replications: int,
    seed: int,
    alpha: float,
    chunk: int = 2_000,
) -> float:
    """Simulated rejection rate of the estimator at a design and true components.

    Drawn rather than derived because the analytic power above uses the true
    components while the test uses estimated ones. The null case of this
    function is the calibration check the sizing rule must pass before it is
    accepted: an estimator whose null rejection rate is not near the nominal
    level cannot be sized, whatever its power curve says.
    """
    generator = np.random.default_rng(seed)
    rejected = 0
    completed = 0
    df_family = families - 1
    df_name = names - 1
    df_residual = df_family * df_name

    while completed < replications:
        batch = min(chunk, replications - completed)
        values = (
            effect
            + generator.normal(0.0, np.sqrt(parts.family), (batch, families, 1))
            + generator.normal(0.0, np.sqrt(parts.name), (batch, 1, names))
            + generator.normal(
                0.0, np.sqrt(parts.residual), (batch, families, names)
            )
        )

        grand = values.mean(axis=(1, 2))
        row_means = values.mean(axis=2)
        column_means = values.mean(axis=1)
        ms_family = (
            names * ((row_means - grand[:, None]) ** 2).sum(axis=1) / df_family
        )
        ms_name = (
            families * ((column_means - grand[:, None]) ** 2).sum(axis=1) / df_name
        )
        centred = (
            values
            - row_means[:, :, None]
            - column_means[:, None, :]
            + grand[:, None, None]
        )
        ms_residual = (centred * centred).sum(axis=(1, 2)) / df_residual

        variance_family = np.maximum((ms_family - ms_residual) / names, 0.0) / families
        variance_name = np.maximum((ms_name - ms_residual) / families, 0.0) / names
        variance_residual = ms_residual / (families * names)
        variance = variance_family + variance_name + variance_residual

        denominator = (
            np.square(variance_family) / df_family
            + np.square(variance_name) / df_name
            + np.square(variance_residual) / df_residual
        )
        degrees_freedom = np.square(variance) / denominator
        statistic = grand / np.sqrt(variance)
        critical = t.ppf(1.0 - alpha / 2.0, degrees_freedom)
        rejected += int(np.count_nonzero(np.abs(statistic) > critical))
        completed += batch

    return rejected / replications


def _cell_seed(seed: int, label: str) -> int:
    """Deterministic per-cell seed so a sizing report reproduces exactly."""
    import hashlib

    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def calibration(
    parts: VarianceComponents,
    replications: int | None = None,
    family_grid: Sequence[int] | None = None,
    name_grid: Sequence[int] | None = None,
) -> dict[tuple[int, int], float]:
    """Null rejection rate at every registered grid cell."""
    settings = config.study()["sizing"]
    alpha = float(config.study()["inference"]["two_sided_alpha"])
    family_grid = family_grid or [int(value) for value in settings["family_grid"]]
    name_grid = name_grid or [int(value) for value in settings["name_grid"]]
    replications = replications or int(settings["calibration_replications"])
    seed = int(settings["simulation_seed"])

    return {
        (families, names): rejection_rate(
            families=families,
            names=names,
            effect=0.0,
            parts=parts,
            replications=replications,
            seed=_cell_seed(seed, f"null:{families}:{names}"),
            alpha=alpha,
        )
        for families in family_grid
        for names in name_grid
    }


def select_design(
    parts: VarianceComponents,
    replications: int | None = None,
    name_grid: Sequence[int] | None = None,
) -> dict:
    """Apply the registered variance-only sizing rule.

    Only the variance components reach this function. The observed interaction
    is not a parameter of it, which is what makes the selected size independent
    of the result being tested.
    """
    settings = config.study()["sizing"]
    inference = config.study()["inference"]
    alpha = float(inference["two_sided_alpha"])
    effect = float(inference["minimum_meaningful_effect"])
    target_power = float(settings["target_power"])
    family_grid = [int(value) for value in settings["family_grid"]]
    name_grid = (
        [int(value) for value in name_grid]
        if name_grid is not None
        else [int(value) for value in settings["name_grid"]]
    )
    replications = replications or int(settings["sizing_replications"])
    seed = int(settings["simulation_seed"])

    grid = []
    for families in family_grid:
        for names in name_grid:
            power = rejection_rate(
                families=families,
                names=names,
                effect=effect,
                parts=parts,
                replications=replications,
                seed=_cell_seed(seed, f"power:{families}:{names}"),
                alpha=alpha,
            )
            grid.append(
                {
                    "families": families,
                    "names": names,
                    "cells": families * names,
                    "power": power,
                    "analytic_power": analytic_power(
                        families=families,
                        names=names,
                        effect=effect,
                        parts=parts,
                        alpha=alpha,
                    ),
                }
            )

    reaching = [entry for entry in grid if entry["power"] >= target_power]
    selected = (
        min(reaching, key=lambda entry: (entry["cells"], entry["families"], entry["names"]))
        if reaching
        else None
    )
    return {
        "components": parts.as_dict(),
        "minimum_meaningful_effect": effect,
        "target_power": target_power,
        "family_grid": family_grid,
        "name_grid": name_grid,
        "grid": grid,
        "selected": selected,
        # No grid cell reaching the target is a stopping rule, not a prompt to
        # enlarge the grid: the largest registered design is the largest the
        # collection budget supports.
        "decision": "collect" if selected else "do_not_collect",
    }


DIRECT_PSEUDO_PAIR = "direct"


def _arm_buckets(
    rows: Iterable[dict], cue_mode: str, prompt_form: str | None
) -> dict[tuple[str, str, str, str], list[float]]:
    """Readings grouped by family, name pair, context level and identity arm.

    The direct conditions carry no name pair: the cue is a stated sentence
    rather than a name draw, so there is no name factor to generalise over and
    the rows are collected under a single pseudo-pair. That is why a direct
    estimate is family-clustered rather than crossed, and it is reported as the
    weaker design it is.

    Prestige is averaged over rather than entering the key. It is a crossed
    nuisance factor: the estimand generalises over credential presentation, and
    splitting cells by it would halve the observations behind each without
    changing what is being estimated.

    Only prompts in the primary role contribute. Families whose candidate
    objectively fails are a rule-following control, and on them the outcome
    saturates, so including them would dilute the estimand with cells that
    cannot carry an effect.
    """
    from . import plan

    buckets: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        if row.get("role") != plan.PRIMARY or row.get("cue_mode") != cue_mode:
            continue
        if prompt_form is not None and row.get("prompt_form") != prompt_form:
            continue
        arm = row.get("identity_group")
        pair = row.get("name_pair_id")
        if arm is None:
            continue
        if pair is None:
            # Only the direct conditions legitimately carry no name pair. A
            # concealed row without one is a defective record, and bucketing it
            # under the pseudo-pair would mix a name draw into an estimate that
            # reports itself as generalising over no names at all.
            if cue_mode not in ("direct",):
                continue
            pair = DIRECT_PSEUDO_PAIR
        key = (row["family_id"], pair, row["context_level"], arm)
        buckets.setdefault(key, []).append(float(row["token_log_odds"]))
    return buckets


def identity_cells(
    rows: Iterable[dict],
    context_level: str,
    cue_mode: str = "concealed",
    prompt_form: str | None = None,
) -> list[tuple[str, str, float]]:
    """One Black-minus-White contrast per family and name pair, at one context."""
    buckets = _arm_buckets(rows, cue_mode, prompt_form)
    cells: list[tuple[str, str, float]] = []
    for family, pair in sorted({(family, pair) for family, pair, _, _ in buckets}):
        try:
            arms = {
                arm: float(np.mean(buckets[(family, pair, context_level, arm)]))
                for arm in ("black", "white")
            }
        except KeyError as exc:
            raise EstimationError(
                f"cell ({family}, {pair}) is missing {exc.args[0][3]!r} at context "
                f"{context_level!r}; the identity effect needs both arms"
            ) from exc
        cells.append((family, pair, arms["black"] - arms["white"]))
    return cells


def interaction_cells(
    rows: Iterable[dict],
    cue_mode: str = "concealed",
    realistic_level: str = "employer",
    prompt_form: str | None = None,
) -> list[tuple[str, str, float]]:
    """Form one interaction contrast per scenario family and name pair.

        [(Black - White)_realistic - (Black - White)_bare]

    `realistic_level` selects which rich level is contrasted against the shared
    bare baseline. Every rich level declared by the context factor is a valid
    argument and each is reported separately, because the levels differ in the
    property under test and a pooled figure would average across it.
    """
    buckets = _arm_buckets(rows, cue_mode, prompt_form)
    cells: list[tuple[str, str, float]] = []
    for family, pair in sorted({(family, pair) for family, pair, _, _ in buckets}):
        try:
            parts = {
                (level, arm): float(np.mean(buckets[(family, pair, level, arm)]))
                for level in (realistic_level, "bare")
                for arm in ("black", "white")
            }
        except KeyError as exc:
            raise EstimationError(
                f"cell ({family}, {pair}) is missing {exc.args[0][2:]}; the interaction "
                "needs both identity arms at both context levels"
            ) from exc
        cells.append(
            (
                family,
                pair,
                (parts[(realistic_level, "black")] - parts[(realistic_level, "white")])
                - (parts[("bare", "black")] - parts[("bare", "white")]),
            )
        )
    return cells


def clustered(
    contrasts: Iterable[tuple[str, str, float]],
    alpha: float | None = None,
    interval_level: float | None = None,
) -> Estimate:
    """Family-clustered mean of a contrast that has no name factor to cross over.

    Used for the direct-identity conditions, where the cue is a stated sentence
    rather than a draw from the name pool. The crossed estimator's name term is
    undefined there, and supplying a single pseudo-level to it would report a
    name variance of zero as though it had been estimated. Collapsing to family
    means and clustering on family states the design honestly instead: the
    estimate generalises over scenario families and over nothing else.
    """
    settings = config.study()["inference"]
    alpha = float(settings["two_sided_alpha"]) if alpha is None else alpha
    interval_level = (
        float(settings["interval_level"]) if interval_level is None else interval_level
    )

    grouped: dict[str, list[float]] = {}
    for family, _, value in contrasts:
        grouped.setdefault(family, []).append(float(value))
    if len(grouped) < 2:
        raise EstimationError(
            f"family-clustered inference needs at least two families; got {len(grouped)}"
        )

    means = np.array([float(np.mean(values)) for _, values in sorted(grouped.items())])
    family_count = len(means)
    degrees_freedom = float(family_count - 1)
    mean = float(means.mean())
    standard_error = float(means.std(ddof=1) / np.sqrt(family_count))
    statistic = mean / standard_error if standard_error > 0 else float("inf")
    half_width = (
        float(t.ppf(1.0 - (1.0 - interval_level) / 2.0, degrees_freedom)) * standard_error
    )
    return Estimate(
        mean=mean,
        standard_error=standard_error,
        degrees_freedom=degrees_freedom,
        statistic=float(statistic),
        p_value=float(2.0 * t.sf(abs(statistic), degrees_freedom)),
        interval_lower=mean - half_width,
        interval_upper=mean + half_width,
        interval_level=interval_level,
        families=family_count,
        names=0,
        components=VarianceComponents(
            family=float(means.var(ddof=1)), name=0.0, residual=0.0
        ),
    )
