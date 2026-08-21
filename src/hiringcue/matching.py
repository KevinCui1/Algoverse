"""Name-pair construction by matching the two arms on attribution accuracy.

*Attribution accuracy* is the share of survey respondents who assigned a name
the group it was written to signal. It is the strength of the cue the stimulus
delivers, and the two arms of this design do not share its distribution: in the
source data the White-associated roster spans 0.61-0.93 with a mean of 0.82,
while the Black-associated roster spans 0.23-0.89 with a mean of 0.65.

Applying one threshold to both arms therefore selects them differently. A 0.75
floor retains most of the White-associated roster near its centre and only the
upper tail of the Black-associated one, so the two arms differ in cue strength
as well as in the group signalled, and any measured contrast confounds the two.
Comparability across arms is the property the design needs; a high absolute
floor is a different property that does not deliver it.

The pool is built instead as a set of *pairs*, each holding one name from each
arm whose attribution accuracies differ by no more than a caliper. Pairs are
chosen to maximise the number retained and, among the matchings that achieve
that number, to minimise the total within-pair difference. The retained floor is
whatever the matching implies rather than a value chosen in advance.

Balance is reported and checked two ways, because a caliper and an arm-level
summary constrain different things: the caliper bounds the difference inside
each pair, and the standardised mean difference bounds the difference between
the arms taken as wholes. A pool passes only if both hold.

Inputs are the per-name attribution accuracies of the two arms. Outputs are the
matched pairs and the balance statistics. Nothing here reads a model, and the
module raises rather than returning an unbalanced pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp

WHITE = "white"
BLACK = "black"


class MatchingError(ValueError):
    """Raised when no caliper on the ladder produces a balanced pool."""


@dataclass(frozen=True)
class Candidate:
    """One roster name and the accuracy with which it is attributed."""

    stimulus_id: str
    full_name: str
    group: str
    attribution_accuracy: float


@dataclass(frozen=True)
class Pair:
    pair_id: str
    white: Candidate
    black: Candidate

    @property
    def accuracy_difference(self) -> float:
        return self.white.attribution_accuracy - self.black.attribution_accuracy


@dataclass(frozen=True)
class MatchedPool:
    pairs: list[Pair]
    caliper: float
    balance: dict[str, float]

    @property
    def floor(self) -> float:
        return min(
            min(pair.white.attribution_accuracy, pair.black.attribution_accuracy)
            for pair in self.pairs
        )


def _standardised_mean_difference(white: np.ndarray, black: np.ndarray) -> float:
    """Arm mean difference in units of the pooled spread of the matched pool.

    Standardised rather than raw so the criterion does not depend on the scale
    of the accuracy measure, which is what makes a single conventional bound
    (0.10) meaningful across rebuilds of the pool.
    """
    spread = float(np.std(np.concatenate([white, black])))
    if spread == 0.0:
        return 0.0
    return float(white.mean() - black.mean()) / spread


def match(
    candidates: Iterable[Candidate], caliper: float
) -> list[Pair]:
    """Pair the two arms under a caliper, maximising pairs then minimising distance.

    The assignment is solved rather than taken greedily. A greedy sweep from
    either end of the accuracy range is optimal in cardinality but biases the
    retained pool towards that end, and which end it sweeps from is an arbitrary
    choice that moves the retained floor by several points of accuracy.
    """
    if caliper <= 0:
        raise MatchingError(f"caliper must be positive, got {caliper}")

    pool = list(candidates)
    white = sorted(
        (item for item in pool if item.group == WHITE),
        key=lambda item: (item.attribution_accuracy, item.stimulus_id),
    )
    black = sorted(
        (item for item in pool if item.group == BLACK),
        key=lambda item: (item.attribution_accuracy, item.stimulus_id),
    )
    if not white or not black:
        raise MatchingError(
            f"both arms must be non-empty; got {len(white)} white and {len(black)} black"
        )

    distance = np.abs(
        np.array([item.attribution_accuracy for item in white])[:, None]
        - np.array([item.attribution_accuracy for item in black])[None, :]
    )
    # Forbidden cells are given a cost far above any admissible total, so the
    # solver spends them only when no admissible assignment exists for that row.
    # They are then discarded, which is what makes the result a maximum-
    # cardinality matching rather than a full one-to-one assignment.
    forbidden = float(distance.size) * (float(distance.max()) + 1.0) + 1.0
    cost = np.where(distance <= caliper, distance, forbidden)

    rows, columns = linear_sum_assignment(cost)
    selected = [
        (row, column)
        for row, column in zip(rows, columns)
        if distance[row, column] <= caliper
    ]
    selected.sort(key=lambda item: distance[item[0], item[1]])

    return [
        Pair(
            pair_id=f"pair_{index:03d}",
            white=white[row],
            black=black[column],
        )
        for index, (row, column) in enumerate(selected)
    ]


def balance(pairs: Sequence[Pair]) -> dict[str, float]:
    white = np.array([pair.white.attribution_accuracy for pair in pairs])
    black = np.array([pair.black.attribution_accuracy for pair in pairs])
    differences = np.abs(white - black)
    return {
        "pairs": len(pairs),
        "white_mean": float(white.mean()),
        "black_mean": float(black.mean()),
        "standardised_mean_difference": _standardised_mean_difference(white, black),
        "maximum_within_pair_difference": float(differences.max()),
        "mean_within_pair_difference": float(differences.mean()),
        "kolmogorov_smirnov_p": float(ks_2samp(white, black).pvalue),
        "floor": float(min(white.min(), black.min())),
    }


def build_pool(
    candidates: Iterable[Candidate],
    caliper_ladder: Sequence[float],
    minimum_pairs: int,
    maximum_standardised_mean_difference: float,
) -> MatchedPool:
    """Select the caliper that balances the arms best while supplying enough pairs.

    The ladder is walked from tightest to loosest and the first caliper meeting
    the pair requirement wins. Tightest-first rather than largest-admissible is
    deliberate: a looser caliper retains more names, but the pair count is a
    requirement to be met rather than a quantity to be maximised, and every
    additional pair bought by loosening is bought with arm imbalance, which is
    the defect the rebuild exists to remove.
    """
    pool = list(candidates)
    attempts: list[dict[str, float]] = []
    for caliper in sorted(caliper_ladder):
        pairs = match(pool, caliper)
        if not pairs:
            attempts.append({"caliper": caliper, "pairs": 0})
            continue
        statistics = balance(pairs)
        attempts.append({"caliper": caliper, **statistics})
        if len(pairs) < minimum_pairs:
            continue
        if (
            abs(statistics["standardised_mean_difference"])
            > maximum_standardised_mean_difference
        ):
            continue
        return MatchedPool(pairs=pairs, caliper=caliper, balance=statistics)

    best = max((entry.get("pairs", 0) for entry in attempts), default=0)
    raise MatchingError(
        f"no caliper on the ladder {sorted(caliper_ladder)} yields {minimum_pairs} "
        f"pairs within a standardised mean difference of "
        f"{maximum_standardised_mean_difference}; the best attempt retained {best} "
        "pairs. Report the shortfall and re-run the sizing rule against the "
        "achievable pair count rather than relaxing either bound."
    )


def split(
    pool: MatchedPool, development_pairs: int, seed: int
) -> tuple[list[Pair], list[Pair]]:
    """Reserve development pairs and leave the rest for confirmation.

    The split is random against a recorded seed rather than by accuracy rank, so
    the development pairs are not systematically the strongest or the weakest
    cues in the pool. Development and confirmatory pools must stay disjoint for
    the whole study, so this assignment is made once and frozen.
    """
    if development_pairs >= len(pool.pairs):
        raise MatchingError(
            f"{development_pairs} development pairs requested from a pool of "
            f"{len(pool.pairs)}; no pairs would remain for confirmation"
        )
    order = np.random.default_rng(seed).permutation(len(pool.pairs))
    reserved = sorted(order[:development_pairs])
    held = set(reserved)
    development = [pool.pairs[index] for index in reserved]
    confirmatory = [
        pair for index, pair in enumerate(pool.pairs) if index not in held
    ]
    return development, confirmatory
