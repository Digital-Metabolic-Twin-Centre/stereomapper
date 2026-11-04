"""One-to-one participant alignment under both orientations (research R4, align-v1).

Each side is solved separately. The objective, compared lexicographically, is:
(supporting pairs, uncertain pairs, sum of align-v1 class weights, pairs with equal
coefficients). Every participant entry is used at most once.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

from stereomapper.reconciliation.model import (
    ALIGN_V1_WEIGHTS,
    Orientation,
    PairEvidence,
    ParticipantEntry,
    Side,
    SourceReaction,
)
from stereomapper.reconciliation.normalize import sides

PairFn = Callable[[ParticipantEntry, ParticipantEntry], PairEvidence]
Score = tuple[int, int, int, int]
ZERO: Score = (0, 0, 0, 0)
MAX_ALTERNATIVES = 32


@dataclass(frozen=True)
class AlignedPair:
    vmh: ParticipantEntry
    rhea: ParticipantEntry
    evidence: PairEvidence

    @property
    def coefficient_equal(self) -> bool:
        return self.vmh.coefficient is not None and self.vmh.coefficient == self.rhea.coefficient


@dataclass
class AlignmentResult:
    orientation: Orientation
    pairs: list[AlignedPair]
    unmatched_vmh: list[ParticipantEntry]
    unmatched_rhea: list[ParticipantEntry]
    objective: Score
    exactness: str = "exact"
    ambiguous: bool = False
    alternatives: list[list[AlignedPair]] = field(default_factory=list)
    alternatives_truncated: bool = False


def _add(a: Score, b: Score) -> Score:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


def _edge_score(pair: AlignedPair) -> Score | None:
    ev = pair.evidence
    if ev.is_supporting:
        kind = (1, 0)
    elif ev.is_uncertain:
        kind = (0, 1)
    else:
        return None
    return (kind[0], kind[1], ALIGN_V1_WEIGHTS.get(ev.term_id, 0), int(pair.coefficient_equal))


def _profile(pairs: list[AlignedPair]) -> tuple[str, ...]:
    return tuple(sorted(p.evidence.term_id for p in pairs))


def _pair_order(pairs: list[AlignedPair]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((p.vmh.entry_key, p.rhea.entry_key) for p in pairs))


@dataclass
class _SideSolution:
    assignments: list[list[AlignedPair]]  # optimal assignments, deterministic order
    objective: Score
    exact: bool
    truncated: bool = False


def _solve_side(
    vmh: list[ParticipantEntry],
    rhea: list[ParticipantEntry],
    pair_fn: PairFn,
    exact_max_side: int,
) -> _SideSolution:
    edges: dict[tuple[int, int], tuple[Score, AlignedPair]] = {}
    for i, v in enumerate(vmh):
        for j, r in enumerate(rhea):
            pair = AlignedPair(v, r, pair_fn(v, r))
            score = _edge_score(pair)
            if score is not None:
                edges[(i, j)] = (score, pair)
    if not edges:
        return _SideSolution([[]], ZERO, True)

    if min(len(vmh), len(rhea)) > exact_max_side:
        return _greedy(edges)

    # Mask over the smaller side; walk the larger side.
    vmh_small = len(vmh) <= len(rhea)
    n_large = len(rhea) if vmh_small else len(vmh)
    n_small = len(vmh) if vmh_small else len(rhea)

    def edge(large: int, small: int) -> tuple[Score, AlignedPair] | None:
        return edges.get((small, large) if vmh_small else (large, small))

    memo: dict[tuple[int, int], tuple[Score, list[tuple[AlignedPair, ...]], bool]] = {}

    def best(i: int, mask: int) -> tuple[Score, list[tuple[AlignedPair, ...]], bool]:
        if i == n_large:
            return ZERO, [()], False
        key = (i, mask)
        if key in memo:
            return memo[key]
        top, top_assignments, truncated = best(i + 1, mask)
        candidates = [(top, top_assignments, truncated)]
        for j in range(n_small):
            if mask & (1 << j):
                continue
            hit = edge(i, j)
            if hit is None:
                continue
            sub_score, sub_assignments, sub_trunc = best(i + 1, mask | (1 << j))
            total = _add(hit[0], sub_score)
            candidates.append((total, [(hit[1],) + a for a in sub_assignments], sub_trunc))
        best_score = max(c[0] for c in candidates)
        merged: list[tuple[AlignedPair, ...]] = []
        any_trunc = False
        for score, assignments, trunc in candidates:
            if score != best_score:
                continue
            any_trunc = any_trunc or trunc
            merged.extend(assignments)
        if len(merged) > MAX_ALTERNATIVES:
            merged, any_trunc = merged[:MAX_ALTERNATIVES], True
        memo[key] = (best_score, merged, any_trunc)
        return memo[key]

    score, assignments, truncated = best(0, 0)
    unique = {_pair_order(list(a)): list(a) for a in assignments}
    ordered = [unique[k] for k in sorted(unique)]
    return _SideSolution(ordered, score, True, truncated)


def _greedy(edges: dict[tuple[int, int], tuple[Score, AlignedPair]]) -> _SideSolution:
    ranked = sorted(
        edges.items(),
        key=lambda kv: (
            tuple(-x for x in kv[1][0]),
            kv[1][1].vmh.entry_key,
            kv[1][1].rhea.entry_key,
        ),
    )
    used_v: set[int] = set()
    used_r: set[int] = set()
    chosen: list[AlignedPair] = []
    total = ZERO
    for (i, j), (score, pair) in ranked:
        if i in used_v or j in used_r:
            continue
        used_v.add(i)
        used_r.add(j)
        chosen.append(pair)
        total = _add(total, score)
    chosen.sort(key=lambda p: (p.vmh.entry_key, p.rhea.entry_key))
    return _SideSolution([chosen], total, False)


def _align(
    vmh: SourceReaction,
    rhea: SourceReaction,
    orientation: Orientation,
    pair_fn: PairFn,
    exact_max_side: int,
) -> AlignmentResult:
    rhea_left, rhea_right = sides(rhea, orientation)
    solutions = []
    for vmh_side, rhea_side in (
        (vmh.side_entries(Side.LEFT), rhea_left),
        (vmh.side_entries(Side.RIGHT), rhea_right),
    ):
        key = lambda e: e.entry_key  # noqa: E731
        solutions.append(
            _solve_side(
                sorted(vmh_side, key=key), sorted(rhea_side, key=key), pair_fn, exact_max_side
            )
        )
    selected = [pair for sol in solutions for pair in sol.assignments[0]]
    used_v = {p.vmh.entry_key for p in selected}
    used_r = {p.rhea.entry_key for p in selected}
    ambiguous = any(len({_profile(a) for a in sol.assignments}) > 1 for sol in solutions)
    alternatives: list[list[AlignedPair]] = []
    truncated = any(sol.truncated for sol in solutions)
    if ambiguous:
        for combo in itertools.islice(
            itertools.product(*(sol.assignments for sol in solutions)), MAX_ALTERNATIVES + 1
        ):
            alternatives.append([pair for part in combo for pair in part])
        if len(alternatives) > MAX_ALTERNATIVES:
            alternatives, truncated = alternatives[:MAX_ALTERNATIVES], True
    return AlignmentResult(
        orientation=orientation,
        pairs=selected,
        unmatched_vmh=[e for e in vmh.participants if e.entry_key not in used_v],
        unmatched_rhea=[e for e in rhea.participants if e.entry_key not in used_r],
        objective=_add(solutions[0].objective, solutions[1].objective),
        exactness="exact" if all(sol.exact for sol in solutions) else "greedy",
        ambiguous=ambiguous,
        alternatives=alternatives,
        alternatives_truncated=truncated,
    )


def align_pair(
    vmh: SourceReaction, rhea: SourceReaction, pair_fn: PairFn, exact_max_side: int
) -> dict[Orientation, AlignmentResult]:
    return {o: _align(vmh, rhea, o, pair_fn, exact_max_side) for o in Orientation}


def select_orientation(results: dict[Orientation, AlignmentResult]) -> Orientation:
    """Best objective; ties go to ``same``."""
    same, rev = results[Orientation.SAME], results[Orientation.REVERSED]
    return Orientation.REVERSED if rev.objective > same.objective else Orientation.SAME
