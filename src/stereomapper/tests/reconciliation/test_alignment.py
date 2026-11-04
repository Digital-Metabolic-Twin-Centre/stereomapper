from __future__ import annotations

import random
from collections.abc import Sequence
from fractions import Fraction
from typing import Union

from stereomapper.reconciliation.alignment import AlignmentResult, PairFn, align_pair
from stereomapper.reconciliation.model import (
    EvidenceType,
    Orientation,
    PairEvidence,
    ParticipantEntry,
    ParticipantKind,
    Side,
    SourceReaction,
)

Spec = Union[str, tuple[str, str]]

LABELS = {
    "SMRO:0001": "Identical structures",
    "SMRO:0003": "Protomers",
    "SMRO:0005": "Enantiomers",
    "SMRO:0006": "Diastereomers",
    "SMRO:0007": "Stereo-resolution pairs",
    "SMRO:0009": "Unresolved",
}


def entry(source: str, rid: str, side: str, met: str, coef: str = "1") -> ParticipantEntry:
    return ParticipantEntry(
        source=source,
        reaction_id=rid,
        side=Side(side),
        metabolite_id=met,
        name=met,
        compartment="",
        coefficient_raw=coef,
        coefficient=Fraction(coef),
        kind=ParticipantKind.SMALL_MOLECULE,
        structure_curie=f"x:{met}",
        formula="",
        charge="",
        source_row=0,
    )


def reaction(source: str, rid: str, left: Sequence[Spec], right: Sequence[Spec]) -> SourceReaction:
    r = SourceReaction(source=source, reaction_id=rid, equation="", source_row=0)
    for spec in left:
        met, coef = spec if isinstance(spec, tuple) else (spec, "1")
        r.participants.append(entry(source, rid, "left", met, coef))
    for spec in right:
        met, coef = spec if isinstance(spec, tuple) else (spec, "1")
        r.participants.append(entry(source, rid, "right", met, coef))
    return r


def pair_fn(edges: dict[tuple[str, str], str]) -> PairFn:
    def fn(v: ParticipantEntry, r: ParticipantEntry) -> PairEvidence:
        term = edges.get((v.metabolite_id, r.metabolite_id))
        if not term:
            return PairEvidence(EvidenceType.NONE)
        return PairEvidence(EvidenceType.RELATIONSHIP, LABELS[term], term)

    return fn


def pairs(result: AlignmentResult) -> set[tuple[str, str]]:
    return {(p.vmh.metabolite_id, p.rhea.metabolite_id) for p in result.pairs}


def test_one_entry_is_never_reused() -> None:
    v = reaction("vmh", "V", [("A", "2")], [])
    r = reaction("rhea", "R", ["A1", "A2"], [])
    res = align_pair(v, r, pair_fn({("A", "A1"): "SMRO:0001", ("A", "A2"): "SMRO:0001"}), 16)
    same = res[Orientation.SAME]
    assert len(same.pairs) == 1
    assert len(same.unmatched_rhea) == 1
    assert not same.ambiguous  # equal class profiles are not ambiguous


def test_supporting_edge_beats_uncertain_edge() -> None:
    v = reaction("vmh", "V", ["X"], [])
    r = reaction("rhea", "R", ["U", "E"], [])
    res = align_pair(v, r, pair_fn({("X", "U"): "SMRO:0009", ("X", "E"): "SMRO:0005"}), 16)
    assert pairs(res[Orientation.SAME]) == {("X", "E")}


def test_class_order_breaks_ties() -> None:
    v = reaction("vmh", "V", ["X"], [])
    r = reaction("rhea", "R", ["E", "I"], [])
    res = align_pair(v, r, pair_fn({("X", "E"): "SMRO:0005", ("X", "I"): "SMRO:0001"}), 16)
    assert pairs(res[Orientation.SAME]) == {("X", "I")}


def test_more_supporting_pairs_beat_heavier_single_pair() -> None:
    v = reaction("vmh", "V", ["X1", "X2"], [])
    r = reaction("rhea", "R", ["Y1", "Y2"], [])
    edges = {("X1", "Y1"): "SMRO:0001", ("X1", "Y2"): "SMRO:0005", ("X2", "Y1"): "SMRO:0005"}
    res = align_pair(v, r, pair_fn(edges), 16)
    assert pairs(res[Orientation.SAME]) == {("X1", "Y2"), ("X2", "Y1")}


def test_equal_coefficient_preferred() -> None:
    v = reaction("vmh", "V", [("X", "2")], [])
    r = reaction("rhea", "R", [("Y1", "1"), ("Y2", "2")], [])
    res = align_pair(v, r, pair_fn({("X", "Y1"): "SMRO:0001", ("X", "Y2"): "SMRO:0001"}), 16)
    assert pairs(res[Orientation.SAME]) == {("X", "Y2")}


def test_tied_assignments_with_different_profiles_are_ambiguous() -> None:
    v = reaction("vmh", "V", ["X1", "X2"], [])
    r = reaction("rhea", "R", ["Y1", "Y2"], [])
    edges = {
        ("X1", "Y1"): "SMRO:0006",  # 3
        ("X2", "Y2"): "SMRO:0007",  # 4
        ("X1", "Y2"): "SMRO:0003",  # 5
        ("X2", "Y1"): "SMRO:0005",  # 2
    }
    same = align_pair(v, r, pair_fn(edges), 16)[Orientation.SAME]
    assert same.ambiguous
    assert len(same.alternatives) == 2
    profiles = {tuple(sorted(p.evidence.term_id for p in alt)) for alt in same.alternatives}
    assert len(profiles) == 2


def test_reversed_orientation_swaps_sides() -> None:
    v = reaction("vmh", "V", ["A"], ["B"])
    r = reaction("rhea", "R", ["B2"], ["A2"])
    res = align_pair(v, r, pair_fn({("A", "A2"): "SMRO:0001", ("B", "B2"): "SMRO:0001"}), 16)
    assert pairs(res[Orientation.SAME]) == set()
    assert pairs(res[Orientation.REVERSED]) == {("A", "A2"), ("B", "B2")}


def test_greedy_fallback_is_flagged_and_deterministic() -> None:
    mets = ["M1", "M2", "M3"]
    edges = {(m, m + "r"): "SMRO:0001" for m in mets}
    edges[("M1", "M2r")] = "SMRO:0005"
    expected = None
    for seed in range(5):
        left_v = mets[:]
        left_r = [m + "r" for m in mets]
        random.Random(seed).shuffle(left_v)
        random.Random(seed + 100).shuffle(left_r)
        v = reaction("vmh", "V", left_v, [])
        r = reaction("rhea", "R", left_r, [])
        same = align_pair(v, r, pair_fn(edges), 2)[Orientation.SAME]
        assert same.exactness == "greedy"
        if expected is None:
            expected = pairs(same)
        assert pairs(same) == expected
    assert expected == {("M1", "M1r"), ("M2", "M2r"), ("M3", "M3r")}
