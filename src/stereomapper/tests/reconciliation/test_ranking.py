from __future__ import annotations

from typing import Any

from stereomapper.reconciliation.proposals import Proposal
from stereomapper.reconciliation.ranking import RANK_NOTE, rank_proposals


def make(pid: str, **fields: Any) -> Proposal:
    base = {
        "proposal_id": pid,
        "n_aligned": 2,
        "n_vmh_entries": 2,
        "n_rhea_entries": 2,
        "stoichiometry_consistent": True,
        "n_non_assessable": 0,
        "n_SMRO_0001": 0,
        "n_SMRO_0002": 0,
    }
    base.update(fields)
    return Proposal(fields=base, evidence_rows=[], alternative_rows=[])


def order(proposals: list[Proposal]) -> list[str]:
    return [p.fields["proposal_id"] for p in rank_proposals(proposals)]


def test_min_coverage_compared_exactly() -> None:
    # 2/3 vs 3/5 on the weaker side: 2/3 > 3/5 even though floats would be close.
    a = make("RXP-a", n_aligned=2, n_vmh_entries=3, n_rhea_entries=3)
    b = make("RXP-b", n_aligned=3, n_vmh_entries=3, n_rhea_entries=5)
    assert order([b, a]) == ["RXP-a", "RXP-b"]


def test_key_order() -> None:
    base = make("RXP-0")
    more_aligned = make("RXP-1", n_aligned=4, n_vmh_entries=4, n_rhea_entries=4)
    inconsistent = make("RXP-2", stoichiometry_consistent=False)
    non_assessable = make("RXP-3", n_non_assessable=1)
    identical = make("RXP-4", n_SMRO_0001=2)
    ranked = order([base, more_aligned, inconsistent, non_assessable, identical])
    assert ranked == ["RXP-1", "RXP-4", "RXP-0", "RXP-3", "RXP-2"]


def test_tie_broken_by_proposal_id_and_ranks_are_contiguous() -> None:
    proposals = [make(f"RXP-{c}") for c in "dcba"]
    ranked = rank_proposals(proposals)
    assert [p.fields["proposal_id"] for p in ranked] == ["RXP-a", "RXP-b", "RXP-c", "RXP-d"]
    assert [p.fields["rank"] for p in ranked] == [1, 2, 3, 4]
    for p in ranked:
        assert p.fields["ranking_version"] == "rank-v1"
        assert p.fields["rank_note"] == RANK_NOTE == "review order, not a probability"


def test_low_ranked_proposals_kept() -> None:
    proposals = [make(f"RXP-{i}", n_aligned=1, n_vmh_entries=10) for i in range(50)]
    assert len(rank_proposals(proposals)) == 50
