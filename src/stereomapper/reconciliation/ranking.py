"""rank-v1: a reproducible review order (research R11). Not a probability of equivalence."""

from __future__ import annotations

from fractions import Fraction

from stereomapper.reconciliation.config import RANK_V1
from stereomapper.reconciliation.proposals import Proposal

RANK_NOTE = "review order, not a probability"


def _key(p: Proposal) -> tuple[Fraction, int, int, int, int, str]:
    f = p.fields
    n_aligned = int(f["n_aligned"])
    coverage = min(
        Fraction(n_aligned, max(int(f["n_vmh_entries"]), 1)),
        Fraction(n_aligned, max(int(f["n_rhea_entries"]), 1)),
    )
    identical = int(f.get("n_SMRO_0001", 0)) + int(f.get("n_SMRO_0002", 0))
    return (
        -coverage,
        -n_aligned,
        0 if f["stoichiometry_consistent"] else 1,
        int(f["n_non_assessable"]),
        -identical,
        f["proposal_id"],
    )


def rank_proposals(proposals: list[Proposal]) -> list[Proposal]:
    ranked = sorted(proposals, key=_key)
    for i, proposal in enumerate(ranked, start=1):
        proposal.fields["rank"] = i
        proposal.fields["ranking_version"] = RANK_V1
        proposal.fields["rank_note"] = RANK_NOTE
    return ranked
