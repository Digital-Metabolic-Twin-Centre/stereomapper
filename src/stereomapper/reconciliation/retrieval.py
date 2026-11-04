"""Candidate retrieval through participant evidence (research R5, retrieve-v1).

Routes: ``shared_cluster`` (a Rhea participant in the same cluster), ``related_cluster`` (a
StereoMapper relationship of any class), and ``xref`` (existing cross-references). There is no
cap. Cofactor clusters are not used as retrieval keys: a plausible proposal needs a supporting
non-cofactor pair (FR-017), which always shares or relates a non-cofactor cluster, so this cannot
omit a plausible candidate. Cofactor-only candidates can still arrive by ``xref``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from stereomapper.reconciliation.evidence import CofactorIndex, StereoMapperEvidence
from stereomapper.reconciliation.model import SourceReaction, StructureState
from stereomapper.reconciliation.normalize import Reactions


class RheaIndex:
    def __init__(
        self,
        rhea_in_scope: dict[str, SourceReaction],
        evidence: StereoMapperEvidence,
        cofactors: CofactorIndex,
    ):
        self._by_cluster: dict[int, set[str]] = {}
        for rid, reaction in rhea_in_scope.items():
            for entry in reaction.participants:
                if evidence.structure_state(entry) != StructureState.ASSESSABLE:
                    continue
                if cofactors.is_cofactor(entry):
                    continue
                for cid in evidence.clusters_for(entry):
                    self._by_cluster.setdefault(cid, set()).add(rid)
        self.in_scope = set(rhea_in_scope)

    def masters_with_cluster(self, cluster_id: int) -> set[str]:
        return self._by_cluster.get(cluster_id, set())


def retrieve(
    vmh: SourceReaction,
    index: RheaIndex,
    evidence: StereoMapperEvidence,
    cofactors: CofactorIndex,
    reactions: Reactions,
    routes: tuple[str, ...],
) -> dict[str, set[str]]:
    """Map Rhea master ID -> routes that found it, for one VMH reaction."""
    found: dict[str, set[str]] = {}
    clusters: set[int] = set()
    for entry in vmh.participants:
        if evidence.structure_state(entry) != StructureState.ASSESSABLE:
            continue
        if cofactors.is_cofactor(entry):
            continue
        clusters.update(evidence.clusters_for(entry))
    if "shared_cluster" in routes:
        for cid in clusters:
            for rid in index.masters_with_cluster(cid):
                found.setdefault(rid, set()).add("shared_cluster")
    if "related_cluster" in routes:
        for cid in clusters:
            for other in evidence.related_clusters(cid):
                for rid in index.masters_with_cluster(other):
                    found.setdefault(rid, set()).add("related_cluster")
    if "xref" in routes:
        for xref in vmh.xref_rhea:
            master = reactions.master_for(xref)
            if master is not None and master in index.in_scope:
                found.setdefault(master, set()).add("xref")
    return found


FUNNEL_STAGES = (
    "vmh_reactions_read",
    "rhea_reactions_read",
    "rhea_excluded_obsolete",
    "vmh_in_scope",
    "rhea_in_scope",
    "vmh_with_candidates",
    "candidates_shared_cluster",
    "candidates_related_cluster",
    "candidates_xref",
    "candidates_total",
    "candidates_aligned",
    "candidates_plausible",
    "not_plausible_cofactor_only",
    "not_plausible_no_supporting_pair",
    "proposals",
    "vmh_reactions_with_proposals",
    "proposals_greedy_alignment",
)

_DENOMINATORS = {
    "rhea_excluded_obsolete": "rhea_reactions_read",
    "vmh_in_scope": "vmh_reactions_read",
    "rhea_in_scope": "rhea_reactions_read",
    "vmh_with_candidates": "vmh_in_scope",
    "candidates_shared_cluster": "candidates_total",
    "candidates_related_cluster": "candidates_total",
    "candidates_xref": "candidates_total",
    "candidates_aligned": "candidates_total",
    "candidates_plausible": "candidates_total",
    "not_plausible_cofactor_only": "candidates_total",
    "not_plausible_no_supporting_pair": "candidates_total",
    "proposals": "candidates_total",
    "vmh_reactions_with_proposals": "vmh_in_scope",
    "proposals_greedy_alignment": "proposals",
}

_NOTES = {
    "rhea_reactions_read": "Rhea master reactions (directional variants grouped)",
    "rhea_excluded_obsolete": "see scope_exclusions.tsv",
    "vmh_in_scope": "no pathway or subsystem exclusion (FR-005)",
    "candidates_shared_cluster": "routes overlap; a candidate may be found by several",
    "candidates_related_cluster": "any StereoMapper relationship class",
    "candidates_total": "retrieval uses non-cofactor clusters; no cap (retrieve-v1)",
    "not_plausible_cofactor_only": "only cofactor pairs aligned (FR-017)",
    "proposals_greedy_alignment": "a side exceeded alignment.exact_max_side",
}


@dataclass
class Funnel:
    counts: OrderedDict[str, int] = field(
        default_factory=lambda: OrderedDict((s, 0) for s in FUNNEL_STAGES)
    )

    def add(self, stage: str, n: int = 1) -> None:
        self.counts[stage] += n

    def set(self, stage: str, n: int) -> None:
        self.counts[stage] = n

    def rows(self) -> list[dict[str, object]]:
        out = []
        for stage, count in self.counts.items():
            denominator: int | None = None
            if stage in _DENOMINATORS:
                denominator = self.counts[_DENOMINATORS[stage]]
            out.append(
                {
                    "stage": stage,
                    "count": count,
                    "denominator": denominator,
                    "note": _NOTES.get(stage, ""),
                }
            )
        return out
