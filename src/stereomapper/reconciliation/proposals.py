"""Assemble reaction proposals and their participant evidence (FR-008 to FR-021)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stereomapper.reconciliation.alignment import AlignedPair, AlignmentResult, select_orientation
from stereomapper.reconciliation.evidence import CofactorIndex, StereoMapperEvidence
from stereomapper.reconciliation.hashing import canonical_json, proposal_id, sha256_text
from stereomapper.reconciliation.model import (
    ALL_TERMS,
    PROTOMER_TERMS,
    SPEC_STATE,
    STEREO_TERMS,
    AlignmentState,
    Orientation,
    ParticipantEntry,
    RheaStatus,
    Side,
    SourceReaction,
    StructureState,
)
from stereomapper.reconciliation.normalize import (
    directional_kind,
    expected_directional_kind,
    is_transport_rhea,
    is_transport_vmh,
    normalize_rhea_id,
)
from stereomapper.reconciliation.outputs import fraction_columns

TERM_COLUMNS = [f"n_{t.replace(':', '_')}" for t in ALL_TERMS]
FLAG_COLUMNS = [
    "proton_balance_difference",
    "has_stereo_relationship",
    "has_protomer",
    "has_uncertain",
    "has_not_assessable",
    "has_unavailable_structure",
    "ambiguous_alignment",
    "alignment_exactness",
    "rhea_preliminary",
    "vmh_transport",
    "rhea_transport",
    "compartment_difference",
    "direction_difference",
    "direction_note",
]
PROPOSAL_COLUMNS = [
    "rank",
    "proposal_id",
    "evidence_hash",
    "vmh_reaction_id",
    "rhea_master_id",
    "rhea_source_ids",
    "rhea_matching_directional_id",
    "vmh_equation",
    "rhea_equation",
    "selected_orientation",
    "coverage_vmh",
    "coverage_vmh_decimal",
    "coverage_rhea",
    "coverage_rhea_decimal",
    "n_aligned",
    "n_aligned_uncertain",
    "n_unmatched_vmh",
    "n_unmatched_rhea",
    *TERM_COLUMNS,
    "stoichiometry_consistent",
    *FLAG_COLUMNS,
    "in_strict_baseline",
    "relationship_profile",
    "ranking_version",
    "rank_note",
    # Additional fields (data-model.md ReactionProposal)
    "n_vmh_entries",
    "n_rhea_entries",
    "n_assessable_vmh",
    "n_assessable_rhea",
    "n_non_assessable",
    "vmh_direction",
    "retrieval_routes",
    "vmh_subsystems",
    "run_id",
]
EVIDENCE_COLUMNS = [
    "proposal_id",
    "orientation",
    "side",
    "vmh_entry_key",
    "rhea_entry_key",
    "alignment_state",
    "spec_state",
    "evidence_type",
    "relationship_label",
    "smro_term_id",
    "relationship_direction",
    "score",
    "coefficient_vmh",
    "coefficient_rhea",
    "coefficient_equal",
    "formula_agreement",
    "vmh_reaction_id",
    "vmh_metabolite_id",
    "vmh_name",
    "vmh_compartment",
    "vmh_kind",
    "vmh_structure_curie",
    "vmh_structure_state",
    "vmh_structure_error",
    "vmh_is_cofactor",
    "rhea_reaction_id",
    "rhea_metabolite_id",
    "rhea_name",
    "rhea_compartment",
    "rhea_kind",
    "rhea_structure_curie",
    "rhea_structure_state",
    "rhea_structure_error",
    "rhea_is_cofactor",
    "vmh_source_row",
    "rhea_source_row",
    "sm_cluster_ids",
    "sm_version_tag",
]
ALTERNATIVE_COLUMNS = [
    "proposal_id",
    "alternative_group",
    "orientation",
    "vmh_entry_key",
    "rhea_entry_key",
    "relationship_label",
    "smro_term_id",
]
# Fields excluded from the evidence hash: rank depends on other proposals; run_id and
# subsystems are not evidence (subsystem flags are applied after the run, R16).
_NOT_HASHED = {"rank", "ranking_version", "rank_note", "run_id", "evidence_hash", "vmh_subsystems"}


@dataclass
class Proposal:
    fields: dict[str, Any]
    evidence_rows: list[dict[str, Any]]
    alternative_rows: list[dict[str, Any]]


@dataclass
class CandidateOutcome:
    proposal: Proposal | None
    plausible: bool
    reason: str  # "", "cofactor_only", or "no_supporting_pair"
    exactness: str


class ProposalBuilder:
    def __init__(
        self,
        evidence: StereoMapperEvidence,
        cofactors: CofactorIndex,
        baseline_pairs: set[tuple[str, str]],
        run_id: str,
    ):
        self.evidence = evidence
        self.cofactors = cofactors
        self.baseline_pairs = baseline_pairs
        self.run_id = run_id

    # -- per-entry helpers ---------------------------------------------------------------

    def _state(self, entry: ParticipantEntry) -> StructureState:
        return self.evidence.structure_state(entry)

    def _entry_columns(self, prefix: str, entry: ParticipantEntry | None) -> dict[str, Any]:
        if entry is None:
            return {}
        return {
            f"{prefix}_entry_key": entry.entry_key,
            f"{prefix}_reaction_id": entry.reaction_id,
            f"{prefix}_metabolite_id": entry.metabolite_id,
            f"{prefix}_name": entry.name,
            f"{prefix}_compartment": entry.compartment,
            f"{prefix}_kind": entry.kind,
            f"{prefix}_structure_curie": entry.structure_curie,
            f"{prefix}_structure_state": self._state(entry),
            f"{prefix}_structure_error": self.evidence.structure_error(entry),
            f"{prefix}_is_cofactor": self.cofactors.is_cofactor(entry),
            f"{prefix}_source_row": entry.source_row,
            f"coefficient_{prefix}": entry.coefficient_raw,
        }

    def _unmatched_state(self, entry: ParticipantEntry) -> AlignmentState:
        if self._state(entry) == StructureState.ASSESSABLE:
            return AlignmentState.SUPPORTED_NONMATCH
        return AlignmentState.UNMATCHED

    def _evidence_rows(
        self, pid: str, vmh: SourceReaction, result: AlignmentResult
    ) -> list[dict[str, Any]]:
        orientation = result.orientation
        rows: list[dict[str, Any]] = []
        for pair in result.pairs:
            alignment = (
                AlignmentState.ALIGNED
                if pair.evidence.is_supporting
                else AlignmentState.ALIGNED_UNCERTAIN
            )
            ev = pair.evidence
            clusters = [self.evidence.cluster_for(pair.vmh), self.evidence.cluster_for(pair.rhea)]
            rows.append(
                {
                    "proposal_id": pid,
                    "orientation": orientation,
                    "side": pair.vmh.side,
                    **self._entry_columns("vmh", pair.vmh),
                    **self._entry_columns("rhea", pair.rhea),
                    "alignment_state": alignment,
                    "spec_state": SPEC_STATE[(StructureState.ASSESSABLE, alignment)],
                    "evidence_type": ev.evidence_type,
                    "relationship_label": ev.label,
                    "smro_term_id": ev.term_id,
                    "relationship_direction": ev.direction,
                    "score": ev.score,
                    "coefficient_equal": pair.coefficient_equal,
                    "formula_agreement": bool(pair.vmh.formula)
                    and pair.vmh.formula == pair.rhea.formula,
                    "sm_cluster_ids": ";".join(str(c) for c in clusters if c is not None),
                    "sm_version_tag": ev.version_tag,
                }
            )
        rhea_side_for_vmh = {
            Side.LEFT: Side.LEFT if orientation == Orientation.SAME else Side.RIGHT,
            Side.RIGHT: Side.RIGHT if orientation == Orientation.SAME else Side.LEFT,
        }
        vmh_side_for_rhea = {v: k for k, v in rhea_side_for_vmh.items()}
        unmatched_formulas: dict[Side, set[str]] = {Side.LEFT: set(), Side.RIGHT: set()}
        for entry in result.unmatched_rhea:
            if entry.formula:
                unmatched_formulas[vmh_side_for_rhea[entry.side]].add(entry.formula)
        vmh_formulas: dict[Side, set[str]] = {Side.LEFT: set(), Side.RIGHT: set()}
        for entry in result.unmatched_vmh:
            if entry.formula:
                vmh_formulas[entry.side].add(entry.formula)
        for entry in result.unmatched_vmh:
            state = self._unmatched_state(entry)
            cluster = self.evidence.cluster_for(entry)
            rows.append(
                {
                    "proposal_id": pid,
                    "orientation": orientation,
                    "side": entry.side,
                    **self._entry_columns("vmh", entry),
                    "alignment_state": state,
                    "spec_state": SPEC_STATE[(self._state(entry), state)],
                    "evidence_type": "none",
                    "formula_agreement": bool(entry.formula)
                    and entry.formula in unmatched_formulas[entry.side],
                    "sm_cluster_ids": "" if cluster is None else str(cluster),
                }
            )
        for entry in result.unmatched_rhea:
            state = self._unmatched_state(entry)
            cluster = self.evidence.cluster_for(entry)
            side = vmh_side_for_rhea[entry.side]
            rows.append(
                {
                    "proposal_id": pid,
                    "orientation": orientation,
                    "side": side,
                    **self._entry_columns("rhea", entry),
                    "alignment_state": state,
                    "spec_state": SPEC_STATE[(self._state(entry), state)],
                    "evidence_type": "none",
                    "formula_agreement": bool(entry.formula)
                    and entry.formula in vmh_formulas[side],
                    "sm_cluster_ids": "" if cluster is None else str(cluster),
                }
            )
        rows.sort(
            key=lambda r: (
                r["orientation"].value,
                r["side"].value,
                r.get("vmh_entry_key") or "~",
                r.get("rhea_entry_key") or "",
            )
        )
        return rows

    # -- proposal ------------------------------------------------------------------------

    def _direction(
        self, vmh: SourceReaction, rhea: SourceReaction, orientation: Orientation
    ) -> tuple[str, bool, str]:
        expected = expected_directional_kind(vmh.vmh_direction, orientation)
        kinds = {directional_kind(rhea.reaction_id, d): d for d in rhea.rhea_directional_ids}
        matching = kinds.get(expected, "") if expected else ""
        notes: list[str] = []
        difference = False
        if expected and kinds and not matching:  # R8 case (a)
            difference = True
            notes.append(f"Rhea {expected} directional ID missing from extract")
        if not rhea.rhea_directional_ids:
            notes.append("no Rhea directional IDs in extract")
        opposite = {"LR": "RL", "RL": "LR"}.get(expected, "")
        master_norm = normalize_rhea_id(rhea.reaction_id)
        for xref in vmh.xref_rhea:  # R8 case (b)
            kind = directional_kind(master_norm, normalize_rhea_id(xref))
            if opposite and kind == opposite:
                difference = True
                notes.append(f"VMH xref {xref} is {kind}, opposite to VMH direction ({expected})")
        return matching, difference, "; ".join(notes)

    def build(
        self,
        vmh: SourceReaction,
        rhea: SourceReaction,
        results: dict[Orientation, AlignmentResult],
        routes: set[str],
    ) -> CandidateOutcome:
        orientation = select_orientation(results)
        result = results[orientation]
        exactness = "greedy" if any(r.exactness == "greedy" for r in results.values()) else "exact"
        supporting = [p for p in result.pairs if p.evidence.is_supporting]
        if not supporting:
            return CandidateOutcome(None, False, "no_supporting_pair", exactness)
        if all(
            self.cofactors.is_cofactor(p.vmh) or self.cofactors.is_cofactor(p.rhea)
            for p in supporting
        ):
            return CandidateOutcome(None, False, "cofactor_only", exactness)

        pid = proposal_id(vmh.reaction_id, rhea.reaction_id)
        uncertain = [p for p in result.pairs if p.evidence.is_uncertain]
        n_vmh, n_rhea = len(vmh.participants), len(rhea.participants)
        n_aligned = len(supporting)
        terms = [p.evidence.term_id for p in result.pairs]
        states = [self._state(e) for e in vmh.participants + rhea.participants]
        unavailable = {
            StructureState.STRUCTURE_UNAVAILABLE,
            StructureState.PROCESSING_FAILED,
            StructureState.NO_RESULT,
        }
        stoich_ok = (
            all(p.coefficient_equal for p in result.pairs)
            and not result.unmatched_vmh
            and not result.unmatched_rhea
        )
        proton_only = not stoich_ok and (
            all(
                p.coefficient_equal
                or (self.cofactors.is_proton(p.vmh) and self.cofactors.is_proton(p.rhea))
                for p in result.pairs
            )
            and all(self.cofactors.is_proton(e) for e in result.unmatched_vmh)
            and all(self.cofactors.is_proton(e) for e in result.unmatched_rhea)
        )
        has_stereo = any(t in STEREO_TERMS for t in terms)
        has_protomer = any(t in PROTOMER_TERMS for t in terms)
        partial = bool(result.unmatched_vmh or result.unmatched_rhea)
        if uncertain:
            profile = "includes_uncertain"
        elif has_stereo:
            profile = "includes_stereo"
        elif has_protomer:
            profile = "includes_protomer"
        elif partial:
            profile = "partial"
        else:
            profile = "identical_only"
        matching, direction_difference, direction_note = self._direction(vmh, rhea, orientation)
        vmh_transport, rhea_transport = is_transport_vmh(vmh), is_transport_rhea(rhea)
        cov_vmh, cov_vmh_dec = fraction_columns(n_aligned, n_vmh)
        cov_rhea, cov_rhea_dec = fraction_columns(n_aligned, n_rhea)

        fields: dict[str, Any] = {
            "proposal_id": pid,
            "vmh_reaction_id": vmh.reaction_id,
            "rhea_master_id": rhea.reaction_id,
            "rhea_source_ids": (rhea.reaction_id,) + rhea.rhea_directional_ids,
            "rhea_matching_directional_id": matching,
            "vmh_equation": vmh.equation,
            "rhea_equation": rhea.equation,
            "selected_orientation": orientation,
            "coverage_vmh": cov_vmh,
            "coverage_vmh_decimal": cov_vmh_dec,
            "coverage_rhea": cov_rhea,
            "coverage_rhea_decimal": cov_rhea_dec,
            "n_aligned": n_aligned,
            "n_aligned_uncertain": len(uncertain),
            "n_unmatched_vmh": len(result.unmatched_vmh),
            "n_unmatched_rhea": len(result.unmatched_rhea),
            **{col: terms.count(term) for col, term in zip(TERM_COLUMNS, ALL_TERMS)},
            "stoichiometry_consistent": stoich_ok,
            "proton_balance_difference": proton_only,
            "has_stereo_relationship": has_stereo,
            "has_protomer": has_protomer,
            "has_uncertain": bool(uncertain),
            "has_not_assessable": StructureState.NOT_ASSESSABLE in states,
            "has_unavailable_structure": any(s in unavailable for s in states),
            "ambiguous_alignment": result.ambiguous,
            "alignment_exactness": exactness,
            "rhea_preliminary": rhea.rhea_status == RheaStatus.PRELIMINARY,
            "vmh_transport": vmh_transport,
            "rhea_transport": rhea_transport,
            "compartment_difference": vmh_transport != rhea_transport,
            "direction_difference": direction_difference,
            "direction_note": direction_note,
            "in_strict_baseline": (vmh.reaction_id, rhea.reaction_id) in self.baseline_pairs,
            "relationship_profile": profile,
            "n_vmh_entries": n_vmh,
            "n_rhea_entries": n_rhea,
            "n_assessable_vmh": sum(
                self._state(e) == StructureState.ASSESSABLE for e in vmh.participants
            ),
            "n_assessable_rhea": sum(
                self._state(e) == StructureState.ASSESSABLE for e in rhea.participants
            ),
            "n_non_assessable": sum(s != StructureState.ASSESSABLE for s in states),
            "vmh_direction": vmh.vmh_direction,
            "retrieval_routes": tuple(sorted(routes)),
            "vmh_subsystems": vmh.subsystems,
            "run_id": self.run_id,
        }
        evidence_rows: list[dict[str, Any]] = []
        for o in Orientation:
            evidence_rows.extend(self._evidence_rows(pid, vmh, results[o]))
        alternative_rows: list[dict[str, Any]] = []
        if result.ambiguous:
            for k, alternative in enumerate(result.alternatives, start=1):
                for pair in sorted(alternative, key=lambda p: (p.vmh.entry_key, p.rhea.entry_key)):
                    alternative_rows.append(_alternative_row(pid, k, orientation, pair))
        fields["evidence_hash"] = _evidence_hash(fields, evidence_rows, alternative_rows)
        return CandidateOutcome(
            Proposal(fields, evidence_rows, alternative_rows), True, "", exactness
        )


def _alternative_row(
    pid: str, k: int, orientation: Orientation, pair: AlignedPair
) -> dict[str, Any]:
    return {
        "proposal_id": pid,
        "alternative_group": f"{pid}:{k}",
        "orientation": orientation,
        "vmh_entry_key": pair.vmh.entry_key,
        "rhea_entry_key": pair.rhea.entry_key,
        "relationship_label": pair.evidence.label,
        "smro_term_id": pair.evidence.term_id,
    }


def _evidence_hash(
    fields: dict[str, Any], rows: list[dict[str, Any]], alternatives: list[dict[str, Any]]
) -> str:
    from stereomapper.reconciliation.outputs import format_value

    payload = {
        "fields": {k: format_value(v) for k, v in fields.items() if k not in _NOT_HASHED},
        "evidence": [{k: format_value(v) for k, v in r.items()} for r in rows],
        "alternatives": [{k: format_value(v) for k, v in r.items()} for r in alternatives],
    }
    return "sha256:" + sha256_text(canonical_json(payload))
