"""Enumerations and entities for VMH-Rhea reaction reconciliation (see data-model.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction


class Side(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class Orientation(str, Enum):
    SAME = "same"
    REVERSED = "reversed"


class VmhDirection(str, Enum):
    FORWARD = "forward"
    REVERSE = "reverse"
    REVERSIBLE = "reversible"
    UNKNOWN = "unknown"


class RheaStatus(str, Enum):
    APPROVED = "approved"
    PRELIMINARY = "preliminary"
    OBSOLETE = "obsolete"


class ParticipantKind(str, Enum):
    SMALL_MOLECULE = "small_molecule"
    GENERIC = "generic"
    POLYMER = "polymer"
    MACROMOLECULE = "macromolecule"


class StructureState(str, Enum):
    ASSESSABLE = "assessable"
    STRUCTURE_UNAVAILABLE = "structure_unavailable"
    NO_RESULT = "no_result"
    PROCESSING_FAILED = "processing_failed"
    NOT_ASSESSABLE = "not_assessable"


class AlignmentState(str, Enum):
    ALIGNED = "aligned"
    ALIGNED_UNCERTAIN = "aligned_uncertain"
    SUPPORTED_NONMATCH = "supported_nonmatch"
    UNMATCHED = "unmatched"


class EvidenceType(str, Enum):
    CLUSTER_MEMBERSHIP = "cluster_membership"
    RELATIONSHIP = "relationship"
    NONE = "none"


class DecisionValue(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class BenchmarkStatus(str, Enum):
    RECOVERED_NEW = "recovered_new"
    ALREADY_IN_BASELINE = "already_in_baseline"
    UNRECOVERED = "unrecovered"
    NOT_A_POSITIVE = "not_a_positive"


class UnrecoveredReason(str, Enum):
    VMH_ID_ABSENT = "vmh_id_absent"
    RHEA_ID_ABSENT = "rhea_id_absent"
    RHEA_OBSOLETE = "rhea_obsolete"
    STRUCTURE_INSUFFICIENT = "structure_insufficient"
    NOT_RETRIEVED = "not_retrieved"
    RETRIEVED_NOT_PLAUSIBLE = "retrieved_not_plausible"


class ExclusionRule(str, Enum):
    RHEA_OBSOLETE = "rhea_obsolete"


# SMRO term groups used by alignment (research R3). No class is excluded.
SUPPORTING_TERMS = ("SMRO:0001", "SMRO:0002", "SMRO:0003", "SMRO:0007", "SMRO:0006", "SMRO:0005")
UNCERTAIN_TERMS = ("SMRO:0004", "SMRO:0009", "SMRO:0008")
ALL_TERMS = tuple(f"SMRO:{i:04d}" for i in range(1, 10))
STEREO_TERMS = ("SMRO:0005", "SMRO:0006", "SMRO:0007")
PROTOMER_TERMS = ("SMRO:0003",)
IDENTICAL_TERMS = ("SMRO:0001", "SMRO:0002")

# align-v1 class order, used only to choose among competing assignments.
ALIGN_V1_WEIGHTS = {
    "SMRO:0001": 7,
    "SMRO:0002": 6,
    "SMRO:0003": 5,
    "SMRO:0007": 4,
    "SMRO:0006": 3,
    "SMRO:0005": 2,
    "SMRO:0004": 1,
    "SMRO:0009": 1,
    "SMRO:0008": 1,
}

# Mapping of (structure_state, alignment_state) to the spec's participant states (FR-014).
SPEC_STATE = {
    (StructureState.ASSESSABLE, AlignmentState.ALIGNED): "aligned",
    (StructureState.ASSESSABLE, AlignmentState.ALIGNED_UNCERTAIN): "uncertain",
    (StructureState.ASSESSABLE, AlignmentState.SUPPORTED_NONMATCH): "supported_nonmatch",
    (StructureState.ASSESSABLE, AlignmentState.UNMATCHED): "supported_nonmatch",
    (StructureState.STRUCTURE_UNAVAILABLE, AlignmentState.UNMATCHED): "structure_unavailable",
    (StructureState.PROCESSING_FAILED, AlignmentState.UNMATCHED): "processing_failed",
    (StructureState.NO_RESULT, AlignmentState.UNMATCHED): "no_result",
    (StructureState.NOT_ASSESSABLE, AlignmentState.UNMATCHED): "not_assessable",
}


@dataclass(frozen=True)
class ParticipantEntry:
    source: str
    reaction_id: str
    side: Side
    metabolite_id: str
    name: str
    compartment: str
    coefficient_raw: str
    coefficient: Fraction | None
    kind: ParticipantKind
    structure_curie: str
    formula: str
    charge: str
    source_row: int

    @property
    def entry_key(self) -> str:
        return "|".join(
            (self.source, self.reaction_id, self.side.value, self.metabolite_id, self.compartment)
        )


@dataclass
class SourceReaction:
    source: str
    reaction_id: str
    equation: str
    source_row: int
    rhea_master_id: str = ""
    rhea_directional_ids: tuple[str, ...] = ()
    rhea_status: RheaStatus | None = None
    vmh_direction: VmhDirection | None = None
    xref_rhea: tuple[str, ...] = ()
    subsystems: tuple[str, ...] = ()
    participants: list[ParticipantEntry] = field(default_factory=list)

    def side_entries(self, side: Side) -> list[ParticipantEntry]:
        return [p for p in self.participants if p.side == side]


@dataclass(frozen=True)
class PairEvidence:
    evidence_type: EvidenceType
    label: str = ""
    term_id: str = ""
    direction: str = ""
    score: float | None = None
    version_tag: str = ""

    @property
    def is_supporting(self) -> bool:
        return self.term_id in SUPPORTING_TERMS

    @property
    def is_uncertain(self) -> bool:
        return self.term_id in UNCERTAIN_TERMS


@dataclass(frozen=True)
class ScopeExclusion:
    source: str
    reaction_id: str
    rule: ExclusionRule
    rule_parameter: str
    evidence: str
    reason: str


@dataclass(frozen=True)
class CurationDecision:
    decision_id: str
    proposal_id: str
    evidence_hash: str
    run_id: str
    vmh_reaction_id: str
    rhea_master_id: str
    decision: DecisionValue
    reviewer: str
    date: str
    reason: str
    supersedes: str = ""
