from __future__ import annotations

from collections.abc import Iterator

import pytest

from stereomapper.domain.exceptions import ConfigurationError
from stereomapper.reconciliation.evidence import StereoMapperEvidence
from stereomapper.reconciliation.hashing import sha256_file
from stereomapper.reconciliation.inputs import load_extract
from stereomapper.reconciliation.model import (
    EvidenceType,
    ParticipantEntry,
    SourceReaction,
    StructureState,
)
from stereomapper.reconciliation.normalize import Reactions, build_reactions
from stereomapper.tests.reconciliation.conftest import Scenario

Case = tuple[Reactions, StereoMapperEvidence]


@pytest.fixture
def case(scenario: Scenario) -> Iterator[Case]:
    scenario.vmh(
        "R1",
        [
            ("a", "vmh:a"),
            ("u", ""),
            ("n", "vmh:n"),
            ("f", "vmh:f"),
            {"metabolite_id": "g", "kind": "generic", "structure_curie": "vmh:g"},
        ],
        [("b", "vmh:b")],
    )
    scenario.rhea(10, [("CHEBI:1", "chebi:1")], [("CHEBI:5", "chebi:5"), ("CHEBI:6", "chebi:6")])
    scenario.same("vmh:a", "chebi:1")
    scenario.rel("vmh:b", "chebi:5", "Stereo-resolution pairs", "A_to_B")
    scenario.other_version_rels.append(("vmh:b", "chebi:6", "Enantiomers"))
    scenario.missing("vmh:n")
    scenario.failed("vmh:f")
    scenario.build()
    reactions = build_reactions(load_extract(scenario.root / "extract"))
    curies = {p.structure_curie for r in reactions.all_reactions() for p in r.participants}
    db = scenario.root / "results.sqlite"
    before = sha256_file(db)
    evidence = StereoMapperEvidence(db, "v-test", curies)
    yield reactions, evidence
    evidence.close()
    assert sha256_file(db) == before  # FR-004: read-only


def _entry(reaction: SourceReaction, met: str) -> ParticipantEntry:
    return next(p for p in reaction.participants if p.metabolite_id == met)


def test_structure_states(case: Case) -> None:
    reactions, evidence = case
    r1 = reactions.vmh["R1"]
    assert evidence.structure_state(_entry(r1, "a")) == StructureState.ASSESSABLE
    assert evidence.structure_state(_entry(r1, "u")) == StructureState.STRUCTURE_UNAVAILABLE
    assert evidence.structure_state(_entry(r1, "n")) == StructureState.NO_RESULT
    assert evidence.structure_state(_entry(r1, "f")) == StructureState.PROCESSING_FAILED
    assert "rdkit_sanitize_failed" in evidence.structure_error(_entry(r1, "f"))
    # FR-012: generic participants are never assessable, even with a structure.
    assert evidence.structure_state(_entry(r1, "g")) == StructureState.NOT_ASSESSABLE


def test_pair_evidence(case: Case) -> None:
    reactions, evidence = case
    r1, rh = reactions.vmh["R1"], reactions.rhea_masters["RHEA:10"]
    same = evidence.pair_evidence(_entry(r1, "a"), _entry(rh, "CHEBI:1"))
    assert same.evidence_type == EvidenceType.CLUSTER_MEMBERSHIP
    assert (same.label, same.term_id) == ("Identical structures", "SMRO:0001")

    rel = evidence.pair_evidence(_entry(r1, "b"), _entry(rh, "CHEBI:5"))
    assert rel.evidence_type == EvidenceType.RELATIONSHIP
    assert rel.term_id == "SMRO:0007"
    assert rel.direction == "vmh_more_resolved"
    assert rel.score == pytest.approx(0.9)
    assert rel.version_tag == "v-test"

    none = evidence.pair_evidence(_entry(r1, "a"), _entry(rh, "CHEBI:5"))
    assert none.evidence_type == EvidenceType.NONE


def test_only_configured_version_tag_is_used(case: Case) -> None:
    reactions, evidence = case
    r1, rh = reactions.vmh["R1"], reactions.rhea_masters["RHEA:10"]
    other = evidence.pair_evidence(_entry(r1, "b"), _entry(rh, "CHEBI:6"))
    assert other.evidence_type == EvidenceType.NONE


def test_non_assessable_entries_have_no_pair_evidence(case: Case) -> None:
    reactions, evidence = case
    r1, rh = reactions.vmh["R1"], reactions.rhea_masters["RHEA:10"]
    assert evidence.pair_evidence(_entry(r1, "g"), _entry(rh, "CHEBI:1")).evidence_type == (
        EvidenceType.NONE
    )


def test_unknown_version_tag_rejected(scenario: Scenario) -> None:
    scenario.vmh("R1", [("a", "vmh:a")], [("b", "vmh:b")])
    scenario.build()
    with pytest.raises(ConfigurationError, match="version_tag"):
        StereoMapperEvidence(scenario.root / "results.sqlite", "v-missing", {"vmh:a"})
