"""US1 acceptance scenarios and FR-011/012/013/014/016/017/019 regression cases."""

from __future__ import annotations

from typing import Any

import pytest

from stereomapper.reconciliation.model import SPEC_STATE, AlignmentState, StructureState
from stereomapper.tests.reconciliation.conftest import RunOutput, Scenario, propose, read_rows

S, P = ("s", "vmh:s"), ("p", "vmh:p")
S_R, P_R = ("CHEBI:s", "chebi:s"), ("CHEBI:p", "chebi:p")


def _simple(scenario: Scenario, master: int = 100, **vmh_kwargs: Any) -> Scenario:
    scenario.vmh("E1", [S], [P], **vmh_kwargs)
    scenario.rhea(master, [S_R], [P_R])
    scenario.same("vmh:s", "chebi:s")
    return scenario


def _run(scenario: Scenario) -> RunOutput:
    return propose(scenario.build())


def _rows(out: RunOutput, row: dict[str, str]) -> list[dict[str, str]]:
    return out.evidence_for(row["proposal_id"], row["selected_orientation"])


def test_exact_pair(scenario: Scenario) -> None:
    _simple(scenario).same("vmh:p", "chebi:p")
    row = _run(scenario).proposal("E1", "RHEA:100")
    assert row["n_SMRO_0001"] == "2"
    assert row["coverage_vmh"] == "2/2"
    assert row["coverage_vmh_decimal"] == "1.0000"
    assert row["coverage_rhea"] == "2/2"
    assert row["selected_orientation"] == "same"
    assert row["stoichiometry_consistent"] == "true"
    assert row["relationship_profile"] == "identical_only"
    assert row["rhea_matching_directional_id"] == "RHEA:101"


def test_stereo_resolution_pair_identified(scenario: Scenario) -> None:  # US1-1
    _simple(scenario).rel("vmh:p", "chebi:p", "Stereo-resolution pairs", "A_to_B")
    out = _run(scenario)
    row = out.proposal("E1", "RHEA:100")
    assert row["n_SMRO_0007"] == "1"
    assert row["has_stereo_relationship"] == "true"
    p_row = next(r for r in _rows(out, row) if r["vmh_metabolite_id"] == "p")
    assert p_row["relationship_label"] == "Stereo-resolution pairs"
    assert p_row["smro_term_id"] == "SMRO:0007"
    assert p_row["relationship_direction"] == "vmh_more_resolved"
    assert p_row["alignment_state"] == "aligned"


@pytest.mark.parametrize(
    "label,term,flag",
    [
        ("Enantiomers", "SMRO:0005", "has_stereo_relationship"),
        ("Diastereomers", "SMRO:0006", "has_stereo_relationship"),
        ("Protomers", "SMRO:0003", "has_protomer"),
    ],
)
def test_stereo_and_protomer_relationships_are_proposed(
    scenario: Scenario, label: str, term: str, flag: str
) -> None:  # US1-2, FR-011
    _simple(scenario).rel("vmh:p", "chebi:p", label)
    row = _run(scenario).proposal("E1", "RHEA:100")
    assert row[f"n_{term.replace(':', '_')}"] == "1"
    assert row[flag] == "true"
    assert row["n_aligned"] == "2"


def test_four_aligned_one_unmatched(scenario: Scenario) -> None:  # US1-3
    left = [(f"m{i}", f"vmh:m{i}") for i in range(1, 5)]
    left_r = [(f"CHEBI:m{i}", f"chebi:m{i}") for i in range(1, 5)]
    scenario.vmh("E1", left, [("x", "vmh:x")])
    scenario.rhea(100, left_r, [("CHEBI:y", "chebi:y")])
    for i in range(1, 5):
        scenario.same(f"vmh:m{i}", f"chebi:m{i}")
    out = _run(scenario)
    row = out.proposal("E1", "RHEA:100")
    assert row["coverage_vmh"] == "4/5"
    assert row["coverage_rhea"] == "4/5"
    assert row["n_unmatched_vmh"] == "1"
    rows = _rows(out, row)
    assert {r["vmh_entry_key"] for r in rows if r["vmh_entry_key"]} == {
        f"vmh|E1|left|m{i}|" for i in range(1, 5)
    } | {"vmh|E1|right|x|"}
    x_row = next(r for r in rows if r["vmh_metabolite_id"] == "x")
    assert x_row["rhea_entry_key"] == ""
    assert x_row["spec_state"] == "supported_nonmatch"


def test_reversed_orientation_gives_one_proposal(scenario: Scenario) -> None:  # US1-4
    scenario.vmh("E1", [S], [P])
    scenario.rhea(200, [P_R], [S_R])
    scenario.same("vmh:s", "chebi:s").same("vmh:p", "chebi:p")
    out = _run(scenario)
    assert len([r for r in out.proposals if r["vmh_reaction_id"] == "E1"]) == 1
    row = out.proposal("E1", "RHEA:200")
    assert row["selected_orientation"] == "reversed"
    assert row["vmh_direction"] == "forward"
    assert row["rhea_matching_directional_id"] == "RHEA:202"  # RL
    assert {r["orientation"] for r in out.evidence_for(row["proposal_id"])} == {
        "same",
        "reversed",
    }


def test_stoichiometry_difference_not_hidden(scenario: Scenario) -> None:  # US1-5
    scenario.vmh("E1", [("s", "vmh:s", "2")], [P])
    scenario.rhea(100, [S_R], [P_R])
    scenario.same("vmh:s", "chebi:s").same("vmh:p", "chebi:p")
    out = _run(scenario)
    row = out.proposal("E1", "RHEA:100")
    assert row["stoichiometry_consistent"] == "false"
    s_rows = [r for r in _rows(out, row) if r["vmh_metabolite_id"] == "s"]
    assert len(s_rows) == 1
    assert s_rows[0]["coefficient_equal"] == "false"


def test_one_molecule_cannot_fill_two_slots(scenario: Scenario) -> None:  # US1-5
    scenario.vmh("E1", [S], [P])
    scenario.rhea(100, [S_R, ("CHEBI:s2", "chebi:s2")], [P_R])
    scenario.same("vmh:s", "chebi:s", "chebi:s2").same("vmh:p", "chebi:p")
    row = _run(scenario).proposal("E1", "RHEA:100")
    assert row["n_aligned"] == "2"
    assert row["n_unmatched_rhea"] == "1"
    assert row["stoichiometry_consistent"] == "false"


def test_missing_structure_is_unavailable_not_different(scenario: Scenario) -> None:  # US1-6
    scenario.vmh("E1", [S, ("u", "")], [P])
    scenario.rhea(100, [S_R], [P_R])
    scenario.same("vmh:s", "chebi:s").same("vmh:p", "chebi:p")
    out = _run(scenario)
    row = out.proposal("E1", "RHEA:100")
    assert row["coverage_vmh"] == "2/3"
    assert row["has_unavailable_structure"] == "true"
    u_row = next(r for r in _rows(out, row) if r["vmh_metabolite_id"] == "u")
    assert u_row["vmh_structure_state"] == "structure_unavailable"
    assert u_row["spec_state"] == "structure_unavailable"


def test_formula_agreement_alone_does_not_align(scenario: Scenario) -> None:  # US1-7
    q = {"metabolite_id": "q", "structure_curie": "vmh:q", "formula": "C6H12O6"}
    q_r = {"metabolite_id": "CHEBI:q", "structure_curie": "chebi:q", "formula": "C6H12O6"}
    scenario.vmh("E1", [S], [P, q])
    scenario.rhea(100, [S_R], [P_R, q_r])
    scenario.same("vmh:s", "chebi:s").same("vmh:p", "chebi:p")
    out = _run(scenario)
    row = out.proposal("E1", "RHEA:100")
    assert row["n_aligned"] == "2"
    q_row = next(r for r in _rows(out, row) if r["vmh_metabolite_id"] == "q")
    assert q_row["alignment_state"] == "supported_nonmatch"
    assert q_row["formula_agreement"] == "true"


def test_compartment_difference_flagged_and_kept(scenario: Scenario) -> None:  # US1-8
    s_c = {"metabolite_id": "s", "structure_curie": "vmh:s", "compartment": "c"}
    s_e = {"metabolite_id": "s", "structure_curie": "vmh:s", "compartment": "e"}
    scenario.vmh("T1", [s_c, P], [s_e, P])
    scenario.rhea(100, [S_R, P_R], [S_R, P_R])
    scenario.same("vmh:s", "chebi:s").same("vmh:p", "chebi:p")
    row = _run(scenario).proposal("T1", "RHEA:100")
    assert row["vmh_transport"] == "true"
    assert row["rhea_transport"] == "false"
    assert row["compartment_difference"] == "true"


def test_direction_difference_missing_directional_id(scenario: Scenario) -> None:  # US1-8, R8a
    scenario.vmh("E1", [S], [P], direction="reverse")
    scenario.rhea(100, [S_R], [P_R], directional=(1, 3))  # LR and BI only; RL missing
    scenario.same("vmh:s", "chebi:s").same("vmh:p", "chebi:p")
    row = _run(scenario).proposal("E1", "RHEA:100")
    assert row["direction_difference"] == "true"
    assert "missing" in row["direction_note"]


def test_direction_difference_opposite_xref(scenario: Scenario) -> None:  # US1-8, R8b
    _simple(scenario, xref="RHEA:102").same("vmh:p", "chebi:p")  # VMH forward, xref is RL
    row = _run(scenario).proposal("E1", "RHEA:100")
    assert row["direction_difference"] == "true"
    assert "xref" in row["direction_note"]


def test_no_direction_difference_when_consistent(scenario: Scenario) -> None:
    _simple(scenario, xref="RHEA:101").same("vmh:p", "chebi:p")
    row = _run(scenario).proposal("E1", "RHEA:100")
    assert row["direction_difference"] == "false"


def test_proton_only_difference_flagged(scenario: Scenario) -> None:  # US1-9, FR-008a
    h = ("h", "vmh:h", "1")
    h_r = ("CHEBI:15378", "chebi:15378", "2")
    scenario.vmh("E1", [S, h], [P])
    scenario.rhea(100, [S_R, h_r], [P_R])
    scenario.same("vmh:s", "chebi:s").same("vmh:p", "chebi:p").same("vmh:h", "chebi:15378")
    row = _run(scenario).proposal("E1", "RHEA:100")
    assert row["n_aligned"] == "3"  # H+ counts as an ordinary participant
    assert row["stoichiometry_consistent"] == "false"
    assert row["proton_balance_difference"] == "true"


def test_non_proton_difference_not_flagged_as_proton(scenario: Scenario) -> None:
    scenario.vmh("E1", [("s", "vmh:s", "2")], [P])
    scenario.rhea(100, [S_R], [P_R])
    scenario.same("vmh:s", "chebi:s").same("vmh:p", "chebi:p")
    row = _run(scenario).proposal("E1", "RHEA:100")
    assert row["proton_balance_difference"] == "false"


def test_generic_participant_not_assessable(scenario: Scenario) -> None:  # FR-012
    g_r = {"metabolite_id": "CHEBI:g", "kind": "generic", "structure_curie": "chebi:g"}
    scenario.vmh("E1", [S, ("g", "vmh:g")], [P])
    scenario.rhea(100, [S_R, g_r], [P_R])
    scenario.same("vmh:s", "chebi:s").same("vmh:p", "chebi:p").same("vmh:g", "chebi:g")
    out = _run(scenario)
    row = out.proposal("E1", "RHEA:100")
    assert row["has_not_assessable"] == "true"
    assert row["n_aligned"] == "2"
    assert row["coverage_rhea"] == "2/3"
    g_row = next(r for r in _rows(out, row) if r["rhea_metabolite_id"] == "CHEBI:g")
    assert g_row["rhea_structure_state"] == "not_assessable"
    assert g_row["rhea_kind"] == "generic"
    assert g_row["spec_state"] == "not_assessable"


def test_cofactor_only_overlap_not_proposed(scenario: Scenario) -> None:  # FR-017
    scenario.vmh("C1", [("h2o", "vmh:h2o")], [("x", "vmh:x")], xref="RHEA:300")
    scenario.rhea(300, [("CHEBI:15377", "chebi:15377")], [("CHEBI:y", "chebi:y")])
    scenario.same("vmh:h2o", "chebi:15377")
    out = _run(scenario)
    assert not out.has_proposal("C1", "RHEA:300")
    assert out.funnel["candidates_xref"][0] == "1"
    assert out.funnel["not_plausible_cofactor_only"][0] == "1"


def test_uncertain_classes_not_counted_as_aligned(scenario: Scenario) -> None:  # FR-011
    _simple(scenario).rel("vmh:p", "chebi:p", "Unresolved")
    out = _run(scenario)
    row = out.proposal("E1", "RHEA:100")
    assert row["n_aligned"] == "1"
    assert row["n_aligned_uncertain"] == "1"
    assert row["coverage_vmh"] == "1/2"
    assert row["has_uncertain"] == "true"
    assert row["relationship_profile"] == "includes_uncertain"
    p_row = next(r for r in _rows(out, row) if r["vmh_metabolite_id"] == "p")
    assert p_row["alignment_state"] == "aligned_uncertain"
    assert p_row["spec_state"] == "uncertain"


def test_ambiguous_alignment_keeps_alternatives(scenario: Scenario) -> None:  # FR-016
    scenario.vmh("E1", [("x1", "vmh:x1"), ("x2", "vmh:x2")], [P])
    scenario.rhea(100, [("CHEBI:y1", "chebi:y1"), ("CHEBI:y2", "chebi:y2")], [P_R])
    scenario.same("vmh:p", "chebi:p")
    scenario.rel("vmh:x1", "chebi:y1", "Diastereomers")
    scenario.rel("vmh:x2", "chebi:y2", "Stereo-resolution pairs")
    scenario.rel("vmh:x1", "chebi:y2", "Protomers")
    scenario.rel("vmh:x2", "chebi:y1", "Enantiomers")
    out = _run(scenario)
    row = out.proposal("E1", "RHEA:100")
    assert row["ambiguous_alignment"] == "true"
    alts = read_rows(out.run_dir / "alternative_alignments.tsv")
    groups = {a["alternative_group"] for a in alts if a["proposal_id"] == row["proposal_id"]}
    assert len(groups) == 2


@pytest.fixture
def mixed(scenario: Scenario) -> RunOutput:
    _simple(scenario).same("vmh:p", "chebi:p")
    scenario.vmh("E2", [("a", "vmh:a"), ("u", "")], [("b", "vmh:b"), ("z", "vmh:z")])
    scenario.rhea(
        400,
        [
            ("CHEBI:a", "chebi:a"),
            {
                "metabolite_id": "CHEBI:g",
                "kind": "polymer",
                "coefficient": "n",
                "structure_curie": "chebi:g",
            },
        ],
        [("CHEBI:b", "chebi:b"), ("CHEBI:f", "chebi:f")],
    )
    scenario.same("vmh:a", "chebi:a")
    scenario.rel("vmh:b", "chebi:b", "Enantiomers")
    scenario.rel("vmh:z", "chebi:f", "Unclassified")
    scenario.failed("chebi:f")
    return _run(scenario)


def test_no_equivalence_label_anywhere(mixed: RunOutput) -> None:  # FR-013
    for table in (mixed.proposals, mixed.evidence):
        for row in table:
            for key, value in row.items():
                assert "equivalent" not in key.lower()
                assert "equivalent" not in value.lower()


def test_entry_count_invariant(mixed: RunOutput) -> None:
    for row in mixed.proposals:
        total = int(row["n_aligned"]) + int(row["n_aligned_uncertain"])
        assert total + int(row["n_unmatched_vmh"]) == int(row["n_vmh_entries"])
        assert total + int(row["n_unmatched_rhea"]) == int(row["n_rhea_entries"])


def test_every_row_maps_to_one_spec_state(mixed: RunOutput) -> None:  # FR-014, SC-006
    assert mixed.evidence
    for row in mixed.evidence:
        alignment = AlignmentState(row["alignment_state"])
        for side in ("vmh", "rhea"):
            if not row[f"{side}_entry_key"]:
                continue
            state = StructureState(row[f"{side}_structure_state"])
            assert (state, alignment) in SPEC_STATE
            assert SPEC_STATE[(state, alignment)] == row["spec_state"]


def test_every_entry_appears_in_each_orientation(mixed: RunOutput) -> None:  # SC-006
    row = next(r for r in mixed.proposals if r["vmh_reaction_id"] == "E2")
    for orientation in ("same", "reversed"):
        rows = mixed.evidence_for(row["proposal_id"], orientation)
        assert len({r["vmh_entry_key"] for r in rows if r["vmh_entry_key"]}) == 4
        assert len({r["rhea_entry_key"] for r in rows if r["rhea_entry_key"]}) == 4


def test_failed_structure_kept_distinct(mixed: RunOutput) -> None:
    row = next(r for r in mixed.proposals if r["vmh_reaction_id"] == "E2")
    rows = mixed.evidence_for(row["proposal_id"], row["selected_orientation"])
    f_row = next(r for r in rows if r["rhea_metabolite_id"] == "CHEBI:f")
    assert f_row["rhea_structure_state"] == "processing_failed"
    assert "rdkit" in f_row["rhea_structure_error"]


def test_provenance_columns_present(mixed: RunOutput) -> None:  # FR-019
    for row in mixed.evidence:
        if row["vmh_entry_key"]:
            assert row["vmh_source_row"].isdigit()
        if row["rhea_entry_key"]:
            assert row["rhea_source_row"].isdigit()
        if row["evidence_type"] != "none":
            assert row["sm_version_tag"] == "v-test"
            assert len(row["sm_cluster_ids"].split(";")) == 2
