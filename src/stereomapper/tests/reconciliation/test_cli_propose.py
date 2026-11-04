from __future__ import annotations

from pathlib import Path

import pytest
from openpyxl import load_workbook

from stereomapper.cli import main
from stereomapper.tests.reconciliation.conftest import Scenario, read_rows

RUN_FILES = (
    "run_manifest.json",
    "run_config.json",
    "reaction_proposals.tsv",
    "participant_evidence.tsv",
    "alternative_alignments.tsv",
    "scope_exclusions.tsv",
    "vmh_subsystems.tsv",
    "funnel.tsv",
    "reactions_summary.tsv",
    "candidates.tsv",
    "baseline_pairs.tsv",
)
REQUIRED_ORDER = [
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
    *[f"n_SMRO_{i:04d}" for i in range(1, 10)],
    "stoichiometry_consistent",
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
    "in_strict_baseline",
    "relationship_profile",
    "ranking_version",
    "rank_note",
]


def cli(*args: str) -> int:
    with pytest.raises(SystemExit) as exc:
        main(["reconcile", *args])
    code = exc.value.code
    return code if isinstance(code, int) else 1


@pytest.fixture
def config(scenario: Scenario) -> Path:
    scenario.vmh("E1", [("s", "vmh:s")], [("p", "vmh:p")])
    scenario.rhea(100, [("CHEBI:s", "chebi:s")], [("CHEBI:p", "chebi:p")])
    scenario.same("vmh:s", "chebi:s").rel("vmh:p", "chebi:p", "Enantiomers")
    return scenario.build()


def test_propose_writes_contract_files(config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = config.parent / "run"
    assert cli("propose", "--config", str(config), "--out", str(out)) == 0
    for name in RUN_FILES:
        assert (out / name).is_file(), name
    header = (out / "reaction_proposals.tsv").read_text().splitlines()[0].split("\t")
    positions = [header.index(col) for col in REQUIRED_ORDER]
    assert positions == sorted(positions)
    stdout = capsys.readouterr().out
    assert "run_id:" in stdout
    assert "proposals: 1/1" in stdout


def test_non_empty_out_requires_overwrite(config: Path) -> None:
    out = config.parent / "run"
    assert cli("propose", "--config", str(config), "--out", str(out)) == 0
    assert cli("propose", "--config", str(config), "--out", str(out)) == 2
    assert cli("propose", "--config", str(config), "--out", str(out), "--overwrite") == 0


def test_broken_extract_exit_2_names_file(config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = config.parent / "extract" / "participants.tsv"
    lines = path.read_text().splitlines()
    lines[1] = lines[1].replace("\tleft\t", "\tmiddle\t")
    path.write_text("\n".join(lines) + "\n")
    assert cli("propose", "--config", str(config), "--out", str(config.parent / "run")) == 2
    err = capsys.readouterr().err
    assert "participants.tsv" in err and "row 2" in err and "side" in err


def test_review_workbook_shows_pending(config: Path) -> None:
    out = config.parent / "run"
    assert cli("propose", "--config", str(config), "--out", str(out)) == 0
    xlsx = config.parent / "review.xlsx"
    assert cli("review-workbook", "--run", str(out), "--out", str(xlsx)) == 0
    wb = load_workbook(xlsx)
    assert wb.sheetnames == ["proposals", "participants", "decision_template", "about"]
    sheet = wb["proposals"]
    header = [c.value for c in sheet[1]]
    status_col = header.index("decision_status")
    assert [row[status_col] for row in sheet.iter_rows(min_row=2, values_only=True)] == ["pending"]
    for col in ("vmh_equation", "rhea_equation", "vmh_subsystems"):
        assert col in header
    template = wb["decision_template"]
    assert [c.value for c in template[1]] == [
        "proposal_id",
        "evidence_hash",
        "decision",
        "reviewer",
        "date",
        "reason",
        "supersedes",
    ]
    proposal = read_rows(out / "reaction_proposals.tsv")[0]
    assert template.cell(row=2, column=1).value == proposal["proposal_id"]
    assert template.cell(row=2, column=2).value == proposal["evidence_hash"]
    about = {r[0]: r[1] for r in wb["about"].iter_rows(values_only=True)}
    assert about["rank_note"] == "review order, not a probability"
