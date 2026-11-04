"""US3: recovery against the JC + NM worksheet and the strict baseline (FR-025 to FR-027)."""

from __future__ import annotations

from pathlib import Path

import pytest

from stereomapper.reconciliation.benchmark import NOTE, run_benchmark
from stereomapper.reconciliation.flags import apply_flags
from stereomapper.tests.reconciliation.conftest import RunOutput, Scenario, propose, read_rows
from stereomapper.tests.reconciliation.test_flags import write_flag_list

S, P = ("s", "vmh:s"), ("p", "vmh:p")
S_R, P_R = ("CHEBI:s", "chebi:s"), ("CHEBI:p", "chebi:p")

WORKSHEET = [
    ("E1", "RHEA:100", "verified"),  # recovered_new
    ("B1", "RHEA:200", "verified"),  # already_in_baseline
    ("X1", "RHEA:100", "verified"),  # vmh_id_absent
    ("E1", "RHEA:999", "verified"),  # rhea_id_absent
    ("O1", "RHEA:300", "Verified"),  # rhea_obsolete
    ("U1", "RHEA:400", "verified"),  # structure_insufficient
    ("N1", "RHEA:500", "verified"),  # not_retrieved
    ("C1", "RHEA:600", "verified"),  # retrieved_not_plausible
    ("D1", "RHEA:700", "verified"),  # recovered_new, subsystem-flagged
    ("E1", "RHEA:200", "rejected"),  # not a positive
    ("E2", "802", "verified"),  # directional RL ID joins through master 800
]


def populate(sc: Scenario) -> Scenario:
    """One reaction pair per benchmark outcome (see WORKSHEET)."""
    sc.vmh("E1", [S], [P])
    sc.rhea(100, [S_R], [P_R])
    sc.vmh("B1", [("b", "vmh:b")], [P])
    sc.rhea(200, [("CHEBI:b", "chebi:b")], [P_R])
    sc.in_baseline("B1", "RHEA:201")
    sc.vmh("O1", [("o", "vmh:o")], [P])
    sc.rhea(300, [("CHEBI:o", "chebi:o")], [P_R], status="obsolete")
    sc.vmh("U1", [("u1", "")], [("u2", "")])
    sc.rhea(400, [("CHEBI:u", "chebi:u")], [P_R])
    sc.vmh("N1", [("n", "vmh:n")], [("m", "vmh:m")])
    sc.rhea(500, [("CHEBI:n", "chebi:n")], [("CHEBI:m", "chebi:m")])
    sc.vmh("C1", [("h2o", "vmh:h2o")], [("x", "vmh:x")], xref="RHEA:600")
    sc.rhea(600, [("CHEBI:15377", "chebi:15377")], [("CHEBI:y", "chebi:y")])
    sc.vmh("D1", [("d", "vmh:d")], [P], subsystems=("Drug metabolism",))
    sc.rhea(700, [("CHEBI:d", "chebi:d")], [P_R])
    sc.vmh("E2", [("e", "vmh:e")], [P])
    sc.rhea(800, [("CHEBI:e", "chebi:e")], [P_R])
    for a in ("s", "b", "o", "d", "e", "p"):
        sc.same(f"vmh:{a}", f"chebi:{a}")
    sc.same("vmh:h2o", "chebi:15377")
    return sc


@pytest.fixture
def run(scenario: Scenario) -> RunOutput:
    return propose(populate(scenario).build())


@pytest.fixture
def worksheet(tmp_path: Path) -> Path:
    path = tmp_path / "worksheet.tsv"
    lines = ["vmh_reaction_id\trhea_id\tcurator_verdict"] + ["\t".join(r) for r in WORKSHEET]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def flagged(run: RunOutput, tmp_path: Path) -> RunOutput:
    apply_flags(
        run.run_dir, write_flag_list(tmp_path / "f.tsv", "v1", [("Drug metabolism", "drug")])
    )
    return run


def test_every_pair_has_one_status_and_reason(flagged: RunOutput, worksheet: Path) -> None:
    run_benchmark(flagged.run_dir, worksheet, "ws-1", flag_version="v1")
    rows = read_rows(flagged.run_dir / "benchmark" / "ws-1" / "v1" / "benchmark_pairs.tsv")
    got = [(r["status"], r["unrecovered_reason"]) for r in rows]
    assert got == [
        ("recovered_new", ""),
        ("already_in_baseline", ""),
        ("unrecovered", "vmh_id_absent"),
        ("unrecovered", "rhea_id_absent"),
        ("unrecovered", "rhea_obsolete"),
        ("unrecovered", "structure_insufficient"),
        ("unrecovered", "not_retrieved"),
        ("unrecovered", "retrieved_not_plausible"),
        ("recovered_new", ""),
        ("not_a_positive", ""),
        ("recovered_new", ""),
    ]
    assert [r["worksheet_row"] for r in rows] == [str(i) for i in range(2, 13)]
    d1 = rows[8]
    assert d1["subsystem_flagged"] == "true" and d1["counted"] == "false"
    assert rows[10]["rhea_master_id"] == "RHEA:800"
    assert rows[10]["rhea_id_raw"] == "802"
    assert rows[0]["proposal_id"].startswith("RXP-") and rows[0]["rank"]


def test_subsets_and_invariants(flagged: RunOutput, worksheet: Path) -> None:
    report = run_benchmark(flagged.run_dir, worksheet, "ws-1", flag_version="v1")
    subsets = report["subsets"]
    assert subsets["full_worksheet"]["denominator"] == 10
    assert subsets["full_worksheet"]["recovered_new"] == 3
    assert subsets["full_worksheet"]["already_in_baseline"] == 1
    assert subsets["counted"]["denominator"] == 6
    assert subsets["counted"]["recovered_new"] == 2
    assert subsets["counted_sufficient_structure"]["denominator"] == 5
    for subset in subsets.values():
        assert (
            subset["recovered_new"] + subset["already_in_baseline"] + subset["unrecovered"]
            == subset["denominator"]
        )
        assert sum(subset["unrecovered_by_reason"].values()) == subset["unrecovered"]
        assert subset["recovered_new_fraction"] == (
            f"{subset['recovered_new']}/{subset['denominator']}"
        )
    assert report["note"] == NOTE
    assert "not precision" in NOTE
    assert report["flag_list_version"] == "v1"
    effect = report["rule_effect"]
    assert effect["pairs_out_of_scope"] == {"rhea_obsolete": 1}
    assert effect["pairs_flagged_by_category"] == {"drug": 1}
    assert effect["counted_without_flags"]["denominator"] == 7
    assert effect["counted_without_flags"]["recovered_new"] == 3


def test_without_flags_recorded_as_none(run: RunOutput, worksheet: Path) -> None:
    report = run_benchmark(run.run_dir, worksheet, "ws-1", flag_version=None)
    assert report["flag_list_version"] == "none"
    assert report["subsets"]["counted"]["denominator"] == 7
    assert (run.run_dir / "benchmark" / "ws-1" / "none" / "benchmark_report.json").is_file()


def test_baseline_pair_not_counted_as_new_even_when_proposed(
    run: RunOutput, worksheet: Path
) -> None:  # US3-2
    b1 = run.proposal("B1", "RHEA:200")
    assert b1["in_strict_baseline"] == "true"
    report = run_benchmark(run.run_dir, worksheet, "ws-1", flag_version=None)
    assert report["subsets"]["full_worksheet"]["already_in_baseline"] == 1
