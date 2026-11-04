"""US2: decision log and curated export (FR-022 to FR-024, FR-029)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from openpyxl import load_workbook

from stereomapper.reconciliation.curation import (
    LOG_COLUMNS,
    effective_decisions,
    export_curated,
    read_decision_log,
    record_decisions,
)
from stereomapper.reconciliation.flags import apply_flags
from stereomapper.reconciliation.hashing import sha256_file
from stereomapper.reconciliation.inputs import InputContractError
from stereomapper.reconciliation.review import write_review_workbook
from stereomapper.tests.reconciliation.conftest import RunOutput, Scenario, propose, read_rows
from stereomapper.tests.reconciliation.test_flags import write_flag_list

S, P = ("s", "vmh:s"), ("p", "vmh:p")
BATCH_COLUMNS = [
    "proposal_id",
    "evidence_hash",
    "decision",
    "reviewer",
    "date",
    "reason",
    "supersedes",
]


def build_run(root: Path, p_label: str = "") -> RunOutput:
    sc = Scenario(root=root)
    for rid in ("E1", "E2", "E3", "E4"):
        sc.vmh(rid, [S], [P])
    sc.vmh("D1", [S], [P], subsystems=("Drug metabolism",))
    sc.rhea(100, [("CHEBI:s", "chebi:s")], [("CHEBI:p", "chebi:p")])
    sc.same("vmh:s", "chebi:s")
    if p_label:
        sc.rel("vmh:p", "chebi:p", p_label)
    else:
        sc.same("vmh:p", "chebi:p")
    return propose(sc.build())


def write_batch(path: Path, rows: list[dict[str, str]]) -> Path:
    lines = ["\t".join(BATCH_COLUMNS)]
    lines += ["\t".join(r.get(c, "") for c in BATCH_COLUMNS) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def decision(run: RunOutput, vmh_id: str, value: str, **extra: str) -> dict[str, str]:
    p = run.proposal(vmh_id, "RHEA:100")
    row = {
        "proposal_id": p["proposal_id"],
        "evidence_hash": p["evidence_hash"],
        "decision": value,
        "reviewer": "curator-1",
        "date": "2026-11-04",
        "reason": f"{value} after review",
    }
    row.update(extra)
    return row


@pytest.fixture
def run(tmp_path: Path) -> RunOutput:
    return build_run(tmp_path / "a")


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / "curation" / "decisions.tsv"


def test_only_accepted_exported(run: RunOutput, log: Path, tmp_path: Path) -> None:
    batch = write_batch(
        tmp_path / "b1.tsv",
        [
            decision(run, "E1", "accepted"),
            decision(run, "E2", "rejected"),
            decision(run, "E3", "deferred"),
        ],
    )
    assert record_decisions(run.run_dir, batch, log) == 3
    out = tmp_path / "CURATED.tsv"
    summary = export_curated(run.run_dir, log, out, flag_version=None)
    rows = read_rows(out)
    assert [r["vmh_reaction_id"] for r in rows] == ["E1"]  # US2-1, US2-2: E2-E4 absent
    accepted = rows[0]
    assert accepted["rhea_master_id"] == "RHEA:100"
    assert accepted["decision_id"].startswith("DEC-")
    assert accepted["reviewer"] == "curator-1"
    assert accepted["date"] == "2026-11-04"
    assert accepted["reviewed_evidence_hash"] == run.proposal("E1", "RHEA:100")["evidence_hash"]
    assert accepted["run_id"] == run.manifest["run_id"]
    assert accepted["evidence_matches_run"] == "true"
    assert summary["exported"] == 1
    assert "flag_list_version: none" in out.read_text().splitlines()[0]


def test_summary_counts_add_up(run: RunOutput, log: Path, tmp_path: Path) -> None:  # FR-029
    batch = write_batch(
        tmp_path / "b.tsv", [decision(run, "E1", "accepted"), decision(run, "E2", "rejected")]
    )
    record_decisions(run.run_dir, batch, log)
    out = tmp_path / "CURATED.tsv"
    export_curated(run.run_dir, log, out, flag_version=None)
    summary = json.loads((tmp_path / "CURATED.summary.json").read_text())
    counts = summary["all"]
    assert counts == {"accepted": 1, "rejected": 1, "deferred": 0, "pending": 3, "denominator": 5}
    assert sum(v for k, v in counts.items() if k != "denominator") == counts["denominator"]


def test_flagged_accepted_mapping_exported_with_flag(
    run: RunOutput, log: Path, tmp_path: Path
) -> None:  # FR-023
    apply_flags(
        run.run_dir, write_flag_list(tmp_path / "f.tsv", "v1", [("Drug metabolism", "drug")])
    )
    record_decisions(
        run.run_dir,
        write_batch(
            tmp_path / "b.tsv", [decision(run, "D1", "accepted"), decision(run, "E1", "accepted")]
        ),
        log,
    )
    out = tmp_path / "CURATED.tsv"
    export_curated(run.run_dir, log, out, flag_version="v1")
    rows = {r["vmh_reaction_id"]: r for r in read_rows(out)}
    assert rows["D1"]["subsystem_flagged"] == "true"
    assert rows["D1"]["matched_subsystems"] == "Drug metabolism"
    assert rows["E1"]["subsystem_flagged"] == "false"
    summary = json.loads((tmp_path / "CURATED.summary.json").read_text())
    assert summary["unflagged"]["denominator"] == 4
    assert summary["unflagged"]["accepted"] == 1

    out2 = tmp_path / "CURATED_noflags.tsv"
    export_curated(run.run_dir, log, out2, flag_version=None)
    assert {r["subsystem_flagged"] for r in read_rows(out2)} == {""}


def test_stale_evidence_goes_to_rereview(run: RunOutput, log: Path, tmp_path: Path) -> None:
    record_decisions(
        run.run_dir, write_batch(tmp_path / "b.tsv", [decision(run, "E1", "accepted")]), log
    )
    changed = build_run(tmp_path / "b", p_label="Enantiomers")  # same pair, new evidence
    assert (
        changed.proposal("E1", "RHEA:100")["proposal_id"]
        == run.proposal("E1", "RHEA:100")["proposal_id"]
    )
    assert (
        changed.proposal("E1", "RHEA:100")["evidence_hash"]
        != run.proposal("E1", "RHEA:100")["evidence_hash"]
    )

    out = tmp_path / "CURATED.tsv"
    summary = export_curated(changed.run_dir, log, out, flag_version=None)
    assert read_rows(out) == []
    rereview = read_rows(tmp_path / "CURATED.rereview.tsv")
    assert [r["vmh_reaction_id"] for r in rereview] == ["E1"]
    assert rereview[0]["reviewed_evidence_hash"] != rereview[0]["run_evidence_hash"]
    assert summary["rereview"] == 1

    export_curated(changed.run_dir, log, out, flag_version=None, allow_stale=True)
    rows = read_rows(out)
    assert rows[0]["evidence_matches_run"] == "false"

    # US2-4: the workbook shows the decision was made against other evidence.
    xlsx = tmp_path / "review.xlsx"
    write_review_workbook(changed.run_dir, xlsx, decision_log=log)
    sheet = load_workbook(xlsx)["proposals"]
    header = [c.value for c in sheet[1]]
    pid_col, cur_col = header.index("proposal_id"), header.index("decision_evidence_current")
    values = {r[pid_col]: r[cur_col] for r in sheet.iter_rows(min_row=2, values_only=True)}
    assert values[changed.proposal("E1", "RHEA:100")["proposal_id"]] is False


def test_supersede_changes_effective_decision(run: RunOutput, log: Path, tmp_path: Path) -> None:
    record_decisions(
        run.run_dir, write_batch(tmp_path / "b1.tsv", [decision(run, "E1", "accepted")]), log
    )
    first = read_decision_log(log)[0]
    record_decisions(
        run.run_dir,
        write_batch(
            tmp_path / "b2.tsv", [decision(run, "E1", "rejected", supersedes=first.decision_id)]
        ),
        log,
    )
    entries = read_decision_log(log)
    assert len(entries) == 2  # both rows kept (FR-024)
    effective = effective_decisions(entries)
    assert effective[first.proposal_id].decision.value == "rejected"
    out = tmp_path / "CURATED.tsv"
    export_curated(run.run_dir, log, out, flag_version=None)
    assert read_rows(out) == []


def test_new_decision_must_supersede_current(run: RunOutput, log: Path, tmp_path: Path) -> None:
    record_decisions(
        run.run_dir, write_batch(tmp_path / "b1.tsv", [decision(run, "E1", "accepted")]), log
    )
    with pytest.raises(InputContractError, match="supersede"):
        record_decisions(
            run.run_dir, write_batch(tmp_path / "b2.tsv", [decision(run, "E1", "rejected")]), log
        )


def test_conflicting_heads_fail_export(run: RunOutput, log: Path, tmp_path: Path) -> None:
    record_decisions(
        run.run_dir, write_batch(tmp_path / "b1.tsv", [decision(run, "E1", "accepted")]), log
    )
    first = read_rows(log)[0]
    conflict = dict(first, decision_id="DEC-manual", decision="rejected")
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("\t".join(conflict[c] for c in LOG_COLUMNS) + "\n")
    with pytest.raises(InputContractError, match="conflicting"):
        export_curated(run.run_dir, log, tmp_path / "CURATED.tsv", flag_version=None)


def test_proposal_files_unchanged(run: RunOutput, log: Path, tmp_path: Path) -> None:  # FR-024
    names = ("reaction_proposals.tsv", "participant_evidence.tsv", "run_manifest.json")
    before = {n: sha256_file(run.run_dir / n) for n in names}
    record_decisions(
        run.run_dir, write_batch(tmp_path / "b.tsv", [decision(run, "E1", "accepted")]), log
    )
    export_curated(run.run_dir, log, tmp_path / "CURATED.tsv", flag_version=None)
    assert {n: sha256_file(run.run_dir / n) for n in names} == before
