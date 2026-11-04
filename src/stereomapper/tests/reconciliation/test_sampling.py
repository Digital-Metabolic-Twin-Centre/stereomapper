"""US3-5: stratified acceptance sample and acceptance-rate report (FR-028)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stereomapper.reconciliation.benchmark import run_benchmark
from stereomapper.reconciliation.curation import record_decisions
from stereomapper.reconciliation.flags import apply_flags
from stereomapper.reconciliation.sampling import acceptance_report, draw_sample
from stereomapper.tests.reconciliation.conftest import RunOutput, Scenario, propose, read_rows
from stereomapper.tests.reconciliation.test_curation import write_batch
from stereomapper.tests.reconciliation.test_flags import write_flag_list

LABELS = {"I": None, "S": "Enantiomers", "P": "Protomers"}


@pytest.fixture
def run(scenario: Scenario, tmp_path: Path) -> RunOutput:
    for kind, label in LABELS.items():
        for i in range(6):
            met = f"{kind.lower()}{i}"
            scenario.vmh(
                f"{kind}{i}",
                [(met, f"vmh:{met}")],
                [(f"p{met}", f"vmh:p{met}")],
                subsystems=("Drug metabolism",) if (kind, i) == ("I", 5) else ("T",),
            )
            scenario.rhea(
                1000 + 10 * len(scenario.reactions),
                [(f"CHEBI:{met}", f"chebi:{met}")],
                [(f"CHEBI:p{met}", f"chebi:p{met}")],
            )
            scenario.same(f"vmh:p{met}", f"chebi:p{met}")
            if label:
                scenario.rel(f"vmh:{met}", f"chebi:{met}", label)
            else:
                scenario.same(f"vmh:{met}", f"chebi:{met}")
    out = propose(scenario.build())
    assert len(out.proposals) == 18
    apply_flags(
        out.run_dir, write_flag_list(tmp_path / "f.tsv", "v1", [("Drug metabolism", "drug")])
    )
    return out


def _worksheet(tmp_path: Path, run: RunOutput) -> Path:
    first = run.proposals[0]
    path = tmp_path / "ws.tsv"
    path.write_text(
        "vmh_reaction_id\trhea_id\tcurator_verdict\n"
        f"{first['vmh_reaction_id']}\t{first['rhea_master_id']}\tverified\n",
        encoding="utf-8",
    )
    return path


def test_sample_is_seeded_and_excludes_worksheet_and_flagged(
    run: RunOutput, tmp_path: Path
) -> None:
    bench = tmp_path / "bench"
    run_benchmark(run.run_dir, _worksheet(tmp_path, run), "ws", flag_version="v1", out_dir=bench)
    a = draw_sample(
        run.run_dir, 6, 7, tmp_path / "a.tsv", flag_version="v1", exclude_benchmark_dir=bench
    )
    b = draw_sample(
        run.run_dir, 6, 7, tmp_path / "b.tsv", flag_version="v1", exclude_benchmark_dir=bench
    )
    assert (tmp_path / "a.tsv").read_text() == (tmp_path / "b.tsv").read_text()
    assert len(a) == len(b) == 6
    ids = {r["proposal_id"] for r in a}
    worksheet_pid = run.proposals[0]["proposal_id"]
    flagged_pid = next(p["proposal_id"] for p in run.proposals if p["vmh_reaction_id"] == "I5")
    assert worksheet_pid not in ids and flagged_pid not in ids
    header = (tmp_path / "a.tsv").read_text().splitlines()[:6]
    assert any(line.startswith("# seed: 7") for line in header)
    assert any(line.startswith("# flag_list_version: v1") for line in header)


def test_strata_cover_profiles(run: RunOutput, tmp_path: Path) -> None:
    rows = draw_sample(run.run_dir, 17, 1, tmp_path / "s.tsv", flag_version="v1")
    profiles = {r["stratum"].split("|")[0] for r in rows}
    assert profiles == {"identical_only", "includes_stereo", "includes_protomer"}
    for r in rows:
        assert r["stratum"].split("|")[1] in {"top", "middle", "bottom"}


def test_acceptance_report(run: RunOutput, tmp_path: Path) -> None:
    rows = draw_sample(run.run_dir, 6, 3, tmp_path / "s.tsv", flag_version="v1")
    by_id = {p["proposal_id"]: p for p in run.proposals}
    values = ["accepted", "accepted", "rejected", "deferred"]
    batch = [
        {
            "proposal_id": r["proposal_id"],
            "evidence_hash": by_id[r["proposal_id"]]["evidence_hash"],
            "decision": v,
            "reviewer": "c",
            "date": "2026-11-04",
            "reason": "sample review",
        }
        for r, v in zip(rows, values)
    ]
    log = tmp_path / "log.tsv"
    record_decisions(run.run_dir, write_batch(tmp_path / "b.tsv", batch), log)
    report = acceptance_report(
        run.run_dir, tmp_path / "s.tsv", log, tmp_path / "r.json", flag_version="v1"
    )
    overall = report["overall"]
    assert overall == {
        "accepted": 2,
        "rejected": 1,
        "deferred": 1,
        "pending": 2,
        "decided": 3,
        "sample_size": 6,
        "acceptance_rate": "2/3",
        "acceptance_rate_decimal": "0.6667",
    }
    assert json.loads((tmp_path / "r.json").read_text())["overall"] == overall
    for stratum in report["strata"].values():
        if stratum["decided"] == 0:
            assert stratum["acceptance_rate"] is None
    assert "acceptance rate over reviewed" in report["note"].lower()


def test_sample_file_columns(run: RunOutput, tmp_path: Path) -> None:
    draw_sample(run.run_dir, 3, 5, tmp_path / "s.tsv", flag_version=None)
    rows = read_rows(tmp_path / "s.tsv")
    assert list(rows[0]) == ["proposal_id", "stratum", "rank"]
