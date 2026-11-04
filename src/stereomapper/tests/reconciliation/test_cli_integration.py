"""End-to-end: propose -> flag -> benchmark -> review -> record -> export -> sample -> acceptance."""

from __future__ import annotations

import json
from pathlib import Path

from stereomapper.reconciliation.hashing import sha256_file
from stereomapper.tests.reconciliation.conftest import Scenario, read_rows
from stereomapper.tests.reconciliation.test_benchmark import WORKSHEET, populate
from stereomapper.tests.reconciliation.test_cli_propose import cli
from stereomapper.tests.reconciliation.test_curation import write_batch
from stereomapper.tests.reconciliation.test_flags import write_flag_list


def test_full_workflow(tmp_path: Path, scenario: Scenario) -> None:
    config = populate(scenario).build()
    run_dir = tmp_path / "pilot-01"
    assert cli("propose", "--config", str(config), "--out", str(run_dir)) == 0
    proposals_checksum = sha256_file(run_dir / "reaction_proposals.tsv")

    # Funnel stages add up (quickstart §2.2).
    funnel = {r["stage"]: r for r in read_rows(run_dir / "funnel.tsv")}
    assert int(funnel["rhea_excluded_obsolete"]["count"]) == len(
        read_rows(run_dir / "scope_exclusions.tsv")
    )
    assert int(funnel["candidates_total"]["count"]) == (
        int(funnel["candidates_plausible"]["count"])
        + int(funnel["not_plausible_cofactor_only"]["count"])
        + int(funnel["not_plausible_no_supporting_pair"]["count"])
    )
    assert funnel["proposals"]["count"] == funnel["candidates_plausible"]["count"]

    flag_list = write_flag_list(tmp_path / "flags_v1.tsv", "v1", [("Drug metabolism", "drug")])
    assert cli("flag-subsystems", "--run", str(run_dir), "--flag-list", str(flag_list)) == 0

    worksheet = tmp_path / "ws.tsv"
    worksheet.write_text(
        "vmh_reaction_id\trhea_id\tcurator_verdict\n"
        + "\n".join("\t".join(r) for r in WORKSHEET)
        + "\n"
    )
    assert (
        cli(
            "benchmark",
            "--run",
            str(run_dir),
            "--worksheet",
            str(worksheet),
            "--worksheet-version",
            "ws-1",
            "--flags",
            "v1",
        )
        == 0
    )
    report = json.loads(
        (run_dir / "benchmark" / "ws-1" / "v1" / "benchmark_report.json").read_text()
    )
    positives = sum(1 for r in WORKSHEET if r[2].lower() == "verified")
    assert report["subsets"]["full_worksheet"]["denominator"] == positives  # SC-001
    for subset in report["subsets"].values():
        assert (
            subset["recovered_new"] + subset["already_in_baseline"] + subset["unrecovered"]
            == subset["denominator"]
        )

    assert (
        cli(
            "review-workbook",
            "--run",
            str(run_dir),
            "--out",
            str(tmp_path / "review.xlsx"),
            "--flags",
            "v1",
        )
        == 0
    )

    proposals = read_rows(run_dir / "reaction_proposals.tsv")
    batch_rows = [
        {
            "proposal_id": p["proposal_id"],
            "evidence_hash": p["evidence_hash"],
            "decision": "accepted" if i % 2 == 0 else "rejected",
            "reviewer": "curator-1",
            "date": "2026-11-04",
            "reason": "pilot review",
        }
        for i, p in enumerate(proposals)
    ]
    log = tmp_path / "curation" / "decisions.tsv"
    assert (
        cli(
            "record-decisions",
            "--run",
            str(run_dir),
            "--decisions",
            str(write_batch(tmp_path / "batch1.tsv", batch_rows)),
            "--decision-log",
            str(log),
        )
        == 0
    )
    curated = tmp_path / "curation" / "curated_mappings.tsv"
    assert (
        cli(
            "export-curated",
            "--run",
            str(run_dir),
            "--decision-log",
            str(log),
            "--out",
            str(curated),
            "--flags",
            "v1",
        )
        == 0
    )
    accepted = {r["proposal_id"] for r in batch_rows if r["decision"] == "accepted"}
    assert {r["proposal_id"] for r in read_rows(curated)} == accepted  # SC-005

    sample = tmp_path / "curation" / "sample1.tsv"
    assert (
        cli(
            "sample",
            "--run",
            str(run_dir),
            "--size",
            "2",
            "--seed",
            "20261103",
            "--flags",
            "v1",
            "--out",
            str(sample),
        )
        == 0
    )
    acceptance = tmp_path / "curation" / "acceptance1.json"
    assert (
        cli(
            "acceptance",
            "--run",
            str(run_dir),
            "--sample",
            str(sample),
            "--decision-log",
            str(log),
            "--out",
            str(acceptance),
            "--flags",
            "v1",
        )
        == 0
    )
    overall = json.loads(acceptance.read_text())["overall"]
    assert overall["sample_size"] == len(read_rows(sample))

    assert sha256_file(run_dir / "reaction_proposals.tsv") == proposals_checksum
