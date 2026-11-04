from __future__ import annotations

from pathlib import Path

import pytest

from stereomapper.reconciliation.hashing import sha256_file
from stereomapper.tests.reconciliation.conftest import RunOutput
from stereomapper.tests.reconciliation.test_cli_propose import cli
from stereomapper.tests.reconciliation.test_curation import build_run, decision, write_batch


@pytest.fixture
def run(tmp_path: Path) -> RunOutput:
    return build_run(tmp_path / "a")


def test_record_creates_log_with_header(run: RunOutput, tmp_path: Path) -> None:
    log = tmp_path / "log.tsv"
    batch = write_batch(tmp_path / "b.tsv", [decision(run, "E1", "accepted")])
    assert (
        cli(
            "record-decisions",
            "--run",
            str(run.run_dir),
            "--decisions",
            str(batch),
            "--decision-log",
            str(log),
        )
        == 0
    )
    lines = log.read_text().splitlines()
    assert lines[0].startswith("decision_id\tproposal_id\tevidence_hash")
    assert len(lines) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"proposal_id": "RXP-unknown"},
        {"evidence_hash": "sha256:wrong"},
        {"reason": ""},
        {"decision": "maybe"},
        {"date": "04/11/2026"},
        {"supersedes": "DEC-does-not-exist"},
    ],
)
def test_invalid_batch_rejected_atomically(
    run: RunOutput, tmp_path: Path, change: dict[str, str]
) -> None:
    log = tmp_path / "log.tsv"
    first = write_batch(tmp_path / "b0.tsv", [decision(run, "E2", "accepted")])
    assert (
        cli(
            "record-decisions",
            "--run",
            str(run.run_dir),
            "--decisions",
            str(first),
            "--decision-log",
            str(log),
        )
        == 0
    )
    before = sha256_file(log)
    bad = write_batch(
        tmp_path / "b.tsv",
        [decision(run, "E1", "accepted"), decision(run, "E3", "accepted", **change)],
    )
    assert (
        cli(
            "record-decisions",
            "--run",
            str(run.run_dir),
            "--decisions",
            str(bad),
            "--decision-log",
            str(log),
        )
        == 2
    )
    assert sha256_file(log) == before


def test_supersedes_must_name_same_proposal(run: RunOutput, tmp_path: Path) -> None:
    log = tmp_path / "log.tsv"
    first = write_batch(tmp_path / "b0.tsv", [decision(run, "E2", "accepted")])
    cli(
        "record-decisions",
        "--run",
        str(run.run_dir),
        "--decisions",
        str(first),
        "--decision-log",
        str(log),
    )
    other_id = log.read_text().splitlines()[1].split("\t")[0]
    bad = write_batch(tmp_path / "b.tsv", [decision(run, "E1", "accepted", supersedes=other_id)])
    assert (
        cli(
            "record-decisions",
            "--run",
            str(run.run_dir),
            "--decisions",
            str(bad),
            "--decision-log",
            str(log),
        )
        == 2
    )


def test_export_requires_flag_choice(run: RunOutput, tmp_path: Path) -> None:
    log = tmp_path / "log.tsv"
    log.write_text("")
    assert (
        cli(
            "export-curated",
            "--run",
            str(run.run_dir),
            "--decision-log",
            str(log),
            "--out",
            str(tmp_path / "c.tsv"),
        )
        == 2
    )
    assert (
        cli(
            "export-curated",
            "--run",
            str(run.run_dir),
            "--decision-log",
            str(log),
            "--out",
            str(tmp_path / "c.tsv"),
            "--no-subsystem-flags",
        )
        == 0
    )
