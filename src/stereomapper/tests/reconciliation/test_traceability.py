"""US4: manifest completeness and traceability to source records (US4-1, US4-3, SC-004)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from stereomapper.domain import relationship_terms
from stereomapper.reconciliation.manifest import smro_vocabulary
from stereomapper.tests.reconciliation.conftest import RunOutput, propose, read_rows
from stereomapper.tests.reconciliation.test_determinism import build

RUN_RECORD_FIELDS = (
    "run_id",
    "started_at",
    "output_schema_version",
    "extract",
    "inputs",
    "stereomapper_results",
    "baseline_version",
    "cofactors",
    "smro_vocabulary",
    "config",
    "stereomapper_version",
    "git_commit",
    "python_version",
    "funnel",
    "greedy_alignment_count",
    "trace",
    "nondeterminism",
)


@pytest.fixture
def run(tmp_path: Path) -> RunOutput:
    return propose(build(tmp_path / "case"))


def test_manifest_has_every_run_record_field(run: RunOutput) -> None:
    m = run.manifest
    for name in RUN_RECORD_FIELDS:
        assert name in m, name
    assert m["output_schema_version"] == "reconcile-out-v1"
    assert m["extract"]["vmh_release"] == "vmh-test"
    assert m["extract"]["rhea_release"] == "rhea-test"
    assert m["extract"]["kg_snapshot"] == "kg-test-1"
    assert m["stereomapper_results"]["version_tag"] == "v-test"
    assert m["stereomapper_results"]["checksum"].startswith("sha256:")
    assert m["baseline_version"] == "baseline-test"
    assert m["cofactors"]["version"] == "cofactors-v1"
    assert m["smro_vocabulary"]["terms_sha256"].startswith("sha256:")
    for name in (
        "reactions",
        "participants",
        "subsystems",
        "extract_manifest",
        "stereomapper_db",
        "baseline",
    ):
        assert m["inputs"][name]["checksum"].startswith("sha256:")
        assert not Path(m["inputs"][name]["path"]).is_absolute()
    assert m["trace"]["reaction_proposals.tsv"]["stereomapper_db"] == (
        m["inputs"]["stereomapper_db"]["checksum"]
    )


def test_smro_vocabulary_changes_with_term_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    before = smro_vocabulary()["terms_sha256"]
    table = dict(relationship_terms._RELATIONSHIP_TERM_IDS)
    table["Enantiomers"] = "SMRO:9999"
    monkeypatch.setattr(relationship_terms, "_RELATIONSHIP_TERM_IDS", table)
    assert smro_vocabulary()["terms_sha256"] != before


def test_evidence_traces_to_inputs_and_db(run: RunOutput) -> None:
    extract = run.run_dir.parent / "extract"
    participant_lines = (extract / "participants.tsv").read_text().splitlines()
    header = participant_lines[0].split("\t")
    con = sqlite3.connect(run.run_dir.parent / "results.sqlite")
    rels = {
        frozenset((a, b)): (label, term)
        for a, b, label, term in con.execute(
            "SELECT cluster_a, cluster_b, classification, classification_term_id "
            "FROM relationships WHERE version_tag = 'v-test'"
        )
    }
    con.close()
    for row in run.evidence:
        for side in ("vmh", "rhea"):
            if not row[f"{side}_entry_key"]:
                continue
            line = participant_lines[int(row[f"{side}_source_row"]) - 1].split("\t")
            source = dict(zip(header, line))
            assert source["metabolite_id"] == row[f"{side}_metabolite_id"]
            assert source["reaction_id"] == row[f"{side}_reaction_id"]
        if row["evidence_type"] == "relationship":
            a, b = (int(x) for x in row["sm_cluster_ids"].split(";"))
            assert rels[frozenset((a, b))] == (row["relationship_label"], row["smro_term_id"])


def test_proposals_trace_to_reactions(run: RunOutput) -> None:
    reactions = read_rows(run.run_dir.parent / "extract" / "reactions.tsv")
    ids = {(r["source"], r["reaction_id"]) for r in reactions}
    for p in run.proposals:
        assert ("vmh", p["vmh_reaction_id"]) in ids
        assert ("rhea", p["rhea_master_id"]) in ids
