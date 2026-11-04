"""US4: identical pinned inputs give identical outputs and run_id (FR-003, SC-003)."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from stereomapper.reconciliation.pipeline import RUN_FILES
from stereomapper.tests.reconciliation.conftest import Scenario, propose, read_rows


def build(root: Path) -> Path:
    sc = Scenario(root=root)
    sc.vmh("E1", [("s", "vmh:s"), ("h", "vmh:h", "1")], [("p", "vmh:p")])
    sc.vmh("E2", [("a", "vmh:a"), ("u", "")], [("b", "vmh:b")], direction="reversible")
    sc.vmh("E3", [("x1", "vmh:x1"), ("x2", "vmh:x2")], [("p", "vmh:p")])
    sc.rhea(
        100, [("CHEBI:s", "chebi:s"), ("CHEBI:15378", "chebi:15378", "2")], [("CHEBI:p", "chebi:p")]
    )
    sc.rhea(200, [("CHEBI:b", "chebi:b")], [("CHEBI:a", "chebi:a")])
    sc.rhea(300, [("CHEBI:y1", "chebi:y1"), ("CHEBI:y2", "chebi:y2")], [("CHEBI:p", "chebi:p")])
    sc.same("vmh:s", "chebi:s").same("vmh:p", "chebi:p").same("vmh:h", "chebi:15378")
    sc.same("vmh:a", "chebi:a").rel("vmh:b", "chebi:b", "Protomers")
    sc.rel("vmh:x1", "chebi:y1", "Diastereomers").rel(
        "vmh:x2", "chebi:y2", "Stereo-resolution pairs"
    )
    sc.rel("vmh:x1", "chebi:y2", "Protomers").rel("vmh:x2", "chebi:y1", "Enantiomers")
    return sc.build()


def _strip(manifest_path: Path) -> dict[str, object]:
    data: dict[str, object] = json.loads(manifest_path.read_text())
    data.pop("started_at")
    return data


def test_repeat_runs_are_byte_identical(tmp_path: Path) -> None:
    config = build(tmp_path / "case")
    a = propose(config, tmp_path / "run_a")
    b = propose(config, tmp_path / "run_b")
    assert a.manifest["run_id"] == b.manifest["run_id"]
    for name in RUN_FILES:
        if name == "run_manifest.json":
            assert _strip(a.run_dir / name) == _strip(b.run_dir / name)
        else:
            assert (a.run_dir / name).read_bytes() == (b.run_dir / name).read_bytes(), name


def _shuffle_rows(path: Path, seed: int) -> None:
    lines = path.read_text().splitlines()
    body = lines[1:]
    random.Random(seed).shuffle(body)
    path.write_text("\n".join([lines[0], *body]) + "\n")


# Provenance legitimately changes when input rows move: source line numbers, the input
# checksums behind run_id, and the evidence_hash that covers provenance columns.
PROVENANCE = {"run_id", "evidence_hash", "vmh_source_row", "rhea_source_row"}


def test_input_row_order_changes_only_provenance(tmp_path: Path) -> None:
    config = build(tmp_path / "case")
    a = propose(config, tmp_path / "run_a")
    for name in ("reactions.tsv", "participants.tsv", "subsystems.tsv"):
        _shuffle_rows(tmp_path / "case" / "extract" / name, seed=3)
    b = propose(config, tmp_path / "run_b")
    for name in (
        "reaction_proposals.tsv",
        "participant_evidence.tsv",
        "alternative_alignments.tsv",
        "candidates.tsv",
        "funnel.tsv",
    ):
        rows_a = [
            {k: v for k, v in r.items() if k not in PROVENANCE} for r in read_rows(a.run_dir / name)
        ]
        rows_b = [
            {k: v for k, v in r.items() if k not in PROVENANCE} for r in read_rows(b.run_dir / name)
        ]
        assert rows_a == rows_b, name


def test_changed_input_changes_run_id_and_checksum(tmp_path: Path) -> None:
    config = build(tmp_path / "case")
    a = propose(config, tmp_path / "run_a")
    path = tmp_path / "case" / "extract" / "reactions.tsv"
    path.write_text(path.read_text().replace("E1 equation", "E1 equation (edited)"))
    b = propose(config, tmp_path / "run_b")
    assert a.manifest["run_id"] != b.manifest["run_id"]
    assert (
        a.manifest["inputs"]["reactions"]["checksum"]
        != b.manifest["inputs"]["reactions"]["checksum"]
    )


def test_run_id_independent_of_location(tmp_path: Path) -> None:
    config = build(tmp_path / "case")
    a = propose(config, tmp_path / "run_a")
    moved = tmp_path / "elsewhere" / "nested"
    shutil.copytree(tmp_path / "case", moved)
    b = propose(moved / "config.json", tmp_path / "other_out")
    assert a.manifest["run_id"] == b.manifest["run_id"]
