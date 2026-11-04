from __future__ import annotations

import json
from pathlib import Path

import pytest

from stereomapper.domain.exceptions import ConfigurationError
from stereomapper.reconciliation.flags import apply_flags, load_flags
from stereomapper.reconciliation.hashing import sha256_file
from stereomapper.reconciliation.inputs import InputContractError
from stereomapper.tests.reconciliation.conftest import RunOutput, Scenario, propose, read_rows

S, P = ("s", "vmh:s"), ("p", "vmh:p")


def write_flag_list(path: Path, version: str, rows: list[tuple[str, str]]) -> Path:
    lines = [f"# list_version: {version}", "subsystem\tcategory\trationale"]
    lines += [f"{sub}\t{cat}\tconfounds results" for sub, cat in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def run(scenario: Scenario) -> RunOutput:
    scenario.vmh("G1", [S], [P], subsystems=("Glycolysis/gluconeogenesis", "Other"))
    scenario.vmh("D1", [S], [P], subsystems=("Drug metabolism",))
    scenario.vmh("K1", [S], [P], subsystems=("Citric acid cycle",))
    scenario.vmh("N1", [("n", "vmh:n")], [("m", "vmh:m")], subsystems=("Citric acid cycle",))
    scenario.rhea(100, [("CHEBI:s", "chebi:s")], [("CHEBI:p", "chebi:p")])
    scenario.same("vmh:s", "chebi:s").same("vmh:p", "chebi:p")
    return propose(scenario.build())


def test_apply_flags_writes_versioned_directory(run: RunOutput, tmp_path: Path) -> None:
    before = sha256_file(run.run_dir / "reaction_proposals.tsv")
    hashes = {r["evidence_hash"] for r in run.proposals}
    flag_list = write_flag_list(
        tmp_path / "flags.tsv",
        "v1",
        [("Glycolysis/gluconeogenesis", "sugar"), ("Drug metabolism", "drug")],
    )
    manifest = apply_flags(run.run_dir, flag_list)
    flag_dir = run.run_dir / "flags" / "v1"
    for name in (
        "flag_list.tsv",
        "subsystem_flags.tsv",
        "flagged_proposals.tsv",
        "flags_manifest.json",
    ):
        assert (flag_dir / name).is_file()
    flagged = read_rows(flag_dir / "subsystem_flags.tsv")
    # Any-match rule: G1 has two subsystems, one listed.
    assert [r["vmh_reaction_id"] for r in flagged] == ["D1", "G1"]
    assert manifest["flagged_vmh_reactions"] == 2
    assert manifest["flagged_proposals"] == 2
    assert manifest["flagged_by_category"] == {
        "drug": {"proposals": 1, "vmh_reactions": 1},
        "sugar": {"proposals": 1, "vmh_reactions": 1},
    }
    # FR-007: proposals and evidence hashes unchanged.
    assert sha256_file(run.run_dir / "reaction_proposals.tsv") == before
    assert {r["evidence_hash"] for r in read_rows(run.run_dir / "reaction_proposals.tsv")} == hashes


def test_coverage_with_and_without_flags(run: RunOutput, tmp_path: Path) -> None:  # FR-006
    flag_list = write_flag_list(tmp_path / "f.tsv", "v1", [("Drug metabolism", "drug")])
    apply_flags(run.run_dir, flag_list)
    data = json.loads((run.run_dir / "flags" / "v1" / "flags_manifest.json").read_text())
    cov = data["coverage"]
    # 4 VMH reactions in scope; 3 have proposals (G1, D1, K1); D1 flagged.
    assert cov["vmh_reactions_with_proposals"] == {
        "all": "3/4",
        "unflagged": "2/4",
        "flagged": "1/4",
    }
    assert cov["proposals"] == {"all": "3/3", "unflagged": "2/3", "flagged": "1/3"}


def test_unknown_subsystem_rejected_and_nothing_written(run: RunOutput, tmp_path: Path) -> None:
    flag_list = write_flag_list(tmp_path / "f.tsv", "v1", [("Glycolysis/gluconeogenesys", "sugar")])
    with pytest.raises(InputContractError, match="gluconeogenesys"):
        apply_flags(run.run_dir, flag_list)
    assert not (run.run_dir / "flags" / "v1").exists()


def test_existing_version_not_overwritten(run: RunOutput, tmp_path: Path) -> None:
    flag_list = write_flag_list(tmp_path / "f.tsv", "v1", [("Drug metabolism", "drug")])
    apply_flags(run.run_dir, flag_list)
    with pytest.raises(ConfigurationError, match="already exists"):
        apply_flags(run.run_dir, flag_list)


def test_load_flags(run: RunOutput, tmp_path: Path) -> None:
    apply_flags(
        run.run_dir, write_flag_list(tmp_path / "f.tsv", "v1", [("Drug metabolism", "drug")])
    )
    flags = load_flags(run.run_dir, "v1")
    assert flags.version == "v1"
    assert flags.is_vmh_flagged("D1") and not flags.is_vmh_flagged("K1")
    assert flags.categories_for_vmh("D1") == ["drug"]
    none = load_flags(run.run_dir, None)
    assert none.version == "none" and not none.is_vmh_flagged("D1")
    with pytest.raises(ConfigurationError):
        load_flags(run.run_dir, "v9")
