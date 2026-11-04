from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stereomapper.domain.exceptions import ConfigurationError
from stereomapper.reconciliation.config import load_run_config
from stereomapper.reconciliation.hashing import sha256_file
from stereomapper.tests.reconciliation.conftest import Scenario


def _build(scenario: Scenario, **overrides: Any) -> Path:
    scenario.vmh("R1", [("a", "chebi:1")], [("b", "chebi:2")])
    scenario.rhea(10, [("CHEBI:1", "chebi:1")], [("CHEBI:2", "chebi:2")])
    return scenario.build(**overrides)


def test_defaults_merged_and_paths_resolved(scenario: Scenario) -> None:
    config = load_run_config(_build(scenario))
    assert config.raw["alignment"]["version"] == "align-v1"
    assert config.raw["ranking"]["version"] == "rank-v1"
    assert config.rhea_status_include == ("approved", "preliminary")
    assert config.extract_dir == (scenario.root / "extract").resolve()
    assert config.relative(config.stereomapper_db) == "results.sqlite"


def test_checksum_mismatch_rejected(scenario: Scenario) -> None:
    path = _build(scenario)
    raw = json.loads(path.read_text())
    raw["inputs"]["checksums"] = {"results.sqlite": "sha256:" + "0" * 64}
    path.write_text(json.dumps(raw))
    with pytest.raises(ConfigurationError, match="Checksum mismatch"):
        load_run_config(path)


def test_matching_checksum_accepted(scenario: Scenario) -> None:
    path = _build(scenario)
    raw = json.loads(path.read_text())
    raw["inputs"]["checksums"] = {"results.sqlite": sha256_file(scenario.root / "results.sqlite")}
    path.write_text(json.dumps(raw))
    load_run_config(path)


def test_unknown_alignment_version_rejected(scenario: Scenario) -> None:
    with pytest.raises(ConfigurationError, match="alignment.version"):
        load_run_config(_build(scenario, alignment={"version": "align-v9"}))


def test_pathway_exclusion_cannot_be_configured(scenario: Scenario) -> None:
    # FR-005: no pathway or subsystem exclusion before proposals are generated.
    with pytest.raises(ConfigurationError, match="FR-005"):
        load_run_config(
            _build(scenario, scope={"glucose": {"enabled": True, "kg_pathway_id": "X"}})
        )


def test_retrieval_cap_rejected(scenario: Scenario) -> None:
    with pytest.raises(ConfigurationError, match="FR-018"):
        load_run_config(_build(scenario, retrieval={"cap": 50}))


def test_missing_required_input(scenario: Scenario) -> None:
    with pytest.raises(ConfigurationError, match="version_tag"):
        load_run_config(_build(scenario, inputs={"version_tag": ""}))
