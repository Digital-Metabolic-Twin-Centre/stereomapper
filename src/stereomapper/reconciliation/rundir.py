"""Read a finished run directory."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from stereomapper.domain.exceptions import ConfigurationError
from stereomapper.reconciliation.inputs import read_tsv
from stereomapper.reconciliation.outputs import read_json


@dataclass
class RunDir:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not (self.path / "run_manifest.json").is_file():
            raise ConfigurationError(f"{self.path} is not a reconcile run directory")

    def rows(self, name: str) -> list[dict[str, str]]:
        return read_tsv(self.path / name).rows

    @cached_property
    def manifest(self) -> dict[str, Any]:
        data: dict[str, Any] = read_json(self.path / "run_manifest.json")
        return data

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @cached_property
    def proposals(self) -> list[dict[str, str]]:
        return self.rows("reaction_proposals.tsv")

    @cached_property
    def proposals_by_id(self) -> dict[str, dict[str, str]]:
        return {p["proposal_id"]: p for p in self.proposals}

    def funnel_count(self, stage: str) -> int:
        for row in self.manifest["funnel"]:
            if row["stage"] == stage:
                return int(row["count"])
        raise KeyError(stage)
