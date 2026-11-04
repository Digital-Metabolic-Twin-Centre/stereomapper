"""Readers and contract validation for reconciliation inputs (contracts/input-files.md)."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any

from stereomapper.domain.exceptions.base import StereomapperError
from stereomapper.reconciliation.model import ParticipantKind, RheaStatus, Side, VmhDirection


class _Source(str, Enum):
    VMH = "vmh"
    RHEA = "rhea"


class InputContractError(StereomapperError):
    """An input file violates its contract. Raised with the file, row, and field."""

    def __init__(self, file: str, row: int | None, field_name: str, message: str):
        where = f"{file}" + (f", row {row}" if row is not None else "")
        super().__init__(f"{where}, field '{field_name}': {message}")
        self.file = file
        self.row = row
        self.field_name = field_name

    def _get_default_error_code(self) -> str:
        return "INPUT_CONTRACT_ERROR"


@dataclass
class TsvTable:
    path: Path
    columns: list[str]
    rows: list[dict[str, str]]
    comments: list[str] = field(default_factory=list)
    # 1-based line numbers of each data row in the file (header is line 1 after comments)
    line_numbers: list[int] = field(default_factory=list)


def read_tsv(path: Path, required: Iterable[str] = ()) -> TsvTable:
    """Read a UTF-8 TSV. Leading ``#`` lines are kept as comments; extra columns are kept."""
    if not path.is_file():
        raise InputContractError(path.name, None, "-", "file not found")
    comments: list[str] = []
    with open(path, encoding="utf-8", newline="") as handle:
        lines = handle.read().splitlines()
    idx = 0
    while idx < len(lines) and lines[idx].startswith("#"):
        comments.append(lines[idx][1:].strip())
        idx += 1
    if idx >= len(lines):
        raise InputContractError(path.name, None, "-", "missing header row")
    reader = csv.reader(lines[idx:], delimiter="\t", quoting=csv.QUOTE_NONE)
    header = [h.strip() for h in next(reader)]
    missing = [col for col in required if col not in header]
    if missing:
        raise InputContractError(path.name, None, missing[0], "required column is missing")
    rows: list[dict[str, str]] = []
    line_numbers: list[int] = []
    for offset, values in enumerate(reader, start=idx + 2):
        if not any(v.strip() for v in values):
            continue
        if len(values) > len(header):
            raise InputContractError(path.name, offset, "-", "more values than header columns")
        padded = values + [""] * (len(header) - len(values))
        rows.append({h: v.strip() for h, v in zip(header, padded)})
        line_numbers.append(offset)
    return TsvTable(
        path=path, columns=header, rows=rows, comments=comments, line_numbers=line_numbers
    )


def _require_value(table: TsvTable, i: int, column: str) -> str:
    value = table.rows[i].get(column, "")
    if not value:
        raise InputContractError(
            table.path.name, table.line_numbers[i], column, "value is required"
        )
    return value


def _enum_value(table: TsvTable, i: int, column: str, enum: Any, default: str = "") -> str:
    value = table.rows[i].get(column, "") or default
    allowed = {member.value for member in enum}
    if value not in allowed:
        raise InputContractError(
            table.path.name,
            table.line_numbers[i],
            column,
            f"invalid value {value!r}; expected one of {sorted(allowed)}",
        )
    return value


def parse_coefficient(raw: str) -> Fraction | None:
    """Parse a stoichiometric coefficient; None if it is symbolic (for example ``n``)."""
    try:
        value = Fraction(raw)
    except (ValueError, ZeroDivisionError):
        return None
    return value


@dataclass
class Extract:
    reactions: TsvTable
    participants: TsvTable
    subsystems: TsvTable
    manifest: dict[str, Any]
    paths: dict[str, Path]


REACTION_COLUMNS = ("source", "reaction_id", "equation")
PARTICIPANT_COLUMNS = ("source", "reaction_id", "side", "metabolite_id", "coefficient")
SUBSYSTEM_COLUMNS = ("reaction_id", "subsystem")
MANIFEST_FIELDS = ("kg_snapshot", "vmh_release", "rhea_release", "extraction_query", "extracted_at")


def load_extract(extract_dir: Path) -> Extract:
    paths = {
        "reactions": extract_dir / "reactions.tsv",
        "participants": extract_dir / "participants.tsv",
        "subsystems": extract_dir / "subsystems.tsv",
        "extract_manifest": extract_dir / "extract_manifest.json",
    }
    reactions = read_tsv(paths["reactions"], REACTION_COLUMNS)
    participants = read_tsv(paths["participants"], PARTICIPANT_COLUMNS)
    subsystems = read_tsv(paths["subsystems"], SUBSYSTEM_COLUMNS)
    if not paths["extract_manifest"].is_file():
        raise InputContractError("extract_manifest.json", None, "-", "file not found")
    try:
        manifest = json.loads(paths["extract_manifest"].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputContractError("extract_manifest.json", None, "-", f"invalid JSON: {exc}")
    for name in MANIFEST_FIELDS:
        if name not in manifest:
            raise InputContractError("extract_manifest.json", None, name, "field is required")

    seen_reactions: set[tuple[str, str]] = set()
    for i, row in enumerate(reactions.rows):
        source = _enum_value(reactions, i, "source", _Source)
        rid = _require_value(reactions, i, "reaction_id")
        _require_value(reactions, i, "equation")
        key = (source, rid)
        if key in seen_reactions:
            raise InputContractError(
                reactions.path.name, reactions.line_numbers[i], "reaction_id", "duplicate reaction"
            )
        seen_reactions.add(key)
        if source == "rhea":
            _enum_value(reactions, i, "rhea_status", RheaStatus)
        else:
            _enum_value(reactions, i, "direction", VmhDirection)

    seen_entries: set[str] = set()
    for i, row in enumerate(participants.rows):
        source = _enum_value(participants, i, "source", _Source)
        rid = _require_value(participants, i, "reaction_id")
        if (source, rid) not in seen_reactions:
            raise InputContractError(
                participants.path.name,
                participants.line_numbers[i],
                "reaction_id",
                f"no reaction {source}:{rid} in reactions.tsv",
            )
        side = _enum_value(participants, i, "side", Side)
        met = _require_value(participants, i, "metabolite_id")
        kind = _enum_value(participants, i, "kind", ParticipantKind, default="small_molecule")
        raw = _require_value(participants, i, "coefficient")
        coefficient = parse_coefficient(raw)
        if coefficient is None and kind == "small_molecule":
            raise InputContractError(
                participants.path.name,
                participants.line_numbers[i],
                "coefficient",
                f"{raw!r} is not a number (symbolic coefficients are allowed only for "
                "generic, polymer, or macromolecule participants)",
            )
        if coefficient is not None and coefficient <= 0:
            raise InputContractError(
                participants.path.name,
                participants.line_numbers[i],
                "coefficient",
                "must be positive",
            )
        entry_key = "|".join((source, rid, side, met, row.get("compartment", "")))
        if entry_key in seen_entries:
            raise InputContractError(
                participants.path.name,
                participants.line_numbers[i],
                "metabolite_id",
                f"duplicate participant entry {entry_key}",
            )
        seen_entries.add(entry_key)

    vmh_ids = {rid for source, rid in seen_reactions if source == "vmh"}
    for i, _row in enumerate(subsystems.rows):
        rid = _require_value(subsystems, i, "reaction_id")
        _require_value(subsystems, i, "subsystem")
        if rid not in vmh_ids:
            raise InputContractError(
                subsystems.path.name,
                subsystems.line_numbers[i],
                "reaction_id",
                f"no VMH reaction {rid} in reactions.tsv",
            )

    return Extract(reactions, participants, subsystems, manifest, paths)


def read_baseline(path: Path) -> TsvTable:
    table = read_tsv(path, ("vmh_reaction_id", "rhea_id"))
    for i in range(len(table.rows)):
        _require_value(table, i, "vmh_reaction_id")
        _require_value(table, i, "rhea_id")
    return table


def read_worksheet(path: Path) -> TsvTable:
    table = read_tsv(path, ("vmh_reaction_id", "rhea_id", "curator_verdict"))
    for i in range(len(table.rows)):
        _require_value(table, i, "vmh_reaction_id")
        _require_value(table, i, "rhea_id")
    return table


@dataclass
class FlagList:
    list_version: str
    entries: list[dict[str, str]]
    path: Path


def read_flag_list(path: Path) -> FlagList:
    table = read_tsv(path, ("subsystem", "category", "rationale"))
    version = ""
    for comment in table.comments:
        if comment.lower().startswith("list_version:"):
            version = comment.split(":", 1)[1].strip()
    if not version:
        raise InputContractError(
            path.name, 1, "list_version", "first comment line must be '# list_version: <version>'"
        )
    if "/" in version or version in (".", "..", "none"):
        raise InputContractError(path.name, 1, "list_version", f"invalid version {version!r}")
    seen: set[str] = set()
    for i, row in enumerate(table.rows):
        for column in ("subsystem", "category", "rationale"):
            _require_value(table, i, column)
        if row["subsystem"] in seen:
            raise InputContractError(
                path.name, table.line_numbers[i], "subsystem", "duplicate subsystem"
            )
        seen.add(row["subsystem"])
    return FlagList(list_version=version, entries=table.rows, path=path)


DECISION_BATCH_COLUMNS = ("proposal_id", "evidence_hash", "decision", "reviewer", "date", "reason")


def read_decision_batch(path: Path) -> TsvTable:
    return read_tsv(path, DECISION_BATCH_COLUMNS)
