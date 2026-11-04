"""Deterministic TSV/JSON writers (contracts/output-files.md)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any

from stereomapper.reconciliation.hashing import json_default


def format_value(value: Any) -> str:
    """The single formatting rule for every TSV cell."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value) if isinstance(value, (set, frozenset)) else list(value)
        return ";".join(format_value(v) for v in items)
    text = str(value)
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def fraction_columns(numerator: int, denominator: int) -> tuple[str, str]:
    """Return the ``n/d`` text and a 4-place decimal (empty when the denominator is 0)."""
    text = f"{numerator}/{denominator}"
    decimal = f"{numerator / denominator:.4f}" if denominator else ""
    return text, decimal


def write_tsv(
    path: str | Path,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
    comments: Sequence[str] = (),
) -> int:
    count = 0
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for comment in comments:
            handle.write(f"# {comment}\n")
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(format_value(row.get(col)) for col in columns) + "\n")
            count += 1
    return count


def write_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, sort_keys=True, indent=2, default=json_default)
        handle.write("\n")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
