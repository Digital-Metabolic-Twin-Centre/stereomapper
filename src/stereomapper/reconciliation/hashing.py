"""Deterministic hashing helpers (research R10)."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """Return ``sha256:<hex>`` for a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def json_default(obj: Any) -> Any:
    if isinstance(obj, Fraction):
        return f"{obj.numerator}/{obj.denominator}"
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Cannot serialise {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """Serialise with sorted keys and no whitespace so equal content hashes equally."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=json_default)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, n: int = 16) -> str:
    return sha256_text(text)[:n]


def proposal_id(vmh_reaction_id: str, rhea_master_id: str) -> str:
    """Stable identifier for a VMH-Rhea pair across runs."""
    return "RXP-" + short_hash(f"{vmh_reaction_id}|{rhea_master_id}")
