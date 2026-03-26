"""Data models for confidence scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ConfidenceResult:
    score: int  # 0..100
    bin: str  # "high" | "medium" | "low" | "very_low"
    contributors: dict[str, float]  # feature -> weighted contribution in support
    expectations: dict[str, Any]  # IK/charge/stereo expectation checks (True/False)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d
