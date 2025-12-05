"""Data models for confidence scoring."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class ConfidenceResult:
    score: int                      # 0..100
    bin: str                        # "high" | "medium" | "low" | "very_low"
    contributors: Dict[str, float]  # feature -> weighted contribution in support
    expectations: Dict[str, Any]    # IK/charge/stereo expectation checks (True/False)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
