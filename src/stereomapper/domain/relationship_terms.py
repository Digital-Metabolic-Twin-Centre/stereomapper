"""Lookup helpers for StereoMapper relationship ontology terms."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationshipTerm:
    label: str
    term_id: str


_RELATIONSHIP_TERM_IDS = {
    "Identical structures": "SMRO:0001",
    "Identical structures with undetermined charge": "SMRO:0002",
    "Protomers": "SMRO:0003",
    "Indistinguishable structures": "SMRO:0004",
    "Enantiomers": "SMRO:0005",
    "Diastereomers": "SMRO:0006",
    "Stereo-resolution pairs": "SMRO:0007",
    "Unclassified": "SMRO:0008",
    "Unresolved": "SMRO:0009",
}


def _normalize_label(label: str) -> str:
    cleaned = label.strip().lower().replace("-", " ")
    return " ".join(cleaned.split())


_NORMALIZED_TERM_IDS = {
    _normalize_label(label): term_id for label, term_id in _RELATIONSHIP_TERM_IDS.items()
}


def get_relationship_term_id(classification: str | None) -> str | None:
    """Return the SMRO term id for a relationship classification label."""
    if classification is None:
        return None
    normalized = _normalize_label(str(classification))
    if not normalized:
        return None
    return _NORMALIZED_TERM_IDS.get(normalized)


def get_relationship_term(classification: str | None) -> RelationshipTerm | None:
    term_id = get_relationship_term_id(classification)
    if not term_id or classification is None:
        return None
    return RelationshipTerm(label=str(classification).strip(), term_id=term_id)
