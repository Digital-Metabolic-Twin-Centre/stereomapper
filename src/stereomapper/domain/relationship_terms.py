"""Lookup helpers for StereoMapper relationship ontology terms."""

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


def get_relationship_term_id(classification: str | None) -> str | None:
    """Return the SMRO term id for a relationship classification label."""
    if classification is None:
        return None
    return _RELATIONSHIP_TERM_IDS.get(str(classification).strip())
