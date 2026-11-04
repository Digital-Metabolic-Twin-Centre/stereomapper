"""Build reactions and participant entries from extract rows (research R7, R8).

Raw source IDs are never modified; normalized identifiers are only used for lookups.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from stereomapper.reconciliation.inputs import Extract, parse_coefficient
from stereomapper.reconciliation.model import (
    Orientation,
    ParticipantEntry,
    ParticipantKind,
    RheaStatus,
    Side,
    SourceReaction,
    VmhDirection,
)

_RHEA_DIGITS = re.compile(r"^(?:RHEA:)?(\d+)$", re.IGNORECASE)

# Rhea numbering convention: master N, then N+1 (LR), N+2 (RL), N+3 (BI).
_DIRECTIONAL_OFFSETS = {1: "LR", 2: "RL", 3: "BI"}


def normalize_rhea_id(value: str) -> str:
    """``17825``, ``rhea:17825``, ``RHEA:17825`` -> ``RHEA:17825``; anything else is upper-cased."""
    text = value.strip()
    match = _RHEA_DIGITS.match(text)
    return f"RHEA:{match.group(1)}" if match else text.upper()


def _rhea_number(value: str) -> int | None:
    match = _RHEA_DIGITS.match(value.strip())
    return int(match.group(1)) if match else None


def directional_kind(master_id: str, directional_id: str) -> str:
    """``LR``, ``RL``, ``BI``, or empty when the ID doesn't follow the Rhea numbering."""
    master, other = _rhea_number(master_id), _rhea_number(directional_id)
    if master is None or other is None:
        return ""
    return _DIRECTIONAL_OFFSETS.get(other - master, "")


def expected_directional_kind(direction: VmhDirection | None, orientation: Orientation) -> str:
    """Rhea directional kind matching a VMH direction once sides are aligned."""
    if direction is None or direction == VmhDirection.UNKNOWN:
        return ""
    if direction == VmhDirection.REVERSIBLE:
        return "BI"
    forward = direction == VmhDirection.FORWARD
    if orientation == Orientation.REVERSED:
        forward = not forward
    return "LR" if forward else "RL"


def _split(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(";") if part.strip())


@dataclass
class Reactions:
    vmh: dict[str, SourceReaction] = field(default_factory=dict)
    rhea_masters: dict[str, SourceReaction] = field(default_factory=dict)
    # normalized Rhea ID (master or directional) -> master reaction ID as written
    rhea_master_of: dict[str, str] = field(default_factory=dict)
    # rows for Rhea directional variants: reaction_id -> master ID
    rhea_variants: dict[str, str] = field(default_factory=dict)

    def all_reactions(self) -> Iterator[SourceReaction]:
        yield from self.vmh.values()
        yield from self.rhea_masters.values()

    def master_for(self, rhea_id: str) -> str | None:
        return self.rhea_master_of.get(normalize_rhea_id(rhea_id))


def build_reactions(extract: Extract) -> Reactions:
    out = Reactions()
    table = extract.reactions
    variants: list[tuple[str, str, int]] = []
    for row, line in zip(table.rows, table.line_numbers):
        rid = row["reaction_id"]
        if row["source"] == "vmh":
            out.vmh[rid] = SourceReaction(
                source="vmh",
                reaction_id=rid,
                equation=row["equation"],
                source_row=line,
                vmh_direction=VmhDirection(row.get("direction") or "unknown"),
                xref_rhea=_split(row.get("xref_rhea", "")),
            )
            continue
        master = row.get("rhea_master_id", "")
        if master and normalize_rhea_id(master) != normalize_rhea_id(rid):
            variants.append((rid, master, line))
            continue
        out.rhea_masters[rid] = SourceReaction(
            source="rhea",
            reaction_id=rid,
            equation=row["equation"],
            source_row=line,
            rhea_master_id=rid,
            rhea_directional_ids=_split(row.get("rhea_directional_ids", "")),
            rhea_status=RheaStatus(row["rhea_status"]),
        )
        out.rhea_master_of[normalize_rhea_id(rid)] = rid

    by_normalized = {normalize_rhea_id(k): k for k in out.rhea_masters}
    for rid, master, _line in variants:
        master_id = by_normalized.get(normalize_rhea_id(master))
        if master_id is None:
            continue  # variant without a master row: kept out of the index, IDs stay raw
        out.rhea_variants[rid] = master_id
        reaction = out.rhea_masters[master_id]
        if rid not in reaction.rhea_directional_ids:
            reaction.rhea_directional_ids = tuple(sorted(reaction.rhea_directional_ids + (rid,)))
    for master_id, reaction in out.rhea_masters.items():
        for directional in reaction.rhea_directional_ids:
            out.rhea_master_of.setdefault(normalize_rhea_id(directional), master_id)

    for row, line in zip(extract.subsystems.rows, extract.subsystems.line_numbers):
        reaction = out.vmh[row["reaction_id"]]
        if row["subsystem"] not in reaction.subsystems:
            reaction.subsystems = reaction.subsystems + (row["subsystem"],)
    for reaction in out.vmh.values():
        reaction.subsystems = tuple(sorted(reaction.subsystems))

    part = extract.participants
    for row, line in zip(part.rows, part.line_numbers):
        entry = ParticipantEntry(
            source=row["source"],
            reaction_id=row["reaction_id"],
            side=Side(row["side"]),
            metabolite_id=row["metabolite_id"],
            name=row.get("name", "") or row["metabolite_id"],
            compartment=row.get("compartment", ""),
            coefficient_raw=row["coefficient"],
            coefficient=parse_coefficient(row["coefficient"]),
            kind=ParticipantKind(row.get("kind") or "small_molecule"),
            structure_curie=row.get("structure_curie", ""),
            formula=row.get("formula", ""),
            charge=row.get("charge", ""),
            source_row=line,
        )
        if entry.source == "vmh":
            out.vmh[entry.reaction_id].participants.append(entry)
        elif entry.reaction_id in out.rhea_masters:
            out.rhea_masters[entry.reaction_id].participants.append(entry)
        # Participants of directional variants are not used: proposals use the master equation.
    for reaction in out.all_reactions():
        reaction.participants.sort(key=lambda e: e.entry_key)
    return out


def sides(
    reaction: SourceReaction, orientation: Orientation
) -> tuple[list[ParticipantEntry], list[ParticipantEntry]]:
    """(left, right) entries of a Rhea reaction as aligned to the VMH sides."""
    left, right = reaction.side_entries(Side.LEFT), reaction.side_entries(Side.RIGHT)
    return (left, right) if orientation == Orientation.SAME else (right, left)


def is_transport_vmh(reaction: SourceReaction) -> bool:
    return len({p.compartment for p in reaction.participants if p.compartment}) > 1


def is_transport_rhea(reaction: SourceReaction) -> bool:
    return any(p.compartment for p in reaction.participants)
