"""Read-only StereoMapper lookups scoped to one ``version_tag`` (research R2, R6)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from stereomapper.domain.exceptions import ConfigurationError
from stereomapper.domain.relationship_terms import get_relationship_term_id
from stereomapper.reconciliation.config import COFACTORS_V1
from stereomapper.reconciliation.hashing import sha256_file, sha256_text
from stereomapper.reconciliation.inputs import read_tsv
from stereomapper.reconciliation.model import (
    EvidenceType,
    PairEvidence,
    ParticipantEntry,
    ParticipantKind,
    StructureState,
)

_CHUNK = 500
IDENTICAL = ("Identical structures", "SMRO:0001")
PROTON_CURIE = "chebi:15378"


def normalize_curie(curie: str) -> str:
    """Lower-case the namespace prefix only (``CHEBI:15377`` -> ``chebi:15377``)."""
    text = curie.strip()
    if ":" not in text:
        return text
    prefix, local = text.split(":", 1)
    return f"{prefix.lower()}:{local}"


def _chunks(items: list[Any], size: int = _CHUNK) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _map_direction(direction: str | None, vmh_is_a: bool) -> str:
    if not direction:
        return ""
    if direction == "A_to_B":
        return "vmh_more_resolved" if vmh_is_a else "rhea_more_resolved"
    if direction == "B_to_A":
        return "rhea_more_resolved" if vmh_is_a else "vmh_more_resolved"
    return direction


@dataclass(frozen=True)
class _Relationship:
    cluster_a: int
    cluster_b: int
    label: str
    term_id: str
    direction: str
    score: float | None


class StereoMapperEvidence:
    """Loads cluster membership, cluster status, and relationships for a set of CURIEs."""

    def __init__(self, db_path: Path, version_tag: str, curies: Iterable[str]):
        self.db_path = Path(db_path)
        self.version_tag = version_tag
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        self._con = sqlite3.connect(uri, uri=True)
        try:
            self._load(sorted({normalize_curie(c) for c in curies if c}))
        except Exception:
            self._con.close()
            raise

    def close(self) -> None:
        self._con.close()

    def _load(self, curies: list[str]) -> None:
        cur = self._con.cursor()
        found = cur.execute(
            "SELECT 1 FROM relationships WHERE version_tag = ? LIMIT 1", (self.version_tag,)
        ).fetchone()
        if not found:
            raise ConfigurationError(
                f"version_tag {self.version_tag!r} has no relationships in {self.db_path.name}",
                config_field="inputs.version_tag",
            )
        self._clusters_of: dict[str, tuple[int, ...]] = {}
        for chunk in _chunks(curies):
            marks = ",".join("?" * len(chunk))
            rows = cur.execute(
                f"SELECT member_curie, cluster_id FROM cluster_members "
                f"WHERE member_curie IN ({marks})",
                chunk,
            ).fetchall()
            for curie, cid in rows:
                self._clusters_of[curie] = tuple(sorted(self._clusters_of.get(curie, ()) + (cid,)))
        cluster_ids = sorted({c for ids in self._clusters_of.values() for c in ids})
        self._status: dict[int, tuple[str, str]] = {}
        self._members: dict[int, set[str]] = {}
        for chunk in _chunks(cluster_ids):
            marks = ",".join("?" * len(chunk))
            for cid, status, error in cur.execute(
                f"SELECT cluster_id, status, error FROM clusters WHERE cluster_id IN ({marks})",
                chunk,
            ):
                self._status[cid] = (status or "", error or "")
            for cid, curie in cur.execute(
                f"SELECT cluster_id, member_curie FROM cluster_members "
                f"WHERE cluster_id IN ({marks})",
                chunk,
            ):
                self._members.setdefault(cid, set()).add(curie)
        wanted = set(cluster_ids)
        self._relationships: dict[frozenset[int], _Relationship] = {}
        self._related: dict[int, set[int]] = {}
        for chunk in _chunks(cluster_ids):
            marks = ",".join("?" * len(chunk))
            rows = cur.execute(
                f"SELECT cluster_a, cluster_b, classification, classification_term_id, "
                f"direction, score FROM relationships "
                f"WHERE version_tag = ? AND cluster_a IN ({marks})",
                [self.version_tag, *chunk],
            ).fetchall()
            for a, b, label, term_id, direction, score in rows:
                if b not in wanted or a == b:
                    continue
                key = frozenset((a, b))
                rel = _Relationship(
                    a,
                    b,
                    label,
                    term_id or get_relationship_term_id(label) or "",
                    direction or "",
                    score,
                )
                previous = self._relationships.get(key)
                if previous is None or (previous.cluster_a, previous.cluster_b) > (a, b):
                    self._relationships[key] = rel
                self._related.setdefault(a, set()).add(b)
                self._related.setdefault(b, set()).add(a)

    # -- participant level ---------------------------------------------------------------

    def clusters_of_curie(self, curie: str) -> tuple[int, ...]:
        return self._clusters_of.get(normalize_curie(curie), ())

    def clusters_for(self, entry: ParticipantEntry) -> tuple[int, ...]:
        if not entry.structure_curie:
            return ()
        return self._clusters_of.get(normalize_curie(entry.structure_curie), ())

    def cluster_for(self, entry: ParticipantEntry) -> int | None:
        clusters = self.clusters_for(entry)
        return clusters[0] if clusters else None

    def structure_state(self, entry: ParticipantEntry) -> StructureState:
        if entry.kind != ParticipantKind.SMALL_MOLECULE:
            return StructureState.NOT_ASSESSABLE
        if not entry.structure_curie:
            return StructureState.STRUCTURE_UNAVAILABLE
        cluster = self.cluster_for(entry)
        if cluster is None:
            return StructureState.NO_RESULT
        if self._status.get(cluster, ("", ""))[0] != "passed":
            return StructureState.PROCESSING_FAILED
        return StructureState.ASSESSABLE

    def structure_error(self, entry: ParticipantEntry) -> str:
        cluster = self.cluster_for(entry)
        if cluster is None:
            return ""
        status, error = self._status.get(cluster, ("", ""))
        return error if status != "passed" else ""

    def members(self, cluster_id: int) -> set[str]:
        return self._members.get(cluster_id, set())

    def related_clusters(self, cluster_id: int) -> set[int]:
        return self._related.get(cluster_id, set())

    # -- pair level ----------------------------------------------------------------------

    def pair_evidence(self, vmh: ParticipantEntry, rhea: ParticipantEntry) -> PairEvidence:
        if (
            self.structure_state(vmh) != StructureState.ASSESSABLE
            or self.structure_state(rhea) != StructureState.ASSESSABLE
        ):
            return PairEvidence(EvidenceType.NONE)
        cv, cr = self.cluster_for(vmh), self.cluster_for(rhea)
        if cv is None or cr is None:  # unreachable for assessable entries
            return PairEvidence(EvidenceType.NONE)
        if cv == cr:
            return PairEvidence(
                EvidenceType.CLUSTER_MEMBERSHIP,
                IDENTICAL[0],
                IDENTICAL[1],
                "",
                None,
                self.version_tag,
            )
        rel = self._relationships.get(frozenset((cv, cr)))
        if rel is None:
            return PairEvidence(EvidenceType.NONE, version_tag=self.version_tag)
        return PairEvidence(
            EvidenceType.RELATIONSHIP,
            rel.label,
            rel.term_id,
            _map_direction(rel.direction, vmh_is_a=rel.cluster_a == cv),
            rel.score,
            self.version_tag,
        )


@dataclass(frozen=True)
class CofactorList:
    version: str
    checksum: str
    curies: frozenset[str]
    source: str


def load_cofactor_list(spec: str, config_dir: Path | None = None) -> CofactorList:
    """``builtin:cofactors-v1`` or a path to a TSV with a ``curie`` column."""
    if spec == f"builtin:{COFACTORS_V1}":
        text = (
            resources.files("stereomapper.reconciliation")
            .joinpath(f"resources/{COFACTORS_V1}.tsv")
            .read_text(encoding="utf-8")
        )
        curies = {
            normalize_curie(line.split("\t")[0]) for line in text.splitlines()[1:] if line.strip()
        }
        return CofactorList(COFACTORS_V1, "sha256:" + sha256_text(text), frozenset(curies), spec)
    path = Path(spec)
    if not path.is_absolute() and config_dir is not None:
        path = config_dir / path
    table = read_tsv(path, ("curie",))
    curies = {normalize_curie(row["curie"]) for row in table.rows if row["curie"]}
    return CofactorList(path.stem, sha256_file(path), frozenset(curies), spec)


class CofactorIndex:
    """A participant is a cofactor when its cluster contains a listed CURIE (R6)."""

    def __init__(self, cofactors: CofactorList, evidence: StereoMapperEvidence):
        self.cofactors = cofactors
        self._evidence = evidence
        self._proton_clusters = set(evidence.clusters_of_curie(PROTON_CURIE))
        self._clusters = {
            cid for curie in cofactors.curies for cid in evidence.clusters_of_curie(curie)
        }

    def is_cofactor(self, entry: ParticipantEntry) -> bool:
        if normalize_curie(entry.structure_curie) in self.cofactors.curies:
            return True
        return any(cid in self._clusters for cid in self._evidence.clusters_for(entry))

    def is_proton(self, entry: ParticipantEntry) -> bool:
        if normalize_curie(entry.structure_curie) == PROTON_CURIE:
            return True
        return any(cid in self._proton_clusters for cid in self._evidence.clusters_for(entry))
