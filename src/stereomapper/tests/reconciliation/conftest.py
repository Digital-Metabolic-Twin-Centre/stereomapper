"""Fixture builders for reconciliation tests: KG extracts plus a StereoMapper results DB."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from stereomapper.data.results_schema import results_schema
from stereomapper.domain.relationship_terms import get_relationship_term_id

REACTION_COLUMNS = [
    "source",
    "reaction_id",
    "rhea_master_id",
    "rhea_directional_ids",
    "rhea_status",
    "direction",
    "equation",
    "xref_rhea",
]
PARTICIPANT_COLUMNS = [
    "source",
    "reaction_id",
    "side",
    "metabolite_id",
    "name",
    "compartment",
    "coefficient",
    "kind",
    "structure_curie",
    "formula",
    "charge",
]
SUBSYSTEM_COLUMNS = ["reaction_id", "subsystem", "kg_pathway_id"]

# Participant spec: (metabolite_id, structure_curie, coefficient) plus optional keyword fields.
Spec = Any


def write_tsv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row.get(c, "")) for c in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_results_db(
    path: Path,
    clusters: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    version_tag: str = "v-test",
) -> Path:
    """Build a StereoMapper results DB with ``results_schema``.

    ``clusters``: dicts with ``cluster_id``, ``members`` (CURIEs), optional ``status``/``error``.
    ``relationships``: dicts with ``a``, ``b`` (cluster IDs), ``label``, optional ``direction``,
    ``score``, ``version_tag``.
    """
    con = sqlite3.connect(path)
    results_schema(con)
    with con:
        for c in clusters:
            members = sorted(c["members"])
            con.execute(
                "INSERT INTO clusters (cluster_id, inchikey_first, identity_key_strict, "
                "member_count, members_json, members_hash, status, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    c["cluster_id"],
                    f"IK{c['cluster_id']}",
                    f"ID{c['cluster_id']}",
                    len(members),
                    json.dumps(members),
                    f"H{c['cluster_id']}",
                    c.get("status", "passed"),
                    c.get("error"),
                ),
            )
            for m in members:
                con.execute(
                    "INSERT INTO cluster_members (cluster_id, member_curie) VALUES (?, ?)",
                    (c["cluster_id"], m),
                )
        members_by_id = {c["cluster_id"]: sorted(c["members"]) for c in clusters}
        for r in relationships:
            tag = r.get("version_tag", version_tag)
            con.execute(
                "INSERT INTO relationships (cluster_a, cluster_b, cluster_a_members, "
                "cluster_b_members, cluster_a_size, cluster_b_size, classification, "
                "classification_term_id, direction, score, version_tag) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    r["a"],
                    r["b"],
                    json.dumps(members_by_id[r["a"]]),
                    json.dumps(members_by_id[r["b"]]),
                    len(members_by_id[r["a"]]),
                    len(members_by_id[r["b"]]),
                    r["label"],
                    get_relationship_term_id(r["label"]),
                    r.get("direction"),
                    r.get("score", 0.9),
                    tag,
                ),
            )
            for side, cid in (("A", r["a"]), ("B", r["b"])):
                for m in members_by_id[cid]:
                    con.execute(
                        "INSERT INTO relationship_members (cluster_a, cluster_b, version_tag, "
                        "side, member_curie) VALUES (?, ?, ?, ?, ?)",
                        (r["a"], r["b"], tag, side, m),
                    )
    con.close()
    return path


def make_extract(
    directory: Path,
    reactions: list[dict[str, Any]],
    participants: list[dict[str, Any]],
    subsystems: list[dict[str, Any]],
    manifest: dict[str, Any] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    write_tsv(directory / "reactions.tsv", REACTION_COLUMNS, reactions)
    write_tsv(directory / "participants.tsv", PARTICIPANT_COLUMNS, participants)
    write_tsv(directory / "subsystems.tsv", SUBSYSTEM_COLUMNS, subsystems)
    (directory / "extract_manifest.json").write_text(
        json.dumps(
            manifest
            or {
                "kg_snapshot": "kg-test-1",
                "vmh_release": "vmh-test",
                "rhea_release": "rhea-test",
                "extraction_query": {"repository": "test", "commit": "abc", "path": "q.cypher"},
                "extracted_at": "2026-11-01T00:00:00Z",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return directory


def make_config(directory: Path, **overrides: Any) -> Path:
    config: dict[str, Any] = {
        "config_version": 1,
        "inputs": {
            "extract_dir": "extract",
            "stereomapper_db": "results.sqlite",
            "version_tag": "v-test",
            "baseline": "baseline.tsv",
            "baseline_version": "baseline-test",
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    path = directory / "config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def _participant(source: str, rid: str, side: str, spec: Spec) -> dict[str, Any]:
    if isinstance(spec, dict):
        row = dict(spec)
    else:
        met, curie, *rest = spec
        row = {"metabolite_id": met, "structure_curie": curie or ""}
        if rest:
            row["coefficient"] = rest[0]
    row.setdefault("coefficient", "1")
    row.setdefault("name", row["metabolite_id"])
    row.update({"source": source, "reaction_id": rid, "side": side})
    return row


@dataclass
class Scenario:
    """Concise builder: reactions, clusters, and relationships, then ``build()``.

    Every CURIE used by a participant gets a singleton cluster unless it is grouped with
    ``same()``, marked ``missing()`` (absent from the DB), or ``failed()``.
    """

    root: Path
    reactions: list[dict[str, Any]] = field(default_factory=list)
    participants: list[dict[str, Any]] = field(default_factory=list)
    subsystems: list[dict[str, Any]] = field(default_factory=list)
    baseline: list[dict[str, Any]] = field(default_factory=list)
    groups: list[list[str]] = field(default_factory=list)
    rels: list[tuple[str, str, str, str | None]] = field(default_factory=list)
    missing_curies: set[str] = field(default_factory=set)
    failed_curies: set[str] = field(default_factory=set)
    extra_curies: list[str] = field(default_factory=list)
    other_version_rels: list[tuple[str, str, str]] = field(default_factory=list)

    def vmh(
        self,
        rid: str,
        left: list[Spec],
        right: list[Spec],
        direction: str = "forward",
        xref: str = "",
        subsystems: tuple[str, ...] = ("Test metabolism",),
    ) -> Scenario:
        self.reactions.append(
            {
                "source": "vmh",
                "reaction_id": rid,
                "direction": direction,
                "equation": f"{rid} equation",
                "xref_rhea": xref,
            }
        )
        for spec in left:
            self.participants.append(_participant("vmh", rid, "left", spec))
        for spec in right:
            self.participants.append(_participant("vmh", rid, "right", spec))
        for sub in subsystems:
            self.subsystems.append({"reaction_id": rid, "subsystem": sub})
        return self

    def rhea(
        self,
        master: int,
        left: list[Spec],
        right: list[Spec],
        status: str = "approved",
        directional: tuple[int, ...] = (1, 2, 3),
    ) -> Scenario:
        """Rhea master ``RHEA:<master>``; ``directional`` offsets follow Rhea (1 LR, 2 RL, 3 BI)."""
        rid = f"RHEA:{master}"
        ids = [f"RHEA:{master + k}" for k in directional]
        self.reactions.append(
            {
                "source": "rhea",
                "reaction_id": rid,
                "rhea_status": status,
                "rhea_directional_ids": ";".join(ids),
                "equation": f"{rid} equation",
            }
        )
        for spec in left:
            self.participants.append(_participant("rhea", rid, "left", spec))
        for spec in right:
            self.participants.append(_participant("rhea", rid, "right", spec))
        return self

    def same(self, *curies: str) -> Scenario:
        self.groups.append(list(curies))
        return self

    def rel(self, a: str, b: str, label: str, direction: str | None = None) -> Scenario:
        self.rels.append((a, b, label, direction))
        return self

    def missing(self, *curies: str) -> Scenario:
        self.missing_curies.update(curies)
        return self

    def failed(self, *curies: str) -> Scenario:
        self.failed_curies.update(curies)
        return self

    def in_baseline(self, vmh_id: str, rhea_id: str) -> Scenario:
        self.baseline.append({"vmh_reaction_id": vmh_id, "rhea_id": rhea_id})
        return self

    def cluster_map(self) -> dict[str, int]:
        curies = [p["structure_curie"] for p in self.participants if p.get("structure_curie")]
        curies += self.extra_curies
        for a, b, _l, _d in self.rels:
            curies += [a, b]
        mapping: dict[str, int] = {}
        next_id = 1
        for group in self.groups:
            for c in group:
                mapping[c] = next_id
            next_id += 1
        for c in sorted(set(curies)):
            if c in mapping or c in self.missing_curies:
                continue
            mapping[c] = next_id
            next_id += 1
        return mapping

    def build(self, **config_overrides: Any) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        make_extract(self.root / "extract", self.reactions, self.participants, self.subsystems)
        mapping = self.cluster_map()
        members: dict[int, list[str]] = {}
        for curie, cid in mapping.items():
            members.setdefault(cid, []).append(curie)
        clusters = []
        for cid, curies in sorted(members.items()):
            failed = any(c in self.failed_curies for c in curies)
            clusters.append(
                {
                    "cluster_id": cid,
                    "members": curies,
                    "status": "failed" if failed else "passed",
                    "error": "rdkit_sanitize_failed: test" if failed else None,
                }
            )
        relationships = [
            {"a": mapping[a], "b": mapping[b], "label": label, "direction": direction}
            for a, b, label, direction in self.rels
        ]
        relationships += [
            {"a": mapping[a], "b": mapping[b], "label": label, "version_tag": "v-other"}
            for a, b, label in self.other_version_rels
        ]
        if not relationships:
            # The DB must contain the configured version_tag; add a harmless self-unrelated row.
            spare = max(members) + 1 if members else 1
            clusters.append({"cluster_id": spare, "members": ["test:spare-a"]})
            clusters.append({"cluster_id": spare + 1, "members": ["test:spare-b"]})
            relationships.append({"a": spare, "b": spare + 1, "label": "Unclassified"})
        make_results_db(self.root / "results.sqlite", clusters, relationships)
        write_tsv(self.root / "baseline.tsv", ["vmh_reaction_id", "rhea_id"], self.baseline)
        return make_config(self.root, **config_overrides)


@pytest.fixture
def scenario(tmp_path: Path) -> Scenario:
    return Scenario(root=tmp_path / "case")


def read_rows(path: Path) -> list[dict[str, str]]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


@dataclass
class RunOutput:
    run_dir: Path
    proposals: list[dict[str, str]]
    evidence: list[dict[str, str]]
    funnel: dict[str, tuple[str, str]]
    manifest: dict[str, Any]

    def proposal(self, vmh_id: str, rhea_master: str) -> dict[str, str]:
        for row in self.proposals:
            if row["vmh_reaction_id"] == vmh_id and row["rhea_master_id"] == rhea_master:
                return row
        raise KeyError((vmh_id, rhea_master))

    def has_proposal(self, vmh_id: str, rhea_master: str) -> bool:
        return any(
            r["vmh_reaction_id"] == vmh_id and r["rhea_master_id"] == rhea_master
            for r in self.proposals
        )

    def evidence_for(
        self, proposal_id: str, orientation: str | None = None
    ) -> list[dict[str, str]]:
        return [
            r
            for r in self.evidence
            if r["proposal_id"] == proposal_id
            and (orientation is None or r["orientation"] == orientation)
        ]


def propose(config_path: Path, out: Path | None = None) -> RunOutput:
    from stereomapper.reconciliation.config import load_run_config
    from stereomapper.reconciliation.pipeline import run_propose

    run_dir = out or config_path.parent / "run"
    run_propose(load_run_config(config_path), run_dir)
    funnel = {r["stage"]: (r["count"], r["denominator"]) for r in read_rows(run_dir / "funnel.tsv")}
    return RunOutput(
        run_dir=run_dir,
        proposals=read_rows(run_dir / "reaction_proposals.tsv"),
        evidence=read_rows(run_dir / "participant_evidence.tsv"),
        funnel=funnel,
        manifest=json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")),
    )
