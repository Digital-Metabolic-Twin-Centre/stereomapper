"""Post-processing subsystem flags (FR-005b, FR-006, FR-007, research R16).

Flags never change the run's proposals or evidence hashes. Flagged proposals stay visible,
are left out of counted results, and are exported with their flag if a curator accepts them.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stereomapper.domain.exceptions import ConfigurationError
from stereomapper.reconciliation.hashing import sha256_file
from stereomapper.reconciliation.inputs import InputContractError, read_flag_list, read_tsv
from stereomapper.reconciliation.outputs import write_json, write_tsv
from stereomapper.reconciliation.rundir import RunDir

NO_FLAGS = "none"


def _fraction(n: int, d: int) -> str:
    return f"{n}/{d}"


def apply_flags(run_dir: Path, flag_list_path: Path) -> dict[str, Any]:
    run = RunDir(run_dir)
    flag_list = read_flag_list(Path(flag_list_path))
    target = run.path / "flags" / flag_list.list_version
    if target.exists():
        raise ConfigurationError(
            f"Flag list version {flag_list.list_version!r} already exists in {run.path}; "
            "assign a new list_version"
        )

    assignments: dict[str, list[str]] = {}
    for row in run.rows("vmh_subsystems.tsv"):
        assignments.setdefault(row["reaction_id"], []).append(row["subsystem"])
    known = {s for subs in assignments.values() for s in subs}
    table = read_tsv(Path(flag_list_path))
    for row, line in zip(table.rows, table.line_numbers):
        if row["subsystem"] not in known:
            raise InputContractError(
                Path(flag_list_path).name,
                line,
                "subsystem",
                f"{row['subsystem']!r} does not occur in the run's VMH subsystems",
            )
    category_of = {e["subsystem"]: e["category"] for e in flag_list.entries}

    flagged: dict[str, list[str]] = {}
    for rid, subs in assignments.items():
        matched = sorted(s for s in subs if s in category_of)  # any-match rule
        if matched:
            flagged[rid] = matched

    proposals_by_vmh: dict[str, list[str]] = {}
    for p in run.proposals:
        proposals_by_vmh.setdefault(p["vmh_reaction_id"], []).append(p["proposal_id"])

    target.mkdir(parents=True)
    try:
        shutil.copyfile(flag_list_path, target / "flag_list.tsv")
        write_tsv(
            target / "subsystem_flags.tsv",
            [
                "vmh_reaction_id",
                "matched_subsystems",
                "categories",
                "all_subsystems",
                "list_version",
                "proposal_ids",
            ],
            (
                {
                    "vmh_reaction_id": rid,
                    "matched_subsystems": flagged[rid],
                    "categories": sorted({category_of[s] for s in flagged[rid]}),
                    "all_subsystems": sorted(assignments[rid]),
                    "list_version": flag_list.list_version,
                    "proposal_ids": sorted(proposals_by_vmh.get(rid, [])),
                }
                for rid in sorted(flagged)
            ),
        )
        flagged_proposals = sorted(
            (pid, rid) for rid in flagged for pid in proposals_by_vmh.get(rid, [])
        )
        write_tsv(
            target / "flagged_proposals.tsv",
            ["proposal_id", "vmh_reaction_id", "matched_subsystems"],
            (
                {"proposal_id": pid, "vmh_reaction_id": rid, "matched_subsystems": flagged[rid]}
                for pid, rid in flagged_proposals
            ),
        )
        by_category: dict[str, dict[str, int]] = {}
        for rid, matched in flagged.items():
            for category in sorted({category_of[s] for s in matched}):
                entry = by_category.setdefault(category, {"vmh_reactions": 0, "proposals": 0})
                entry["vmh_reactions"] += 1
                entry["proposals"] += len(proposals_by_vmh.get(rid, []))

        in_scope = run.funnel_count("vmh_in_scope")
        with_props = set(proposals_by_vmh)
        flagged_with = with_props & set(flagged)
        total = len(run.proposals)
        n_flagged_props = len(flagged_proposals)
        manifest = {
            "run_id": run.run_id,
            "list_version": flag_list.list_version,
            "list_checksum": sha256_file(flag_list_path),
            "rule": "any-match on VMH subsystem assignments",
            "flagged_vmh_reactions": len(flagged),
            "flagged_proposals": n_flagged_props,
            "flagged_by_category": by_category,
            "coverage": {
                "vmh_reactions_with_proposals": {
                    "all": _fraction(len(with_props), in_scope),
                    "unflagged": _fraction(len(with_props) - len(flagged_with), in_scope),
                    "flagged": _fraction(len(flagged_with), in_scope),
                },
                "proposals": {
                    "all": _fraction(total, total),
                    "unflagged": _fraction(total - n_flagged_props, total),
                    "flagged": _fraction(n_flagged_props, total),
                },
            },
        }
        write_json(target / "flags_manifest.json", manifest)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return manifest


@dataclass
class FlagSet:
    version: str
    vmh: dict[str, list[str]] = field(default_factory=dict)  # vmh id -> matched subsystems
    categories: dict[str, list[str]] = field(default_factory=dict)
    category_of_subsystem: dict[str, str] = field(default_factory=dict)

    def is_vmh_flagged(self, vmh_id: str) -> bool:
        return vmh_id in self.vmh

    def matched_subsystems(self, vmh_id: str) -> list[str]:
        return self.vmh.get(vmh_id, [])

    def categories_for_vmh(self, vmh_id: str) -> list[str]:
        return self.categories.get(vmh_id, [])


def load_flags(run_dir: Path, list_version: str | None) -> FlagSet:
    """``None`` means ``--no-subsystem-flags`` (recorded as ``none``)."""
    if list_version is None:
        return FlagSet(NO_FLAGS)
    target = Path(run_dir) / "flags" / list_version
    if not (target / "flags_manifest.json").is_file():
        raise ConfigurationError(
            f"No flag list {list_version!r} in {run_dir}; run `reconcile flag-subsystems` first"
        )
    flags = FlagSet(list_version)
    for row in read_tsv(target / "flag_list.tsv").rows:
        flags.category_of_subsystem[row["subsystem"]] = row["category"]
    for row in read_tsv(target / "subsystem_flags.tsv").rows:
        rid = row["vmh_reaction_id"]
        flags.vmh[rid] = [s for s in row["matched_subsystems"].split(";") if s]
        flags.categories[rid] = [c for c in row["categories"].split(";") if c]
    return flags
