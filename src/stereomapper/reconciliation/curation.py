"""Append-only curator decision log and curated export (FR-022 to FR-024, research R12)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from stereomapper.reconciliation.flags import load_flags
from stereomapper.reconciliation.hashing import short_hash
from stereomapper.reconciliation.inputs import (
    DECISION_BATCH_COLUMNS,
    InputContractError,
    read_decision_batch,
    read_tsv,
)
from stereomapper.reconciliation.model import CurationDecision, DecisionValue
from stereomapper.reconciliation.outputs import format_value, write_json, write_tsv
from stereomapper.reconciliation.rundir import RunDir

LOG_COLUMNS = [
    "decision_id",
    "proposal_id",
    "evidence_hash",
    "run_id",
    "vmh_reaction_id",
    "rhea_master_id",
    "decision",
    "reviewer",
    "date",
    "reason",
    "supersedes",
]
CURATED_COLUMNS = [
    "vmh_reaction_id",
    "rhea_master_id",
    "rhea_matching_directional_id",
    "proposal_id",
    "decision_id",
    "reviewer",
    "date",
    "reason",
    "reviewed_evidence_hash",
    "decision_run_id",
    "run_id",
    "evidence_matches_run",
    "subsystem_flagged",
    "matched_subsystems",
]
REREVIEW_COLUMNS = [
    "proposal_id",
    "vmh_reaction_id",
    "rhea_master_id",
    "decision_id",
    "reviewed_evidence_hash",
    "run_evidence_hash",
    "issue",
]


def decision_id(proposal_id: str, evidence_hash: str, reviewer: str, day: str, value: str) -> str:
    return "DEC-" + short_hash("|".join((proposal_id, evidence_hash, reviewer, day, value)))


def read_decision_log(path: Path) -> list[CurationDecision]:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return []
    table = read_tsv(path, LOG_COLUMNS)
    out = []
    for row, line in zip(table.rows, table.line_numbers):
        try:
            value = DecisionValue(row["decision"])
        except ValueError:
            raise InputContractError(path.name, line, "decision", f"invalid {row['decision']!r}")
        out.append(
            CurationDecision(
                decision_id=row["decision_id"],
                proposal_id=row["proposal_id"],
                evidence_hash=row["evidence_hash"],
                run_id=row["run_id"],
                vmh_reaction_id=row["vmh_reaction_id"],
                rhea_master_id=row["rhea_master_id"],
                decision=value,
                reviewer=row["reviewer"],
                date=row["date"],
                reason=row["reason"],
                supersedes=row.get("supersedes", ""),
            )
        )
    return out


def _heads(decisions: list[CurationDecision]) -> dict[str, list[CurationDecision]]:
    superseded = {d.supersedes for d in decisions if d.supersedes}
    heads: dict[str, list[CurationDecision]] = {}
    for d in decisions:
        if d.decision_id not in superseded:
            heads.setdefault(d.proposal_id, []).append(d)
    return heads


def effective_decisions(decisions: list[CurationDecision]) -> dict[str, CurationDecision]:
    """Latest decision per proposal; conflicting heads are an error until resolved."""
    out: dict[str, CurationDecision] = {}
    for pid, heads in _heads(decisions).items():
        if len(heads) > 1:
            ids = ", ".join(sorted(h.decision_id for h in heads))
            raise InputContractError(
                "decision log",
                None,
                "supersedes",
                f"conflicting decisions for {pid} ({ids}); record one that supersedes the other",
            )
        out[pid] = heads[0]
    return out


def validate_batch(
    run: RunDir, batch_path: Path, existing: list[CurationDecision]
) -> list[CurationDecision]:
    table = read_decision_batch(batch_path)
    by_id = {d.decision_id: d for d in existing}
    heads = {pid: hs for pid, hs in _heads(existing).items()}
    errors: list[str] = []
    accepted: list[CurationDecision] = []
    seen_ids: set[str] = set()
    for row, line in zip(table.rows, table.line_numbers):
        problems = []
        pid = row["proposal_id"]
        proposal = run.proposals_by_id.get(pid)
        if proposal is None:
            problems.append(f"proposal {pid!r} is not in run {run.run_id}")
        elif row["evidence_hash"] != proposal["evidence_hash"]:
            problems.append("evidence_hash does not match the run's proposal")
        if row["decision"] not in {v.value for v in DecisionValue}:
            problems.append(f"decision {row['decision']!r} must be accepted, rejected, or deferred")
        for col in ("reviewer", "date", "reason"):
            if not row.get(col):
                problems.append(f"{col} is required")
        if row.get("date"):
            try:
                date.fromisoformat(row["date"])
            except ValueError:
                problems.append(f"date {row['date']!r} is not ISO 8601 (YYYY-MM-DD)")
        supersedes = row.get("supersedes", "")
        current = [h.decision_id for h in heads.get(pid, [])]
        if supersedes:
            target = by_id.get(supersedes)
            if target is None:
                problems.append(f"supersedes {supersedes!r} is not in the decision log")
            elif target.proposal_id != pid:
                problems.append(f"supersedes {supersedes!r} belongs to another proposal")
            elif supersedes not in current:
                problems.append(f"supersedes {supersedes!r} is not the current decision")
        elif current:
            problems.append(f"proposal already has decision {current[0]}; supersede it")
        if problems or proposal is None:
            errors.append(f"row {line}: " + "; ".join(problems))
            continue
        did = decision_id(pid, row["evidence_hash"], row["reviewer"], row["date"], row["decision"])
        if did in by_id or did in seen_ids:
            errors.append(f"row {line}: duplicate decision {did}")
            continue
        seen_ids.add(did)
        decision = CurationDecision(
            decision_id=did,
            proposal_id=pid,
            evidence_hash=row["evidence_hash"],
            run_id=run.run_id,
            vmh_reaction_id=proposal["vmh_reaction_id"],
            rhea_master_id=proposal["rhea_master_id"],
            decision=DecisionValue(row["decision"]),
            reviewer=row["reviewer"],
            date=row["date"],
            reason=row["reason"],
            supersedes=supersedes,
        )
        accepted.append(decision)
        heads[pid] = [decision]
    if errors:
        raise InputContractError(
            batch_path.name, None, "-", "batch rejected: " + " | ".join(errors)
        )
    return accepted


def record_decisions(run_dir: Path, batch_path: Path, log_path: Path) -> int:
    run = RunDir(run_dir)
    log_path = Path(log_path)
    existing = read_decision_log(log_path)
    effective_decisions(existing)  # refuse to extend a log with conflicts
    new = validate_batch(run, Path(batch_path), existing)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.is_file() or not log_path.read_text(encoding="utf-8").strip()
    with open(log_path, "a", encoding="utf-8", newline="\n") as handle:  # append only
        if write_header:
            handle.write("\t".join(LOG_COLUMNS) + "\n")
        for d in new:
            values = {**vars(d), "decision": d.decision.value}
            handle.write("\t".join(format_value(values[c]) for c in LOG_COLUMNS) + "\n")
    return len(new)


def _sibling(out_path: Path, suffix: str) -> Path:
    return out_path.with_name(out_path.stem + suffix)


def export_curated(
    run_dir: Path,
    log_path: Path,
    out_path: Path,
    flag_version: str | None,
    allow_stale: bool = False,
) -> dict[str, Any]:
    run = RunDir(run_dir)
    flags = load_flags(run.path, flag_version)
    effective = effective_decisions(read_decision_log(Path(log_path)))
    out_path = Path(out_path)
    curated: list[dict[str, Any]] = []
    rereview: list[dict[str, Any]] = []
    for pid, d in sorted(effective.items()):
        if d.decision != DecisionValue.ACCEPTED:
            continue
        proposal = run.proposals_by_id.get(pid)
        if proposal is None:
            rereview.append(
                {
                    "proposal_id": pid,
                    "vmh_reaction_id": d.vmh_reaction_id,
                    "rhea_master_id": d.rhea_master_id,
                    "decision_id": d.decision_id,
                    "reviewed_evidence_hash": d.evidence_hash,
                    "issue": "proposal absent from this run",
                }
            )
            continue
        matches = d.evidence_hash == proposal["evidence_hash"]
        if not matches:
            rereview.append(
                {
                    "proposal_id": pid,
                    "vmh_reaction_id": d.vmh_reaction_id,
                    "rhea_master_id": d.rhea_master_id,
                    "decision_id": d.decision_id,
                    "reviewed_evidence_hash": d.evidence_hash,
                    "run_evidence_hash": proposal["evidence_hash"],
                    "issue": "evidence changed since review",
                }
            )
            if not allow_stale:
                continue
        flagged = flags.is_vmh_flagged(proposal["vmh_reaction_id"])
        curated.append(
            {
                "vmh_reaction_id": proposal["vmh_reaction_id"],
                "rhea_master_id": proposal["rhea_master_id"],
                "rhea_matching_directional_id": proposal["rhea_matching_directional_id"],
                "proposal_id": pid,
                "decision_id": d.decision_id,
                "reviewer": d.reviewer,
                "date": d.date,
                "reason": d.reason,
                "reviewed_evidence_hash": d.evidence_hash,
                "decision_run_id": d.run_id,
                "run_id": run.run_id,
                "evidence_matches_run": matches,
                "subsystem_flagged": flagged if flag_version is not None else None,
                "matched_subsystems": flags.matched_subsystems(proposal["vmh_reaction_id"]),
            }
        )
    curated.sort(key=lambda r: (r["vmh_reaction_id"], r["rhea_master_id"]))
    comments = [
        f"flag_list_version: {flags.version}",
        f"run_id: {run.run_id}",
        "only curator-accepted mappings; subsystem_flagged rows can be filtered",
    ]
    write_tsv(out_path, CURATED_COLUMNS, curated, comments=comments)
    write_tsv(_sibling(out_path, ".rereview.tsv"), REREVIEW_COLUMNS, rereview)

    def counts(pids: list[str]) -> dict[str, int]:
        out = {"accepted": 0, "rejected": 0, "deferred": 0, "pending": 0}
        for pid in pids:
            d = effective.get(pid)
            out[d.decision.value if d else "pending"] += 1
        out["denominator"] = len(pids)
        return out

    all_ids = [p["proposal_id"] for p in run.proposals]
    unflagged = [
        p["proposal_id"] for p in run.proposals if not flags.is_vmh_flagged(p["vmh_reaction_id"])
    ]
    summary = {
        "run_id": run.run_id,
        "flag_list_version": flags.version,
        "all": counts(all_ids),
        "unflagged": counts(unflagged),
        "exported": len(curated),
        "rereview": len(rereview),
        "note": "Only curator-accepted decisions are exported; rank and coverage never "
        "promote a proposal.",
    }
    write_json(_sibling(out_path, ".summary.json"), summary)
    return summary


__all__ = [
    "DECISION_BATCH_COLUMNS",
    "LOG_COLUMNS",
    "effective_decisions",
    "export_curated",
    "read_decision_log",
    "record_decisions",
]
