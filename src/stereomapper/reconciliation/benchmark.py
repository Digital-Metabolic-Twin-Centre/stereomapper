"""Recovery of known positives against the JC + NM worksheet (FR-025 to FR-027, research R13).

Worksheet recovery is not precision: the worksheet holds verified positives, not a complete set
of negatives. Acceptance rate for new proposals comes from curator review of a sample.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stereomapper.reconciliation.flags import load_flags
from stereomapper.reconciliation.inputs import read_worksheet
from stereomapper.reconciliation.model import BenchmarkStatus, UnrecoveredReason
from stereomapper.reconciliation.normalize import normalize_rhea_id
from stereomapper.reconciliation.outputs import write_json, write_tsv
from stereomapper.reconciliation.rundir import RunDir

NOTE = "Recovery of known positives; not precision."
POSITIVE_VERDICTS = {"verified"}
PAIR_COLUMNS = [
    "worksheet_row",
    "worksheet_version",
    "vmh_reaction_id",
    "rhea_id_raw",
    "rhea_master_id",
    "curator_verdict",
    "status",
    "unrecovered_reason",
    "in_scope",
    "subsystem_flagged",
    "flag_categories",
    "counted",
    "sufficient_structure_evidence",
    "proposal_id",
    "rank",
    "alignment_exactness",
]
_RANK_BUCKETS = ((10, "1-10"), (100, "11-100"), (1000, "101-1000"))


def _rank_bucket(rank: int) -> str:
    for limit, label in _RANK_BUCKETS:
        if rank <= limit:
            return label
    return ">1000"


def _subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    ranks: dict[str, int] = {}
    counts = {
        s.value: 0
        for s in (
            BenchmarkStatus.RECOVERED_NEW,
            BenchmarkStatus.ALREADY_IN_BASELINE,
            BenchmarkStatus.UNRECOVERED,
        )
    }
    for row in rows:
        counts[row["status"]] += 1
        if row["status"] == BenchmarkStatus.UNRECOVERED.value:
            reasons[row["unrecovered_reason"]] = reasons.get(row["unrecovered_reason"], 0) + 1
        if row["status"] == BenchmarkStatus.RECOVERED_NEW.value:
            bucket = _rank_bucket(int(row["rank"]))
            ranks[bucket] = ranks.get(bucket, 0) + 1
    denominator = len(rows)
    out = {
        "denominator": denominator,
        **counts,
        "recovered_new_fraction": f"{counts['recovered_new']}/{denominator}",
        "unrecovered_by_reason": dict(sorted(reasons.items())),
        "rank_distribution": ranks,
    }
    assert sum(counts.values()) == denominator  # invariant (contracts/output-files.md)
    return out


def run_benchmark(
    run_dir: Path,
    worksheet_path: Path,
    worksheet_version: str,
    flag_version: str | None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    run = RunDir(run_dir)
    flags = load_flags(run.path, flag_version)
    worksheet = read_worksheet(Path(worksheet_path))
    out_dir = (
        Path(out_dir) if out_dir else run.path / "benchmark" / worksheet_version / flags.version
    )

    summary = {(r["source"], r["reaction_id"]): r for r in run.rows("reactions_summary.tsv")}
    rhea_by_norm = {
        normalize_rhea_id(rid): row for (src, rid), row in summary.items() if src == "rhea"
    }
    baseline = {
        (r["vmh_reaction_id"], r["rhea_master_id"])
        for r in run.rows("baseline_pairs.tsv")
        if r["rhea_master_id"]
    }
    candidates = {
        (c["vmh_reaction_id"], c["rhea_master_id"]): c for c in run.rows("candidates.tsv")
    }
    proposals = {(p["vmh_reaction_id"], p["rhea_master_id"]): p for p in run.proposals}

    rows: list[dict[str, Any]] = []
    for row, line in zip(worksheet.rows, worksheet.line_numbers):
        vmh_id, rhea_raw = row["vmh_reaction_id"], row["rhea_id"]
        rhea_row = rhea_by_norm.get(normalize_rhea_id(rhea_raw))
        master = rhea_row["rhea_master_id"] if rhea_row else ""
        vmh_row = summary.get(("vmh", vmh_id))
        out = {
            "worksheet_row": line,
            "worksheet_version": worksheet_version,
            "vmh_reaction_id": vmh_id,
            "rhea_id_raw": rhea_raw,
            "rhea_master_id": master,
            "curator_verdict": row["curator_verdict"],
            "unrecovered_reason": "",
        }
        if row["curator_verdict"].strip().lower() not in POSITIVE_VERDICTS:
            rows.append({**out, "status": BenchmarkStatus.NOT_A_POSITIVE.value})
            continue
        rhea_out_of_scope = bool(rhea_row and rhea_row["in_scope"] != "true")
        in_scope = bool(vmh_row and rhea_row and not rhea_out_of_scope)
        flagged = flags.is_vmh_flagged(vmh_id)
        sufficient = bool(
            in_scope
            and vmh_row is not None
            and rhea_row is not None
            and vmh_row["all_assessable"] == "true"
            and rhea_row["all_assessable"] == "true"
        )
        proposal = proposals.get((vmh_id, master))
        candidate = candidates.get((vmh_id, master))
        out.update(
            {
                "in_scope": in_scope,
                "subsystem_flagged": flagged,
                "flag_categories": flags.categories_for_vmh(vmh_id),
                "counted": in_scope and not flagged,
                "sufficient_structure_evidence": sufficient,
                "proposal_id": proposal["proposal_id"] if proposal else "",
                "rank": proposal["rank"] if proposal else "",
                "alignment_exactness": candidate["alignment_exactness"] if candidate else "",
                "_rhea_out_of_scope": rhea_out_of_scope,
            }
        )
        if (vmh_id, master) in baseline:
            status = BenchmarkStatus.ALREADY_IN_BASELINE
        elif proposal is not None:
            status = BenchmarkStatus.RECOVERED_NEW
        else:
            status = BenchmarkStatus.UNRECOVERED
            if vmh_row is None:
                reason = UnrecoveredReason.VMH_ID_ABSENT
            elif rhea_row is None:
                reason = UnrecoveredReason.RHEA_ID_ABSENT
            elif rhea_out_of_scope:
                reason = UnrecoveredReason.RHEA_OBSOLETE
            elif (
                vmh_row["has_assessable_noncofactor"] != "true"
                or rhea_row["has_assessable_noncofactor"] != "true"
            ):
                reason = UnrecoveredReason.STRUCTURE_INSUFFICIENT
            elif candidate is None:
                reason = UnrecoveredReason.NOT_RETRIEVED
            else:
                reason = UnrecoveredReason.RETRIEVED_NOT_PLAUSIBLE
            out["unrecovered_reason"] = reason.value
        rows.append({**out, "status": status.value})

    positives = [r for r in rows if r["status"] != BenchmarkStatus.NOT_A_POSITIVE.value]
    counted = [r for r in positives if r["counted"]]
    report: dict[str, Any] = {
        "run_id": run.run_id,
        "worksheet_version": worksheet_version,
        "baseline_version": run.manifest.get("baseline_version", ""),
        "flag_list_version": flags.version,
        "note": NOTE,
        "subsets": {
            "full_worksheet": _subset(positives),
            "counted": _subset(counted),
            "counted_sufficient_structure": _subset(
                [r for r in counted if r["sufficient_structure_evidence"]]
            ),
        },
        "non_positive_rows": len(rows) - len(positives),
        "rule_effect": {
            "pairs_out_of_scope": {
                "rhea_obsolete": sum(1 for r in positives if r["_rhea_out_of_scope"])
            },
            "pairs_flagged_by_category": _by_category(positives),
            "counted_without_flags": _subset([r for r in positives if r["in_scope"]]),
        },
        "greedy_alignment_pairs": sum(1 for r in positives if r["alignment_exactness"] == "greedy"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "benchmark_pairs.tsv", PAIR_COLUMNS, rows)
    write_json(out_dir / "benchmark_report.json", report)
    return report


def _by_category(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        if row["subsystem_flagged"]:
            for category in row["flag_categories"]:
                out[category] = out.get(category, 0) + 1
    return dict(sorted(out.items()))
