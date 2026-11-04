"""Seeded, stratified review sample and acceptance-rate report (FR-028, research R13).

Strata are relationship profile x rank tercile. Acceptance rate is accepted / (accepted +
rejected) over reviewed sample members; deferred and pending are reported, not counted.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from stereomapper.reconciliation.curation import effective_decisions, read_decision_log
from stereomapper.reconciliation.flags import load_flags
from stereomapper.reconciliation.inputs import InputContractError, read_tsv
from stereomapper.reconciliation.outputs import fraction_columns, write_json, write_tsv
from stereomapper.reconciliation.rundir import RunDir

TERCILES = ("top", "middle", "bottom")
NOTE = (
    "Acceptance rate over reviewed sample members (accepted / (accepted + rejected)); "
    "deferred and pending are reported but not counted as decided."
)


def _tercile(position: int, total: int) -> str:
    return TERCILES[min(2, (3 * position) // max(total, 1))]


def _allocate(sizes: dict[str, int], size: int) -> dict[str, int]:
    """Proportional allocation; remainders by largest fraction, then stratum name."""
    total = sum(sizes.values())
    if size >= total:
        return dict(sizes)
    quotas = {k: size * n / total for k, n in sizes.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    remainder = size - sum(alloc.values())
    order = sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), k))
    for k in order:
        if remainder == 0:
            break
        if alloc[k] < sizes[k]:
            alloc[k] += 1
            remainder -= 1
    return alloc


def draw_sample(
    run_dir: Path,
    size: int,
    seed: int,
    out_path: Path,
    flag_version: str | None,
    exclude_benchmark_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if size < 1:
        raise InputContractError("--size", None, "size", "must be at least 1")
    run = RunDir(run_dir)
    flags = load_flags(run.path, flag_version)
    excluded: set[tuple[str, str]] = set()
    if exclude_benchmark_dir is not None:
        pairs = read_tsv(Path(exclude_benchmark_dir) / "benchmark_pairs.tsv").rows
        excluded = {(r["vmh_reaction_id"], r["rhea_master_id"]) for r in pairs}
    eligible = [
        p
        for p in sorted(run.proposals, key=lambda p: int(p["rank"]))
        if (p["vmh_reaction_id"], p["rhea_master_id"]) not in excluded
        and not flags.is_vmh_flagged(p["vmh_reaction_id"])
    ]
    strata: dict[str, list[dict[str, str]]] = {}
    for position, p in enumerate(eligible):
        stratum = f"{p['relationship_profile']}|{_tercile(position, len(eligible))}"
        strata.setdefault(stratum, []).append(p)
    allocation = _allocate({k: len(v) for k, v in strata.items()}, size)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for stratum in sorted(strata):
        members = sorted(strata[stratum], key=lambda p: p["proposal_id"])
        for p in rng.sample(members, allocation[stratum]):
            selected.append(
                {"proposal_id": p["proposal_id"], "stratum": stratum, "rank": int(p["rank"])}
            )
    selected.sort(key=lambda r: (r["stratum"], r["rank"]))
    comments = [
        f"seed: {seed}",
        f"size: {size}",
        f"eligible: {len(eligible)}",
        f"run_id: {run.run_id}",
        f"flag_list_version: {flags.version}",
        "strata: relationship_profile|rank tercile; "
        + ", ".join(f"{k}={allocation[k]}/{len(strata[k])}" for k in sorted(strata)),
    ]
    write_tsv(out_path, ["proposal_id", "stratum", "rank"], selected, comments=comments)
    return selected


def _rate(counts: dict[str, int]) -> dict[str, Any]:
    decided = counts["accepted"] + counts["rejected"]
    rate, decimal = fraction_columns(counts["accepted"], decided)
    return {
        **counts,
        "decided": decided,
        "acceptance_rate": rate if decided else None,
        "acceptance_rate_decimal": decimal if decided else None,
    }


def acceptance_report(
    run_dir: Path,
    sample_path: Path,
    log_path: Path,
    out_path: Path,
    flag_version: str | None,
) -> dict[str, Any]:
    run = RunDir(run_dir)
    flags = load_flags(run.path, flag_version)
    sample = read_tsv(Path(sample_path), ("proposal_id", "stratum")).rows
    effective = effective_decisions(read_decision_log(Path(log_path)))
    empty = {"accepted": 0, "rejected": 0, "deferred": 0, "pending": 0, "sample_size": 0}
    overall = dict(empty)
    strata: dict[str, dict[str, int]] = {}
    excluded_flagged = 0
    for row in sample:
        proposal = run.proposals_by_id.get(row["proposal_id"])
        if proposal is not None and flags.is_vmh_flagged(proposal["vmh_reaction_id"]):
            excluded_flagged += 1  # flagged proposals are not counted (FR-005b)
            continue
        d = effective.get(row["proposal_id"])
        status = d.decision.value if d else "pending"
        bucket = strata.setdefault(row["stratum"], dict(empty))
        for target in (overall, bucket):
            target[status] += 1
            target["sample_size"] += 1
    report = {
        "run_id": run.run_id,
        "flag_list_version": flags.version,
        "note": NOTE,
        "overall": _rate(overall),
        "strata": {k: _rate(v) for k, v in sorted(strata.items())},
        "excluded_flagged": excluded_flagged,
    }
    write_json(out_path, report)
    return report
