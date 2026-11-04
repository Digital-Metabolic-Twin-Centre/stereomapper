"""``stereomapper reconcile`` subcommands (contracts/cli.md).

Exit codes: 0 success; 2 invalid arguments or input-contract violations; 1 other failures.
No command modifies its inputs or the StereoMapper results database.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

from stereomapper.domain.exceptions import ConfigurationError
from stereomapper.reconciliation.inputs import InputContractError

logger = logging.getLogger(__name__)


def _add_flag_option(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--flags",
        metavar="LIST_VERSION",
        help="Apply this subsystem flag list (written by `reconcile flag-subsystems`).",
    )
    group.add_argument(
        "--no-subsystem-flags",
        action="store_true",
        help="Apply no subsystem flags; recorded as flag_list_version: none.",
    )


def _flag_version(args: argparse.Namespace) -> str | None:
    return None if args.no_subsystem_flags else args.flags


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    rec = subparsers.add_parser(
        "reconcile",
        help="Propose VMH-Rhea reaction mappings for curator review (downstream workflow)",
        description=(
            "Downstream VMH-Rhea reaction reconciliation. Proposals are a curator review "
            "queue; only curator-accepted mappings are exported for GEM use."
        ),
    )
    sub = rec.add_subparsers(dest="reconcile_cmd", required=True)

    p = sub.add_parser("propose", help="Generate ranked reaction proposals from pinned inputs")
    p.add_argument("--config", required=True, type=Path, help="Run configuration JSON")
    p.add_argument("--out", required=True, type=Path, help="Run directory to create")
    p.add_argument("--overwrite", action="store_true", help="Allow a non-empty run directory")
    p.set_defaults(handler=_cmd_propose)

    p = sub.add_parser("flag-subsystems", help="Apply a versioned subsystem flag list to a run")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--flag-list", required=True, type=Path)
    p.set_defaults(handler=_cmd_flag_subsystems)

    p = sub.add_parser("review-workbook", help="Write an .xlsx workbook for curator review")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--decision-log", type=Path)
    p.add_argument("--proposals", type=Path, help="File with one proposal_id per line")
    p.add_argument("--flags", metavar="LIST_VERSION", help="Show this subsystem flag list")
    p.set_defaults(handler=_cmd_review_workbook)

    p = sub.add_parser("record-decisions", help="Validate and append curator decisions")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--decisions", required=True, type=Path)
    p.add_argument("--decision-log", required=True, type=Path)
    p.set_defaults(handler=_cmd_record_decisions)

    p = sub.add_parser("export-curated", help="Export curator-accepted mappings for GEM use")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--decision-log", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--allow-stale-evidence", action="store_true")
    _add_flag_option(p)
    p.set_defaults(handler=_cmd_export_curated)

    p = sub.add_parser("benchmark", help="Report recovery against the verified worksheet")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--worksheet", required=True, type=Path)
    p.add_argument("--worksheet-version", required=True)
    p.add_argument("--out", type=Path)
    _add_flag_option(p)
    p.set_defaults(handler=_cmd_benchmark)

    p = sub.add_parser("sample", help="Draw a stratified sample of proposals for review")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--size", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--exclude-worksheet", type=Path, metavar="BENCH_DIR")
    _add_flag_option(p)
    p.set_defaults(handler=_cmd_sample)

    p = sub.add_parser("acceptance", help="Report acceptance rate over a reviewed sample")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--sample", required=True, type=Path)
    p.add_argument("--decision-log", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    _add_flag_option(p)
    p.set_defaults(handler=_cmd_acceptance)


def run(args: argparse.Namespace) -> int:
    """Dispatch a parsed ``reconcile`` command and return its exit code."""
    handler: Callable[[argparse.Namespace], None] = args.handler
    try:
        handler(args)
    except (InputContractError, ConfigurationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - report and map to exit code 1
        logger.exception("reconcile %s failed", args.reconcile_cmd)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_propose(args: argparse.Namespace) -> None:
    from stereomapper.reconciliation.config import load_run_config
    from stereomapper.reconciliation.pipeline import run_propose

    record = run_propose(load_run_config(args.config), args.out, overwrite=args.overwrite)
    print(f"run_id: {record['run_id']}")
    for stage in record["funnel"]:
        denominator = stage["denominator"]
        suffix = f"/{denominator}" if denominator not in (None, "") else ""
        print(f"{stage['stage']}: {stage['count']}{suffix}")


def _cmd_flag_subsystems(args: argparse.Namespace) -> None:
    from stereomapper.reconciliation.flags import apply_flags

    manifest = apply_flags(args.run, args.flag_list)
    print(
        f"flag list {manifest['list_version']}: {manifest['flagged_vmh_reactions']} VMH "
        f"reactions, {manifest['flagged_proposals']} proposals flagged"
    )


def _cmd_review_workbook(args: argparse.Namespace) -> None:
    from stereomapper.reconciliation.review import write_review_workbook

    ids = None
    if args.proposals:
        ids = [
            line.strip()
            for line in args.proposals.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    write_review_workbook(
        args.run, args.out, decision_log=args.decision_log, proposal_ids=ids, flags=args.flags
    )
    print(f"wrote {args.out}")


def _cmd_record_decisions(args: argparse.Namespace) -> None:
    from stereomapper.reconciliation.curation import record_decisions

    count = record_decisions(args.run, args.decisions, args.decision_log)
    print(f"appended {count} decisions to {args.decision_log}")


def _cmd_export_curated(args: argparse.Namespace) -> None:
    from stereomapper.reconciliation.curation import export_curated

    summary = export_curated(
        args.run,
        args.decision_log,
        args.out,
        flag_version=_flag_version(args),
        allow_stale=args.allow_stale_evidence,
    )
    print(f"exported {summary['exported']} accepted mappings; {summary['rereview']} need re-review")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    from stereomapper.reconciliation.benchmark import run_benchmark

    report = run_benchmark(
        args.run,
        args.worksheet,
        args.worksheet_version,
        flag_version=_flag_version(args),
        out_dir=args.out,
    )
    for name, subset in report["subsets"].items():
        print(
            f"{name}: recovered_new {subset['recovered_new']}/{subset['denominator']}, "
            f"already_in_baseline {subset['already_in_baseline']}, "
            f"unrecovered {subset['unrecovered']}"
        )
    print(report["note"])


def _cmd_sample(args: argparse.Namespace) -> None:
    from stereomapper.reconciliation.sampling import draw_sample

    selected = draw_sample(
        args.run,
        args.size,
        args.seed,
        args.out,
        flag_version=_flag_version(args),
        exclude_benchmark_dir=args.exclude_worksheet,
    )
    print(f"sampled {len(selected)} proposals into {args.out}")


def _cmd_acceptance(args: argparse.Namespace) -> None:
    from stereomapper.reconciliation.sampling import acceptance_report

    report = acceptance_report(
        args.run, args.sample, args.decision_log, args.out, flag_version=_flag_version(args)
    )
    overall = report["overall"]
    print(
        f"acceptance rate {overall['accepted']}/{overall['decided']} decided "
        f"(deferred {overall['deferred']}, pending {overall['pending']}, "
        f"sample {overall['sample_size']})"
    )
