"""Curator review workbook (research R12). The workbook is a view, not the store of record."""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

from stereomapper.reconciliation.flags import load_flags
from stereomapper.reconciliation.ranking import RANK_NOTE
from stereomapper.reconciliation.rundir import RunDir

TEMPLATE_COLUMNS = [
    "proposal_id",
    "evidence_hash",
    "decision",
    "reviewer",
    "date",
    "reason",
    "supersedes",
]
DECISION_COLUMNS = [
    "decision_status",
    "decision_evidence_current",
    "decision_id",
    "decision_reviewer",
    "decision_date",
    "subsystem_flagged",
    "matched_subsystems",
]
_STALE = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
_BOLD = Font(bold=True)


def write_review_workbook(
    run_dir: Path,
    out_path: Path,
    decision_log: Path | None = None,
    proposal_ids: list[str] | None = None,
    flags: str | None = None,
) -> None:
    run = RunDir(run_dir)
    flag_set = load_flags(run.path, flags)
    effective = {}
    if decision_log is not None and Path(decision_log).is_file():
        from stereomapper.reconciliation.curation import effective_decisions, read_decision_log

        effective = effective_decisions(read_decision_log(Path(decision_log)))

    proposals = run.proposals
    if proposal_ids is not None:
        wanted = set(proposal_ids)
        proposals = [p for p in proposals if p["proposal_id"] in wanted]
    columns = list(run.rows("reaction_proposals.tsv")[0].keys()) if run.proposals else []

    wb = Workbook()
    sheet = wb.active
    sheet.title = "proposals"
    sheet.append(DECISION_COLUMNS + columns)
    for cell in sheet[1]:
        cell.font = _BOLD
    for p in proposals:
        decision = effective.get(p["proposal_id"])
        current = None if decision is None else decision.evidence_hash == p["evidence_hash"]
        row = [
            decision.decision.value if decision else "pending",
            current,
            decision.decision_id if decision else "",
            decision.reviewer if decision else "",
            decision.date if decision else "",
            flag_set.is_vmh_flagged(p["vmh_reaction_id"]),
            ";".join(flag_set.matched_subsystems(p["vmh_reaction_id"])),
        ] + [p[c] for c in columns]
        sheet.append(row)
        if current is False:
            for cell in sheet[sheet.max_row]:
                cell.fill = _STALE
    sheet.freeze_panes = "A2"

    selected = {p["proposal_id"]: p["selected_orientation"] for p in proposals}
    evidence = [
        r
        for r in run.rows("participant_evidence.tsv")
        if selected.get(r["proposal_id"]) == r["orientation"]
    ]
    part = wb.create_sheet("participants")
    ev_columns = list(evidence[0].keys()) if evidence else []
    part.append(ev_columns)
    for cell in part[1]:
        cell.font = _BOLD
    for r in evidence:
        part.append([r[c] for c in ev_columns])
    part.freeze_panes = "A2"

    template = wb.create_sheet("decision_template")
    template.append(TEMPLATE_COLUMNS)
    for cell in template[1]:
        cell.font = _BOLD
    for p in proposals:
        template.append([p["proposal_id"], p["evidence_hash"], "", "", "", "", ""])

    about = wb.create_sheet("about")
    manifest = run.manifest
    for key, value in (
        ("run_id", run.run_id),
        ("output_schema_version", manifest.get("output_schema_version")),
        ("stereomapper_version", manifest.get("stereomapper_version")),
        ("git_commit", manifest.get("git_commit")),
        ("vmh_release", manifest["extract"].get("vmh_release")),
        ("rhea_release", manifest["extract"].get("rhea_release")),
        ("kg_snapshot", manifest["extract"].get("kg_snapshot")),
        ("version_tag", manifest["stereomapper_results"]["version_tag"]),
        ("ranking_version", manifest["config"]["ranking"]["version"]),
        ("flag_list_version", flag_set.version),
        ("rank_note", RANK_NOTE),
        (
            "decisions",
            "Record decisions in decision_template, export it as TSV, and run "
            "`stereomapper reconcile record-decisions`. Values: accepted, rejected, "
            "deferred. Only accepted mappings are exported for GEM use.",
        ),
    ):
        about.append([key, value])
    wb.save(out_path)
