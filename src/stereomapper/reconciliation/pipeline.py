"""``reconcile propose``: pinned inputs -> run directory (contracts/output-files.md)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from stereomapper.domain.exceptions import ConfigurationError
from stereomapper.reconciliation.alignment import align_pair
from stereomapper.reconciliation.config import RunConfig
from stereomapper.reconciliation.evidence import (
    PROTON_CURIE,
    CofactorIndex,
    StereoMapperEvidence,
    load_cofactor_list,
)
from stereomapper.reconciliation.inputs import load_extract, read_baseline
from stereomapper.reconciliation.manifest import build_run_record, trace_section
from stereomapper.reconciliation.model import SourceReaction, StructureState
from stereomapper.reconciliation.normalize import Reactions, build_reactions
from stereomapper.reconciliation.outputs import write_json, write_tsv
from stereomapper.reconciliation.proposals import (
    ALTERNATIVE_COLUMNS,
    EVIDENCE_COLUMNS,
    PROPOSAL_COLUMNS,
    Proposal,
    ProposalBuilder,
)
from stereomapper.reconciliation.ranking import rank_proposals
from stereomapper.reconciliation.retrieval import Funnel, RheaIndex, retrieve
from stereomapper.reconciliation.scope import apply_rhea_status_rule

logger = logging.getLogger(__name__)

RUN_FILES = (
    "run_manifest.json",
    "run_config.json",
    "reaction_proposals.tsv",
    "participant_evidence.tsv",
    "alternative_alignments.tsv",
    "scope_exclusions.tsv",
    "vmh_subsystems.tsv",
    "funnel.tsv",
    "reactions_summary.tsv",
    "candidates.tsv",
    "baseline_pairs.tsv",
)
DERIVED_DIRS = ("flags", "benchmark")

SUMMARY_COLUMNS = [
    "source",
    "reaction_id",
    "rhea_master_id",
    "rhea_status",
    "in_scope",
    "n_entries",
    "n_assessable",
    "has_assessable_noncofactor",
    "all_assessable",
]
CANDIDATE_COLUMNS = [
    "vmh_reaction_id",
    "rhea_master_id",
    "retrieval_routes",
    "plausible",
    "outcome",
    "proposal_id",
    "alignment_exactness",
]
BASELINE_COLUMNS = ["vmh_reaction_id", "rhea_id", "rhea_master_id", "source_note"]
SCOPE_COLUMNS = ["rule", "source", "reaction_id", "rule_parameter", "evidence", "reason"]


def _prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise ConfigurationError(
                f"Run directory {out_dir} is not empty; use --overwrite to replace the run"
            )
        derived = [d for d in DERIVED_DIRS if (out_dir / d).exists()]
        if derived:
            raise ConfigurationError(
                f"Run directory {out_dir} has derived outputs {derived} from the previous run; "
                "remove them before overwriting"
            )
        for name in RUN_FILES:
            target = out_dir / name
            if target.exists():
                target.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)


def _summary_rows(
    reactions: Reactions,
    rhea_in_scope: set[str],
    evidence: StereoMapperEvidence,
    cofactors: CofactorIndex,
) -> list[dict[str, Any]]:
    rows = []

    def stats(reaction: SourceReaction) -> dict[str, Any]:
        states = [evidence.structure_state(e) for e in reaction.participants]
        assessable = [
            e for e, s in zip(reaction.participants, states) if s == StructureState.ASSESSABLE
        ]
        return {
            "n_entries": len(states),
            "n_assessable": len(assessable),
            "has_assessable_noncofactor": any(not cofactors.is_cofactor(e) for e in assessable),
            "all_assessable": bool(states) and len(assessable) == len(states),
        }

    for rid, reaction in reactions.vmh.items():
        rows.append({"source": "vmh", "reaction_id": rid, "in_scope": True, **stats(reaction)})
    for rid, reaction in reactions.rhea_masters.items():
        base = {
            "source": "rhea",
            "rhea_master_id": rid,
            "rhea_status": reaction.rhea_status,
            "in_scope": rid in rhea_in_scope,
            **stats(reaction),
        }
        rows.append({**base, "reaction_id": rid})
        for directional in reaction.rhea_directional_ids:
            if directional != rid:
                rows.append({**base, "reaction_id": directional})
    rows.sort(key=lambda r: (r["source"], r["reaction_id"]))
    return rows


def run_propose(config: RunConfig, out_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    out_dir = Path(out_dir)
    _prepare_out_dir(out_dir, overwrite)

    extract = load_extract(config.extract_dir)
    reactions = build_reactions(extract)
    cofactor_list = load_cofactor_list(config.cofactor_spec, config.config_dir)

    input_paths = dict(extract.paths)
    input_paths["stereomapper_db"] = config.stereomapper_db
    baseline_rows: list[dict[str, str]] = []
    if config.baseline is not None:
        input_paths["baseline"] = config.baseline
        baseline_rows = read_baseline(config.baseline).rows
    record = build_run_record(config, extract.manifest, input_paths, cofactor_list)

    rhea_in_scope, exclusions = apply_rhea_status_rule(
        reactions.rhea_masters, config.rhea_status_include
    )
    curies = {p.structure_curie for r in reactions.all_reactions() for p in r.participants}
    curies |= set(cofactor_list.curies) | {PROTON_CURIE}
    evidence = StereoMapperEvidence(config.stereomapper_db, config.version_tag, curies)
    try:
        cofactors = CofactorIndex(cofactor_list, evidence)
        baseline_out = []
        baseline_pairs: set[tuple[str, str]] = set()
        for row in baseline_rows:
            master = reactions.master_for(row["rhea_id"]) or ""
            baseline_out.append({**row, "rhea_master_id": master})
            if master:
                baseline_pairs.add((row["vmh_reaction_id"], master))
        builder = ProposalBuilder(evidence, cofactors, baseline_pairs, record["run_id"])
        index = RheaIndex(rhea_in_scope, evidence, cofactors)

        funnel = Funnel()
        funnel.set("vmh_reactions_read", len(reactions.vmh))
        funnel.set("rhea_reactions_read", len(reactions.rhea_masters))
        funnel.set("rhea_excluded_obsolete", len(exclusions))
        funnel.set("vmh_in_scope", len(reactions.vmh))
        funnel.set("rhea_in_scope", len(rhea_in_scope))

        proposals: list[Proposal] = []
        candidate_rows: list[dict[str, Any]] = []
        vmh_with_proposals: set[str] = set()
        for vmh_id in sorted(reactions.vmh):
            vmh = reactions.vmh[vmh_id]
            found = retrieve(vmh, index, evidence, cofactors, reactions, config.routes)
            if found:
                funnel.add("vmh_with_candidates")
            for rhea_id in sorted(found):
                routes = found[rhea_id]
                for route in routes:
                    funnel.add(f"candidates_{route}")
                funnel.add("candidates_total")
                rhea = rhea_in_scope[rhea_id]
                results = align_pair(vmh, rhea, evidence.pair_evidence, config.exact_max_side)
                funnel.add("candidates_aligned")
                outcome = builder.build(vmh, rhea, results, routes)
                if outcome.proposal is None:
                    funnel.add(f"not_plausible_{outcome.reason}")
                else:
                    funnel.add("candidates_plausible")
                    proposals.append(outcome.proposal)
                    vmh_with_proposals.add(vmh_id)
                    if outcome.exactness == "greedy":
                        funnel.add("proposals_greedy_alignment")
                candidate_rows.append(
                    {
                        "vmh_reaction_id": vmh_id,
                        "rhea_master_id": rhea_id,
                        "retrieval_routes": tuple(sorted(routes)),
                        "plausible": outcome.plausible,
                        "outcome": "proposed" if outcome.proposal else outcome.reason,
                        "proposal_id": (
                            outcome.proposal.fields["proposal_id"] if outcome.proposal else ""
                        ),
                        "alignment_exactness": outcome.exactness,
                    }
                )
        funnel.set("proposals", len(proposals))
        funnel.set("vmh_reactions_with_proposals", len(vmh_with_proposals))
        summary_rows = _summary_rows(reactions, set(rhea_in_scope), evidence, cofactors)
    finally:
        evidence.close()

    ranked = rank_proposals(proposals)
    write_tsv(out_dir / "reaction_proposals.tsv", PROPOSAL_COLUMNS, (p.fields for p in ranked))
    by_id = sorted(ranked, key=lambda p: p.fields["proposal_id"])
    write_tsv(
        out_dir / "participant_evidence.tsv",
        EVIDENCE_COLUMNS,
        (row for p in by_id for row in p.evidence_rows),
    )
    write_tsv(
        out_dir / "alternative_alignments.tsv",
        ALTERNATIVE_COLUMNS,
        (row for p in by_id for row in p.alternative_rows),
    )
    write_tsv(
        out_dir / "scope_exclusions.tsv",
        SCOPE_COLUMNS,
        (vars(e) for e in sorted(exclusions, key=lambda e: (e.rule, e.source, e.reaction_id))),
    )
    write_tsv(
        out_dir / "vmh_subsystems.tsv",
        ["reaction_id", "subsystem"],
        sorted(
            (
                {"reaction_id": rid, "subsystem": s}
                for rid, r in reactions.vmh.items()
                for s in r.subsystems
            ),
            key=lambda r: (r["reaction_id"], r["subsystem"]),
        ),
    )
    write_tsv(out_dir / "funnel.tsv", ["stage", "count", "denominator", "note"], funnel.rows())
    write_tsv(out_dir / "reactions_summary.tsv", SUMMARY_COLUMNS, summary_rows)
    write_tsv(out_dir / "candidates.tsv", CANDIDATE_COLUMNS, candidate_rows)
    write_tsv(
        out_dir / "baseline_pairs.tsv",
        BASELINE_COLUMNS,
        sorted(baseline_out, key=lambda r: (r["vmh_reaction_id"], r["rhea_id"])),
    )
    write_json(out_dir / "run_config.json", record["config"])

    record["funnel"] = funnel.rows()
    record["greedy_alignment_count"] = funnel.counts["proposals_greedy_alignment"]
    record["counts"] = {"proposals": len(ranked), "scope_exclusions": len(exclusions)}
    record["trace"] = trace_section(record)
    write_json(out_dir / "run_manifest.json", record)
    logger.info("run %s: %d proposals", record["run_id"], len(ranked))
    return record
