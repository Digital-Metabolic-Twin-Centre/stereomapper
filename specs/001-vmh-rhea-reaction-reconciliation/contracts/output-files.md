# Output Contract

All TSVs are UTF-8 and tab-separated, with a header row. Rows are sorted as stated for each file.
Fractions are written as `n/d` with a separate decimal column at 4 decimal places. The files in
the run directory are immutable once written for a pinned run.

## Run directory

| File | One row per | Sort order | Key content |
|------|-------------|------------|-------------|
| `run_manifest.json` | — | — | RunRecord ([data-model.md](../data-model.md#runrecord-manifest)): pinned versions (including `smro_vocabulary` and `output_schema_version`), checksums, config, `run_id`, funnel counts, greedy-alignment count, `trace`, nondeterminism notes |
| `reaction_proposals.tsv` | proposal | `rank` | every ReactionProposal field; `rank_note` column fixed to `review order, not a probability` |
| `participant_evidence.tsv` | participant entry or aligned pair, per orientation | `proposal_id`, `orientation`, `side`, `entry_key` | ParticipantEvidence plus `structure_state`, `is_cofactor`, `kind`, raw IDs, and provenance columns `vmh_source_row`, `rhea_source_row`, `sm_cluster_ids`, `sm_version_tag` |
| `alternative_alignments.tsv` | alternative optimal assignment row | `proposal_id`, `alternative_group` | only for proposals marked `ambiguous_alignment` |
| `scope_exclusions.tsv` | excluded reaction (obsolete Rhea only) | `rule`, `source`, `reaction_id` | ScopeExclusion |
| `vmh_subsystems.tsv` | VMH reaction × subsystem | `reaction_id`, `subsystem` | copy of the pinned subsystem assignments, used by `flag-subsystems` |
| `run_config.json` | — | — | merged configuration used for the run |
| `reactions_summary.tsv` | source reaction (VMH, Rhea master, Rhea directional ID) | `source`, `reaction_id` | `rhea_master_id`, `rhea_status`, `in_scope`, `n_entries`, `n_assessable`, `has_assessable_noncofactor`, `all_assessable` (used by `benchmark`) |
| `candidates.tsv` | retrieved candidate pair | `vmh_reaction_id`, `rhea_master_id` | `retrieval_routes`, `plausible`, `outcome` (`proposed`, `cofactor_only`, `no_supporting_pair`), `proposal_id`, `alignment_exactness` |
| `baseline_pairs.tsv` | strict-baseline row | `vmh_reaction_id`, `rhea_id` | raw Rhea ID plus normalized `rhea_master_id` |
| `funnel.tsv` | stage | stage order | `stage`, `count`, `denominator`, `note` (stages: VMH read, Rhea read, excluded per rule, in scope, candidates per route, candidates total, aligned, plausible, not plausible because only cofactors aligned, proposals) |

### Required proposal columns, in order (excerpt)

`rank`, `proposal_id`, `evidence_hash`, `vmh_reaction_id`, `rhea_master_id`, `rhea_source_ids`,
`rhea_matching_directional_id`, `vmh_equation`, `rhea_equation`, `selected_orientation`,
`coverage_vmh`, `coverage_vmh_decimal`, `coverage_rhea`, `coverage_rhea_decimal`,
`n_aligned`, `n_aligned_uncertain`, `n_unmatched_vmh`, `n_unmatched_rhea`,
`n_SMRO_0001` … `n_SMRO_0009`, `stoichiometry_consistent`, flag columns (see
[data-model.md](../data-model.md#reactionproposal)), `in_strict_baseline`,
`relationship_profile`, `ranking_version`, `rank_note`.

## Flags directory

`RUN_DIR/flags/<list_version>/`, written by `flag-subsystems`:

- `flag_list.tsv`: copy of the applied list, plus its checksum in `flags_manifest.json`.
- `subsystem_flags.tsv`: one row per flagged VMH reaction (SubsystemFlag), sorted by
  `vmh_reaction_id`.
- `flagged_proposals.tsv`: `proposal_id`, `vmh_reaction_id`, `matched_subsystems`, sorted by
  `proposal_id`.
- `flags_manifest.json`: `run_id`, `list_version`, list checksum, counts of flagged VMH reactions
  and proposals by category, and a `coverage` object (FR-006):
  `vmh_reactions_with_proposals` and `proposals`, each with `all` and `unflagged` values written
  as `n/d`. Denominators are in-scope VMH reactions and total proposals respectively, and
  `all = unflagged + flagged` holds for each.

## Benchmark directory

- **`benchmark_pairs.tsv`**: one row per worksheet row, sorted by `worksheet_row`, with the
  BenchmarkPair fields. Non-positive worksheet rows appear with `status=not_a_positive`.
- **`benchmark_report.json`**:

```json
{
  "run_id": "...", "worksheet_version": "...", "baseline_version": "...",
  "flag_list_version": "<version or none>",
  "note": "Recovery of known positives; not precision.",
  "subsets": {
    "full_worksheet":  {"denominator": 0, "recovered_new": 0, "already_in_baseline": 0,
                        "unrecovered": 0, "unrecovered_by_reason": {}, "rank_distribution": {}},
    "counted": {"...": "same fields"},
    "counted_sufficient_structure": {"...": "same fields"}
  },
  "rule_effect": {"pairs_out_of_scope": {"rhea_obsolete": 0},
                  "pairs_flagged_by_category": {"sugar": 0, "drug": 0},
                  "counted_without_flags": {"...": "same fields as a subset"}},
  "greedy_alignment_pairs": 0
}
```

Invariant: for every subset, `recovered_new + already_in_baseline + unrecovered == denominator`.

## Curation outputs

- **Decision log** (`LOG.tsv`, append-only): CurationDecision fields in insertion order.
- **`CURATED.tsv`**: CuratedMapping fields plus `evidence_matches_run`, `subsystem_flagged`, and
  `matched_subsystems`, sorted by `vmh_reaction_id`, `rhea_master_id`. A header comment records
  `flag_list_version`.
- **`CURATED.rereview.tsv`**: accepted decisions whose evidence changed, with both hashes.
- **`CURATED.summary.json`**: effective decisions counted by status (`pending`, `accepted`,
  `rejected`, `deferred`), each for all proposals and for unflagged proposals, with the total
  proposal count as denominator (FR-029).
- **`SAMPLE.tsv`**: header comment lines with the seed, size, and strata, then
  `proposal_id`, `stratum`, `rank`.
- **Acceptance `REPORT.json`**: per-stratum and overall `accepted`, `rejected`, `deferred`,
  `pending`, `decided` (= accepted + rejected), `sample_size`, and `acceptance_rate`
  (= accepted/decided, given with both counts; deferred and pending are reported, not counted as
  decided).
