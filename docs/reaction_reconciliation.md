# VMH–Rhea reaction reconciliation

`stereomapper reconcile` is a downstream workflow that uses StereoMapper's metabolite-level
relationships to **propose** VMH–Rhea reaction mappings for curator review. It is a pilot.

> **Proposals are not mappings.** Every proposal starts as `pending`. Only proposals a curator
> explicitly accepts are exported for genome-scale metabolic model (GEM) use. Rank, coverage, and
> scores never promote a proposal, and nothing labels a reaction pair "equivalent".

The specification, plan, and design decisions are in
[`specs/001-vmh-rhea-reaction-reconciliation/`](../specs/001-vmh-rhea-reaction-reconciliation/).

## Compatibility

- `stereomapper run`, its arguments, and the StereoMapper SQLite results schema are unchanged.
  The results database is opened read-only.
- New output files are TSV/JSON, versioned as `reconcile-out-v1` (recorded in every run
  manifest).
- `stereomapper.domain.relationship_terms` gains `relationship_term_table()`, which is used to
  pin the SMRO vocabulary in run manifests.
- Python ≥3.9, per `pyproject.toml`.

## Workflow

```bash
stereomapper reconcile propose --config pilot_config.json --out runs/pilot-01
stereomapper reconcile flag-subsystems --run runs/pilot-01 --flag-list subsystem_flags_v1.tsv
stereomapper reconcile benchmark --run runs/pilot-01 --worksheet jc_nm.tsv \
    --worksheet-version 2026-11 --flags v1
stereomapper reconcile review-workbook --run runs/pilot-01 --out review.xlsx --flags v1
# curators fill the decision_template sheet and save it as TSV
stereomapper reconcile record-decisions --run runs/pilot-01 --decisions batch1.tsv \
    --decision-log curation/decisions.tsv
stereomapper reconcile export-curated --run runs/pilot-01 \
    --decision-log curation/decisions.tsv --out curation/curated_mappings.tsv --flags v1
stereomapper reconcile sample --run runs/pilot-01 --size 100 --seed 20261103 --flags v1 \
    --exclude-worksheet runs/pilot-01/benchmark/2026-11/v1 --out curation/sample1.tsv
stereomapper reconcile acceptance --run runs/pilot-01 --sample curation/sample1.tsv \
    --decision-log curation/decisions.tsv --out curation/acceptance1.json --flags v1
```

Every command exits with 0 on success, 2 on invalid arguments or input-contract violations (the
message names the file, row, and field), and 1 on other failures. No command modifies its inputs.

`benchmark`, `export-curated`, `sample`, and `acceptance` require either `--flags LIST_VERSION`
or `--no-subsystem-flags`, so applying subsystem flags is always an explicit, recorded choice.

### Commands

| Command | Purpose | Writes |
|---------|---------|--------|
| `propose --config C --out RUN [--overwrite]` | Generate ranked proposals from pinned inputs | the run directory (below) |
| `flag-subsystems --run RUN --flag-list F` | Apply a versioned subsystem flag list | `RUN/flags/<list_version>/` |
| `review-workbook --run RUN --out X.xlsx [--decision-log L] [--proposals IDS] [--flags V]` | Curator view | `.xlsx` with `proposals`, `participants`, `decision_template`, `about` |
| `record-decisions --run RUN --decisions B --decision-log L` | Validate and append decisions | appends to `L` (whole batch rejected if any row is invalid) |
| `export-curated --run RUN --decision-log L --out C.tsv (--flags V \| --no-subsystem-flags) [--allow-stale-evidence]` | Export accepted mappings | `C.tsv`, `C.rereview.tsv`, `C.summary.json` |
| `benchmark --run RUN --worksheet W --worksheet-version V (--flags … ) [--out DIR]` | Recovery against the verified worksheet | `benchmark_pairs.tsv`, `benchmark_report.json` (default `RUN/benchmark/<V>/<flags>/`) |
| `sample --run RUN --size N --seed S --out F (--flags … ) [--exclude-worksheet BENCH_DIR]` | Stratified review sample | `F` with seed, size, and strata in header comments |
| `acceptance --run RUN --sample F --decision-log L --out R.json (--flags … )` | Acceptance rate over the reviewed sample | `R.json` |

## Inputs

### Run configuration (JSON)

```json
{
  "config_version": 1,
  "inputs": {
    "extract_dir": "extract/", "stereomapper_db": "results.sqlite", "version_tag": "<tag>",
    "baseline": "baseline.tsv", "baseline_version": "<id>",
    "checksums": {"results.sqlite": "sha256:<hex>"}
  },
  "scope": {"rhea_status": {"include": ["approved", "preliminary"], "exclude": ["obsolete"]}},
  "cofactors": {"list": "builtin:cofactors-v1"},
  "alignment": {"version": "align-v1", "exact_max_side": 16},
  "retrieval": {"version": "retrieve-v1", "routes": ["shared_cluster", "related_cluster", "xref"],
                "cap": null},
  "ranking": {"version": "rank-v1"}
}
```

Paths are relative to the config file. Only `inputs.extract_dir`, `inputs.stereomapper_db`, and
`inputs.version_tag` are required; the other values have the defaults shown. Pathway or subsystem
exclusion can't be configured: nothing is excluded before proposals (FR-005). `retrieval.cap`
must stay `null`; a cap needs measured benchmark impact before adoption (FR-018).

### KG extract (directory)

The extract comes from a versioned KG query kept outside this repository. All files are UTF-8,
tab-separated, and have a header row.

- `reactions.tsv`: `source` (`vmh`/`rhea`), `reaction_id`, `equation`. VMH rows also need
  `direction` (`forward`, `reverse`, `reversible`, `unknown`) and optionally `xref_rhea`
  (`;`-separated). Rhea rows need `rhea_status` (`approved`, `preliminary`, `obsolete`) and
  optionally `rhea_master_id` and `rhea_directional_ids`. Directional IDs follow Rhea numbering
  (master N, then N+1 LR, N+2 RL, N+3 BI).
- `participants.tsv`: `source`, `reaction_id`, `side` (`left`/`right`), `metabolite_id`,
  `coefficient`, plus the optional `name`, `compartment`, `kind` (`small_molecule` (default),
  `generic`, `polymer`, `macromolecule`), `structure_curie` (the CURIE used when the molfile was
  loaded into StereoMapper), `formula`, and `charge`. Coefficients must be positive numbers. A
  symbolic coefficient (for example `n`) is accepted only for generic, polymer, or macromolecule
  participants. Proposals use the Rhea master reaction's participants.
- `subsystems.tsv`: `reaction_id` (VMH), `subsystem`, and optionally `kg_pathway_id`.
- `extract_manifest.json`: `kg_snapshot`, `vmh_release`, `rhea_release`, `extraction_query`
  (repository, commit, path), and `extracted_at`. "Latest release" is decided when the extract is
  made; the workflow records it but can't verify it.

### Other inputs

- **Strict baseline** (`baseline.tsv`): `vmh_reaction_id`, `rhea_id`. Rhea IDs are normalized
  to master IDs for joins, and the raw values are kept.
- **JC + NM worksheet**: `vmh_reaction_id`, `rhea_id`, `curator_verdict`. Only `verified` rows
  (case-insensitive) are benchmark positives. Other rows are reported as `not_a_positive`.
- **Subsystem flag list**: a first line `# list_version: <version>`, then `subsystem`,
  `category`, and `rationale` (all required). Subsystem names must match the run's VMH
  subsystems exactly; an unknown name is rejected (exit 2), so a typo can't silently flag
  nothing.
- **Decision batch**: `proposal_id`, `evidence_hash`, `decision` (`accepted`, `rejected`,
  `deferred`), `reviewer`, `date` (YYYY-MM-DD), `reason`, and optionally `supersedes`.

## How proposals are built

1. **Scope.** Approved and preliminary Rhea reactions are in scope; preliminary ones are flagged.
   Obsolete reactions are written to `scope_exclusions.tsv`. No reaction is excluded by pathway
   or subsystem.
2. **Evidence.** Each participant is linked to a StereoMapper cluster through its
   `structure_curie`, using the configured `version_tag` only. Pair evidence is either the same
   cluster (SMRO:0001, `cluster_membership`) or a `relationships` row (`relationship`).
3. **Retrieval** (`retrieve-v1`). Rhea reactions are found through shared clusters, related
   clusters (any relationship class), and existing cross-references. Cofactor clusters aren't
   used as retrieval keys. This can't omit a plausible candidate, because plausibility needs a
   supporting non-cofactor pair. There is no cap.
4. **Alignment** (`align-v1`). Participants are aligned one-to-one per side, under both
   orientations. The objective, compared in order, is: supporting pairs, uncertain pairs, class
   weights (Identical > Identical with undetermined charge > Protomers > Stereo-resolution >
   Diastereomers > Enantiomers > uncertain classes), and equal coefficients. The class order only
   chooses among competing assignments and never excludes a class. Sides with more than
   `exact_max_side` entries use a deterministic greedy assignment, flagged
   `alignment_exactness=greedy` and counted in the funnel. Tied assignments with different class
   profiles set `ambiguous_alignment` and are listed in `alternative_alignments.tsv`.
5. **Plausibility.** A candidate is proposed when at least one supporting pair involves no
   cofactor (`cofactors-v1`: water, H⁺, O₂, CO₂, phosphate, diphosphate, ATP/ADP/AMP,
   NAD(P)(H), FAD(H2), CoA, ammonium; a participant counts as a cofactor if its cluster contains a
   listed CURIE).
6. **Ranking** (`rank-v1`). Proposals are sorted by the minimum of the two coverages (compared
   as exact fractions), then supporting pairs, consistent stoichiometry, fewer non-assessable
   entries, more SMRO:0001/0002 pairs, then `proposal_id`. Rank is a **review order, not a
   probability**. Low-ranked proposals are kept.

### Participant states

Each evidence row has a `structure_state` and an `alignment_state`, which map to one `spec_state`:

| structure_state | alignment_state | spec_state |
|---|---|---|
| assessable | aligned | aligned (supporting class: SMRO:0001/0002/0003/0005/0006/0007) |
| assessable | aligned_uncertain | uncertain (SMRO:0004/0008/0009; not counted as aligned) |
| assessable | supported_nonmatch | supported_nonmatch |
| structure_unavailable | unmatched | structure_unavailable (no `structure_curie`) |
| no_result | unmatched | no_result (CURIE absent from the results DB) |
| processing_failed | unmatched | processing_failed (cluster `status` ≠ `passed`; error kept) |
| not_assessable | unmatched | not_assessable (generic, polymer, macromolecule; never aligned) |

Unavailable, failed, and not-assessable participants stay in every coverage denominator.
Formula agreement is shown (`formula_agreement`) but never aligns participants on its own.

### Flags on each proposal

`proton_balance_difference` is set when H⁺ is the only stoichiometric difference; H⁺ is
otherwise an ordinary participant. The other flags are:
- relationship content: `has_stereo_relationship` (SMRO:0005/0006/0007), `has_protomer`,
  `has_uncertain`;
- data availability: `has_not_assessable`, `has_unavailable_structure`;
- alignment: `ambiguous_alignment`, `alignment_exactness`;
- Rhea status: `rhea_preliminary`;
- compartments: `vmh_transport`, `rhea_transport`, `compartment_difference`;
- direction: `direction_difference` and `direction_note`. A direction difference is set when the
  extract lacks the Rhea directional ID that matches the VMH direction under the chosen
  orientation, or when the VMH cross-reference names the opposite directional ID;
- baseline: `in_strict_baseline`.

`relationship_profile` is the first that applies, in this order: `includes_uncertain`,
`includes_stereo`, `includes_protomer`, `partial` (any unmatched entry), `identical_only`.

## Outputs

### Run directory (`propose`)

| File | Content |
|------|---------|
| `run_manifest.json` | `run_id`, extract releases and KG snapshot, input paths (relative) and checksums, results DB checksum and `version_tag`, baseline and cofactor versions, `smro_vocabulary`, `output_schema_version`, configuration, code version and git commit, funnel, `trace`, `nondeterminism` |
| `run_config.json` | the merged configuration that was used |
| `reaction_proposals.tsv` | one row per proposal, in rank order |
| `participant_evidence.tsv` | every participant entry, for both orientations, with provenance (`vmh_source_row`, `rhea_source_row`, `sm_cluster_ids`, `sm_version_tag`) |
| `alternative_alignments.tsv` | alternatives for ambiguous proposals |
| `scope_exclusions.tsv` | obsolete Rhea reactions |
| `vmh_subsystems.tsv` | pinned subsystem assignments used by `flag-subsystems` |
| `funnel.tsv` | stage counts with denominators |
| `reactions_summary.tsv` | per-reaction structure availability and scope, used by `benchmark` |
| `candidates.tsv` | every retrieved candidate and its outcome (`proposed`, `cofactor_only`, `no_supporting_pair`) |
| `baseline_pairs.tsv` | the strict baseline with normalized Rhea master IDs |

`proposal_id` (`RXP-…`) is stable for a VMH–Rhea pair across runs. `evidence_hash` versions the
evidence a curator reviews; it covers the proposal's evidence and provenance, but not rank,
`run_id`, or subsystems. `run_id` (`RUN-…`) hashes the pinned inputs and configuration, so
identical inputs give identical outputs and the same `run_id` from any location.

### Subsystem flags (`flag-subsystems`)

Flags apply **after** proposals and never change the run's files or evidence hashes, so a list
can be revised without regenerating proposals. A VMH reaction is flagged if any of its subsystems
is listed. Flagged proposals stay visible, are left out of counted results (benchmark `counted`
subsets, sampling, acceptance rates), and **are exported with `subsystem_flagged=true` if a
curator accepts them**, so GEM users can filter them. `flags_manifest.json` gives coverage
with and without the flags.

### Curation

The decision log is append-only. A later decision on the same proposal must name the current
decision in `supersedes`; rows are never edited. Two decisions for one proposal that don't
supersede each other are a conflict, and `export-curated` refuses to run until it is resolved.
By default, an accepted decision whose `evidence_hash` differs from the exported run goes to
`*.rereview.tsv` instead of the export. `*.summary.json` counts decisions by status for all and
for unflagged proposals.

### Benchmark

Each positive worksheet pair gets exactly one status: `recovered_new`, `already_in_baseline`
(never counted as newly recovered), or `unrecovered`. An unrecovered pair gets one reason,
checked in this order:
1. `vmh_id_absent` or `rhea_id_absent`;
2. `rhea_obsolete`;
3. `structure_insufficient` (no assessable non-cofactor participant on one side);
4. `not_retrieved`;
5. `retrieved_not_plausible`.

Results are reported for `full_worksheet`, `counted` (in scope and not flagged), and
`counted_sufficient_structure` (every participant of both reactions assessable), each with
numerator and denominator. `rule_effect` shows obsolete and flagged pairs and the counted results
without flags.

**Worksheet recovery is not precision.** The worksheet holds verified positives, not negatives.
Use `sample` and `acceptance` to estimate how often new proposals are accepted. The acceptance
rate is accepted / (accepted + rejected); deferred and pending counts are shown alongside it.

## Versions

When behaviour changes, bump the matching version string: `align-v1` (alignment), `retrieve-v1`
(retrieval), `rank-v1` (ranking), `cofactors-v1` (cofactor list), or `reconcile-out-v1` (output
files). Benchmark results must be reported against the exact versions used.
