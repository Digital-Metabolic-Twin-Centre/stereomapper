---

description: "Task list for VMH–Rhea reaction reconciliation"
---

# Tasks: VMH–Rhea Reaction Reconciliation

**Input**: Design documents from `specs/001-vmh-rhea-reaction-reconciliation/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/ (cli.md, input-files.md,
output-files.md), quickstart.md

**Tests**: Required. Constitution Principle V requires regression tests for chemistry rules,
missing-data handling, and mapping decisions, and integration tests for changed joins and
contracts. Spec SC-009 lists the regression cases. Write each story's tests first and confirm they
fail before implementing.

**Organization**: Tasks are grouped by user story so each story can be implemented and tested on
its own.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on incomplete tasks)
- **[Story]**: The user story the task belongs to (US1–US4)

## Path Conventions

- Production code: `src/stereomapper/reconciliation/`
- Tests: `src/stereomapper/tests/reconciliation/`
- Every new module starts with `from __future__ import annotations` and must run on Python 3.9
  (no `match`, no 3.10+ APIs), per research R14.
- Identifiers in task text (R1–R16, FR-xxx, US1-n) refer to research.md and spec.md.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Package skeleton, packaging, and the Python 3.9 import fix the feature depends on.

- [X] T001 Add `from __future__ import annotations` (after any module docstring) to every module that uses `X | Y` annotations so the package imports on Python 3.9 (R14): src/stereomapper/domain/relationship_terms.py, src/stereomapper/domain/exceptions/base.py, src/stereomapper/domain/chemistry/validation.py, src/stereomapper/classification/inchi.py, src/stereomapper/classification/engine.py, src/stereomapper/data/results_repo.py, src/stereomapper/comparison/compare.py, src/stereomapper/results/assemblers.py, src/stereomapper/utils/timing.py, src/stereomapper/utils/itertools.py, src/stereomapper/utils/suppress.py. Rewrite any union evaluated at runtime outside annotations (in `isinstance`, type aliases, or `cast`) with `Optional`/`Union`. Run `pytest -ra` to confirm nothing breaks
- [X] T002 Create the package skeleton: src/stereomapper/reconciliation/__init__.py (module docstring only), an empty src/stereomapper/reconciliation/resources/ directory, src/stereomapper/tests/reconciliation/__init__.py, and an empty src/stereomapper/tests/reconciliation/fixtures/ directory
- [X] T003 Add `[tool.setuptools.package-data]` with `"stereomapper.reconciliation" = ["resources/*.tsv", "resources/*.json"]` to pyproject.toml, and confirm `pip install -e .` still succeeds
- [X] T004 [P] Create src/stereomapper/reconciliation/resources/cofactors-v1.tsv with header `curie	name	reason` and rows for water, H+, O2, CO2, phosphate, diphosphate, ATP, ADP, AMP, NAD+, NADH, NADP+, NADPH, FAD, FADH2, CoA, and ammonium, using the ChEBI CURIEs (lower-case `chebi:` prefix, matching StereoMapper member CURIEs) for the forms Rhea uses at pH 7.3 (R6)
- [X] T005 [P] Create src/stereomapper/reconciliation/resources/default_config.json with the run-configuration defaults from contracts/input-files.md#run-configuration (`config_version` 1, `scope.rhea_status` include approved/preliminary and exclude obsolete, `cofactors.list` `builtin:cofactors-v1`, `alignment` `align-v1` with `exact_max_side` 16, `retrieval` `retrieve-v1` with routes `shared_cluster`, `related_cluster`, `xref` and `cap` null, `ranking` `rank-v1`); input paths stay empty

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared types, configuration, input contracts, deterministic writers, CLI wiring, and
test fixture builders. All user stories depend on these.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T006 [P] Implement every enumeration and entity from data-model.md as frozen dataclasses and `str`-valued `Enum`s in src/stereomapper/reconciliation/model.py: Side, Orientation, VmhDirection, RheaStatus, ParticipantKind, StructureState, AlignmentState, EvidenceType, DecisionValue, BenchmarkStatus, UnrecoveredReason, ExclusionRule, SourceReaction, ParticipantEntry (with an `entry_key` property `source|reaction_id|side|metabolite_id|compartment`), ParticipantEvidence, ScopeExclusion, SubsystemFlagList, SubsystemFlag, ReactionProposal, CurationDecision, CuratedMapping, BenchmarkPair, ReviewSample. Store coefficients as `fractions.Fraction` parsed from text
- [X] T007 [P] Implement src/stereomapper/reconciliation/hashing.py: `sha256_file(path) -> str` (returns `sha256:<hex>`, streamed), `canonical_json(obj) -> str` (sorted keys, no whitespace, Fractions as `n/d` strings), `short_hash(text, n=16) -> str`, and `proposal_id(vmh_id, rhea_master_id)` returning `RXP-` + the first 16 hex characters of SHA-256 over `vmh_id|rhea_master_id` (R10)
- [X] T008 Implement src/stereomapper/reconciliation/config.py: `load_run_config(path) -> RunConfig`, which deep-merges the user JSON over resources/default_config.json (loaded via `importlib.resources`), resolves input paths relative to the config file, verifies any `inputs.checksums` entries with `hashing.sha256_file`, and raises `ConfigurationError` (reuse `stereomapper.domain.exceptions`) on a missing required input, unknown version strings, or a checksum mismatch. Also expose the version constants `ALIGN_V1`, `RETRIEVE_V1`, `RANK_V1`, `COFACTORS_V1`, and `OUTPUT_SCHEMA_VERSION = "reconcile-out-v1"` (depends on T005, T007)
- [X] T009 Implement src/stereomapper/reconciliation/inputs.py: readers for `reactions.tsv`, `participants.tsv`, `subsystems.tsv`, `extract_manifest.json`, `baseline.tsv`, `worksheet.tsv`, the subsystem flag list (with its `# list_version:` comment line), and decision batch TSVs, exactly per contracts/input-files.md. Keep unknown extra columns, apply defaults (`kind` = `small_molecule`), and raise `InputContractError(file, row, field, message)` (defined here) on a missing required column, an invalid enumeration value, a non-positive coefficient, or a duplicate `entry_key` within one reaction side. Never merge duplicates silently (depends on T006)
- [X] T010 [P] Implement src/stereomapper/reconciliation/outputs.py: `write_tsv(path, columns, rows)` (UTF-8, tab-separated, `\n` line endings, fixed column order, values formatted by one `format_value` helper: Fractions as `n/d`, decimals to 4 places, booleans as `true`/`false`, None as empty), `write_json(path, obj)` (sorted keys, indent 2, trailing newline), and `fraction_columns(n, d)` returning the `n/d` string and the 4-place decimal (depends on T006)
- [X] T011 Create the `reconcile` subcommand group in src/stereomapper/reconciliation/cli.py with `register(subparsers)` that adds the subparsers `propose`, `flag-subsystems`, `review-workbook`, `record-decisions`, `export-curated`, `benchmark`, `sample`, `acceptance` with the arguments in contracts/cli.md (handlers may raise `NotImplementedError` for now). Add a shared helper for the mutually exclusive, required `--flags LIST_VERSION` / `--no-subsystem-flags` option, and map exit codes: 0 success, 2 for `InputContractError`/`ConfigurationError`/invalid arguments (message names the file, row, and field), 1 otherwise
- [X] T012 Register the `reconcile` group in src/stereomapper/cli.py `build_parser()` by calling `reconciliation.cli.register(sub)` and dispatching `args.cmd == "reconcile"` in `main`, leaving the existing `run` subcommand's arguments and behaviour unchanged (depends on T011)
- [X] T013 Write fixture builders in src/stereomapper/tests/reconciliation/conftest.py:
  - `make_results_db(tmp_path, clusters, relationships, version_tag="v-test")` builds a StereoMapper results DB with `stereomapper.data.results_schema.results_schema`, inserts `clusters` (with `status`/`error`), `cluster_members`, `relationships` (label, `classification_term_id` via `get_relationship_term_id`, `direction`, `score`), and `relationship_members`.
  - `make_extract(tmp_path, reactions, participants, subsystems)` writes the three TSVs and `extract_manifest.json`.
  - `make_config(tmp_path, **overrides)` writes a run config pointing to them.
  - Helpers `vmh_rxn(...)`, `rhea_rxn(...)`, `participant(...)` keep test data short.
- [X] T014 [P] Write src/stereomapper/tests/reconciliation/test_inputs.py covering: valid fixtures load; a missing required column raises `InputContractError` naming file and column; invalid `rhea_status`, `kind`, or `direction` values are rejected; a coefficient of 0 or below is rejected; a duplicate `entry_key` is rejected; extra columns are kept; the flag-list `list_version` is parsed and an empty `rationale` is rejected
- [X] T015 [P] Write src/stereomapper/tests/reconciliation/test_config.py covering: defaults are merged; relative paths are resolved; a checksum mismatch raises `ConfigurationError`; an unknown `alignment.version` is rejected; there is no pathway or glucose setting, so pathway exclusion cannot be configured (FR-005)

**Checkpoint**: Foundation ready. `stereomapper reconcile --help` lists all subcommands, and
`stereomapper run --help` is unchanged.

---

## Phase 3: User Story 1 - Review evidence-backed reaction proposals (Priority: P1) 🎯 MVP

**Goal**: `stereomapper reconcile propose` turns pinned extracts and a StereoMapper DB into ranked,
plausible proposals with full per-participant evidence. Subsystem flags can be applied afterwards,
and curators can open everything in a review workbook.

**Independent Test**: Run `propose` on fixture extracts plus a fixture DB and check that each US1
acceptance scenario (US1-1 to US1-9) and the FR-012, FR-016, and FR-017 cases appear in
`reaction_proposals.tsv` / `participant_evidence.tsv` exactly as quickstart.md §1 describes.

### Tests for User Story 1 (write first; must fail) ⚠️

- [X] T016 [P] [US1] Write src/stereomapper/tests/reconciliation/test_evidence.py:
  - State cases: a participant without `structure_curie` → `structure_unavailable`; a CURIE absent from `cluster_members` → `no_result`; a cluster with `status` ≠ `passed` → `processing_failed` with its `error` kept; `kind` generic/polymer/macromolecule → `not_assessable`.
  - Pair evidence: same cluster → SMRO:0001 `cluster_membership`; a relationship row → its label, term ID, direction, and score; neither → `supported_nonmatch`.
  - **Version isolation**: a DB holding the same cluster pair under two `version_tag`s returns only the configured one (R2).
  - **Read-only**: the DB file's checksum is unchanged after lookups (FR-004).
- [X] T017 [P] [US1] Write src/stereomapper/tests/reconciliation/test_alignment.py:
  - Exactness: one-to-one assignment never reuses an entry (2 vs 1 coefficients, US1-5), and the lexicographic objective prefers supporting over uncertain edges and then the `align-v1` class order.
  - Ties: tied optimal assignments with different class profiles → `ambiguous` with every alternative returned (FR-016); identical-profile ties are not ambiguous.
  - Orientation: the `reversed` orientation swaps sides.
  - Fallback: sides larger than `exact_max_side` use the greedy method, flagged `greedy`, and are deterministic for shuffled inputs.
- [X] T018 [P] [US1] Write src/stereomapper/tests/reconciliation/test_proposals.py with one fixture-driven test per case in quickstart.md §1 rows US1 through FR-016:
  - exact pair;
  - stereo-resolution pair labelled SMRO:0007 with direction (US1-1);
  - enantiomer, diastereomer, and protomer proposals present with `has_stereo_relationship` / `has_protomer` (US1-2);
  - 4 aligned + 1 unmatched gives coverage `4/5` and lists all 5 entries (US1-3);
  - a reversed Rhea equation yields one proposal with `selected_orientation=reversed` (US1-4);
  - a stoichiometry mismatch gives `stoichiometry_consistent=false` (US1-5);
  - a missing structure gives `structure_unavailable` and is kept in denominators (US1-6);
  - formula-only agreement is not aligned but sets `formula_agreement` (US1-7);
  - compartment differences are flagged and the proposal is kept (US1-8);
  - `direction_difference` is set, with `direction_note` naming the case, for both R8 cases: (a) the matching Rhea directional ID is missing from the extract, and (b) the VMH `xref_rhea` names the opposite directional ID; the proposal is kept (US1-8);
  - an H+-only difference sets `proton_balance_difference` (US1-9);
  - a generic or polymer participant is `not_assessable` and the proposal is flagged (FR-012);
  - a cofactor-only overlap produces no proposal and is counted in the funnel (FR-017);
  - uncertain classes (SMRO:0004/0008/0009) are `aligned_uncertain` and excluded from aligned numerators (FR-011);
  - no output column or value contains "equivalent" (FR-013);
  - the invariant `n_aligned + n_aligned_uncertain + n_unmatched_vmh == n_vmh_entries` holds;
  - every evidence row's (`structure_state`, `alignment_state`) maps to exactly one spec state in the data-model.md state mapping table, and no invalid combination occurs (FR-014, SC-006);
  - every evidence row carries `vmh_source_row`/`rhea_source_row`, `sm_cluster_ids`, and `sm_version_tag` (FR-019).
- [X] T019 [P] [US1] Write src/stereomapper/tests/reconciliation/test_scope.py: obsolete Rhea reactions go to `scope_exclusions.tsv` with their status and are never candidates; preliminary reactions are proposed with `rhea_preliminary=true` (FR-005a); reactions from every VMH subsystem (including sugar and drug metabolism) are retrieved and proposed (FR-005)
- [X] T020 [P] [US1] Write src/stereomapper/tests/reconciliation/test_ranking.py: the `rank-v1` key order from R11 (min coverage compared as exact Fractions, then supporting count, stoichiometry, non-assessable count ascending, SMRO:0001/0002 count, `proposal_id` tie-break); every proposal gets a rank with no gaps; low-ranked proposals are kept; `rank_note` is `review order, not a probability`
- [X] T021 [P] [US1] Write src/stereomapper/tests/reconciliation/test_flags.py: applying a flag list writes `flags/<list_version>/` with `subsystem_flags.tsv`, `flagged_proposals.tsv`, `flag_list.tsv`, and `flags_manifest.json`; the any-match rule flags a reaction with several subsystems when one is listed; an unknown subsystem name exits 2 and writes nothing; an existing list version is not overwritten; the run's `reaction_proposals.tsv` and every `evidence_hash` are byte-identical before and after (FR-007); `flags_manifest.json` `coverage` gives `vmh_reactions_with_proposals` and `proposals` as `all` and `unflagged` with `n/d` values, and `all = unflagged + flagged` (FR-006)
- [X] T022 [P] [US1] Write src/stereomapper/tests/reconciliation/test_cli_propose.py: `stereomapper reconcile propose` on fixtures exits 0, writes every run-directory file from contracts/output-files.md#run-directory with the required proposal columns in order, prints the `run_id` and funnel, refuses a non-empty `--out` without `--overwrite` (exit 2), and exits 2 with file/row/field on a broken extract; `review-workbook` writes an .xlsx with sheets `proposals`, `participants`, `decision_template`, `about`, where every proposal shows `pending` when no decision log is given

### Implementation for User Story 1

- [X] T023 [P] [US1] Implement src/stereomapper/reconciliation/normalize.py:
  - Build `SourceReaction` and `ParticipantEntry` objects from the input rows. Rhea rows without `rhea_master_id` are masters; directional IDs are attached to their master; participants come from the master's equation.
  - `sides(reaction, orientation)` returns (left, right) entry lists for `same` or `reversed`.
  - `is_transport_vmh(reaction)` (participants span more than one compartment) and `is_transport_rhea(reaction)` (any participant has a location).
  - Keep raw IDs untouched (FR-004, R7, R8).
- [X] T024 [P] [US1] Implement src/stereomapper/reconciliation/scope.py: `apply_rhea_status_rule(reactions, config) -> (in_scope, exclusions)`, which returns `ScopeExclusion(rule=rhea_obsolete, rule_parameter=<status list>, evidence=<status>, reason=...)` for each excluded Rhea reaction. Apply no pathway or subsystem filtering (FR-005, FR-005a)
- [X] T025 [US1] Implement src/stereomapper/reconciliation/evidence.py:
  - `StereoMapperEvidence(db_path, version_tag)` opens `file:<path>?mode=ro` with `uri=True` and loads `cluster_members` and `clusters.status`/`error` for every needed CURIE.
  - It also loads `relationships` joined to the cluster pairs **filtered by `version_tag`**, keyed by unordered cluster pair. The `direction` is kept, and `A_to_B`/`B_to_A` are mapped to `vmh_more_resolved`/`rhea_more_resolved` per pair orientation.
  - `structure_state(entry)` and `pair_evidence(vmh_entry, rhea_entry) -> PairEvidence` follow R2.
  - `related_clusters(cluster_id)` returns related clusters for retrieval.
  - Uses SMRO labels and term IDs from `stereomapper.domain.relationship_terms`; fails with `ConfigurationError` if the `version_tag` is absent from the DB (depends on T023).
- [X] T026 [US1] Implement cofactor resolution in src/stereomapper/reconciliation/evidence.py: `load_cofactor_list(spec)` (`builtin:cofactors-v1` via `importlib.resources`, or a file path) returns the version, checksum, and CURIE set; `is_cofactor(entry)` is true when the entry's cluster contains any listed CURIE (R6) (depends on T004, T025)
- [X] T027 [US1] Implement src/stereomapper/reconciliation/alignment.py:
  - `align(vmh_reaction, rhea_reaction, evidence, config) -> {Orientation: AlignmentResult}` solves each side by exact bitmask dynamic programming over the smaller side (≤ `exact_max_side`), maximizing the lexicographic objective from R4: supporting edges, uncertain edges, `align-v1` class-order weights, equal coefficients.
  - It enumerates every optimal assignment and sets `ambiguous` when their class profiles differ. For larger sides it uses a deterministic greedy fallback (sorted by weight, then `entry_key`) and marks it `greedy`.
  - `AlignmentResult` holds the pairs, unmatched entries per side, alternatives, and exactness (depends on T025).
- [X] T028 [US1] Implement src/stereomapper/reconciliation/retrieval.py:
  - `build_rhea_index(rhea_reactions, evidence)` maps cluster → Rhea master IDs.
  - `retrieve(vmh_reaction, index, evidence, xrefs) -> dict[rhea_master_id, set[route]]` uses the routes `shared_cluster`, `related_cluster` (any class, via `related_clusters`), and `xref` (from `xref_rhea`, normalized to masters).
  - A `Funnel` accumulator records the counts per stage and route listed in contracts/output-files.md (`funnel.tsv` stages). No cap (R5) (depends on T025).
- [X] T029 [US1] Implement src/stereomapper/reconciliation/proposals.py:
  - `build_proposal(vmh, rhea, alignments, evidence, cofactors, baseline_pairs) -> ReactionProposal | None` selects the orientation by objective (tie → `same`) and computes counts per state and per SMRO term, coverage as Fractions with assessable counts, and `stoichiometry_consistent`.
  - Flags: `proton_balance_difference` (R9), `has_stereo_relationship`, `has_protomer`, `has_uncertain`, `has_not_assessable`, `has_unavailable_structure`, `ambiguous_alignment`, `alignment_exactness`, `rhea_preliminary`, `vmh_transport`, `rhea_transport`, `compartment_difference`, `direction_difference` with `direction_note` (both per R8), `in_strict_baseline` (FR-021), `rhea_matching_directional_id` (R8), `relationship_profile` (R13), and `vmh_subsystems`.
  - Plausibility (FR-017): return None (and record it in the funnel, including the "only cofactors aligned" count) unless at least one supporting pair is non-cofactor.
  - `evidence_rows(proposal)` yields ParticipantEvidence rows for **both** orientations, covering every entry, with `formula_agreement` for unmatched pairs with equal formulas, and the provenance columns `vmh_source_row`, `rhea_source_row`, `sm_cluster_ids`, `sm_version_tag` (FR-019).
  - `evidence_hash` is computed with `hashing.canonical_json` over the sorted evidence rows, flags, and counts (R10) (depends on T026, T027, T028).
- [X] T030 [P] [US1] Implement src/stereomapper/reconciliation/ranking.py: `rank_proposals(proposals) -> list[ReactionProposal]` assigns `rank` 1..n by the `rank-v1` key from R11 and sets `ranking_version` and `rank_note` (depends on T006)
- [X] T031 [US1] Implement src/stereomapper/reconciliation/manifest.py: `build_run_record(config, extract_manifest, input_paths, evidence, cofactors, funnel, greedy_count)` records:
  - checksums of every input via `hashing.sha256_file`, plus the StereoMapper DB checksum and `version_tag`;
  - the VMH and Rhea releases, KG snapshot, and query commit from the extract manifest;
  - the baseline version and cofactor version/checksum;
  - the configuration, `stereomapper.__version__`, and the git commit (from `git rev-parse HEAD` if available, otherwise `unknown`);
  - `smro_vocabulary` (SHA-256 of the canonical (label, term_id) table from `stereomapper.domain.relationship_terms`, plus the checksum of docs/ontology/relationship_terms.csv when present) and `output_schema_version` (Principle II, FR-002);
  - the Python version, and `nondeterminism: []`;
  - the run ID: `run_id = short_hash(canonical_json(record without started_at))` (R10). Inputs are recorded by path relative to the config file plus checksum; absolute paths and the output folder are excluded. `started_at` is added afterwards (depends on T007).
- [X] T032 [US1] Implement the `propose` pipeline in src/stereomapper/reconciliation/pipeline.py: `run_propose(config, out_dir, overwrite)`:
  1. Load the inputs.
  2. Apply the scope rule.
  3. Build the evidence and Rhea index.
  4. For each VMH reaction in sorted ID order, retrieve, align, and build proposals.
  5. Rank them.
  6. Write `reaction_proposals.tsv`, `participant_evidence.tsv`, `alternative_alignments.tsv`, `scope_exclusions.tsv`, `vmh_subsystems.tsv`, `funnel.tsv`, and `run_manifest.json` with outputs.py, in the column orders and sort orders of contracts/output-files.md.

  Refuse a non-empty `out_dir` without `overwrite` (depends on T024, T029, T030, T031).
- [X] T033 [US1] Wire the `propose` handler in src/stereomapper/reconciliation/cli.py to `pipeline.run_propose`, printing the `run_id` and each funnel stage as `stage: count/denominator` (depends on T032)
- [X] T034 [US1] Implement src/stereomapper/reconciliation/flags.py:
  - `apply_flags(run_dir, flag_list_path)` reads the run's `vmh_subsystems.tsv` and validates every listed subsystem name against them (exit 2 on unknown names).
  - It flags VMH reactions under the any-match rule and their proposals, then writes `flags/<list_version>/flag_list.tsv`, `subsystem_flags.tsv`, `flagged_proposals.tsv`, and `flags_manifest.json` (`run_id`, list version, checksum, counts by category, and the `coverage` object with all/unflagged `n/d` values for VMH reactions with proposals and for proposals, per contracts/output-files.md#flags-directory). It refuses an existing version directory.
  - `load_flags(run_dir, list_version | None) -> FlagSet` is shared by the downstream commands. `None` means `--no-subsystem-flags`, recorded as `flag_list_version: none`.
  - Wire the `flag-subsystems` handler in src/stereomapper/reconciliation/cli.py (FR-005b, FR-007, R16) (depends on T032).
- [X] T035 [US1] Implement src/stereomapper/reconciliation/review.py: `write_review_workbook(run_dir, out_path, decision_log=None, proposal_ids=None, flags=None)` uses openpyxl and writes four sheets:
  - `proposals`: one row per proposal in rank order with every proposal column, including both full equations, `vmh_subsystems`, and the effective decision status, or `pending` without a log;
  - `participants`: evidence rows for the selected orientation;
  - `decision_template`: columns `proposal_id`, `evidence_hash`, `decision`, `reviewer`, `date`, `reason`, `supersedes`, with the first two pre-filled;
  - `about`: the run ID, versions, and the rank note.

  Wire the `review-workbook` handler in src/stereomapper/reconciliation/cli.py (depends on T032, T034).

**Checkpoint**: User Story 1 is fully functional. T016–T022 pass, and a curator can run `propose`,
`flag-subsystems`, and `review-workbook` on fixture data.

---

## Phase 4: User Story 2 - Record curator decisions and export accepted mappings (Priority: P2)

**Goal**: An append-only decision log plus a curated export that contains only effective accepted
decisions. Flagged mappings are exported with their flag shown, and stale evidence is routed to
re-review.

**Independent Test**: On a fixed proposal set, record accepted, rejected, and deferred decisions,
then export. Only accepted pairs appear, each with a complete decision reference, and the
proposal files are unchanged (US2-1 to US2-4).

### Tests for User Story 2 (write first; must fail) ⚠️

- [X] T036 [P] [US2] Write src/stereomapper/tests/reconciliation/test_curation.py:
  - Pending, rejected, and deferred proposals are absent from `CURATED.tsv` (US2-1, US2-2).
  - An accepted proposal is exported with `decision_id`, `reviewer`, `date`, `reviewed_evidence_hash`, and `run_id` (US2-3).
  - An accepted decision whose evidence hash differs from the run goes to `CURATED.rereview.tsv` by default and is exported with `evidence_matches_run=false` under `--allow-stale-evidence` (US2-4).
  - Accepted flagged proposals are exported with `subsystem_flagged=true` and `matched_subsystems` under `--flags`, and the flag columns are empty under `--no-subsystem-flags` (FR-023).
  - A superseding decision changes the effective decision while both rows stay in the log; two rows that don't supersede each other for one proposal make export fail with exit 2.
  - The proposal files' checksums are unchanged after recording and exporting (FR-024).
  - `CURATED.summary.json` counts by status (all and unflagged) add up to the number of proposals (FR-029).
  - The review workbook shows `decision_evidence_current=false` for a decision recorded against a different `evidence_hash` (US2-4).
- [X] T037 [P] [US2] Write src/stereomapper/tests/reconciliation/test_cli_curation.py: `record-decisions` appends valid rows and creates the log with a header; a batch with one invalid row (unknown proposal, wrong evidence hash, empty reason, bad decision value, a `supersedes` pointing to another proposal) exits 2 and leaves the log byte-identical; `export-curated` without either flag option exits 2 (argparse)

### Implementation for User Story 2

- [X] T038 [US2] Implement src/stereomapper/reconciliation/curation.py:
  - `validate_batch(run_dir, batch_rows, existing_log) -> list[CurationDecision]` checks the proposal exists, the evidence hash matches the run, the decision is in `DecisionValue`, reviewer, date (ISO 8601), and reason are not empty, and `supersedes` refers to an existing decision for the same proposal. It raises `InputContractError` listing every bad row.
  - `decision_id` is `DEC-` + `short_hash` over (`proposal_id`, `evidence_hash`, `reviewer`, `date`, `decision`).
  - `append_decisions(log_path, decisions)` opens the log in append mode only, writing the header if the file is new.
  - `effective_decisions(log_rows) -> dict[proposal_id, CurationDecision]` follows the supersedes chains and raises on conflicting heads.
- [X] T039 [US2] Implement `export_curated(run_dir, log_path, out_path, flags: FlagSet, allow_stale)` in src/stereomapper/reconciliation/curation.py. It writes `CURATED.tsv` (CuratedMapping fields plus `evidence_matches_run`, `subsystem_flagged`, `matched_subsystems`, sorted by `vmh_reaction_id`, `rhea_master_id`, with a `# flag_list_version:` header comment) `CURATED.rereview.tsv`, and `CURATED.summary.json` (effective decisions by status for all and unflagged proposals, with the proposal total as denominator, FR-029), per contracts/output-files.md#curation-outputs (depends on T034, T038)
- [X] T040 [US2] Wire the `record-decisions` and `export-curated` handlers in src/stereomapper/reconciliation/cli.py to curation.py, using the shared flag option helper from T011, and update `review-workbook` to show the effective decision and a `decision_evidence_current` column (true when the decision's `evidence_hash` equals the run's; stale rows highlighted with an openpyxl fill) when `--decision-log` is given (depends on T035, T039)

**Checkpoint**: User Stories 1 and 2 both work on their own.

---

## Phase 5: User Story 3 - Benchmark recovery against verified omissions and the strict baseline (Priority: P3)

**Goal**: Account for every JC + NM worksheet pair, report recovery over three subsets with
explicit denominators, keep baseline pairs separate, report results with and without subsystem
flags, and estimate acceptance rate on a seeded sample.

**Independent Test**: Run a fixture worksheet with recovered, unrecovered (one per reason),
already-mapped, flagged, and obsolete-Rhea pairs through the workflow. Every pair gets exactly one
status, the subset sums equal the denominators, and the report says it is not precision (US3-1
to US3-5).

### Tests for User Story 3 (write first; must fail) ⚠️

- [X] T041 [P] [US3] Write src/stereomapper/tests/reconciliation/test_benchmark.py:
  - Statuses: a worksheet pair in the baseline is `already_in_baseline`, not `recovered_new` (US3-2); each `UnrecoveredReason` is produced by a dedicated fixture, checked in the R13 order (US3-3); a worksheet pair involving a flagged sugar/drug reaction keeps its status, has `subsystem_flagged=true`, appears in the full-worksheet subset, and is absent from `counted` (US3-1).
  - Joins: Rhea directional IDs in the worksheet join through master IDs; non-positive worksheet rows appear as `not_a_positive` and never enter denominators.
  - Report invariants: `recovered_new + already_in_baseline + unrecovered == denominator` for every subset; `note` states recovery of known positives, not precision (US3-4); `rule_effect` gives flagged counts by category and the counted results without flags.
- [X] T042 [P] [US3] Write src/stereomapper/tests/reconciliation/test_sampling.py: the same seed gives the same sample; strata follow relationship profile × rank tercile; worksheet pairs and (with `--flags`) flagged proposals are never sampled; the acceptance report gives `accepted`, `rejected`, `deferred`, `pending`, `decided` (= accepted + rejected), `sample_size`, and `acceptance_rate` per stratum and overall; a stratum with zero decided reports a null rate, not 0 (US3-5)

### Implementation for User Story 3

- [X] T043 [US3] Implement src/stereomapper/reconciliation/benchmark.py: `run_benchmark(run_dir, worksheet_path, worksheet_version, flags: FlagSet, out_dir)`:
  - Loads the run, worksheet, and baseline (from the run config), normalizes Rhea IDs to masters, and gives each positive row exactly one status and reason in the R13 order. "Sufficient structure evidence" means every entry of both reactions is assessable.
  - Sets `in_scope`, `subsystem_flagged`, `counted`, `proposal_id`, and `rank`.
  - Writes `benchmark_pairs.tsv` and `benchmark_report.json` with the subsets `full_worksheet`, `counted`, `counted_sufficient_structure`, plus `rule_effect` and `greedy_alignment_pairs`, per contracts/output-files.md#benchmark-directory. Asserts the sum invariants before writing (depends on T034).
- [X] T044 [US3] Implement src/stereomapper/reconciliation/sampling.py:
  - `draw_sample(run_dir, size, seed, flags, exclude_benchmark_dir)` uses `random.Random(seed)` over proposals sorted by `proposal_id`, stratified by `relationship_profile` × rank tercile with proportional allocation (remainders by largest fraction, then stratum name). It writes `SAMPLE.tsv` with `#` header lines for the seed, size, strata, `run_id`, and `flag_list_version`.
  - `acceptance_report(sample_path, log_path, flags)` writes `REPORT.json` per contracts/output-files.md#curation-outputs, using `curation.effective_decisions` (depends on T038).
- [X] T045 [US3] Wire the `benchmark`, `sample`, and `acceptance` handlers in src/stereomapper/reconciliation/cli.py, with the default `BENCH_DIR` `RUN_DIR/benchmark/<VERSION>/<LIST_VERSION or none>/` and the shared flag option (depends on T043, T044)

**Checkpoint**: User Stories 1, 2, and 3 work on their own.

---

## Phase 6: User Story 4 - Reproduce and audit a run (Priority: P3)

**Goal**: Identical pinned inputs give byte-identical outputs and the same `run_id`; every
proposal traces back to raw source records and StereoMapper rows.

**Independent Test**: Run `propose` twice into different directories and diff all outputs
(US4-1 to US4-3, SC-003, SC-004).

### Tests for User Story 4 (write first; must fail) ⚠️

- [X] T046 [P] [US4] Write src/stereomapper/tests/reconciliation/test_determinism.py: two `propose` runs with identical inputs produce byte-identical TSVs and identical `run_manifest.json` apart from `started_at`, plus an identical `run_id`; shuffling input row order changes nothing except provenance (source row numbers, the input checksums behind `run_id`, and `evidence_hash`, which covers provenance columns); changing one input byte changes the `run_id` and the recorded checksum (SC-003)
- [X] T047 [P] [US4] Write src/stereomapper/tests/reconciliation/test_traceability.py: `run_manifest.json` contains every field listed for RunRecord in data-model.md, including `smro_vocabulary` and `output_schema_version`, and `smro_vocabulary` changes when a term ID in the vocabulary table is altered (monkeypatched) (US4-1, Principle II); running from a different working directory or output folder gives the same `run_id`; for every proposal, each `participant_evidence.tsv` row's raw `metabolite_id`/`reaction_id` exists in the input extract and each `relationship` row matches a `relationships` row under the configured `version_tag` (US4-3, SC-004); every input participant entry appears in the evidence for each orientation (SC-006)

### Implementation for User Story 4

- [X] T048 [US4] Close any gaps T046–T047 reveal in src/stereomapper/reconciliation/manifest.py and src/stereomapper/reconciliation/outputs.py: make all iteration orders explicitly sorted, remove dict-order and set-order dependence, format floats only through `outputs.format_value`, and list any unavoidable nondeterminism in the manifest's `nondeterminism` field (FR-003)
- [X] T049 [US4] Add a `trace` section to `run_manifest.json` in src/stereomapper/reconciliation/manifest.py that maps each output file to the input checksums it derives from (FR-004, US4-3)

**Checkpoint**: All user stories are independently functional.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, compatibility notes, end-to-end validation, and QA gates
(Principle V).

- [X] T050 [P] Write src/stereomapper/tests/reconciliation/test_cli_integration.py: an end-to-end fixture runs `propose` → `flag-subsystems` → `benchmark` → `review-workbook` → `record-decisions` → `export-curated` → `sample` → `acceptance` through `stereomapper.cli.main`, asserting exit codes and the quickstart.md §2–§4 checks (funnel sums, benchmark invariants, only accepted rows in `CURATED.tsv`, unchanged proposals)
- [X] T051 [P] Write docs/reaction_reconciliation.md as the canonical user documentation:
  - purpose and the proposal-vs-curated distinction;
  - every `reconcile` subcommand with its arguments and exit codes (from contracts/cli.md);
  - input and output file layouts (from contracts/input-files.md and output-files.md);
  - participant states and SMRO handling;
  - subsystem flags and the `--flags`/`--no-subsystem-flags` choice;
  - versions `align-v1`, `retrieve-v1`, `rank-v1`, `cofactors-v1`;
  - interpretation cautions (rank is not a probability; worksheet recovery is not precision);
  - a compatibility note that `stereomapper run` and the SQLite schema are unchanged.
- [X] T052 [P] Add a "Reaction reconciliation (downstream)" section with a short example and a link to docs/reaction_reconciliation.md to README.md
- [X] T053 [P] Update CONTRIBUTING.md: note that reconciliation tests live in src/stereomapper/tests/reconciliation/, that new modules must stay Python 3.9 compatible, and that version strings (`align-v1`, `rank-v1`, cofactor list) must be bumped when behaviour changes
- [X] T054 [P] Update docs/reaction_reconciliation_design_notes.md and docs/speckit_reaction_reconciliation_input.md so they no longer describe a glucose-pathway exclusion. Replace it with the post-processing subsystem flag design (spec clarification 2026-11-04) and note that spec.md is authoritative
- [X] T055 Run quickstart.md §1 (`pytest -ra src/stereomapper/tests/reconciliation`) and fix any failing case
- [X] T056 Run the QA gates from the repository root (`pytest -ra`, `ruff check .`, `black --check .`, `mypy .`, plus `mypy --python-version 3.9 src/stereomapper/reconciliation`), fix any issues in src/stereomapper/reconciliation/ and src/stereomapper/tests/reconciliation/, and record in the PR description any check that could not run and why (Principle V)

- [X] T057 Verify Python 3.9 compatibility: under a Python 3.9 interpreter (for example `uv run --python 3.9`), run `python -c "import stereomapper.cli"` and `pytest -ra src/stereomapper/tests/reconciliation`; fix any failure in src/stereomapper/, and record the result, or why it could not run, in the PR description (Principle V)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: none.
- **Foundational (Phase 2)**: depends on Setup and blocks every user story.
- **US1 (Phase 3)**: depends on Foundational. It produces the run directory that every later
  story reads.
- **US2 (Phase 4)**: depends on US1 (`propose`, `flags.load_flags`, `review.py`). It doesn't
  depend on US3 or US4.
- **US3 (Phase 5)**: depends on US1. It needs `curation.effective_decisions` (T038) only for the
  acceptance report (T044), so T038 is the single cross-story dependency.
- **US4 (Phase 6)**: depends on US1. It can run in parallel with US2 and US3.
- **Polish (Phase 7)**: depends on all the stories it documents or tests.

### Within Each User Story

- Write the tests first and confirm they fail.
- normalize/scope → evidence → cofactors → alignment/retrieval → proposals → ranking/manifest →
  pipeline → CLI → flags → review (US1).
- curation core → export → CLI (US2); benchmark/sampling → CLI (US3).

### Parallel Opportunities

- T004 and T005; T006, T007, T010; T014 and T015.
- All US1 test tasks T016–T022 together; T023, T024, T030 in parallel.
- After US1: US2, US3 (except T044, which waits on T038), and US4 can proceed in parallel.
- Polish: T050–T054 in parallel; T057 after T056.

---

## Parallel Example: User Story 1

```bash
# Tests first, in parallel:
Task: "T016 test_evidence.py"   Task: "T017 test_alignment.py"   Task: "T018 test_proposals.py"
Task: "T019 test_scope.py"      Task: "T020 test_ranking.py"     Task: "T021 test_flags.py"
Task: "T022 test_cli_propose.py"

# Independent modules in parallel:
Task: "T023 normalize.py"   Task: "T024 scope.py"   Task: "T030 ranking.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 1 and Phase 2.
2. Phase 3 (US1): `propose`, `flag-subsystems`, `review-workbook`.
3. **Stop and validate**: run quickstart.md §1 for the US1 rows and a pilot `propose` on real
   extracts. Curators can start reviewing in the workbook.

### Incremental Delivery

1. US1, so curators can review proposals (MVP).
2. US2, so decisions are recorded and accepted mappings exported for GEM use.
3. US3, so recovery against the JC + NM worksheet and acceptance rate can be reported.
4. US4, to harden reproducibility for publication.
5. Polish: documentation and the QA gates before the PR.

---

## Notes

- [P] tasks touch different files and have no dependency on incomplete tasks.
- Never write to the StereoMapper results DB or any input file; tests T016 and T036 check this.
- Commit after each task or logical group, and stop at any checkpoint to validate.
