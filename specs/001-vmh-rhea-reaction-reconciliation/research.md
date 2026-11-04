# Research: VMH–Rhea Reaction Reconciliation

**Feature**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Date**: 2026-11-03 (updated 2026-11-04)

Each entry records a decision, why it was chosen, and the alternatives considered. Values that
must come from project data (for example, the subsystem flag list) are curator-supplied, versioned
inputs. The tool doesn't infer them (Principle IV).

## R1. How the workflow gets VMH and Rhea data from the knowledge graph

- **Decision**: The library does not connect to the KG. A versioned extraction query, kept with
  the KG project and referenced by commit, writes three tab-separated extract files that follow
  [contracts/input-files.md](contracts/input-files.md): reactions, participants, and VMH
  subsystem assignments. A small extract manifest records the KG snapshot, VMH and Rhea release IDs, query
  commit, and extraction timestamp. `reconcile propose` checksums every file and copies this
  information into the run manifest.
- **Rationale**: The repository has no KG client, and the KG technology is outside
  StereoMapper's scope. File extracts can be checksummed and pinned, so reruns are reproducible
  (FR-001–FR-003), and tests can use small fixture extracts.
- **Alternatives considered**: (a) Query the KG directly from the CLI. This couples StereoMapper
  to a KG client and makes "latest" drift during a run. (b) Hand-curated spreadsheets. These
  can't be audited for provenance.

## R2. Linking participants to StereoMapper results

- **Decision**: Each participant row in the extract carries a `structure_curie`, the accession
  CURIE used when the molfile was loaded into StereoMapper (for example, `chebi:15903` for Rhea
  participants). The workflow resolves CURIE → `cluster_members.cluster_id`, then cluster pair →
  `relationships` for one pinned `version_tag`, always joining
  `relationship_members`/`relationships` **on `version_tag` as well as the cluster pair**. The
  database is opened read-only (`file:...?mode=ro`).
- **Rationale**: The design notes ask for the normalized membership tables. The existing
  experiment query `experiments/vmh_chebi_mappings/extract_vmh_chebi_rels.py` joins without
  `version_tag`, which would mix runs in a database holding more than one version.
- **Participant structure states** (FR-014), decided per participant entry:
  - `structure_unavailable`: no `structure_curie` in the extract.
  - `no_result`: the CURIE is absent from `cluster_members` for the pinned database.
  - `processing_failed`: the cluster's `status` is not `passed` (the `error` text is kept).
  - `not_assessable`: generic, polymer, or macromolecule participant (FR-012; type comes from the
    extract's `participant_kind` column).
  - Otherwise the participant is **assessable**.
- **Pair evidence** between two assessable participants:
  - Same cluster → `Identical structures` (SMRO:0001), `evidence_type=cluster_membership`.
  - A `relationships` row for the cluster pair and version → that row's class, term ID,
    direction, and score, `evidence_type=relationship`.
  - Neither → `supported_nonmatch` (both structures processed, and StereoMapper relates them by
    neither membership nor relationship).
- **Alternatives considered**: Matching by InChIKey in the workflow. This duplicates
  StereoMapper's job and breaks the boundary in the spec's assumptions.

## R3. Relationship classes in alignment

- **Decision**: Every SMRO class can create an alignment edge. Edges fall into two groups:
  - **Supporting** (counted as aligned): SMRO:0001 Identical, SMRO:0002 Identical with
    undetermined charge, SMRO:0003 Protomers, SMRO:0007 Stereo-resolution pairs, SMRO:0006
    Diastereomers, SMRO:0005 Enantiomers.
  - **Uncertain** (the pair is shown as `aligned_uncertain`, not counted in aligned numerators,
    FR-011): SMRO:0004 Indistinguishable, SMRO:0009 Unresolved, SMRO:0008 Unclassified.

  Within the alignment objective the classes are ordered Identical > Identical (charge) >
  Protomer > Stereo-resolution > Diastereomer > Enantiomer > uncertain classes. The order is
  used only to choose among competing assignments. It is versioned as `align-v1` and never
  excludes a class.
- **Rationale**: FR-011 lets all classes support a proposal and forbids exclusion by class. A
  tie-break order is still needed when one participant could pair with several others.
- **Alternatives considered**: Treating every class equally. This makes optimal assignments
  ambiguous much more often and gives curators less useful defaults.

## R4. Participant alignment method

- **Decision**: For each candidate pair and each orientation (`same`, `reversed`), solve a
  one-to-one assignment per side (VMH substrates ↔ Rhea substrates, VMH products ↔ Rhea
  products; swapped for `reversed`).
  - **Objective** (compared lexicographically): (1) number of supporting edges, (2) number of
    uncertain edges, (3) sum of class-order weights, (4) number of pairs with equal
    coefficients.
  - **Method**: exact dynamic programming over bitmasks. Sides with at most 16 entries on the
    smaller side cover typical metabolic reactions. The method also enumerates all optimal
    assignments, so the workflow can detect ambiguity (FR-016) when the optimal assignments
    differ in relationship-class profile.
  - **Larger sides**: a deterministic greedy assignment is used, and the proposal is flagged
    `alignment_exactness=greedy`. The proposal is never dropped. Greedy cases are counted in the
    manifest and in benchmark reports so their impact is measurable (FR-018).
  - **One entry per slot**: each participant entry is used at most once. Coefficients are
    compared separately (FR-008).
- **Rationale**: Exact methods are cheap at reaction sizes. Enumeration is needed for ambiguity.
  This avoids adding SciPy as a dependency.
- **Alternatives considered**: (a) Hungarian algorithm via SciPy. This adds a dependency and
  doesn't enumerate ties. (b) Greedy only. Results would depend on input order and could hide
  ambiguity.

## R5. Candidate retrieval and the plausibility rule

- **Decision**: Build an inverted index from cluster ID → Rhea reactions that contain a
  participant in that cluster. Then, for each VMH reaction (every subsystem; nothing is excluded by
  pathway, FR-005):
  1. Collect the clusters of its assessable participants, plus every cluster related to them in
     `relationships` for the pinned version (any class).
  2. Look up the Rhea reactions that contain those clusters.
  3. Add Rhea reactions linked by existing cross-references from the extract.

  Every candidate is aligned. A candidate becomes a proposal when it has at least one supporting
  alignment edge that isn't on the cofactor list (FR-017).
- **No retrieval cap in v1.** Stage counts are recorded in the manifest's `funnel`: VMH and Rhea
  reactions read, excluded by each rule, in scope, candidates retrieved by each route, aligned,
  plausible, proposed. Any future cap follows FR-018: benchmark impact is measured before it is
  adopted.
- **Rationale**: Retrieval goes through participant evidence, so candidates are bounded by real
  links and no Cartesian product is formed (design notes §6). Cofactor-only links are kept out
  of plausibility but still counted.
- **Alternatives considered**: (a) A cap on candidates per VMH reaction. This needs benchmark
  measurement first. (b) A participant-count or formula-profile prefilter. This could drop
  valid partial matches, so it is deferred.

## R6. Cofactor/currency-metabolite list

- **Decision**: Ship `cofactors-v1` as a package resource TSV (`curie`, `name`, `reason`) that
  lists structure CURIEs for water, H⁺, O₂, CO₂, phosphate, diphosphate, ATP/ADP/AMP,
  NAD(H), NADP(H), FAD(H2), CoA, and ammonium in the ChEBI forms Rhea uses. A run can point to
  another list. The list's checksum and version go in the manifest. Cofactors match by cluster:
  a participant is a cofactor if its cluster contains any listed CURIE.
  - The benchmark report gives the number of worksheet pairs that are aligned but not plausible
    because only cofactors aligned (FR-017).
- **Rationale**: Gives a versioned, auditable rule, as agreed in clarification Q3.
- **Alternatives considered**: Frequency-based cofactor detection. It is data-dependent and hard
  to audit.

## R7. Rhea directional groups, status, and identity

- **Decision**: Each Rhea reaction is normalized to its master ID for candidate identity, and
  the proposal key is `(vmh_reaction_id, rhea_master_id)`. Participants come from the master
  reaction's equation as written (left/right).
  - Directional IDs (LR, RL, BI) are kept. Under the selected orientation, the proposal reports
    which Rhea directional ID matches the VMH direction.
  - Status rule (clarification Q5): approved and preliminary are in scope, and preliminary is
    flagged; obsolete reactions are recorded in `scope_exclusions.tsv`.
  - Worksheet and baseline Rhea IDs are normalized to master IDs for joins. The source ID is
    kept.
- **Rationale**: Gives one proposal per pair whatever the orientation or variant (FR-009) and
  keeps source IDs (FR-004).

## R8. Direction and compartment comparison

- **Decision**:
  - **VMH direction** comes from the extract's `direction` field (`forward`, `reverse`,
    `reversible`, derived by the extraction query from bounds and recorded as-is).
  - **Rhea direction**: the master reaction is undirected, so the reported comparison is the
    Rhea directional ID equivalent to the VMH direction under the selected orientation.
    `direction_difference` is set when (a) the extract's `rhea_directional_ids` for the master
    don't include the directional ID matching the VMH direction under the selected orientation,
    or (b) the VMH reaction's `xref_rhea` names a directional ID whose direction is opposite to
    the VMH direction under the selected orientation. `direction_note` says which case applies.
    It also notes when a worksheet or baseline row references a different directional ID.
  - **Compartments**: VMH compartments are kept per participant entry. A VMH reaction whose
    participants span more than one compartment is flagged `vmh_transport`. A Rhea reaction
    whose participants carry location annotations is flagged `rhea_transport`.
    `compartment_difference` is set when only one side is a transport reaction or when the
    location patterns differ. No compartment flag removes a proposal (FR-010).
- **Rationale**: Keeps differences visible without inventing a compartment model for Rhea.

## R9. Proton handling

- **Decision**: H⁺ entries are ordinary participants (clarification Q4). After alignment, if
  every coefficient difference and unmatched amount involves only H⁺ entries, the proposal
  gets the flag `proton_balance_difference`.

## R10. Identifiers, evidence versions, and determinism

- **Decision**:
  - **`proposal_id`**: `RXP-` + the first 16 hex characters of SHA-256 over
    `vmh_reaction_id|rhea_master_id`. It is stable across runs for the same pair.
  - **`evidence_hash`**: SHA-256 over a canonical serialization of the proposal's evidence (all
    participant rows, flags, and coverage counts, in sorted order, with no timestamps).
  - **`run_id`**: SHA-256 over the manifest's pinned inputs and configuration. It excludes the
    wall-clock timestamp, absolute paths, and the output folder (inputs are identified by
    relative path and checksum), so identical inputs give the same run ID from any location.
  - Outputs are written in a fixed sort order with fixed float formatting. No randomness is
    used except sampling, which takes a recorded seed.
- **Rationale**: SC-003 requires identical proposals, evidence, and ranking. Curator decisions
  need a stable pair ID plus a version of the evidence they reviewed (FR-022, FR-024).

## R11. Ranking

- **Decision**: `rank-v1` sorts proposals by this key; earlier fields decide first:
  1. minimum of `coverage_vmh` and `coverage_rhea`, descending;
  2. number of supporting aligned pairs, descending;
  3. stoichiometry consistent (yes before no);
  4. number of non-assessable entries (unavailable, failed, no result, not assessable),
     ascending;
  5. number of SMRO:0001/0002 pairs, descending;
  6. tie-break by `proposal_id`.

  Coverage ratios are compared as exact fractions, not floats. The rank is a review order only.
  The output says so, and no probability is shown (FR-020).
- **Alternatives considered**: A single weighted score. It is harder to explain and looks like a
  probability.

## R12. Curator review surface and decision log

- **Decision**:
  - `reconcile review-workbook` writes an `.xlsx` review workbook with openpyxl, which is already
    a dependency. It has one row per proposal, full equations, and a linked participant sheet,
    plus a decision template sheet. The workbook is not the store of record.
  - Curators record decisions in a template TSV, and `reconcile record-decisions` validates the
    rows and appends them to an **append-only decision log** TSV at a path the curator group
    controls. Earlier rows are never rewritten. A later decision on the same proposal adds a row
    that names the decision it supersedes (FR-024).
  - `reconcile export-curated` produces the curated mapping set from accepted effective
    decisions only. By default it also requires the reviewed `evidence_hash` to match the run
    being exported. Mismatches go to a re-review list instead (User Story 2, scenario 4).
- **Rationale**: Curators already work in spreadsheets, and a plain append-only TSV is easy to
  audit and diff. Access control is the file location's permissions, as the spec assumes a
  small trusted group.
- **Alternatives considered**: A web review app (out of scope for the pilot). Writing decisions
  into the StereoMapper results database (that database is an input and must stay unchanged,
  FR-004).

## R13. Benchmark and acceptance sampling

- **Decision**:
  - `reconcile benchmark` joins worksheet pairs (VMH ID plus normalized Rhea master) with the
    run. Each pair gets exactly one status: `recovered_new`, `already_in_baseline`, or
    `unrecovered`. Unrecovered pairs get a reason, checked in this order:
    1. `vmh_id_absent` or `rhea_id_absent`;
    2. `rhea_obsolete`;
    3. `structure_insufficient` (no assessable non-cofactor participant on one side);
    4. `not_retrieved`;
    5. `retrieved_not_plausible`.

    Subsystem flagging is not an unrecovered reason: a flagged worksheet pair keeps its status
    and gets `subsystem_flagged=true` (R16). Recovery is reported over three denominators (full
    worksheet, counted subset, and counted subset with sufficient structure evidence), each with counts and the rank distribution (FR-025–FR-027).
    "Sufficient structure evidence" means every participant entry of both reactions is
    assessable.
  - `reconcile sample` draws a seeded, stratified sample of proposals outside the worksheet.
    The strata are relationship profile × rank tercile, and the relationship profiles are:
    - `identical_only`
    - `includes_protomer`
    - `includes_stereo` (SMRO:0005/0006/0007)
    - `includes_uncertain`
    - `partial` (any unmatched)

    The seed, size, and strata go into the sample file. Acceptance rate is then reported from the
    decision log over the reviewed denominator (FR-028).
- **Rationale**: Gives a deterministic, explicit accounting where every worksheet pair is
  covered (SC-001).

## R14. Python and packaging constraints

- **Decision**:
  - New modules target Python ≥3.9 (`requires-python`): `from __future__ import annotations`,
    no `match`, and no 3.10+ standard-library APIs. CI or local checks should run on 3.9 where
    possible.
  - Package resources (default configuration and cofactor list) need
    `[tool.setuptools.package-data]` in `pyproject.toml`.
- **Finding to fix**: 11 existing modules use `X | Y` annotations without
  `from __future__ import annotations`, so importing them fails on Python 3.9. They include
  `domain/exceptions/base.py`, which the feature reuses, and `domain/chemistry/validation.py`,
  which `stereomapper/cli.py` imports:
  - `domain/relationship_terms.py`, `domain/exceptions/base.py`, `domain/chemistry/validation.py`
  - `classification/inchi.py`, `classification/engine.py`
  - `data/results_repo.py`, `comparison/compare.py`, `results/assemblers.py`
  - `utils/timing.py`, `utils/itertools.py`, `utils/suppress.py`

  The fix (a future import in each, plus `Optional`/`Union` for any union used at runtime
  outside annotations) is task T001. A Python 3.9 check is task T057.
- **Finding to note**: `[tool.mypy] python_version = "3.11"` doesn't match the 3.9 minimum. This
  is flagged in the constitution's Sync Impact Report. The feature doesn't depend on fixing it.

## R15. Performance and scale

- **Decision**: Target a full run (about 10⁴ VMH reactions × about 1.6·10⁴ Rhea master
  reactions, with the full VMH–ChEBI StereoMapper database) in under 30 minutes and under 8 GB
  of memory on a workstation. Indexes are held in memory, and the StereoMapper database is
  streamed with a version-filtered query.
- **Rationale**: Retrieval through participant evidence keeps the candidate count well below
  the Cartesian product. The real counts are recorded in the funnel, so this estimate can be
  checked on the first full run.

## R16. Post-processing subsystem flags (spec clarification 2026-11-04)

- **Decision**: `reconcile propose` excludes no pathway (FR-005). A separate command,
  `reconcile flag-subsystems`, applies a curator-supplied, versioned **subsystem flag list** to a
  finished run and writes the result to `RUN_DIR/flags/<list_version>/`. The list is a TSV of VMH
  subsystem names, each with a category (`sugar`, `drug`, ...) and a rationale. The initial list
  covers glucose and other sugar metabolism and drug metabolism; the exact names come from the
  pinned VMH release. A VMH reaction is flagged if any of its subsystems is listed; a proposal is
  flagged through its VMH reaction (Rhea reactions have no subsystems).
- **Downstream behaviour**: `benchmark`, `export-curated`, `sample`, and `acceptance` each
  require either `--flags <list_version>` or an explicit `--no-subsystem-flags`, so applying (or
  skipping) the flags is always a recorded choice, never a silent default. With flags applied:
  - flagged proposals stay in `reaction_proposals.tsv` (unchanged) and are listed with their
    subsystems in `flagged_proposals.tsv`;
  - they are left out of the counted subset in benchmark reports and out of acceptance sampling
    and acceptance rates;
  - if a curator accepts one, it is exported in `CURATED.tsv` with `subsystem_flagged=true` and
    its matched subsystems, so GEM users can filter it (spec clarification 2026-11-04);
  - every report gives results both with and without the flags (FR-006).
- **Rationale**: The proposal run stays immutable and its `evidence_hash` values don't depend on
  the list, so curators can revise the list without regenerating proposals or invalidating
  decisions (FR-007).
- **Alternatives considered**: (a) Excluding subsystems before retrieval (the previous design). It
  hides those reactions from review and forces a rerun to change the list. (b) A column inside
  the proposal file. It ties the immutable run to one list version.
