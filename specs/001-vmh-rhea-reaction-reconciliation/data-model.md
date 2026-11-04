# Data Model: VMH–Rhea Reaction Reconciliation

**Feature**: [spec.md](spec.md) | **Research**: [research.md](research.md)

These are logical entities. The file layouts are in [contracts/](contracts/). Every entity
keeps raw source IDs. Normalized IDs are added in separate fields (FR-004).

## Enumerations

| Name | Values | Source |
|------|--------|--------|
| `Side` | `left`, `right` | as written in the source equation |
| `Orientation` | `same`, `reversed` | R4 |
| `VmhDirection` | `forward`, `reverse`, `reversible`, `unknown` | extract (R8) |
| `RheaStatus` | `approved`, `preliminary`, `obsolete` | extract (R7) |
| `ParticipantKind` | `small_molecule`, `generic`, `polymer`, `macromolecule` | extract (FR-012) |
| `StructureState` | `assessable`, `structure_unavailable`, `no_result`, `processing_failed`, `not_assessable` | R2 |
| `AlignmentState` | `aligned`, `aligned_uncertain`, `supported_nonmatch`, `unmatched` | R2, R3 |
| `EvidenceType` | `cluster_membership`, `relationship`, `none` | R2 |
| `DecisionValue` | `accepted`, `rejected`, `deferred` | FR-022 (`pending` = no decision recorded) |
| `BenchmarkStatus` | `recovered_new`, `already_in_baseline`, `unrecovered` | R13 |
| `UnrecoveredReason` | `vmh_id_absent`, `rhea_id_absent`, `rhea_obsolete`, `structure_insufficient`, `not_retrieved`, `retrieved_not_plausible` | R13 |
| `ExclusionRule` | `rhea_obsolete` | FR-005a (no pathway exclusion, FR-005) |

## Participant state mapping (FR-014, SC-006)

Each participant evidence row has one `structure_state` and one `alignment_state`. Together they
map to exactly one spec state:

| structure_state | alignment_state | Spec state |
|---|---|---|
| assessable | aligned | aligned (with class) |
| assessable | aligned_uncertain | uncertain (FR-011) |
| assessable | supported_nonmatch or unmatched | supported nonmatch |
| structure_unavailable | unmatched | structure unavailable |
| processing_failed | unmatched | structure processing failed |
| no_result | unmatched | no StereoMapper result |
| not_assessable | unmatched | not assessable (FR-012) |

Any other combination is invalid.

## Entities

### SourceReaction

| Field | Notes |
|-------|-------|
| `source` | `vmh` or `rhea` |
| `reaction_id` | raw source ID |
| `rhea_master_id` | Rhea only; normalized identity (R7) |
| `rhea_directional_ids` | Rhea only; LR/RL/BI IDs |
| `rhea_status` | Rhea only |
| `vmh_direction` | VMH only |
| `equation` | complete equation text as given by the source |
| `subsystems` | VMH only; all VMH subsystem assignments |
| `cross_references` | existing VMH→Rhea cross-references, if any |

Validation: IDs are unique per source within one run. A Rhea row without a master ID is its own
master.

### ParticipantEntry

One metabolite on one side of one reaction in one compartment (design notes §6).

| Field | Notes |
|-------|-------|
| `entry_key` | `source|reaction_id|side|metabolite_id|compartment` (unique) |
| `metabolite_id` | raw source ID (VMH abbreviation or ChEBI) |
| `name` | display name |
| `compartment` | VMH compartment code, or Rhea location, or empty |
| `coefficient` | positive decimal; stored as text, compared exactly |
| `kind` | `ParticipantKind` |
| `structure_curie` | CURIE used in StereoMapper, or empty |
| `formula`, `charge` | optional diagnostics (FR-015) |
| `cluster_id` | resolved from `cluster_members`, or empty |
| `structure_state` | `StructureState` |
| `structure_error` | `clusters.error` when `processing_failed` |
| `is_cofactor` | the cluster contains a CURIE from the cofactor list (R6) |

Validation: an entry appears once per reaction side. A duplicate key is an input error and is
reported, not merged silently.

### ParticipantEvidence (pair-level)

| Field | Notes |
|-------|-------|
| `vmh_entry_key`, `rhea_entry_key` | one may be empty for an unmatched entry |
| `orientation` | `Orientation` |
| `alignment_state` | `AlignmentState` |
| `evidence_type` | `EvidenceType` |
| `relationship_label`, `smro_term_id` | from SMRO; required whenever the label is set |
| `relationship_direction` | from `relationships.direction` (for example, `A_to_B`), mapped to VMH/Rhea terms |
| `score` | StereoMapper score as recorded; never interpreted as proof |
| `version_tag` | StereoMapper run tag |
| `coefficient_vmh`, `coefficient_rhea`, `coefficient_equal` | stoichiometry comparison |
| `formula_agreement` | optional diagnostic only |
| `alternative_group` | ID shared by alternative optimal assignments (FR-016) |

### ScopeExclusion

| Field | Notes |
|-------|-------|
| `source`, `reaction_id` | raw ID |
| `rule` | `ExclusionRule` |
| `rule_parameter` | the configured status list |
| `evidence` | the Rhea status |
| `reason` | readable text |

### SubsystemFlagList

| Field | Notes |
|-------|-------|
| `list_version` | curator-assigned version string |
| `subsystem` | exact VMH subsystem name from the pinned release |
| `category` | for example, `sugar`, `drug` |
| `rationale` | required, not empty |

The list's checksum is recorded wherever it is applied (R16).

### SubsystemFlag

| Field | Notes |
|-------|-------|
| `vmh_reaction_id` | flagged reaction |
| `matched_subsystems`, `all_subsystems` | evidence |
| `list_version` | list applied |
| `proposal_ids` | proposals flagged through this reaction |

Flags live under `RUN_DIR/flags/<list_version>/`. They never alter the run's proposals or
`evidence_hash` values.

### ReactionProposal

| Field | Notes |
|-------|-------|
| `proposal_id` | `RXP-` + hash of the pair (R10) |
| `evidence_hash` | hash of the canonical evidence (R10); this is the spec's "evidence version" |
| `run_id` | run |
| `vmh_reaction_id`, `rhea_master_id`, `rhea_source_ids` | IDs |
| `rhea_matching_directional_id` | R8 |
| `vmh_equation`, `rhea_equation` | complete equations |
| `vmh_subsystems` | all VMH subsystem assignments, shown to curators |
| `selected_orientation` | the orientation with the best objective; both orientations are kept in evidence |
| `retrieval_routes` | the routes that found the candidate |
| `n_vmh_entries`, `n_rhea_entries` | denominators |
| `n_aligned`, `n_aligned_uncertain`, `n_unmatched_vmh`, `n_unmatched_rhea` | counts |
| `n_by_class` | one count per SMRO term |
| `n_assessable_vmh`, `n_assessable_rhea` | assessable counts |
| `coverage_vmh`, `coverage_rhea` | written as `numerator/denominator` and as a decimal |
| `stoichiometry_consistent` | true only if every aligned pair has equal coefficients and nothing is unmatched |
| Flags | `proton_balance_difference`, `has_stereo_relationship`, `has_protomer`, `has_uncertain`, `has_not_assessable`, `has_unavailable_structure`, `ambiguous_alignment`, `alignment_exactness`, `rhea_preliminary`, `vmh_transport`, `rhea_transport`, `compartment_difference`, `direction_difference`, `direction_note` (R8) |
| `in_strict_baseline` | FR-021 |
| `in_worksheet` | set by the benchmark |
| `relationship_profile` | stratum label (R13) |
| `rank`, `ranking_version` | R11 |

**Invariants**:
- Every participant entry of both reactions appears in ParticipantEvidence for each
  orientation (SC-006).
- `n_aligned + n_aligned_uncertain + n_unmatched_vmh == n_vmh_entries`.
- No field labels the reaction pair as "equivalent" (FR-013).
- A proposal is emitted only when it is plausible (FR-017). A candidate that isn't plausible is
  counted in the funnel only.

### RunRecord (manifest)

Pinned inputs with checksums:
- extract files and the extract manifest (KG snapshot, VMH release, Rhea release, query commit);
- StereoMapper database path, checksum, and `version_tag`;
- worksheet and baseline versions;
- cofactor list version and checksum.

Also recorded:
- configuration (Rhea status rule, alignment, ranking, and retrieval versions);
- `smro_vocabulary`: SHA-256 of the canonical (label, term_id) table from
  `stereomapper.domain.relationship_terms`, plus the checksum of
  `docs/ontology/relationship_terms.csv` when that file is present (Principle II);
- `output_schema_version`: `reconcile-out-v1`;
- StereoMapper code version and git commit, Python version, `run_id`;
- start timestamp (excluded from `run_id`), funnel counts per stage, greedy-alignment count;
- a `trace` section mapping each output file to the input checksums it derives from;
- documented nondeterminism (expected: none).

Inputs are identified by relative path and checksum; absolute paths and the output folder are
not part of `run_id`.

### CurationDecision (append-only log row)

| Field | Notes |
|-------|-------|
| `decision_id` | `DEC-` + hash of (`proposal_id`, `evidence_hash`, `reviewer`, `date`, `decision`) |
| `proposal_id`, `evidence_hash`, `run_id` | the evidence version reviewed |
| `vmh_reaction_id`, `rhea_master_id` | stored redundantly for readability |
| `decision` | `DecisionValue` |
| `reviewer`, `date` (ISO 8601), `reason` | all required, not empty |
| `supersedes` | earlier `decision_id` for the same proposal, or empty |

**State transitions**: `pending → accepted | rejected | deferred`. Any decided state can move to
another decided state only by appending a new row with `supersedes` set. Rows are never edited
or deleted. The effective decision is the latest row in the chain that nothing supersedes. Two
rows that don't supersede each other for the same proposal are a conflict: export fails until
someone resolves it.

### CuratedMapping

| Field | Notes |
|-------|-------|
| `vmh_reaction_id`, `rhea_master_id`, `rhea_matching_directional_id` | the mapping |
| `proposal_id`, `decision_id`, `reviewer`, `date`, `reviewed_evidence_hash`, `run_id` | decision reference |

Also `subsystem_flagged`, `matched_subsystems`, and `flag_list_version` (FR-023).

**Invariant**: produced only from an effective `accepted` decision. Subsystem-flagged mappings are
included with their flag shown (FR-023). By default the reviewed evidence hash must equal the
exported run's `evidence_hash` (R12).

### BenchmarkPair

| Field | Notes |
|-------|-------|
| `worksheet_row`, `worksheet_version` | source |
| `vmh_reaction_id`, `rhea_id_raw`, `rhea_master_id` | IDs |
| `in_scope`, `subsystem_flagged`, `counted`, `sufficient_structure_evidence` | subset membership (`counted` = in scope and not flagged under the applied list) |
| `flag_list_version` | list applied, or `none` |
| `status`, `unrecovered_reason` | `BenchmarkStatus`, `UnrecoveredReason` |
| `proposal_id`, `rank` | when proposed |

### ReviewSample

`sample_id`, `run_id`, seed, size, strata definition, the selected `proposal_id`s with their
strata, and the review deadline (optional).
