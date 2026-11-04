# Feature Specification: VMH–Rhea Reaction Reconciliation

**Feature Branch**: `001-vmh-rhea-reaction-reconciliation`

**Created**: 2026-11-03

**Status**: Clarified (ready for implementation)

**Input**: User description: "Downstream reaction reconciliation workflow that uses StereoMapper's
metabolite relationships to propose VMH–Rhea reaction mappings, in a non-stereo-specific search,
for curator review; benchmark against the JC + NM verified worksheet and the strict-mapping
baseline." (Full request: `docs/speckit_reaction_reconciliation_input.md`.)

## Overview

The current strict VMH–Rhea reaction mapping workflow misses mappings that human curators have
verified. StereoMapper already relates metabolites by structure, including exact identity and
stereochemical or protomeric relationships. This feature uses those participant-level
relationships to **propose** VMH–Rhea reaction mappings. The proposals are a curator review queue:
a proposal is not an accepted mapping and is not used for genome-scale metabolic model (GEM) work
until a curator accepts it.

Primary users:

- **Reaction curators**, who inspect proposals and record decisions.
- **Researchers**, who measure whether the workflow recovers known gaps compared with the strict
  baseline.

## Clarifications

### Session 2026-11-03

- Q: What counts as "glucose metabolism" for the pilot exclusion? → A: Only the VMH subsystem
  "Glycolysis/gluconeogenesis", applied to VMH reactions with an any-match rule. *(Superseded on
  2026-11-04: no pathway is excluded; see below.)*
- Q: May generic Rhea participants, polymers, and macromolecules count as aligned? → A: No; they
  are never aligned, are reported as "not assessable" by type, stay in denominators, and the
  proposal is still produced and flagged for curator review.
- Q: What minimum evidence makes a partial match plausible enough to propose? → A: At least one
  aligned participant that is not on a versioned cofactor/currency-metabolite list; cofactors
  still count in coverage and are shown.
- Q: How should proton (H+) differences be handled? → A: H+ stays a normal participant in
  coverage; a separate "proton-balance difference" flag is set when H+ is the only stoichiometric
  difference.
- Q: Which Rhea reaction statuses are in scope? → A: Approved and preliminary (preliminary
  flagged on each proposal); obsolete reactions are recorded as scope exclusions.

### Session 2026-11-04

- Q: Should any pathway be excluded before proposals are generated? → A: No. All VMH subsystems
  stay in scope. After proposals are generated, VMH reactions in a curator-supplied list of
  subsystems (initially glucose and other sugar metabolism, and drug metabolism) are flagged and
  not counted in any resulting mapping: they stay visible in the proposal output but are left out
  of counted results. Rationale: Rhea has no drug metabolism, and sugar metabolism reactions are
  so similar that they would confound the results.
- Q: Can a curator-accepted, subsystem-flagged mapping enter the curated GEM mapping set? → A:
  Yes, for now, to be safe: it is exported with its subsystem flag shown, so GEM users can see
  and filter it.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Review evidence-backed reaction proposals (Priority: P1)

A curator runs the workflow on pinned VMH, Rhea, knowledge graph (KG), and StereoMapper inputs and
receives a ranked list of proposed VMH–Rhea reaction pairs. For each proposal they can see both
complete reaction equations, every participant on both sides, how each participant was aligned
and with which StereoMapper relationship class, what is unmatched, and every difference in
stoichiometry, orientation, direction, and compartment.

**Why this priority**: Without reviewable proposals there is nothing to curate or benchmark. This
alone delivers value: curators gain a queue of candidate mappings the strict workflow misses.

**Independent Test**: Run the workflow on a small fixture set of VMH and Rhea reactions with known
participant relationships and confirm that the expected proposals appear with complete,
correct evidence.

**Acceptance Scenarios**:

1. **Given** a VMH and a Rhea reaction that differ only because one participant is represented
   with unresolved stereochemistry in one source and resolved stereochemistry in the other,
   **When** the workflow runs, **Then** the pair appears as a proposal and that participant pair
   is labelled as a stereo-resolution relationship with its SMRO term ID.
2. **Given** reaction pairs whose aligned participants include enantiomer, diastereomer, or
   protomer relationships, **When** the workflow runs, **Then** those pairs appear in the
   proposal output, each such participant pair shows its exact relationship class, and the
   proposal is flagged for curator interpretation.
3. **Given** a candidate pair where four participants align and one does not, **When** the curator
   views the proposal, **Then** all five participants are shown, the unmatched participant is
   identified by source ID and name, and each coverage measure shows its numerator and
   denominator.
4. **Given** a VMH reaction and a Rhea reaction describing the same transformation with the
   sides written in opposite orientations, **When** the workflow runs, **Then** exactly one
   proposal exists for the pair, the orientation used for alignment is shown, and each source's
   original direction remains visible.
5. **Given** a candidate pair in which one reaction has a participant with coefficient 2 and the
   other has the matching participant with coefficient 1, **When** the workflow aligns them,
   **Then** the stoichiometry difference is reported and the single participant is not counted
   as satisfying more than one participant slot.
6. **Given** a participant with no usable structure, **When** the workflow aligns the reactions,
   **Then** that participant is marked as structure-unavailable, not as chemically different, and
   it remains in every coverage denominator.
7. **Given** two reactions that agree in molecular formula for a participant but have no
   StereoMapper relationship for it, **When** the curator views the proposal, **Then** formula
   agreement may be shown as supporting evidence but the participant is not reported as matched
   on that basis alone.
8. **Given** a VMH reaction and a Rhea reaction that differ in compartment or reaction direction,
   **When** the curator views the proposal, **Then** the difference is shown explicitly and does
   not by itself remove the proposal.
9. **Given** a candidate pair whose participants align through protomer relationships and differ
   only in H+ stoichiometry, **When** the curator views the proposal, **Then** the H+ difference is
   shown, H+ is counted in coverage, and the proposal carries a proton-balance difference flag.

---

### User Story 2 - Record curator decisions and export accepted mappings (Priority: P2)

A curator marks each proposal as accepted, rejected, or deferred, giving a reason. Only accepted
proposals form the curated mapping set used for GEM tasks. Each decision keeps its link to the
proposal and evidence that the curator reviewed.

**Why this priority**: Proposals become useful for GEM work only through an auditable decision.
This depends on User Story 1.

**Independent Test**: Record a mix of accepted, rejected, and deferred decisions on a fixed
proposal set and confirm that only accepted pairs appear in the curated set, each with a complete
decision trail and unchanged evidence.

**Acceptance Scenarios**:

1. **Given** a proposal with no decision, **When** the curated mapping set is produced, **Then**
   the proposal is absent and is shown as pending in the review view.
2. **Given** a proposal a curator has rejected or deferred, **When** the curated mapping set is
   produced, **Then** the proposal is absent, and its decision remains retrievable with
   reviewer, date, and reason.
3. **Given** a proposal a curator has accepted, **When** the curated mapping set is produced,
   **Then** it appears with a reference to its decision (reviewer, date, reason, and evidence
   version) and its original proposal evidence is unchanged.
4. **Given** a decision recorded against one run's evidence, **When** a later run changes that
   proposal's evidence, **Then** the earlier decision remains linked to the evidence version it
   reviewed and the change is visible to the curator.

---

### User Story 3 - Benchmark recovery against verified omissions and the strict baseline (Priority: P3)

A researcher compares the proposals with the JC + NM worksheet of manually verified missing
VMH–Rhea mappings and with the existing strict-mapping baseline. They see which verified
omissions are recovered, which are not and why, and how the scope rule and subsystem
flags change the result.

**Why this priority**: This measures whether the workflow closes known gaps and guides further
development. It depends on User Story 1 but not on User Story 2.

**Independent Test**: Run the benchmark on a fixture worksheet with recovered, unrecovered,
already-mapped, and out-of-scope pairs and confirm that every pair is accounted for with the
correct status and that full-worksheet and counted-subset results are reported separately.

**Acceptance Scenarios**:

1. **Given** a VMH reaction in a flagged subsystem (for example, glucose metabolism or drug
   metabolism), **When** the workflow runs, **Then** its proposals appear in the proposal output
   marked with the flagged subsystem, they are left out of counted results, the reaction appears
   in the subsystem-flag accounting, any verified worksheet pair involving it is still listed in
   full-worksheet recovery reporting, labelled as flagged, and if a curator accepts one of its
   proposals, the mapping enters the curated mapping set with its subsystem flag shown.
2. **Given** a verified worksheet pair that is already present in the strict baseline, **When**
   the benchmark is reported, **Then** it is counted as already mapped and not as a newly
   recovered omission.
3. **Given** a verified worksheet pair that the workflow does not propose, **When** the benchmark
   is reported, **Then** the pair is listed as unrecovered with a stated reason (for example,
   out of scope, structure unavailable, no participant relationship, or not retrieved as a
   candidate).
4. **Given** a completed benchmark, **When** the researcher reads the report, **Then** recovery is
   presented as recovery of known positives with numerator and denominator, separately for the
   full worksheet and the counted subset, and is not described as precision.
5. **Given** proposals not present in the worksheet, **When** a researcher sets up a review of
   them, **Then** the workflow can provide a documented sample of those proposals for curator
   review and report the acceptance rate over the reviewed denominator.

---

### User Story 4 - Reproduce and audit a run (Priority: P3)

A researcher or reviewer reruns a published pilot run from its recorded inputs and configuration
and obtains the same proposals, evidence, and ranking, and can trace each proposal back to its
source reaction records and StereoMapper results.

**Why this priority**: Results that cannot be reproduced cannot support curation or publication.
It is required for every shared run but adds no new curator-facing capability.

**Independent Test**: Run the workflow twice with the same pinned inputs and configuration and
confirm identical proposal identifiers, evidence, and ranking.

**Acceptance Scenarios**:

1. **Given** a completed run, **When** a reviewer inspects its record, **Then** it states the
   exact VMH and Rhea releases, KG snapshot, StereoMapper result version, worksheet version,
   strict-baseline version, configuration (including the scope rule, subsystem flag list, any
   retrieval limits, and ranking version), and input checksums or stable identifiers.
2. **Given** the same pinned inputs and configuration, **When** the workflow is run again,
   **Then** proposal identifiers, participant evidence, and ranking are identical.
3. **Given** any proposal, **When** a reviewer traces it, **Then** they can reach the raw VMH and
   Rhea source records and the StereoMapper results behind every participant pair.

---

### Edge Cases

- **Several equally good participant alignments** that imply different relationship profiles:
  the proposal is marked ambiguous and the alternatives remain inspectable.
- **Participants that only StereoMapper marks as unresolved, unclassified, or indistinguishable**:
  shown as uncertainty within the proposal, never as a confirmed match or a confirmed nonmatch.
- **Participant with a structure that failed processing**: distinguished from both "structure
  unavailable" and "supported nonmatch".
- **No StereoMapper result for a participant pair**: recorded as "no result", not as a chemical
  difference.
- **Rhea reactions that belong to the same directional group** (master and directional
  variants): they produce one proposal per VMH reaction and reconciled Rhea reaction, while each
  source Rhea ID and direction is retained.
- **Transport reactions** (same metabolite, different compartments): remain identifiable and
  show the compartment difference rather than being silently treated as identical reactions.
- **VMH reaction assigned to several subsystems, one of which is on the flag list**: flagged
  under FR-005b and listed in the subsystem-flag accounting with all of its subsystem
  assignments.
- **Rhea reaction with no pathway assignment**: never flagged by subsystem itself; a proposal is
  flagged only through its VMH reaction.
- **Generic Rhea participants, polymers, and macromolecules**: never aligned; shown as not
  assessable with their type and kept in denominators (FR-012).
- **Worksheet pairs whose IDs are absent from the pinned releases or are obsolete in Rhea**:
  reported as unrecovered with that reason (absent or obsolete), not dropped from the denominator.
- **Candidate pair aligned only through cofactors** (for example, sharing only water and ATP): not
  proposed under FR-017, but counted at the retrieval step where it drops out, and reported as the
  reason if it is a worksheet pair.
- **Repeated participant within one reaction side or across compartments**: each participant
  entry is counted once and its coefficient compared separately.

## Requirements *(mandatory)*

### Functional Requirements

**Inputs, provenance, and reproducibility**

- **FR-001**: The workflow MUST use reactions and participant information extracted from the
  project KG for the latest VMH and Rhea releases available when a run starts, and MUST record the
  exact releases resolved as "latest" so a run never silently switches releases. "Latest" is
  decided when the extract is made and recorded in it; the workflow records it but cannot verify
  it.
- **FR-002**: Each run MUST record the VMH and Rhea releases, KG snapshot, StereoMapper result
  version, SMRO relationship vocabulary version, output schema version, JC + NM worksheet version, strict-baseline version, workflow version, full
  configuration (including scope rules, the subsystem flag list version, any retrieval limits,
  and the ranking version), and checksums or stable identifiers for every input.
- **FR-003**: Repeating a run with the same pinned inputs and configuration MUST produce
  identical proposal identifiers, participant evidence, and ranking; any unavoidable
  nondeterminism MUST be documented in the run record.
- **FR-004**: The workflow MUST read source reactions, existing mappings, and StereoMapper results
  without modifying them, and MUST retain raw source IDs alongside any normalized IDs.

**Scope**

- **FR-005**: The workflow MUST NOT exclude any VMH or Rhea reaction by pathway or subsystem
  before generating proposals. Reactions from every VMH subsystem MUST be retrieved, aligned, and
  proposed under the same rules.
- **FR-005a**: Approved and preliminary Rhea reactions MUST be in scope; every proposal involving a
  preliminary Rhea reaction MUST carry a "Rhea preliminary" flag. Obsolete Rhea reactions MUST be
  excluded and recorded as scope exclusions with their IDs, status, and counts.
- **FR-005b**: After proposals are generated, the workflow MUST flag every VMH reaction
  assigned to a subsystem on a curator-supplied subsystem flag list, and every proposal involving
  such a reaction. The list MUST be versioned and recorded in the run with each subsystem's name
  and the rationale (initially glucose and other sugar metabolism, and drug metabolism: Rhea has
  no drug metabolism, and sugar reactions are highly similar). A VMH reaction assigned to several
  subsystems MUST be flagged if any of them is on the list. Flagged proposals MUST stay in the
  proposal output, showing the flagged subsystem, and MUST NOT be counted in any resulting
  mapping: they are left out of the counted results (coverage, benchmark recovery, acceptance
  rate). Curators MAY still record decisions on them, and accepted ones are exported per FR-023.
- **FR-006**: Every excluded reaction (FR-005a) and every subsystem-flagged reaction (FR-005b)
  MUST be retained in an accounting record with its ID, the evidence (status or all subsystem
  assignments), and the rule that applies. Coverage and benchmark results MUST be reported both
  with and without each rule applied.
- **FR-007**: The subsystem flag list MUST apply only where the run states it. Changing the list
  MUST produce a new list version and MUST NOT require regenerating proposals. The list MUST NOT
  become a default for other workflows.

**Reaction comparison**

- **FR-008**: The workflow MUST compare whole reactions using participants on both sides with
  their stoichiometric coefficients, and MUST assign each participant entry at most once per
  alignment so that a stoichiometry difference cannot be hidden by matching one participant
  twice.
- **FR-008a**: Protons (H+) MUST be treated as ordinary participants in alignment and coverage.
  When H+ is the only participant whose stoichiometry differs between the aligned reactions, the
  proposal MUST carry a "proton-balance difference" flag in addition to reporting the
  coefficient difference.
- **FR-009**: The workflow MUST compare both orientations of each candidate pair, MUST report the
  orientation used, and MUST NOT create duplicate proposals for the same VMH–Rhea pair because of
  orientation or Rhea directional variants.
- **FR-010**: The workflow MUST retain each source's original reaction direction and compartment
  information and MUST show any direction or compartment difference to curators. Neither
  difference alone removes a proposal. What counts as a direction difference is defined in the
  plan (research R8).
- **FR-011**: The workflow MUST use StereoMapper participant-level relationships as evidence. Every
  relationship class (including exact, identical with undetermined charge, stereo-resolution,
  enantiomer, diastereomer, and protomer) MAY support a proposal, and no candidate pair MAY be
  excluded solely because an aligned participant pair belongs to any one class. Unresolved,
  unclassified, and indistinguishable results MUST be shown as uncertainty, not as matches.
- **FR-012**: Generic Rhea participants (chemical classes), polymers, and macromolecules MUST NOT
  be counted as aligned. Each MUST be shown in the proposal with the state "not assessable" and
  its type (generic, polymer, or macromolecule), MUST remain in coverage denominators, and MUST NOT
  by itself prevent a proposal; proposals containing such participants MUST be flagged for curator
  review.
- **FR-013**: (See also FR-011.) The workflow MUST preserve the exact relationship class and SMRO
  term ID for each
  aligned participant pair, and MUST NOT label a whole reaction pair "equivalent" based on a
  match percentage or score.
- **FR-014**: The workflow MUST distinguish, for every participant, among: aligned (with class),
  uncertain (FR-011), supported nonmatch, structure unavailable, structure processing failed, no
  StereoMapper result, and not assessable (FR-012). Unavailable or failed structures MUST NOT be reported as
  chemically different and MUST NOT be removed from coverage denominators.
- **FR-015**: Molecular formula agreement MAY be shown as supporting evidence, but MUST NOT on its
  own establish that participants or reactions are the same.
- **FR-016**: When more than one equally supported participant alignment yields a different
  relationship profile, the proposal MUST be marked ambiguous and the alternatives MUST remain
  available for review.

**Candidates, proposals, and ranking**

- **FR-017**: The proposal output MUST include partial matches, with unmatched participants
  shown, not only complete matches. A candidate pair is plausible, and MUST be proposed, when at
  least one aligned participant pair is not on the cofactor/currency-metabolite list (for example,
  water, protons, ATP, NAD(P)H, CO2). The list MUST be documented and versioned in the run
  configuration, and its effect on JC + NM worksheet pairs MUST be measured and reported per
  FR-018. Cofactor participants MUST still be shown and counted in coverage.
- **FR-018**: Any candidate retrieval limit or cap MUST be recorded in the run configuration,
  MUST NOT be based on a StereoMapper relationship class, and MUST NOT be adopted until its
  effect on JC + NM worksheet pairs has been measured and reported. Candidate counts MUST be
  reported at each retrieval step.
- **FR-019**: Each proposal MUST present: VMH and Rhea IDs (including the reconciled Rhea group
  identity and source Rhea ID); complete reaction equations for both sources; aligned participant
  pairs with relationship class and SMRO term ID; unmatched participants; stoichiometric
  coefficients and differences; orientation; direction and compartment differences;
  data-availability flags; provenance links; and the numerator and denominator of every
  coverage measure.
- **FR-020**: Proposals MUST be ordered for review by a documented, versioned, reproducible
  ranking. The rank MUST NOT be presented as a probability of equivalence, and low-ranked
  proposals MUST remain in the final proposal output.
- **FR-021**: Each proposal MUST indicate whether the VMH–Rhea pair is already in the strict
  baseline.

**Curation**

- **FR-022**: Each proposal MUST start as pending. Curators MUST be able to record accepted,
  rejected, or deferred decisions, each with reviewer, date, reason, and the evidence version
  reviewed (in the plan, the proposal's `evidence_hash`).
- **FR-023**: Only proposals with an accepted decision MAY enter the curated mapping set for GEM
  use. An accepted, subsystem-flagged proposal (FR-005b) MUST be exported with its flag and
  matched subsystems shown, so GEM users can filter it. Score, rank, or coverage MUST NOT promote a
  proposal to accepted.
- **FR-024**: Recording a decision MUST NOT alter the original proposal or its evidence. Rejected
  and deferred decisions MUST remain auditable, and later decisions on the same proposal MUST be
  retained as history rather than overwriting earlier ones.

**Benchmarking and reporting**

- **FR-025**: The workflow MUST compare proposals against the JC + NM worksheet and report, for
  every worksheet pair, whether it is recovered, already in the strict baseline, or unrecovered,
  with a reason for each unrecovered pair.
- **FR-026**: Recovery MUST be reported separately over the full worksheet, the counted subset
  (in scope and not subsystem-flagged), and the counted subset with sufficient structure
  evidence, each with numerator and denominator. Worksheet pairs already in the strict baseline
  MUST NOT be counted as newly recovered.
- **FR-027**: Reports MUST describe worksheet results as recovery of known positives and MUST
  NOT present them as precision or false-positive rate.
- **FR-028**: The workflow MUST support selecting a documented sample of proposals not in the
  worksheet for separate curator review, and MUST report acceptance rate over the reviewed
  denominator, stratified by relationship profile and rank.
- **FR-029**: Run reports MUST give counts at each stage (reactions read, excluded, with usable
  structures, candidates retrieved, proposals, decisions by status), each with its denominator.

### Key Entities

- **Source reaction**: A VMH or Rhea reaction as recorded in its pinned release: source ID, Rhea
  status (approved, preliminary, obsolete), participants by side with coefficients, direction,
  compartments (VMH), pathway assignments, Rhea directional group, and existing
  cross-references.
- **Participant entry**: One metabolite on one side of one source reaction in one compartment,
  with its coefficient, source metabolite ID, structure availability and processing status.
- **Participant relationship**: StereoMapper's evidence relating a VMH participant to a Rhea
  participant: relationship class, SMRO term ID, direction where present, score, and the
  StereoMapper result version.
- **Scope exclusion**: A Rhea reaction removed from scope because it is obsolete (FR-005a), with
  its status and reason.
- **Subsystem flag**: A VMH reaction (and its proposals) marked as not counted because one of its
  subsystems is on the versioned flag list (FR-005b), with all subsystem assignments, the list
  version, and the rationale.
- **Reaction proposal**: A candidate VMH–Rhea pair for one run: orientation, participant
  alignment (and alternatives if ambiguous), unmatched participants, stoichiometry, direction
  and compartment comparisons, coverage counts, data-availability flags, rank, strict-baseline
  status, and provenance. Immutable for a pinned run.
- **Curation decision**: A curator's accepted, rejected, or deferred verdict on a proposal, with
  reviewer, date, reason, and evidence version; decisions form an append-only history.
- **Curated mapping**: An accepted VMH–Rhea pair eligible for GEM use, linked to its decision and
  proposal evidence.
- **Benchmark pair**: A verified VMH–Rhea pair from the JC + NM worksheet, with worksheet
  version, and its recovery status and reason in a run.
- **Run record**: The pinned inputs, versions, configuration, checksums, and stage counts for a
  run.

## Constitution Alignment *(mandatory)*

- **I. Chemical meaning**: Each participant pair keeps its relationship class and SMRO term ID
  (FR-011, FR-013). Failed, unavailable, no-result, unresolved, and supported-nonmatch states stay
  distinct (FR-014). Scores and ranks are not presented as proof (FR-013, FR-020). Direction,
  compartment, and stoichiometry stay visible (FR-008–FR-010). StereoMapper classification
  semantics are not changed.
- **II. Traceability & reproducibility**: Raw source IDs retained (FR-004); every run pins
  releases, KG snapshot, StereoMapper version, configuration, and checksums (FR-001–FR-003);
  proposals trace to source records (User Story 4).
- **III. Proposal vs. curated status**: Proposals start pending and need explicit acceptance
  (FR-022, FR-023); evidence is preserved and decisions are auditable (FR-024); no relationship
  class silently excludes a candidate (FR-011).
- **IV. Scope & evaluation**: No pathway is excluded up front (FR-005). The Rhea-status rule and
  the subsystem flag list are explicit, versioned, and recorded, with results reported with and
  without each (FR-005a–FR-007); the cofactor list is versioned (FR-017); all
  percentages carry numerators and denominators (FR-019, FR-026, FR-029);
  recovery is separated from acceptance rate and from the strict baseline (FR-025–FR-028);
  retrieval limits require measured benchmark impact (FR-018).
- **V. Contracts, tests, docs**: This feature adds new downstream outputs and does not change
  existing CLI behavior, SQLite schema, or SMRO vocabulary. If planning finds it must (for example,
  the documented schema omits a runtime `direction` column used as evidence), that change and its
  documentation follow Principle V. The regression cases listed in the acceptance scenarios are
  required tests.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of JC + NM worksheet pairs are accounted for in every benchmark report as
  recovered, already in the strict baseline, or unrecovered with a stated reason; none are
  silently dropped.
- **SC-002**: The pilot recovers at least one verified worksheet omission that the strict
  baseline misses, and reports recovery for the full worksheet and counted subset separately
  with numerators and denominators. (No minimum recovery percentage is set before the pilot data
  are examined.)
- **SC-003**: Two runs with identical pinned inputs and configuration produce identical
  proposals, participant evidence, and ranking in 100% of cases.
- **SC-004**: 100% of proposals can be traced by a reviewer to their raw VMH and Rhea source
  records and to the StereoMapper results behind each aligned participant pair.
- **SC-005**: 0 proposals without an accepted decision appear in the curated mapping set.
- **SC-006**: 100% of participant entries in every proposal are shown with one of the defined
  states (aligned with class, supported nonmatch, structure unavailable, processing failed, no
  result, uncertain, or not assessable with its type), and every coverage figure shows its
  numerator and denominator.
- **SC-007**: A curator can determine from a single proposal view, without consulting other
  sources, which participants differ and why (relationship class, unmatched, stoichiometry,
  direction, compartment, or missing data), confirmed by a curator walkthrough of a sample of
  proposals.
- **SC-008**: An acceptance rate for proposals outside the worksheet is reported over a
  documented reviewed sample, stratified by relationship profile and rank.
- **SC-009**: Every regression case named in the acceptance scenarios (stereo-resolution,
  enantiomer/diastereomer/protomer, partial match, reversed orientation, stoichiometry mismatch,
  unavailable structure, subsystem flag, rejected/deferred exclusion from the curated set,
  already-mapped worksheet pair) passes.

## Assumptions

- StereoMapper remains responsible for metabolite-level relationship assignment; this feature
  proposes reaction-level mappings from its results and does not change its classifications.
- The JC + NM worksheet contains manually verified positive reaction mappings, not an exhaustive
  set of negatives.
- A strict-baseline mapping set exists and can be versioned for comparison.
- Participant-to-structure links for VMH and Rhea metabolites are available through the KG and
  StereoMapper results; participants without them are kept and flagged.
- The first release is a pilot for curator review, not an automated GEM mapping publisher.
- The exact VMH subsystem names on the flag list are taken from the pinned VMH release during
  post-processing and recorded in the list version; the workflow does not infer them.
- Rhea directional variants (master and directional reactions) are reconciled to one reaction
  identity for comparison while each source ID and direction is retained.
- A compartment difference, including for transport reactions, is review evidence rather than an
  automatic rejection in this pilot.
- The specific retrieval rules, ranking order, and alignment method are decided in planning,
  subject to FR-018 and FR-020.
- Storage location and access control for curator decisions and curated exports are decided in
  planning; the pilot assumes a small trusted curator group.
