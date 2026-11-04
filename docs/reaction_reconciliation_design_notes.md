# VMH-Rhea reaction reconciliation: design notes

**Status:** Background for Spec Kit input
**Scope:** Candidate reaction mappings for manual curation
**Companion diagram:** [Reaction reconciliation workflow](figures/reaction_reconciliation_workflow.pdf)

> **Update 2026-11-04:** the pilot no longer excludes glucose-pathway reactions. All VMH
> subsystems are proposed; a curator-supplied, versioned subsystem flag list (initially
> glucose and other sugar metabolism, and drug metabolism) is applied afterwards, and flagged
> proposals are left out of counted results. The specification in
> [`specs/001-vmh-rhea-reaction-reconciliation/spec.md`](../specs/001-vmh-rhea-reaction-reconciliation/spec.md)
> is authoritative; user documentation is in [`reaction_reconciliation.md`](reaction_reconciliation.md).

## 1. Purpose

Identify VMH-Rhea reaction pairs missed by the current strict mapping workflow by
using StereoMapper's metabolite-level structure relationships. The result is a
traceable list of **proposed reaction mappings**, including pairs supported by
exact, stereo-resolution, enantiomer, diastereomer, protomer, or mixed metabolite
relationships. A proposal is not an assertion of reaction equivalence. Curators
must accept a proposal before it is used in a genome-scale metabolic model (GEM).

The primary validation set is the JC + NM worksheet of manually verified missing
VMH-Rhea reaction mappings. It measures recovery of known omissions. Precision
requires separate curator review of newly proposed pairs.

## 2. Pilot scope

### Included

- Reactions from the latest VMH and Rhea releases available at the start of a run,
  extracted from the project knowledge graph (KG).
- Reaction participants with structures or identifiers that can be related to
  StereoMapper results, plus participants whose structures are unavailable.
- All StereoMapper relationship classes as candidate evidence. No enantiomer,
  diastereomer, or protomer class is excluded from proposals.
- Exact and partial reaction matches. Partial matches must expose unmatched
  participants and coverage rather than silently dropping them.

### Flagged after proposals (not excluded)

- No reaction is excluded by pathway or subsystem. After proposals are generated,
  VMH reactions in a curator-supplied, versioned list of subsystems (initially
  glucose and other sugar metabolism, and drug metabolism) are flagged. Rhea has no
  drug metabolism, and sugar reactions are so similar that they would confound the
  results. Flagged proposals stay visible, are left out of counted results, and are
  exported with their flag if a curator accepts them. Coverage and benchmark results
  are reported with and without the flags.
- Obsolete Rhea reactions are excluded and recorded.

The pilot does not alter StereoMapper's metabolite classifications, existing
strict reaction mappings, or the authoritative VMH/Rhea records.

## 3. Run inputs and provenance

| Input | Minimum information |
| --- | --- |
| VMH reactions | Stable reaction ID; participant IDs; side; stoichiometric coefficient; compartment where present; reaction direction; pathway assignment; current cross-references. |
| Rhea reactions | Rhea reaction ID and quartet/master ID where available; participant IDs; side; coefficient; direction; status; structures or resolvable ChEBI IDs. |
| Structure mapping | VMH and Rhea participant-to-structure identifiers; molfile provenance; StereoMapper result database and run tag. |
| Benchmark | JC + NM verified VMH-Rhea pairs, curator verdict, and worksheet version. |
| Existing mapping baseline | Current strict-workflow VMH-Rhea mappings, to distinguish recovered omissions from already known links. |

At run start, write a manifest containing VMH and Rhea release identifiers, KG
snapshot, extraction query or commit, StereoMapper commit and result `version_tag`,
normalization configuration, benchmark version, timestamp, and checksums of input
files. “Latest” is resolved to explicit versions in this manifest; a running job
must not silently switch releases.

## 4. Data normalization

1. Preserve every source ID and raw participant record. Normalize identifiers
   only in additional fields; never replace the raw source value.
2. Represent each reaction as two **multisets** of participants, with explicit
   stoichiometric coefficients. Do not treat an equation as an unordered bag or
   count one participant twice in an alignment.
3. Normalize Rhea quartet members to a common reaction identity for candidate
   search, while retaining the source Rhea ID and direction for the final proposal.
   Test both side orientations during comparison; record which orientation fits.
4. Preserve VMH compartments. A compartment difference is evidence for curator
   review, not an automatic rejection in this pilot. Keep transport reactions
   identifiable in the output.
5. Do not exclude reactions by pathway. Subsystem flags are applied after the run
   from a versioned list; a reaction is flagged if any of its subsystems is listed.
6. Normalize formula and charge for diagnostics, but do not use formula agreement
   as proof that participants or reactions are equivalent.

## 5. Metabolite relationship layer

Use StereoMapper `cluster_members` to resolve source metabolite IDs to strict
structure clusters, and `relationships`/`relationship_members` to retrieve
inter-cluster classifications. Prefer the normalized membership tables over JSON
member lists. For each VMH-Rhea participant pair, retain:

- source metabolite IDs and structure IDs;
- StereoMapper cluster IDs, relationship label and term ID, score, direction when
  present, and run tag;
- whether the evidence is exact-cluster membership or an inter-cluster
  relationship;
- structure availability and processing status; and
- a provenance pointer to the underlying source records.

Relationship categories include exact structure, stereo-resolution pair,
enantiomer, diastereomer, protomer, identical with undetermined charge,
indistinguishable, unresolved, and unclassified. The last two categories do not
by themselves establish a chemical match; they remain visible as uncertainty
within a proposed pair. Absence of a StereoMapper relationship must be recorded
as “no result” or “structure unavailable,” as appropriate, rather than assumed
to be a negative chemical finding.

## 6. Reaction candidate generation and alignment

### Candidate retrieval

Generate plausible VMH-Rhea pairs using an index of cross-database participant
relationships, existing reaction IDs/cross-references, and coarse reaction
features such as participant count and formula profile. This is a retrieval step,
not a final equivalence test. It must permit candidates whose participant links
contain enantiomer, diastereomer, or protomer relationships. Avoid an exhaustive
Cartesian product of all VMH and Rhea reactions.

The pilot must log candidate counts at each retrieval rule. Any candidate
threshold or cap must be stored in the run configuration and tested against the
JC + NM benchmark; it must not be an undocumented hard filter based on a
StereoMapper relationship class.

### Participant alignment

For each candidate, compute one-to-one participant assignments under both
reaction orientations. Respect sides and stoichiometric multiplicity when
calculating alignment. Compare compartments and direction separately. If more
than one optimal assignment changes the relationship-class summary, mark the
pair as ambiguous and retain the alternatives or their audit references.

For each orientation, report:

- counts of participants aligned by each StereoMapper relationship class;
- matched stoichiometric amount and unmatched amount on each side;
- number of participants with unavailable or failed structures;
- per-side and whole-reaction coverage, with numerator and denominator;
- whether source direction and compartments agree; and
- the full identities of unmatched or conflicting participants.

For reproducibility, count a participant **entry** once per source reaction ID,
side, metabolite ID, and compartment; keep its stoichiometric coefficient as a
separate value. Define `coverage_vmh` as aligned VMH entries divided by all VMH
entries, and `coverage_rhea` analogously. Define `exact_fraction` and one
fraction per StereoMapper relationship class as that class's aligned VMH entries
divided by all VMH entries. Also report `assessable_fraction` for entries with
usable structures. Never use a denominator that silently removes unavailable
structures. Show counts beside every fraction, and compare coefficients
separately so a repeated participant cannot hide a stoichiometry mismatch.

Select the best-supported orientation for ranking, but preserve both results.
Do not collapse a mixed relationship profile into a single “percent matched”
value. A proposed pair may contain a defined stereo conflict, a protomer, or an
unmatched participant; these are review flags, not automatic exclusions.

### Ranking

Rank proposals for curator attention using a reproducible ordering over
coverage, relationship profile, stoichiometry, and data completeness. Do not
interpret this rank as a probability of equivalence. The initial ordering and
any later changes must be versioned; benchmark performance must be reported
against the exact ordering used. Keep all generated proposals in the output,
including low-ranked ones.

## 7. Outputs

The run produces five machine-readable artifacts and a human review view.
Curation adds a decision log and an accepted-mapping export:

| Artifact | Required content |
| --- | --- |
| `run_manifest.json` | Versions, configuration, checksums, counts, and software provenance. |
| `reaction_proposals.tsv` | Unique VMH-Rhea pair, source IDs, Rhea quartet ID, selected orientation, relationship-class counts, per-side coverage, stoichiometry/direction/compartment flags, and rank. Immutable for a pinned run. |
| `participant_evidence.tsv` | One row per aligned or unmatched participant, including source IDs, side, coefficient, StereoMapper evidence, structure status, and ambiguity reference. |
| `scope_exclusions.tsv` | Excluded source reaction ID (obsolete Rhea), status evidence, and exclusion rule. |
| `benchmark_report.json` | In-scope benchmark denominator, recovered pairs, unrecovered pairs with reasons, rank distribution, and coverage by reaction class. |
| `curation_decisions.tsv` | Proposal ID, curator, decision, date, reason, and evidence version; appended or versioned after review. |
| `curated_mappings.tsv` | Accepted VMH-Rhea pairs with curator decision reference, for downstream GEM use. |

The review view must show the complete two reaction equations and the
participant-to-participant evidence alongside each proposal. All proposals start
with `pending` status in the review view. Curator decisions are `accepted`,
`rejected`, or `deferred`; each decision records curator, date, reason, and
evidence version in a separate log. Only `accepted` proposals are exported as
curated mappings eligible for GEM use.

## 8. Evaluation

1. **Benchmark recovery:** Join the JC + NM verified pairs to proposals by
   stable VMH ID and normalized Rhea quartet. Report recall over the full
   worksheet, recall over in-scope pairs, and recall over in-scope pairs with
   sufficient structure evidence. Report every unrecovered pair with a reason.
2. **New-proposal review:** Have curators assess a documented sample of proposals
   absent from the worksheet, stratified by relationship profile and rank.
   Report acceptance rate with the reviewed denominator. The worksheet alone
   cannot estimate precision or false-positive rate because it contains
   verified positives rather than a complete negative set.
3. **Strict baseline:** Compare recovered omissions and new accepted mappings to
   the current strict workflow. Do not count already mapped pairs as novel.
4. **Scope and coverage:** Report VMH/Rhea reaction counts, subsystem flags, obsolete exclusions,
   structure availability, candidate counts, and review outcomes at each stage.

No numerical performance threshold is asserted before the pilot data are
examined. Acceptance of the implementation requires complete accounting of
benchmark pairs and reproducible output, not a chosen percentage alone.

## 9. Acceptance criteria and test cases

- The same pinned inputs and configuration produce the same proposal IDs,
  evidence rows, and ranking.
- Every proposed pair can be traced to source reaction records and the
  StereoMapper run; no participant disappears during normalization or alignment.
- Enantiomer, diastereomer, and protomer evidence can appear in final proposals.
- A reversed Rhea equation can align without creating a duplicate proposal;
  direction differences remain visible.
- Stoichiometric multiplicity is respected, and one molecule cannot satisfy two
  participant slots in a one-to-one assignment.
- Missing/failed structures remain distinguishable from confirmed nonmatches.
- Subsystem-flagged reactions appear in the flag accounting and in full-worksheet
  benchmark reporting, with counted results reported with and without the flags.
- Curator acceptance is required before a mapping appears in the GEM export.
- Regression fixtures include: an exact pair; the example from the handwritten
  notes; a defined stereo conflict; a protomer case; a stoichiometry mismatch;
  a reversed reaction; an unavailable structure; and a subsystem-flagged reaction.

## 10. Decisions to finalize before implementation

1. ~~The KG pathway identifier(s) defining “glucose metabolism.”~~ Superseded: the
   subsystem flag list is curator-supplied and applied after proposals.
2. The candidate retrieval rules and any computational cap, calibrated on a
   development subset without silently losing benchmark positives.
3. The compartment comparison policy for transport reactions and the precise
   handling of Rhea generic participants, macromolecules, and polymers.
4. The initial ranking order and how alternative optimal participant assignments
   are represented in the review view.
5. The location and access controls for curator decisions and GEM exports.

## 11. Repository integration notes

- StereoMapper's normalized `cluster_members` and `relationship_members` tables
  are documented in [SQLite schema](sqlite_schema.md). Relationship labels and
  term IDs are listed in [the ontology table](ontology/relationship_terms.csv).
- The runtime schema includes `relationships.direction`, while
  `docs/sqlite_schema.sql` currently omits that column. Verify the actual result
  database schema before relying on this field, and reconcile the documentation
  and DDL as part of implementation.
- This specification describes a downstream reaction-reconciliation workflow;
  it does not redefine StereoMapper's metabolite classification semantics.
