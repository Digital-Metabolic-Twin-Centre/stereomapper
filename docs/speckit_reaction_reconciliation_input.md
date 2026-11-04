# Input for `/speckit.specify`: VMH-Rhea reaction reconciliation

> **Update 2026-11-04:** the pilot no longer excludes glucose-pathway reactions. All VMH
> subsystems are proposed; a curator-supplied, versioned subsystem flag list (initially
> glucose and other sugar metabolism, and drug metabolism) is applied afterwards, and flagged
> proposals are left out of counted results. The specification in
> [`specs/001-vmh-rhea-reaction-reconciliation/spec.md`](../specs/001-vmh-rhea-reaction-reconciliation/spec.md)
> is authoritative; user documentation is in [`reaction_reconciliation.md`](reaction_reconciliation.md).
> The original request is kept below for the record; item 2 and the glucose acceptance
> scenario were changed by that clarification.

Create a feature specification for a **downstream reaction reconciliation
workflow** in the StereoMapper project. This text is the feature request to pass
to Spec Kit's specify stage, not an implementation plan. Focus the resulting
specification on user outcomes, testable behavior, scope, and acceptance
scenarios. Leave architecture, algorithms, file formats, and task breakdown for
later Spec Kit stages.

## Problem and intended outcome

The current strict VMH-Rhea reaction mapping workflow misses mappings that
human curators have verified. StereoMapper already relates metabolites from
molecular structures, including exact identity and stereochemical or protomeric
relationships. Use those metabolite relationships to **propose VMH-Rhea reaction
mappings in a non-stereo-specific search**. The proposals are a curator review
queue. A proposal is not automatically an accepted reaction mapping and must
not be used for GEM work until a curator accepts it.

The primary users are reaction curators who need to inspect proposed mappings
and researchers who need to measure whether the workflow recovers known gaps.

## Scope and required behavior

1. Use reactions and participant information extracted from the project
   knowledge graph for the latest available VMH and Rhea releases. Record the
   exact releases, KG snapshot, and StereoMapper result version used by each run
   so the same results can be reproduced.
2. In the initial study, exclude reactions assigned to the glucose metabolic
   pathway because stereoisomerism is especially consequential there. Make the
   scope rule explicit, retain the excluded IDs and reasons, and show how the
   exclusion changes coverage and benchmark results. Do not silently extend
   this exclusion to other pathways.
3. Compare whole reactions using participants on both sides and their
   stoichiometry. Account for reversed reaction orientation and retain original
   reaction direction and compartment information. A difference in direction or
   compartment must be visible to curators.
4. Use StereoMapper's participant-level relationships as evidence. **All
   relationship classes may contribute to a proposed reaction mapping**,
   including exact, stereo-resolution, enantiomer, diastereomer, and protomer
   relationships. Do not exclude a proposed reaction pair solely because an
   aligned metabolite pair belongs to any of these classes. Preserve the exact
   class for each participant pair; do not label the entire reaction simply
   “equivalent” because it has a high match percentage.
5. Include plausible partial matches in the proposal output, with unmatched
   participants shown. Distinguish a missing or unusable structure from a
   supported nonmatch. Formula agreement may be shown as evidence but must not
   alone establish metabolite or reaction identity.
6. Give curators a reviewable proposal for each candidate: VMH and Rhea IDs,
   complete reaction equations, aligned participant pairs, StereoMapper
   relationship types, unmatched participants, stoichiometry, orientation,
   direction and compartment differences, data-availability flags, provenance,
   and counts and denominators behind any match percentages. Rank proposals
   for review without discarding low-ranked candidates from the final proposal
   output.
7. Support curator decisions of accepted, rejected, or deferred, with reviewer,
   reason, and date. Only accepted mappings may enter the downstream curated
   mapping set used for GEM tasks. Preserve the original proposal and evidence
   when decisions are made.
8. Compare the proposals with JC + NM's manually verified worksheet of missing
   VMH-Rhea mappings and with the existing strict-mapping baseline. Report which
   verified omissions are recovered, which are not, and why. Report recovery
   over the full worksheet and the in-scope subset separately. The worksheet is
   a positive reference set; do not present recovery on it as precision. Enable
   separate curator review of new proposals to estimate their acceptance rate.

## Acceptance scenarios to include in the generated spec

- A reaction pair that differs only because one participant is represented with
  unresolved versus resolved stereochemistry appears as a proposal with that
  relationship clearly identified.
- Reaction pairs containing enantiomer, diastereomer, or protomer relationships
  appear in the proposal output and are flagged for curator interpretation.
- A reaction with four aligned participants and one unmatched participant shows
  all five participants, the unmatched identity, and the numerator and
  denominator of each coverage measure.
- Equivalent reaction equations written in opposite orientations are compared
  without creating duplicate proposals; source direction remains visible.
- A stoichiometry difference cannot be hidden by matching the same participant
  more than once.
- A participant without a usable structure is marked as unavailable rather than
  chemically different.
- A glucose-pathway reaction is absent from the in-scope proposal set but appears
  in the exclusion accounting; a verified worksheet pair involving it remains
  visible in full-workbook recovery reporting.
- A curator-rejected or deferred proposal cannot appear in the accepted GEM
  mapping set. An accepted proposal retains its evidence and decision trail.
- A verified worksheet pair already present in the strict baseline is not
  counted as a newly recovered omission.

## Boundaries and assumptions

- StereoMapper remains responsible for metabolite-level relationship assignment;
  this feature proposes reaction-level mappings from those results.
- The JC + NM worksheet contains manually verified positive reaction mappings,
  not an exhaustive set of negatives.
- Source records, existing mappings, and StereoMapper results are read as inputs;
  the proposal workflow does not rewrite them.
- The first release is a pilot for curator review, not an automated GEM mapping
  publisher.

## Clarifications to resolve in the Spec Kit workflow

Resolve these without inventing a silent filter: the precise KG definition of
“glucose metabolism”; treatment of generic Rhea participants, polymers, and
macromolecules; and any candidate retrieval limit or ranking policy needed for
the scale of the data. If a decision cannot be made from project evidence, mark
it for `/speckit.clarify` before planning.

Reference material in this repository: the current workflow diagram at
`docs/figures/reaction_reconciliation_workflow.pdf`, the design notes at
`docs/reaction_reconciliation_design_notes.md`, and StereoMapper's relationship
vocabulary at `docs/ontology/relationship_terms.csv`. The feature description
above is authoritative if those references differ.
