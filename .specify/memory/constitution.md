<!--
Sync Impact Report
==================
Version change: (unversioned template) → 1.0.0
Bump rationale: initial ratification; all placeholders replaced with project principles.

Modified principles (template placeholder → adopted title):
  - [PRINCIPLE_1_NAME] → I. Chemical Meaning and Uncertainty Are Preserved
  - [PRINCIPLE_2_NAME] → II. Evidence Is Traceable and Runs Are Reproducible
  - [PRINCIPLE_3_NAME] → III. Proposals and Curated Assertions Have Separate Status
  - [PRINCIPLE_4_NAME] → IV. Scope and Evaluation Are Explicit
  - [PRINCIPLE_5_NAME] → V. Contracts, Tests, and Documentation Move Together

Added sections:
  - Interfaces and Repository Layout (was [SECTION_2_NAME])
  - Specification and Review Workflow (was [SECTION_3_NAME])
  - Governance (filled)

Removed sections: none

Templates and guidance:
  ✅ .specify/templates/plan-template.md — Constitution Check gates per principle; StereoMapper
     source layout hint
  ✅ .specify/templates/spec-template.md — added mandatory Constitution Alignment section
  ✅ .specify/templates/tasks-template.md — tests mandatory where Principle V requires them;
     path conventions; contract/docs and QA-gate polish tasks
  ✅ .claude/skills/speckit-*/SKILL.md — reviewed; no outdated agent-specific references found
  ⚠ CONTRIBUTING.md — states "strict mypy (Python 3.11)" and tests under `tests/`, while
     pyproject.toml declares requires-python >=3.9 and tests live in src/stereomapper/tests/
  ⚠ pyproject.toml — [tool.mypy] python_version = "3.11" exceeds the supported minimum (3.9);
     resolve in a dedicated change (Principle V)
  ⚠ docs/sqlite_schema.md / docs/sqlite_schema.sql — reconcile with the runtime schema during
     the next implementation change that touches the schema (per ratification input)

Follow-up TODOs: none deferred in this document.
-->

# StereoMapper Constitution

This constitution governs the StereoMapper Python library and CLI, its chemical relationship
data (including the StereoMapper Relationship Ontology, SMRO), and downstream workflows built
from those relationships. Feature-specific algorithms and numerical thresholds belong in feature
specifications and plans, not here.

## Core Principles

### I. Chemical Meaning and Uncertainty Are Preserved

- A change MUST preserve the distinction between exact structure identity, unresolved versus
  resolved stereochemistry, enantiomerism, diastereomerism, protomerism, and every other
  controlled relationship class. Any human-readable relationship label MUST be accompanied by
  the applicable SMRO term ID.
- Confidence scores and ranking values MUST NOT be presented as proof of chemical identity or
  of reaction equivalence.
- An unavailable or failed structure, an unresolved result, and a supported nonmatch MUST
  remain distinguishable in every output that reports them.
- Directional relationships, charge and tautomer state, and structure limitations MUST remain
  visible wherever they affect interpretation.
- A change to classification semantics MUST include documented examples and regression tests.

**Rationale**: StereoMapper's value is precise chemical meaning. Merging distinct relationship
classes or hiding uncertainty produces confident-looking errors that propagate into models.

### II. Evidence Is Traceable and Runs Are Reproducible

- Results MUST retain source accession or reaction IDs and enough provenance to trace each
  conclusion to its input structures and source records. Normalized IDs MAY be added but MUST
  NOT replace raw source IDs.
- A published or shared run MUST pin its input data versions, StereoMapper code version or
  release, configuration, relationship vocabulary version, schema version, and checksums or
  stable identifiers for its inputs.
- Repeating a run with the same pinned inputs and configuration MUST produce reproducible
  classifications and stable downstream identifiers; any nondeterminism MUST be documented.
- Every example molfile MUST have an entry in `examples/manifest.csv` with the fields needed
  to audit provenance, licence, checksum, and expected relationship term.

**Rationale**: Conclusions about chemical relationships are only useful if others can audit
and regenerate them.

### III. Proposals and Curated Assertions Have Separate Status

- Downstream mapping workflows MUST preserve the evidence for each proposal and MUST
  distinguish a candidate mapping from a curator-accepted mapping. A score, percentage, or rank
  alone MUST NOT promote a proposal to an accepted mapping.
- For VMH–Rhea reaction reconciliation, every StereoMapper relationship class, including
  enantiomer, diastereomer, and protomer, MAY support a proposal. Each proposal MUST expose,
  where relevant to review: each participant relationship, unmatched participants,
  stoichiometric differences, orientation, direction, compartment differences, and
  data-availability limitations. A relationship class alone MUST NOT silently exclude a
  candidate.
- Only explicitly accepted mappings MAY enter a curated export intended for genome-scale
  metabolic model use. Each decision MUST record reviewer, date, reason, and the version of the
  evidence reviewed. Rejected and deferred proposals MUST remain auditable.

**Rationale**: Automated evidence supports curation; it does not replace it. Keeping the two
statuses separate prevents unreviewed mappings from entering models.

### IV. Scope and Evaluation Are Explicit

- Every data inclusion or exclusion rule MUST be recorded with its rationale, the affected IDs
  or counts, and its effect on reported denominators. A pilot-specific exclusion (for example,
  glucose-pathway reactions) MUST NOT become an undocumented default for other workflows.
- Every coverage or match percentage MUST state its numerator and denominator. Missing data
  MUST NOT silently disappear from a denominator.
- Benchmark reports MUST distinguish known-positive recovery from precision or acceptance rate,
  MUST compare against the applicable existing baseline, and MUST NOT count already known
  mappings as newly recovered.
- Candidate retrieval limits and ranking rules MUST be documented and versioned. A limit that
  may omit valid cases MUST NOT be adopted until its impact on relevant benchmark examples has
  been measured and reported.

**Rationale**: Unstated scope and shifting denominators make results look better than they are
and make workflows impossible to compare.

### V. Contracts, Tests, and Documentation Move Together

- A change to CLI behavior, output fields, SQLite schema, SMRO vocabulary, or public
  classification meaning MUST update the canonical documentation and compatibility notes in the
  same change. Documented existing outputs MUST be preserved unless a migration or an
  intentional breaking change is specified.
- New chemistry rules, ambiguity or missing-data handling, and downstream mapping decisions
  MUST be covered by focused pytest regression cases.
- Changed joins or contracts between source IDs, structure results, proposal evidence, and
  curated exports MUST be covered by integration tests.
- Tests MUST exercise meaningful edge cases, not only the successful path.
- Before an implementation change is merged, `pytest -ra`, `ruff check .`, `black --check .`,
  and `mypy .` MUST be run where applicable; any check that could not run MUST be recorded with
  the reason.
- Python code MUST remain compatible with the `requires-python` version declared in
  `pyproject.toml`. A higher minimum MUST NOT be inferred from a local development environment.

**Rationale**: Users depend on the CLI, SQLite outputs, and SMRO terms as contracts; drift
between code, tests, and documentation breaks them silently.

## Interfaces and Repository Layout

- The library/CLI and SQLite outputs are the primary interfaces and MUST remain so unless a
  constitution amendment says otherwise. Downstream joins SHOULD use the normalized membership
  tables.
- Exploratory work belongs in `experiments/`. Production behavior MUST live in
  `src/stereomapper/`, with corresponding tests under `src/stereomapper/tests/`.
- Pilot details MUST NOT be generalized into universal chemistry rules. Conflicts between the
  schema documentation and the runtime schema MUST be resolved within the implementation change
  that touches them.

## Specification and Review Workflow

- Every feature specification and implementation plan MUST include a constitution check that
  addresses each core principle.
- A deviation from a MUST rule is permissible only after an explicit constitution amendment has
  been ratified; implementation convenience is not an exception.
- Pull requests MUST state compliance with this constitution, including test and QA results
  and any contract, schema, or vocabulary changes.

## Governance

- This constitution supersedes conflicting project practices and guidance documents.
- Amendments MUST record the reason, the affected principles, the migration or compatibility
  impact, and the changes required in templates and project guidance, and MUST update the Sync
  Impact Report at the top of this file.
- The constitution uses semantic versioning: MAJOR for incompatible governance changes or
  principle removals or redefinitions; MINOR for new principles or materially expanded rules;
  PATCH for clarifications and wording fixes.
- Compliance MUST be reviewed in each feature plan and in each pull request.

**Version**: 1.0.0 | **Ratified**: 2026-11-03 | **Last Amended**: 2026-11-03
