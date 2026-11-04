# Implementation Plan: VMH–Rhea Reaction Reconciliation

**Branch**: `001-vmh-rhea-reaction-reconciliation` | **Date**: 2026-11-03 (updated 2026-11-04) | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-vmh-rhea-reaction-reconciliation/spec.md`

## Summary

Add a downstream `reconciliation` subpackage and a `stereomapper reconcile` CLI group. The
workflow:

1. Reads pinned VMH and Rhea KG extracts and a pinned StereoMapper results database, read-only.
2. Applies the recorded Rhea status rule. No pathway or subsystem is excluded, so reactions from
   every VMH subsystem are proposed.
3. Retrieves VMH–Rhea candidate pairs through participant cluster evidence.
4. Aligns participants one-to-one, under both orientations, with exact enumeration.
5. Emits ranked, plausible proposals with full per-participant evidence.

A separate post-processing step applies a curator-supplied, versioned subsystem flag list
(initially glucose and other sugar metabolism, and drug metabolism). Flagged proposals stay
visible but are not counted in benchmark results or acceptance rates. If accepted, they are
exported with their flag shown.
Curators record decisions in an append-only log. Only effective accepted decisions are exported
for GEM use. Benchmark and sampling commands report recovery against the JC + NM worksheet and
the strict baseline, and acceptance rate over reviewed samples. Every result is pinned by a
deterministic run manifest.

## Technical Context

**Language/Version**: Python ≥3.9 (`requires-python` in `pyproject.toml`). New modules use
`from __future__ import annotations` and no 3.10+ syntax or APIs.

**Primary Dependencies**: Standard library (`sqlite3`, `csv`, `json`, `hashlib`, `fractions`,
`argparse`) and `openpyxl` (existing dependency) for the review workbook. No new runtime
dependencies.

**Storage**:
- **Inputs**: TSV/JSON KG extracts; the StereoMapper SQLite results database, opened with
  `mode=ro`.
- **Outputs**: a run directory of TSV/JSON files, an append-only decision log TSV, and an
  `.xlsx` review workbook.

**Testing**: pytest under `src/stereomapper/tests/reconciliation/`, using fixture extracts and
results databases built in tests via `results_schema`.

**Target Platform**: Linux/macOS command-line workstation.

**Project Type**: Library + CLI (single project).

**Performance Goals**: A full VMH × Rhea run finishes in under 30 minutes on a workstation (R15).
Stage counts are recorded so the target can be checked.

**Constraints**:
- Byte-identical outputs for identical pinned inputs.
- No modification of any input.
- No undocumented retrieval cap; v1 has no cap.
- Exact alignment for sides of up to 16 entries. Larger sides use a flagged greedy method and
  are counted.

**Scale/Scope**: About 10⁴ VMH reactions and about 1.6·10⁴ Rhea master reactions. Participant
sides are typically 1–6 entries. The decision log is expected to hold hundreds to low thousands
of rows.

All Technical Context unknowns are resolved in [research.md](research.md): KG access (R1),
participant-to-structure linking (R2), alignment (R4), retrieval (R5), and review surface (R12).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design (below).*

| Gate | Status | Evidence |
|------|--------|----------|
| **I. Chemical meaning** | PASS | Each pair keeps its SMRO label, term ID, and direction (data-model ParticipantEvidence). Separate structure states and alignment states keep failed, unavailable, no-result, uncertain, and nonmatch apart (R2, R3). No "equivalent" label exists, and rank is labelled a review order (R11). Direction, compartment, stoichiometry, and proton flags are kept (R8, R9). StereoMapper classification semantics are unchanged. |
| **II. Traceability & reproducibility** | PASS | Raw IDs are kept beside normalized ones. The manifest pins extract releases, KG snapshot, query commit, database checksum, `version_tag`, SMRO vocabulary version, output schema version, code commit, config, cofactor list, and checksums (R1, R10). `run_id` and IDs are deterministic. The `version_tag` join prevents mixing runs (R2). No new example molfiles are added. |
| **III. Proposal vs. curated status** | PASS | Proposals start `pending`. Only effective `accepted` decisions are exported, with stale evidence routed to re-review (R12). The decision log is append-only with a supersedes chain. No class excludes a candidate (R3). All review fields from FR-019 are present. |
| **IV. Scope & evaluation** | PASS | No pathway is excluded up front. The obsolete-Rhea exclusion is written with rule and evidence. The subsystem flag list is versioned, carries a rationale per subsystem, is applied as an explicit post-processing step, and must be chosen or explicitly declined in every downstream command; reports give results with and without it (R16). The funnel gives counts and denominators at each stage. The benchmark uses three subsets with sum invariants, keeps `already_in_baseline` separate, and states it is not precision. Acceptance rate uses accepted/(accepted+rejected), with deferred and pending shown. No cap in v1. Greedy-alignment and cofactor-only counts are reported (R5, R6, R13). |
| **V. Contracts, tests, docs** | PASS (with planned work) | New CLI group and output contracts are documented here and will go into `docs/reaction_reconciliation.md` and the README in the same change. `stereomapper run` and the SQLite schema are unchanged. Regression and integration tests are listed in quickstart §1. The QA gates are planned. Python 3.9 compatibility is required; the 11 existing modules that fail to import on 3.9 are fixed here (T001) and 3.9 is checked (T057) (R14). |
| **Layout** | PASS | Production code in `src/stereomapper/reconciliation/`, tests in `src/stereomapper/tests/reconciliation/`. Pilot-specific rules (the subsystem flag list) are curator-supplied inputs, not code defaults. |

**Note on "SQLite outputs as primary interfaces"**: The new workflow writes TSV/JSON rather
than a new SQLite database. This adds a downstream interface and doesn't replace the existing
primary interfaces. Flat, checksummable, diffable files serve the reproducibility and curator
review requirements better. This is not a deviation from a MUST rule.

## Project Structure

### Documentation (this feature)

```text
specs/001-vmh-rhea-reaction-reconciliation/
├── spec.md
├── plan.md              # this file
├── research.md          # Phase 0
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/
│   ├── cli.md
│   ├── input-files.md
│   └── output-files.md
├── checklists/requirements.md
└── tasks.md             # Phase 2 (/speckit-tasks)
```

### Source Code (repository root)

```text
src/stereomapper/
├── cli.py                       # register `reconcile` subcommand group (existing `run` unchanged)
├── domain/relationship_terms.py # add `from __future__ import annotations` (Py3.9 import fix)
└── reconciliation/
    ├── __init__.py
    ├── config.py        # run config load/validate, defaults, version constants
    ├── hashing.py       # file checksums, canonical JSON, proposal IDs
    ├── inputs.py        # extract/baseline/worksheet readers + contract validation
    ├── model.py         # dataclasses/enums from data-model.md
    ├── normalize.py     # participant entries, Rhea master/directional normalization
    ├── scope.py         # Rhea status rule → ScopeExclusion
    ├── flags.py         # subsystem flag list: validate, apply to run, flagged sets
    ├── evidence.py      # read-only StereoMapper lookups (version_tag-scoped), structure states
    ├── retrieval.py     # inverted index, routes, funnel counts
    ├── alignment.py     # bitmask exact assignment + tie enumeration, greedy fallback
    ├── proposals.py     # coverage, flags, profile, proposal/evidence hashing
    ├── ranking.py       # rank-v1
    ├── manifest.py      # run record, run_id, checksums, SMRO/output versions
    ├── pipeline.py      # `propose` orchestration
    ├── outputs.py       # deterministic TSV/JSON writers
    ├── curation.py      # decision validation, append-only log, effective decision, export
    ├── benchmark.py     # worksheet join, statuses/reasons, subset reports
    ├── sampling.py      # stratified seeded sample, acceptance report
    ├── review.py        # openpyxl review workbook
    ├── cli.py           # argparse wiring for `reconcile` subcommands
    └── resources/
        ├── cofactors-v1.tsv
        └── default_config.json

src/stereomapper/tests/reconciliation/
├── conftest.py          # fixture builders: extracts + results DB via results_schema
├── fixtures/            # small TSV extracts per regression case
├── test_inputs.py, test_scope.py, test_flags.py, test_evidence.py, test_alignment.py,
├── test_proposals.py, test_ranking.py, test_curation.py, test_benchmark.py,
├── test_sampling.py, test_determinism.py
└── test_cli_integration.py   # propose → benchmark → record → export end to end

docs/reaction_reconciliation.md   # canonical user doc: CLI, inputs, outputs, semantics, versions
pyproject.toml                    # [tool.setuptools.package-data] for reconciliation/resources
README.md                         # short section + link
```

**Structure Decision**: Single-project library + CLI, following the existing
`src/stereomapper/<area>/` layout with mirrored tests. KG extraction queries stay outside this
repository (R1). If an extraction script is needed for the pilot, it goes in
`experiments/reaction_reconciliation/`, and its commit is recorded in the extract manifest.

## Implementation Phasing (for /speckit-tasks)

1. **Foundation**: model, config, input contracts, the Python 3.9 import fix, and package data.
2. **US1 (P1)**: normalize, scope, evidence, retrieval, alignment, proposals, ranking, manifest,
   and outputs; then `reconcile propose` and `review-workbook`.
3. **US2 (P2)**: curation log, `record-decisions`, `export-curated`; `flag-subsystems` and the
   shared `--flags`/`--no-subsystem-flags` option.
4. **US3 (P3)**: benchmark, sampling, acceptance.
5. **US4 (P3)**: determinism tests, manifest completeness, traceability checks.
6. **Polish**: `docs/reaction_reconciliation.md`, README, CONTRIBUTING note on reconciliation
   tests, and the QA gates.

## Risks and open items

- **Subsystem flag list**: the curator extracts the sugar and drug metabolism subsystem names
  from the pinned VMH release. `flag-subsystems` refuses names that don't occur in the run, so a
  typo can't silently flag nothing.
- **Extract quality**: participant `kind` (generic/polymer) and `structure_curie` depend on the
  extraction query. A missing value produces `small_molecule` or `structure_unavailable`, both
  of which are visible in the outputs. The benchmark's `structure_insufficient` reason shows
  how often this happens.
- **Rhea location annotations** may be absent from the extract. If so, `rhea_transport` is
  always false. The manifest records whether location data was present.
- **mypy `python_version = "3.11"`** in `pyproject.toml` doesn't match the 3.9 minimum. This is
  tracked under the constitution follow-ups and is out of scope here.

## Constitution Check (post-design re-evaluation)

All gates remain PASS after Phase 1. The data model and contracts introduced no class-based
filter, no unpinned input, no denominator that removes missing data, and no path from proposal
to curated export that skips an accepted decision.

## Complexity Tracking

No constitution violations. Table intentionally empty.
