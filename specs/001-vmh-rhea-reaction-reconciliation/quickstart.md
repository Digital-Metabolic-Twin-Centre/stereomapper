# Quickstart & Validation: VMH–Rhea Reaction Reconciliation

This guide proves the feature works end to end. The command contracts are in
[contracts/cli.md](contracts/cli.md), the file layouts in
[contracts/input-files.md](contracts/input-files.md) and
[contracts/output-files.md](contracts/output-files.md), and the entities in
[data-model.md](data-model.md).

## Prerequisites

- Python ≥3.9 and the package installed in development mode: `pip install -e .[dev]`.
- For a real run: a KG extract directory (`reactions.tsv`, `participants.tsv`, `subsystems.tsv`,
  `extract_manifest.json`), a StereoMapper results database whose `version_tag` covers VMH and
  ChEBI structures, the strict baseline TSV, and the JC + NM worksheet TSV.
- A subsystem flag list TSV (glucose and other sugar metabolism, drug metabolism), with subsystem
  names taken from the pinned VMH release. It is applied after the run, so it can be revised
  without regenerating proposals.

## 1. Fixture validation (no external data)

```bash
pytest -ra src/stereomapper/tests/reconciliation
```

Expected: every regression case passes. Each case is a small fixture extract plus a StereoMapper
database built in the test with `results_schema`.

| Case | Expected outcome | Spec reference |
|------|------------------|----------------|
| exact pair | proposal; all pairs SMRO:0001; coverage `n/n` | US1 |
| stereo-resolution participant | proposal; that pair SMRO:0007 with direction | US1-1 |
| enantiomer / diastereomer / protomer | proposals present; `has_stereo_relationship` / `has_protomer` set | US1-2 |
| 4 aligned + 1 unmatched | all 5 entries listed; coverage `4/5` | US1-3 |
| reversed Rhea equation | one proposal; `selected_orientation=reversed`; directional ID shown | US1-4 |
| coefficient 2 vs 1 | `stoichiometry_consistent=false`; one entry not reused | US1-5 |
| missing structure | `structure_unavailable`; still counted in the denominator | US1-6 |
| formula-only agreement | `formula_agreement` set; not aligned | US1-7 |
| compartment / direction difference | flags set; proposal kept | US1-8 |
| H⁺-only difference | `proton_balance_difference` | US1-9 |
| generic / polymer participant | `not_assessable` by kind; proposal flagged | FR-012 |
| cofactor-only overlap | not proposed; counted in funnel | FR-017 |
| tied alternative alignments | `ambiguous_alignment`; rows in `alternative_alignments.tsv` | FR-016 |
| sugar / drug subsystem reaction | still proposed; listed in `flagged_proposals.tsv`; left out of the counted subset; exported with its flag if accepted; worksheet pair keeps its status with `subsystem_flagged=true` | US3-1, FR-005b |
| multi-subsystem reaction, one listed | flagged (any-match) | FR-005b |
| flag list with an unknown subsystem name | `flag-subsystems` exits 2 | R16 |
| flag list revised | new `flags/<version>/`; proposals and `evidence_hash` unchanged | FR-007 |
| obsolete / preliminary Rhea | obsolete excluded; preliminary flagged | FR-005a |
| baseline pair in worksheet | `already_in_baseline`, not `recovered_new` | US3-2 |
| rejected / deferred / pending | absent from `CURATED.tsv` | US2-1, US2-2 |
| accepted, unchanged evidence | exported with decision reference | US2-3 |
| accepted, evidence changed | written to `CURATED.rereview.tsv` | US2-4 |
| repeat run | byte-identical outputs except the manifest timestamp | US4-2, SC-003 |
| multi-version results database | only the configured `version_tag` is used | R2 |

## 2. End-to-end pilot run

```bash
stereomapper reconcile propose --config pilot_config.json --out runs/pilot-01
stereomapper reconcile flag-subsystems --run runs/pilot-01 --flag-list subsystem_flags_v1.tsv
stereomapper reconcile benchmark --run runs/pilot-01 --worksheet jc_nm.tsv \
    --worksheet-version <version> --flags v1
stereomapper reconcile review-workbook --run runs/pilot-01 --out runs/pilot-01/review.xlsx
```

Check:

1. `run_manifest.json` lists the VMH and Rhea releases, KG snapshot, StereoMapper `version_tag`
   and commit, baseline and cofactor versions, and checksums for every input.
2. In `funnel.tsv`, each stage's count and denominator add up. The excluded count equals the
   number of rows in `scope_exclusions.tsv` (obsolete Rhea only). Proposals exist for reactions
   in every VMH subsystem, including sugar and drug metabolism.
3. In `benchmark_report.json`, for each of the three subsets, `recovered_new +
   already_in_baseline + unrecovered == denominator`. The full-worksheet denominator equals the
   number of positive worksheet rows (SC-001). `rule_effect` gives the counted results without
   flags as well, and flagged pairs by category.
4. Rerun `propose` into `runs/pilot-01b` and diff all TSVs. Expect no differences, and the same
   `run_id` (SC-003).

## 3. Curation loop

```bash
# Curators fill the review workbook's decision_template sheet and export it as TSV
stereomapper reconcile record-decisions --run runs/pilot-01 --decisions batch1.tsv \
    --decision-log curation/decisions.tsv
stereomapper reconcile export-curated --run runs/pilot-01 \
    --decision-log curation/decisions.tsv --out curation/curated_mappings.tsv --flags v1
```

Check: every row in `curated_mappings.tsv` has an `accepted` effective decision in the log
(SC-005); accepted flagged proposals are included with `subsystem_flagged=true` and their
matched subsystems. Rerunning `record-decisions` with an invalid row changes nothing in the log.

## 4. Acceptance-rate sample

```bash
stereomapper reconcile sample --run runs/pilot-01 --size 100 --seed 20261103 \
    --flags v1 --exclude-worksheet runs/pilot-01/benchmark/<version>/v1 \
    --out curation/sample1.tsv
# ...curators review the sampled proposals...
stereomapper reconcile acceptance --sample curation/sample1.tsv \
    --decision-log curation/decisions.tsv --out curation/acceptance1.json --flags v1
```

Check: the report gives accepted, decided, deferred, and pending counts per stratum, and calls
the rate an acceptance rate over the reviewed denominator (SC-008).

## 5. Quality gates (Principle V)

```bash
pytest -ra
ruff check .
black --check .
mypy .
```

Record any check that couldn't run, and why, in the pull request.
