# CLI Contract: `stereomapper reconcile`

This adds a new `reconcile` subcommand group. The existing `stereomapper run` command is unchanged.
Every command:
- exits 0 on success;
- exits 2 on invalid arguments or input-contract violations, with a message naming the file,
  row, and field;
- exits 1 on other failures.

No command modifies input files or the StereoMapper results database (FR-004).

## `stereomapper reconcile propose`

```text
stereomapper reconcile propose --config RUN_CONFIG.json --out RUN_DIR [--overwrite]
```

- Reads the run configuration ([input-files.md](input-files.md#run-configuration)). It refuses
  to run (exit 2) if:
  - any input checksum listed in the configuration doesn't match the file.
- Writes the files listed in [output-files.md](output-files.md#run-directory) to `RUN_DIR`. It
  refuses to write into a non-empty `RUN_DIR` unless `--overwrite` is given.
- Prints the `run_id` and funnel counts to stdout.

## `stereomapper reconcile flag-subsystems`

```text
stereomapper reconcile flag-subsystems --run RUN_DIR --flag-list FLAG_LIST.tsv
```

- Applies a versioned subsystem flag list ([input-files.md](input-files.md#subsystem-flag-list))
  to a finished run. Writes `RUN_DIR/flags/<list_version>/` (see
  [output-files.md](output-files.md#flags-directory)). Never modifies the run's other files.
- Refuses (exit 2) if a listed subsystem name doesn't occur in the run's VMH subsystems, so a
  typo can't silently flag nothing. Refuses to overwrite an existing list version.

## Subsystem flag option (shared)

`benchmark`, `export-curated`, `sample`, and `acceptance` require exactly one of:
- `--flags LIST_VERSION`: apply that flag list (R16);
- `--no-subsystem-flags`: apply none, recorded as `flag_list_version: none` in the output.

## `stereomapper reconcile benchmark`

```text
stereomapper reconcile benchmark --run RUN_DIR --worksheet WORKSHEET.tsv \
    --worksheet-version VERSION (--flags LIST_VERSION | --no-subsystem-flags) [--out BENCH_DIR]
```

- Default `BENCH_DIR` is `RUN_DIR/benchmark/<VERSION>/<LIST_VERSION or none>/`.
- Writes `benchmark_pairs.tsv` and `benchmark_report.json`.
- The strict baseline comes from the run configuration, so it stays pinned with the run.

## `stereomapper reconcile review-workbook`

```text
stereomapper reconcile review-workbook --run RUN_DIR --out REVIEW.xlsx \
    [--decision-log LOG.tsv] [--proposals PROPOSAL_IDS.txt] [--flags LIST_VERSION]
```

- Writes a workbook with the sheets `proposals`, `participants`, `decision_template`, and
  `about`. The `about` sheet holds the run ID and versions and states that rank is a review
  order, not a probability.
- If a decision log is given, each proposal's current effective decision is shown, with
  `decision_evidence_current` (true when the decision's `evidence_hash` equals the run's; stale
  rows are highlighted). Otherwise every proposal shows `pending`.

## `stereomapper reconcile record-decisions`

```text
stereomapper reconcile record-decisions --run RUN_DIR --decisions NEW_DECISIONS.tsv \
    --decision-log LOG.tsv
```

- Validates each row: the proposal ID exists in the run, the evidence hash matches the run,
  the decision value is valid, reviewer, date, and reason are not empty, and any `supersedes`
  value refers to an existing decision for the same proposal.
- Appends the valid rows to `LOG.tsv`, creating the file with a header if it doesn't exist.
- The whole batch is rejected (exit 2) if any row is invalid. Existing rows are never
  rewritten.

## `stereomapper reconcile export-curated`

```text
stereomapper reconcile export-curated --run RUN_DIR --decision-log LOG.tsv \
    --out CURATED.tsv (--flags LIST_VERSION | --no-subsystem-flags) [--allow-stale-evidence]
```

- Exports effective `accepted` decisions only. With flags applied, accepted flagged proposals are
  exported with `subsystem_flagged=true` and `matched_subsystems` filled; with
  `--no-subsystem-flags` those columns are empty.
- **Default**: a decision whose `evidence_hash` differs from the run's proposal goes to
  `CURATED.rereview.tsv` and is not exported.
- **`--allow-stale-evidence`**: such decisions are exported with `evidence_matches_run=false`.
- Also writes `CURATED.summary.json` with decision counts by status (FR-029).
- Fails (exit 2) on conflicting decision chains.

## `stereomapper reconcile sample`

```text
stereomapper reconcile sample --run RUN_DIR --size N --seed SEED --out SAMPLE.tsv \
    (--flags LIST_VERSION | --no-subsystem-flags) [--exclude-worksheet BENCH_DIR]
```

- Draws a stratified sample (R13) of proposals that are not in the worksheet and, with flags
  applied, not flagged. The seed, size,
  and strata go in the file header.

## `stereomapper reconcile acceptance`

```text
stereomapper reconcile acceptance --run RUN_DIR --sample SAMPLE.tsv --decision-log LOG.tsv \
    --out REPORT.json (--flags LIST_VERSION | --no-subsystem-flags)
```

- Reports acceptance rate over decided sample members (accepted + rejected), overall and per
  stratum.
- Each figure gives `accepted`, `decided`, `deferred`, `pending`, and `sample_size`, so the
  reviewed denominator is always explicit.
