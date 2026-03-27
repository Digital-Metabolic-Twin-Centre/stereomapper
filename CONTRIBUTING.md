# Contributing to StereoMapper

Thanks for helping improve StereoMapper! This project benefits from contributions that enhance the stereochemistry engine, documentation, and FAIR assets such as curated example molfiles. These guidelines explain how to get started and what we expect from contributors.

## Ways to Contribute
- **Bug fixes / features** – extend data models, or stereochemical classification modules under `src/stereomapper/`.
- **Data & examples** – add or improve curated molfiles in `examples/` plus matching manifests.
- **Documentation & FAIR assets** – update the README, `CITATIONS.cff`, `codemeta.json`, schemas, or workflow metadata.
- **Testing & QA** – write new tests under `src/stereomapper/tests` or improve existing coverage.

## Development Environment
1. Fork and clone the repository.
2. Create a virtual environment (Conda, `python -m venv`, or `uv venv`).
3. Install StereoMapper in editable mode with the dev extras:
   ```bash
   pip install -e .[dev]
   # or
   uv pip install -e .[dev]
   ```
4. Pre-commit hooks are configured via `.pre-commit-config.yaml`. Run `pre-commit install` after setting up the environment.

## Coding Standards
- **Formatting / linting**: run `black`, `ruff`, and `isort` (configured for 100-character lines). `pre-commit` will enforce these on staged files.
- **Typing**: the repo enables strict `mypy` (Python 3.11). Keep new modules type-annotated and fix type errors.
- **Testing**: add or update `pytest` tests (`tests/` by default). Ensure `pytest -ra` passes before opening a PR.
- **Performance**: large datasets can stress memory and disk; include benchmarks or notes when changing workflows.

## Submitting Changes
1. Create a feature branch (`feature/<topic>`).
2. Commit changes with clear messages; keep unrelated fixes separate.
3. Run the full QA suite before pushing:
   ```bash
   pytest -ra
   ruff check .
   black --check .
   mypy .
   ```
4. Open a pull request on GitHub describing:
   - What changed and why.
   - Testing performed and environments.
   - Any schema/CLI contract changes.
5. Be responsive to review comments; update your branch as needed.

## Data & Example Contributions
- Place new example molfiles under `examples/<relationship>_files/` with descriptive filenames.
- Update any manifests (e.g., `examples/manifest.csv`) and add provenance details (source database, license, relationship class).
- Provide a minimal command to reproduce relevant SQLite outputs. If outputs should be archived, describe them in the PR so maintainers can version them on Zenodo.

## Documentation Updates
- Sync metadata across `README.md`, `CITATIONS.cff`, and `codemeta.json` when adding new DOIs, releases, or contributors.
- Reference the Zenodo DOI (10.5281/zenodo.19251670) when pointing to pinned examples or outputs.

## Reporting Issues
Use GitHub Issues for:
- Bugs or regressions.
- Feature proposals.
- FAIRness/data concerns (missing schema, provenance gaps).
Please include reproduction steps, expected vs. actual behavior, environment details, and any relevant logs.

## Contact
If you need help beyond GitHub Issues (e.g., security disclosures), email the maintainers listed in `CITATIONS.cff`:
- Jack McGoldrick – `j.mcgoldrick9@universityofgalway.ie`
- Ronan M.T. Fleming – `ronan.mt.fleming@gmail.com`

By contributing, you agree to abide by the project’s [Code of Conduct](CODE_OF_CONDUCT.md).
