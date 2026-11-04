# Input Contract

All tabular inputs are UTF-8, tab-separated, and have a header row. Unknown extra columns are
kept and ignored. A missing required column is an input-contract error (exit 2). Empty optional
values are empty strings.

## KG extract

### `reactions.tsv`

| Column | Required | Notes |
|--------|----------|-------|
| `source` | yes | `vmh` or `rhea` |
| `reaction_id` | yes | raw ID (for example, `HEX1`, `RHEA:17825`) |
| `rhea_master_id` | Rhea | master ID; empty means the row is a master |
| `rhea_directional_ids` | no | `;`-separated LR/RL/BI IDs |
| `rhea_status` | Rhea | `approved`, `preliminary`, or `obsolete` |
| `direction` | VMH | `forward`, `reverse`, `reversible`, or `unknown` |
| `equation` | yes | complete equation text |
| `xref_rhea` | no | `;`-separated Rhea IDs cross-referenced from a VMH reaction |

### `participants.tsv`

| Column | Required | Notes |
|--------|----------|-------|
| `source`, `reaction_id` | yes | join to `reactions.tsv` |
| `side` | yes | `left` or `right` as written |
| `metabolite_id` | yes | raw ID |
| `name` | no | display name |
| `compartment` | no | VMH compartment or Rhea location |
| `coefficient` | yes | positive number; a symbolic value (for example `n`) is accepted only for `generic`, `polymer`, or `macromolecule` participants |
| `kind` | no | `small_molecule` (default), `generic`, `polymer`, `macromolecule` |
| `structure_curie` | no | CURIE as loaded into StereoMapper; empty means no structure |
| `formula`, `charge` | no | diagnostics only |

### `subsystems.tsv`

| Column | Required | Notes |
|--------|----------|-------|
| `reaction_id` | yes | VMH reaction ID; one row per subsystem assignment |
| `subsystem` | yes | VMH subsystem name as in the release (for example, `Glycolysis/gluconeogenesis`) |
| `kg_pathway_id` | no | KG identifier, if the KG has one |

### `extract_manifest.json`

```json
{
  "kg_snapshot": "<id or date>",
  "vmh_release": "<id>",
  "rhea_release": "<id>",
  "extraction_query": {"repository": "<url>", "commit": "<sha>", "path": "<file>"},
  "extracted_at": "<ISO 8601>"
}
```

## StereoMapper results database

An existing StereoMapper SQLite results database ([docs/sqlite_schema.md](../../../docs/sqlite_schema.md)),
opened read-only. The workflow uses these tables:
- `clusters`: `cluster_id`, `status`, `error`;
- `cluster_members`;
- `relationships`, filtered to the configured `version_tag`;
- `relationship_members`, joined on `(cluster_a, cluster_b, version_tag)`.

## Strict baseline: `baseline.tsv`

Columns: `vmh_reaction_id`, `rhea_id`, and optionally `source_note`. Rhea IDs are normalized to
master IDs for joins, and the raw value is kept.

## JC + NM worksheet: `worksheet.tsv`

Columns: `vmh_reaction_id`, `rhea_id`, `curator_verdict`, and optionally `notes`. Only rows with
a positive verdict (`verified`, case-insensitive) are benchmark positives. Other rows are
reported separately and never dropped silently.

## Subsystem flag list

`FLAG_LIST.tsv`: a first comment line `# list_version: <version>`, then columns `subsystem`,
`category`, `rationale` (all required). Subsystem names must match `subsystems.tsv` exactly. It
is not part of the run configuration: it is applied after the run (R16).

## Curator decisions batch: `NEW_DECISIONS.tsv`

Columns: `proposal_id`, `evidence_hash`, `decision`, `reviewer`, `date`, `reason`, and
optionally `supersedes`. The review workbook's `decision_template` sheet has these columns
pre-filled with `proposal_id` and `evidence_hash`.

## Run configuration

```json
{
  "config_version": 1,
  "inputs": {
    "extract_dir": "path/", "stereomapper_db": "path.sqlite", "version_tag": "<tag>",
    "baseline": "baseline.tsv", "baseline_version": "<id>",
    "checksums": {"<relative path>": "sha256:<hex>"}
  },
  "scope": {
    "rhea_status": {"include": ["approved", "preliminary"], "exclude": ["obsolete"]}
  },
  "cofactors": {"list": "builtin:cofactors-v1"},
  "alignment": {"version": "align-v1", "exact_max_side": 16},
  "retrieval": {"version": "retrieve-v1", "routes": ["shared_cluster", "related_cluster", "xref"],
                "cap": null},
  "ranking": {"version": "rank-v1"}
}
```

The `inputs.checksums` map is optional. When it is present, every listed file is checked before
the run. Whether present or not, checksums of all inputs are computed and written to the
manifest.
