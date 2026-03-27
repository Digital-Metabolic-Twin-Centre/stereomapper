# StereoMapper SQLite Schema

StereoMapper writes two SQLite tables—`clusters` and `relationships`—that capture canonicalised molecule sets and the stereochemical relationships between them. The schema mirrors the statements in `docs/sqlite_schema.sql`. This document summarizes the purpose of each table and column so downstream tools can read the outputs without reverse-engineering the pipeline.

## Table: clusters

| Column | Type | Description |
| --- | --- | --- |
| `cluster_id` | INTEGER PRIMARY KEY | Auto-incremented identifier for each unique canonical structure cluster. |
| `inchikey_first` | TEXT | First 14 characters of the InChIKey used as the coarse-grained grouping key. |
| `identity_key_strict` | TEXT | Canonical SMILES (strict identity key) that represents the cluster. |
| `is_undef_sru` | BOOLEAN | True if the structure uses an undefined structural repeating unit (SRU). |
| `is_def_sru` | BOOLEAN | True if the structure uses a defined SRU count. |
| `sru_repeat_count` | INTEGER | Repeat count for defined SRUs; NULL when not applicable. |
| `sru_key` | TEXT (generated) | Stored computed key: `def:<count>`, `undef`, or `none`, derived from the SRU flags. |
| `member_count` | INTEGER | Number of accession identifiers aggregated into the cluster. |
| `members_json` | TEXT | JSON array of accession CURIEs that belong to the cluster. |
| `members_hash` | TEXT | SHA-256 hash of the sorted accession list, used for deduplication/integrity checks. |

**Indexes**
- `ux_clusters_ifsmi_disc` enforces uniqueness of `(inchikey_first, identity_key_strict, sru_key)`.
- `idx_ic_inchikey`, `idx_ic_undef_sru`, and `idx_ic_def_sru` accelerate lookups by coarse key and SRU flags.

## Table: relationships

| Column | Type | Description |
| --- | --- | --- |
| `cluster_a` | INTEGER | Foreign-key reference to `clusters.cluster_id` (first structure). |
| `cluster_b` | INTEGER | Foreign-key reference to `clusters.cluster_id` (second structure). |
| `cluster_a_members` | TEXT | JSON string containing member accession IDs for cluster A (subset of `members_json`). |
| `cluster_b_members` | TEXT | JSON string containing member accession IDs for cluster B. |
| `cluster_a_size` | INTEGER | Number of members in cluster A. |
| `cluster_b_size` | INTEGER | Number of members in cluster B. |
| `classification` | TEXT | Assigned stereochemical relationship (e.g., `Enantiomers`, `Diastereomers`, `Protomer`, `Stereo-resolution Pairs`). |
| `score` | REAL | Aggregate confidence or ranking score for the classification. |
| `score_details` | TEXT | JSON payload with per-feature scoring details. |
| `extra_info` | TEXT | JSON payload with provenance/comments (e.g., warnings or conflict notes). |
| `version_tag` | TEXT | Semantic label for the pipeline run (e.g., git SHA, release tag, namespace). |

**Primary key**
- `(cluster_a, cluster_b, version_tag)` to allow multiple pipeline versions to coexist in the same database without overwriting prior results.

**Indexes**
- `idx_cpr_version` on `version_tag` for quick filtering by run.
- `idx_cpr_version_ab` on `(version_tag, cluster_a, cluster_b)` for deduplicated joins.
- `idx_rel_members` on `(cluster_a_members, cluster_b_members)` for membership-based deduplication.

## Usage Notes
- The schema requires SQLite 3.35+ to support the generated column `sru_key`.
- Foreign keys are enabled (`PRAGMA foreign_keys=ON`) in the SQL definition; ensure this pragma remains active when inserting data.
- All JSON columns (`members_json`, `cluster_a_members`, `cluster_b_members`, `score_details`, `extra_info`) store UTF-8 encoded JSON arrays or objects. Consumers should parse them accordingly.
- When shipping derived datasets (e.g., Zenodo archives), include both `docs/sqlite_schema.sql` and this description so downstream workflows can validate schema compatibility automatically.
