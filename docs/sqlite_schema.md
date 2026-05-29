# StereoMapper SQLite Schema

StereoMapper stores canonical clusters and pairwise stereochemical relationships in SQLite. The schema keeps the legacy JSON columns for compatibility, but the normalized membership tables are the preferred access path for filtering, joins, and downstream exports. The statements here mirror `docs/sqlite_schema.sql`.

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
| `members_json` | TEXT | Legacy JSON array of accession CURIEs retained for compatibility. |
| `members_hash` | TEXT | SHA-256 hash of the sorted accession list, used for deduplication and integrity checks. |

**Indexes**
- `ux_clusters_ifsmi_disc` enforces uniqueness of `(inchikey_first, identity_key_strict, sru_key)`.
- `idx_ic_inchikey`, `idx_ic_undef_sru`, and `idx_ic_def_sru` accelerate lookups by coarse key and SRU flags.

## Table: cluster_members

| Column | Type | Description |
| --- | --- | --- |
| `cluster_id` | INTEGER | Foreign-key reference to `clusters.cluster_id`. |
| `member_curie` | TEXT | Normalized accession CURIE for a single cluster member. |

**Primary key**
- `(cluster_id, member_curie)` ensures each member appears once per cluster.

**Indexes**
- `idx_cluster_members_member` supports lookups by accession CURIE.
- `idx_cluster_members_cluster` supports joins back to `clusters`.

## Table: relationships

| Column | Type | Description |
| --- | --- | --- |
| `cluster_a` | INTEGER | Foreign-key reference to `clusters.cluster_id` (first structure). |
| `cluster_b` | INTEGER | Foreign-key reference to `clusters.cluster_id` (second structure). |
| `cluster_a_members` | TEXT | Legacy JSON snapshot of cluster A members retained for compatibility. |
| `cluster_b_members` | TEXT | Legacy JSON snapshot of cluster B members retained for compatibility. |
| `cluster_a_size` | INTEGER | Number of members in cluster A. |
| `cluster_b_size` | INTEGER | Number of members in cluster B. |
| `classification` | TEXT | Assigned stereochemical relationship (e.g., `Enantiomers`, `Diastereomers`, `Protomers`, `Stereo-resolution pairs`). |
| `classification_term_id` | TEXT | SMRO identifier that maps each `classification` to a controlled vocabulary entry defined in `docs/ontology/relationship_terms.csv`. |
| `direction` | TEXT | Directionality for `Stereo-resolution pairs`: `A_to_B` means `cluster_a` is more stereochemically resolved than `cluster_b`, `B_to_A` means the reverse; NULL for symmetric relationships. |
| `score` | REAL | Aggregate confidence or ranking score for the classification. |
| `score_details` | TEXT | JSON payload with per-feature scoring details. |
| `extra_info` | TEXT | JSON payload with provenance/comments (e.g., warnings or conflict notes). |
| `version_tag` | TEXT | Semantic label for the pipeline run (e.g., git SHA, release tag, namespace). |

**Primary key**
- `(cluster_a, cluster_b, version_tag)` allows multiple pipeline versions to coexist in the same database without overwriting prior results.

**Indexes**
- `idx_cpr_version` on `version_tag` for quick filtering by run.
- `idx_cpr_version_ab` on `(version_tag, cluster_a, cluster_b)` for deduplicated joins.
- `idx_rel_members` on `(cluster_a_members, cluster_b_members)` for legacy membership-based deduplication.

## Table: relationship_members

| Column | Type | Description |
| --- | --- | --- |
| `cluster_a` | INTEGER | Foreign-key reference to `relationships.cluster_a`. |
| `cluster_b` | INTEGER | Foreign-key reference to `relationships.cluster_b`. |
| `version_tag` | TEXT | Foreign-key reference to `relationships.version_tag`. |
| `side` | TEXT | Either `A` or `B`, indicating which cluster the member belongs to. |
| `member_curie` | TEXT | Normalized accession CURIE for one side of the relationship. |

**Primary key**
- `(cluster_a, cluster_b, version_tag, side, member_curie)` ensures each relationship-side member appears once.

**Indexes**
- `idx_rel_members_member` supports lookups by accession CURIE.
- `idx_rel_members_pair` supports joins by relationship pair and version.

## Usage Notes
- The schema requires SQLite 3.35+ to support the generated column `sru_key`.
- Foreign keys are enabled (`PRAGMA foreign_keys=ON`) in the SQL definition; ensure this pragma remains active when inserting data.
- The schema initializer backfills `cluster_members` and `relationship_members` from legacy JSON columns when opening older databases.
- JSON columns (`members_json`, `cluster_a_members`, `cluster_b_members`, `score_details`, `extra_info`) remain UTF-8 encoded JSON arrays or objects for compatibility.
- For membership lookups, prefer `cluster_members` and `relationship_members`; use the JSON columns only when reproducing legacy exports.
- When shipping derived datasets (e.g., Zenodo archives), include both `docs/sqlite_schema.sql` and this description so downstream workflows can validate schema compatibility automatically.
