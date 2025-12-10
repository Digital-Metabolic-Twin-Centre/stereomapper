# data/cache_repo.py
import os
import json
import hashlib
import sqlite3
from typing import Optional, Dict, Any
from stereomapper.domain.models import CacheEntry

def get_cached_entry(file_hash: str, conn) -> Optional[CacheEntry]:
    """Fetch a cached structure entry by its file hash, or None if not found."""
    query = """
        SELECT st.*
        FROM sources AS src
        JOIN structures AS st ON st.molecule_id = src.molecule_id
        WHERE src.file_hash = ?
        LIMIT 1;
    """
    cur = conn.execute(query, (str(file_hash),))
    row = cur.fetchone()
    if row is None:
        return None
    
    # Get column names from cursor description
    col_names = [desc[0] for desc in cur.description]
    row_dict = dict(zip(col_names, row))
    
    # Convert to CacheEntry object
    return CacheEntry(
        molecule_id=row_dict["molecule_id"],
        smiles=row_dict.get("smiles"),
        inchikey_first=row_dict.get("inchikey_first"),
        error=row_dict.get("error"),
        is_undef_sru=row_dict.get("is_undef_sru", 0),
        is_def_sru=row_dict.get("is_def_sru", 0),
        sru_repeat_count=row_dict.get("sru_repeat_count"),
        namespace=row_dict.get("namespace", "default")
    )

def inchi_first_by_id(conn, molfile_list, logger=None):
    """
    Retrieves unique inchikey first blocks from a given list of reference molfile paths.
    Uses the sources table to find the corresponding structures.
    Returns a sorted list of unique inchikey_first strings.
    """
    paths = [p for p in molfile_list if os.path.exists(p)]
    if not paths:
        return []

    from pathlib import Path
    def _norm(p): 
        try: return str(Path(p).expanduser().resolve())
        except Exception: return os.path.abspath(p)
    paths = [_norm(p) for p in paths]

    cur = conn.cursor()
    # Speed: one temp table + single join
    cur.execute("DROP TABLE IF EXISTS tmp_paths;")
    cur.execute("CREATE TEMP TABLE tmp_paths(p TEXT PRIMARY KEY);")

    cur.executemany("INSERT OR IGNORE INTO tmp_paths(p) VALUES (?)",
                    ((p,) for p in paths))

    # Make sure there is an index on structures.molfile_path
    cur.execute("""
        SELECT s.inchikey_first,
                sour.source_ref
        FROM structures s
        JOIN sources sour ON sour.molecule_id = s.molecule_id
        JOIN tmp_paths t ON t.p = sour.source_ref
        WHERE s.inchikey_first IS NOT NULL
    """)
    result = [r[0] for r in cur.fetchall()]
    cur.execute("DROP TABLE tmp_paths;")
    return sorted(result)

def streamline_rows(conn: sqlite3.Connection, inchikey_first: str):
    """
    One row per molecule with accession_curies as a Python list.
    Aggregation is scoped to the current inchikey to avoid scanning all sources.
    """
    sql = """
        WITH mols AS (
            SELECT
                molecule_id,
                namespace,
                molecule_key,
                COALESCE(smiles, '')           AS identity_key_strict,
                COALESCE(inchikey_first, '')   AS inchikey_first,
                COALESCE(is_undef_sru, 0)      AS is_undef_sru,
                COALESCE(is_def_sru, 0)        AS is_def_sru,
                sru_repeat_count,
                CASE
                  WHEN is_def_sru = 1 AND sru_repeat_count IS NOT NULL
                       THEN printf('def:%d', sru_repeat_count)
                  WHEN is_undef_sru = 1 THEN 'undef'
                  ELSE 'none'
                END                            AS sru_bucket
            FROM structures
            WHERE inchikey_first = ?
        ),
        src_agg AS (
            SELECT
                s.molecule_id,
                json_group_array(DISTINCT s.accession_curie) AS accession_curies_json
            FROM sources s
            JOIN mols m ON m.molecule_id = s.molecule_id
            GROUP BY s.molecule_id
        )
        SELECT
            m.molecule_id,
            m.namespace,
            m.molecule_key,
            m.identity_key_strict,
            m.inchikey_first,
            m.is_undef_sru,
            m.is_def_sru,
            m.sru_repeat_count,
            m.sru_bucket,
            COALESCE(sa.accession_curies_json, '[]') AS accession_curies_json
        FROM mols m
        LEFT JOIN src_agg sa USING (molecule_id)
        ORDER BY m.inchikey_first, m.identity_key_strict, m.sru_bucket
    """
    cur = conn.execute(sql, (inchikey_first,))
    cols = [d[0] for d in cur.description]
    for r in cur:
        row = dict(zip(cols, r))
        row["smiles"] = row.pop("identity_key_strict")
        row["accession_curies"] = json.loads(row.pop("accession_curies_json"))
        yield row


STRUCT_UPSERT_SQL = """
INSERT INTO structures (
    namespace, molecule_key, smiles, inchikey_first,
    is_undef_sru, is_def_sru, sru_repeat_count, error,
    feature_version, feature_blob, created_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(namespace, molecule_key) DO UPDATE SET
    smiles           = COALESCE(excluded.smiles, smiles),
    inchikey_first   = COALESCE(excluded.inchikey_first, inchikey_first),
    is_undef_sru     = COALESCE(excluded.is_undef_sru, is_undef_sru),
    is_def_sru       = COALESCE(excluded.is_def_sru, is_def_sru),
    sru_repeat_count = COALESCE(excluded.sru_repeat_count, sru_repeat_count),
    error            = COALESCE(excluded.error, error),
    -- only replace features when you provide a blob, and prefer newer versions
    feature_blob     = CASE
                         WHEN excluded.feature_blob IS NOT NULL
                          AND excluded.feature_version >= feature_version
                         THEN excluded.feature_blob
                         ELSE feature_blob
                       END,
    feature_version  = MAX(feature_version, excluded.feature_version),
    updated_at       = CURRENT_TIMESTAMP
RETURNING molecule_id;
"""

SRC_UPSERT_SQL = """
INSERT INTO sources (source_id, molecule_id, source_kind, source_ref, accession_curie, file_hash, created_at)
VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
ON CONFLICT(source_id) DO UPDATE SET
    molecule_id = excluded.molecule_id,  -- in case you re-map a source
    file_hash   = COALESCE(excluded.file_hash, file_hash);
"""

def upsert_structure(
    conn: sqlite3.Connection,
    *,
    namespace: str,
    molecule_key: str,
    smiles: Optional[str],
    inchikey_first: Optional[str],
    is_undef_sru: Optional[int],
    is_def_sru: Optional[int],
    sru_repeat_count: Optional[int],
    error: Optional[str],
    feature_version: int,
    feature_blob: Optional[bytes],
) -> int:
    row = (
        namespace, molecule_key, smiles, inchikey_first,
        is_undef_sru, is_def_sru, sru_repeat_count, error,
        feature_version, feature_blob
    )
    cur = conn.execute(STRUCT_UPSERT_SQL, row)
    (molecule_id,) = cur.fetchone()
    return molecule_id


def link_source(
    conn: sqlite3.Connection,
    *,
    source_id: str,            # e.g., a UUID for this file/input
    molecule_id: int,
    source_kind: str,          # 'file', 'chebi', 'envipath', ...
    source_ref: str,           # path or external ID
    accession_curie: str,  # inferred from molfile basename or SDF id tag
    file_hash: Optional[str],  # content hash if applicable
) -> None:
    conn.execute(SRC_UPSERT_SQL, (source_id, molecule_id, source_kind, source_ref, accession_curie, file_hash))

def ingest_one(
    conn, *, namespace, molecule_key,
    smiles, inchikey_first, is_undef_sru, is_def_sru, sru_repeat_count,
    error, feature_version, feature_blob,
    source_id, source_kind, source_ref, accession_curie, file_hash
):
    with conn:  # single transaction
        mol_id = upsert_structure(
            conn,
            namespace=namespace,
            molecule_key=molecule_key,
            smiles=smiles,
            inchikey_first=inchikey_first,
            is_undef_sru=is_undef_sru,
            is_def_sru=is_def_sru,
            sru_repeat_count=sru_repeat_count,
            error=error,
            feature_version=feature_version,
            feature_blob=feature_blob,
        )
        link_source(
            conn,
            source_id=source_id,
            molecule_id=mol_id,
            source_kind=source_kind,
            source_ref=source_ref,
            accession_curie=accession_curie,
            file_hash=file_hash,
        )

        return int(mol_id)
