# data/results_repo.py
from typing import Iterable, Mapping, Tuple
from itertools import islice
import json
import hashlib
import sqlite3

def bulk_upsert_clusters(conn: sqlite3.Connection, row_tuples: Iterable[Tuple], chunk_size: int = 2000):
    """
    row_tuples must be from cluster_rows(). Uses a single transaction.
    """
    sql = """
    INSERT INTO clusters
      (inchikey_first, identity_key_strict,
       is_undef_sru, is_def_sru, sru_repeat_count,
       member_count, members_json, members_hash)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(inchikey_first, identity_key_strict, sru_key) DO UPDATE SET
       is_undef_sru     = excluded.is_undef_sru,
       is_def_sru       = excluded.is_def_sru,
       sru_repeat_count = excluded.sru_repeat_count,
       member_count     = excluded.member_count,
       members_json     = excluded.members_json,
       members_hash     = excluded.members_hash;
    """
    it = iter(row_tuples)
    with conn:  # single transaction
        while True:
            batch = list(islice(it, chunk_size))
            if not batch:
                break
            conn.executemany(sql, batch)


def fetch_cluster_reps_for_inchikey(results_db_path: str, inchikey_first: str):
    """
    Returns list of (cluster_id, identity_key_strict, rep_identifier)
    """
    with sqlite3.connect(results_db_path) as R:
        rows = R.execute("""
            SELECT cluster_id, identity_key_strict
            FROM clusters
            WHERE inchikey_first = ?
            ORDER BY cluster_id;
        """, (inchikey_first,)).fetchall()
    return rows

def preload_processed_pairs(results_db_path, version_tag, cluster_ids):
    a_min, a_max = min(cluster_ids), max(cluster_ids)
    with sqlite3.connect(results_db_path) as R:
        rows = R.execute("""
            SELECT cluster_a, cluster_b
            FROM relationships
            WHERE version_tag = ?
              AND cluster_a BETWEEN ? AND ?
              AND cluster_b BETWEEN ? AND ?;
        """, (version_tag, a_min, a_max, a_min, a_max)).fetchall()
    return {tuple(row) for row in rows}  # {(a,b), ...}

def load_accession(cache_db_path, smiles_list): # needs to be changed to use accession_curie
    if not smiles_list:
        return {}
    placeholders = ",".join(["?"] * len(smiles_list))
    sql = f"""
    SELECT 
        str.smiles, 
        src.accession_curie 
    FROM structures WHERE smiles IN ({placeholders})"""
    out = {}
    with sqlite3.connect(cache_db_path) as C:
        for smiles, molfile_path in C.execute(sql, smiles_list):
            out[smiles] = molfile_path
    return out


def preload_cluster_sru(results_db_path, cluster_ids):
    if not cluster_ids:
        return {}
    placeholders = ",".join(["?"] * len(cluster_ids))
    sql = f"""
      SELECT cluster_id, is_undef_sru, is_def_sru, sru_repeat_count
      FROM clusters
      WHERE cluster_id IN ({placeholders})
    """
    out = {}
    with sqlite3.connect(results_db_path) as C:
        for cid, is_undef, is_def, repcnt in C.execute(sql, cluster_ids):
            # normalize booleans + repeat count
            is_def   = bool(is_def)
            is_undef = bool(is_undef)
            has_sru  = is_def or is_undef
            repcnt   = int(repcnt) if (repcnt is not None and is_def) else None
            out[cid] = {"has_sru": has_sru, "is_undef": is_undef, "rep": repcnt}
    return out


def batch_insert_cluster_pairs(results_db_path, rows):
    if not rows:
        return
    with sqlite3.connect(results_db_path) as R:
        R.execute("BEGIN")
        R.executemany("""
            INSERT OR REPLACE INTO relationships
            (
                    cluster_a, 
                    cluster_b, 
                    cluster_a_members,
                    cluster_b_members,
                    cluster_a_size,
                    cluster_b_size,
                    classification,
                    score,
                    score_details,
                    extra_info,
                    version_tag
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        R.execute("COMMIT")
