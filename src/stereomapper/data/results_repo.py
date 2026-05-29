# data/results_repo.py
import sqlite3
from collections.abc import Iterable
from itertools import islice


def _sru_key(is_def: bool, is_undef: bool, repeat_count: int | None) -> str:
    if is_def:
        return f"def:{'' if repeat_count is None else repeat_count}"
    if is_undef:
        return "undef"
    return "none"


def _coerce_cluster_row(row) -> dict:
    if isinstance(row, dict):
        row.setdefault("status", "passed")
        row.setdefault("error", None)
        return row

    status = "passed"
    error = None

    if len(row) == 8:
        (
            inchikey_first,
            identity_key_strict,
            is_undef_sru,
            is_def_sru,
            sru_repeat_count,
            member_count,
            members_json,
            members_hash,
        ) = row
    elif len(row) == 10:
        (
            inchikey_first,
            identity_key_strict,
            is_undef_sru,
            is_def_sru,
            sru_repeat_count,
            member_count,
            members_json,
            members_hash,
            status,
            error,
        ) = row
    else:
        raise ValueError(f"Unexpected cluster row length: {len(row)}")

    member_ids = []
    if isinstance(members_json, str) and members_json:
        try:
            import json

            parsed = json.loads(members_json)
        except Exception:
            parsed = []
        if isinstance(parsed, list):
            member_ids = [str(member) for member in parsed if member is not None and str(member)]

    return {
        "inchikey_first": inchikey_first,
        "identity_key_strict": identity_key_strict,
        "is_undef_sru": is_undef_sru,
        "is_def_sru": is_def_sru,
        "sru_repeat_count": sru_repeat_count,
        "member_count": member_count,
        "members_json": members_json,
        "members_hash": members_hash,
        "member_ids": member_ids,
        "status": status,
        "error": error,
    }


def bulk_upsert_clusters(conn: sqlite3.Connection, rows: Iterable[dict], chunk_size: int = 2000):
    """
    rows must be from cluster_rows(). Uses a single transaction.
    Also populates normalized cluster_members.
    """
    sql = """
    INSERT INTO clusters
      (inchikey_first, identity_key_strict,
       is_undef_sru, is_def_sru, sru_repeat_count,
         member_count, members_json, members_hash,
         status, error)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(inchikey_first, identity_key_strict, sru_key) DO UPDATE SET
       is_undef_sru     = excluded.is_undef_sru,
       is_def_sru       = excluded.is_def_sru,
       sru_repeat_count = excluded.sru_repeat_count,
       member_count     = excluded.member_count,
       members_json     = excluded.members_json,
         members_hash     = excluded.members_hash,
         status           = excluded.status,
         error            = excluded.error;
    """
    it = iter(rows)
    with conn:  # single transaction
        conn.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS cluster_members(
                cluster_id INTEGER NOT NULL,
                member_curie TEXT NOT NULL,
                PRIMARY KEY (cluster_id, member_curie)
            );
            CREATE INDEX IF NOT EXISTS idx_cluster_members_member ON cluster_members(member_curie);
            CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster ON cluster_members(cluster_id);
            """
        )

        while True:
            batch = list(islice(it, chunk_size))
            if not batch:
                break

            batch = [_coerce_cluster_row(r) for r in batch]

            cluster_values = [
                (
                    r["inchikey_first"],
                    r["identity_key_strict"],
                    r["is_undef_sru"],
                    r["is_def_sru"],
                    r["sru_repeat_count"],
                    r["member_count"],
                    r["members_json"],
                    r["members_hash"],
                    r.get("status") or "passed",
                    r.get("error"),
                )
                for r in batch
            ]
            conn.executemany(sql, cluster_values)

            key_rows = [
                (
                    r["inchikey_first"],
                    r["identity_key_strict"],
                    _sru_key(r["is_def_sru"], r["is_undef_sru"], r["sru_repeat_count"]),
                )
                for r in batch
            ]
            key_placeholders = ",".join(["(?,?,?)"] * len(key_rows))
            flat_keys = [item for key in key_rows for item in key]
            rows_db = conn.execute(
                f"""
                SELECT cluster_id, inchikey_first, identity_key_strict, sru_key
                FROM clusters
                WHERE (inchikey_first, identity_key_strict, sru_key) IN ({key_placeholders})
                """,
                flat_keys,
            ).fetchall()

            id_by_key = {(ik, smi, sru_key): cid for (cid, ik, smi, sru_key) in rows_db}
            cluster_ids = [cid for cid, _, _, _ in rows_db]

            if cluster_ids:
                placeholders = ",".join(["?"] * len(cluster_ids))
                conn.execute(
                    f"DELETE FROM cluster_members WHERE cluster_id IN ({placeholders})",
                    cluster_ids,
                )

            member_rows = []
            for r in batch:
                key = (
                    r["inchikey_first"],
                    r["identity_key_strict"],
                    _sru_key(r["is_def_sru"], r["is_undef_sru"], r["sru_repeat_count"]),
                )
                cid = id_by_key.get(key)
                if cid is None:
                    continue
                for member in r.get("member_ids", []):
                    member_rows.append((cid, member))

            if member_rows:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO cluster_members (cluster_id, member_curie)
                    VALUES (?, ?)
                    """,
                    member_rows,
                )


def fetch_cluster_reps_for_inchikey(results_db_path: str, inchikey_first: str):
    """
    Returns list of (cluster_id, identity_key_strict, rep_identifier)
    """
    with sqlite3.connect(results_db_path) as r:
        rows = r.execute(
            """
            SELECT cluster_id, identity_key_strict
            FROM clusters
            WHERE inchikey_first = ?
            ORDER BY cluster_id;
        """,
            (inchikey_first,),
        ).fetchall()
    return rows


def update_cluster_statuses(
    results_db_path: str, statuses: list[tuple[str, str | None, int]]
) -> None:
    if not statuses:
        return
    with sqlite3.connect(results_db_path) as conn:
        conn.executemany(
            """
            UPDATE clusters
            SET status = ?, error = ?
            WHERE cluster_id = ?
            """,
            statuses,
        )


def preload_processed_pairs(results_db_path, version_tag, cluster_ids):
    if not cluster_ids:
        return set()
    placeholders = ",".join(["?"] * len(cluster_ids))
    with sqlite3.connect(results_db_path) as r:
        rows = r.execute(
            f"""
            SELECT cluster_a, cluster_b
            FROM relationships
            WHERE version_tag = ?
              AND (cluster_a IN ({placeholders}) OR cluster_b IN ({placeholders}));
        """,
            [version_tag, *cluster_ids, *cluster_ids],
        ).fetchall()
    return {tuple(row) for row in rows}  # {(a,b), ...}


def load_accession(cache_db_path, smiles_list):  # needs to be changed to use accession_curie
    if not smiles_list:
        return {}
    placeholders = ",".join(["?"] * len(smiles_list))
    sql = f"""
    SELECT
        smiles,
        accession_curie
    FROM structures WHERE smiles IN ({placeholders})"""
    out = {}
    with sqlite3.connect(cache_db_path) as c:
        for smiles, accession_curie in c.execute(sql, smiles_list):
            out[smiles] = accession_curie
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
    with sqlite3.connect(results_db_path) as c:
        for cid, is_undef, is_def, repcnt in c.execute(sql, cluster_ids):
            # normalize booleans + repeat count
            is_def = bool(is_def)
            is_undef = bool(is_undef)
            has_sru = is_def or is_undef
            repcnt = int(repcnt) if (repcnt is not None and is_def) else None
            out[cid] = {"has_sru": has_sru, "is_undef": is_undef, "rep": repcnt}
    return out


def _coerce_relationship_row(row: tuple) -> tuple:
    if len(row) == 13:
        return row
    if len(row) == 10:
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
            version_tag,
        ) = row
        return (
            cluster_a,
            cluster_b,
            cluster_a_members,
            cluster_b_members,
            cluster_a_size,
            cluster_b_size,
            classification,
            None,
            None,
            score,
            score_details,
            None,
            version_tag,
        )
    if len(row) == 12:
        (
            cluster_a,
            cluster_b,
            cluster_a_members,
            cluster_b_members,
            cluster_a_size,
            cluster_b_size,
            classification,
            classification_term_id,
            score,
            score_details,
            extra_info,
            version_tag,
        ) = row
        return (
            cluster_a,
            cluster_b,
            cluster_a_members,
            cluster_b_members,
            cluster_a_size,
            cluster_b_size,
            classification,
            classification_term_id,
            None,
            score,
            score_details,
            extra_info,
            version_tag,
        )
    raise ValueError(f"relationship rows must have 10, 12, or 13 fields, got {len(row)}")


def batch_insert_cluster_pairs(results_db_path, rows):
    if not rows:
        return
    rows = [_coerce_relationship_row(row) for row in rows]
    with sqlite3.connect(results_db_path) as r:
        r.execute("BEGIN")
        r.executemany(
            """
            INSERT OR REPLACE INTO relationships
            (
                    cluster_a,
                    cluster_b,
                    cluster_a_members,
                    cluster_b_members,
                    cluster_a_size,
                    cluster_b_size,
                    classification,
                    classification_term_id,
                    direction,
                    score,
                    score_details,
                    extra_info,
                    version_tag
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            rows,
        )
        r.execute("COMMIT")


def batch_insert_relationship_members(results_db_path: str, rows: list[tuple]):
    if not rows:
        return
    keys = {(a, b, version) for (a, b, version, _, _) in rows}
    with sqlite3.connect(results_db_path) as r:
        r.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS relationship_members(
                cluster_a INTEGER NOT NULL,
                cluster_b INTEGER NOT NULL,
                version_tag TEXT NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('A', 'B')),
                member_curie TEXT NOT NULL,
                PRIMARY KEY (cluster_a, cluster_b, version_tag, side, member_curie)
            );
            CREATE INDEX IF NOT EXISTS idx_rel_members_member ON relationship_members(member_curie);
            CREATE INDEX IF NOT EXISTS idx_rel_members_pair ON relationship_members(cluster_a, cluster_b, version_tag);
            """
        )
        with r:
            if keys:
                placeholders = ",".join(["(?,?,?)"] * len(keys))
                flat_keys = [item for key in keys for item in key]
                r.execute(
                    f"""
                    DELETE FROM relationship_members
                    WHERE (cluster_a, cluster_b, version_tag) IN ({placeholders})
                    """,
                    flat_keys,
                )

            r.executemany(
                """
                INSERT OR IGNORE INTO relationship_members
                (
                    cluster_a,
                    cluster_b,
                    version_tag,
                    side,
                    member_curie
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
