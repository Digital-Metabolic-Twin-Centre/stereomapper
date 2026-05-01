# data/results_schema.py
import json
import sqlite3


def _member_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(member) for member in value if member is not None and str(member)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        if isinstance(parsed, list):
            return [str(member) for member in parsed if member is not None and str(member)]
    return []


def _backfill_cluster_members(cur: sqlite3.Cursor) -> None:
    rows = cur.execute(
        """
        SELECT cluster_id, members_json
        FROM clusters
        WHERE members_json IS NOT NULL AND members_json <> ''
        """
    ).fetchall()

    cluster_member_rows = []
    for cluster_id, members_json in rows:
        for member_curie in _member_list(members_json):
            cluster_member_rows.append((cluster_id, member_curie))

    if cluster_member_rows:
        cur.executemany(
            """
            INSERT OR IGNORE INTO cluster_members (cluster_id, member_curie)
            VALUES (?, ?)
            """,
            cluster_member_rows,
        )


def _backfill_relationship_members(cur: sqlite3.Cursor) -> None:
    rows = cur.execute(
        """
        SELECT cluster_a, cluster_b, version_tag, cluster_a_members, cluster_b_members
        FROM relationships
        """
    ).fetchall()

    relationship_member_rows = []
    for cluster_a, cluster_b, version_tag, cluster_a_members, cluster_b_members in rows:
        for member_curie in _member_list(cluster_a_members):
            relationship_member_rows.append((cluster_a, cluster_b, version_tag, 'A', member_curie))
        for member_curie in _member_list(cluster_b_members):
            relationship_member_rows.append((cluster_a, cluster_b, version_tag, 'B', member_curie))

    if relationship_member_rows:
        cur.executemany(
            """
            INSERT OR IGNORE INTO relationship_members
            (cluster_a, cluster_b, version_tag, side, member_curie)
            VALUES (?, ?, ?, ?, ?)
            """,
            relationship_member_rows,
        )


def results_schema(con: sqlite3.Connection) -> sqlite3.Connection:
    """Create the results database schema if it does not exist."""
    with con:
        cur = con.cursor()

        # Which of the core tables exist?
        have = {
            row[0]
            for row in cur.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ('clusters','relationships')
            """
            ).fetchall()
        }

        need = {"clusters", "relationships"}

        if have != need:
            # Drop whatever partial state exists and recreate cleanly
            cur.executescript(
                """
                DROP TABLE IF EXISTS relationship_members;
                DROP TABLE IF EXISTS cluster_members;
                DROP TABLE IF EXISTS relationships;
                DROP TABLE IF EXISTS clusters;

                CREATE TABLE clusters(
                    cluster_id INTEGER PRIMARY KEY,
                    inchikey_first TEXT NOT NULL,
                    identity_key_strict TEXT NOT NULL,
                    is_undef_sru BOOLEAN NOT NULL DEFAULT 0,
                    is_def_sru  BOOLEAN NOT NULL DEFAULT 0,
                    sru_repeat_count INTEGER,
                    sru_key TEXT GENERATED ALWAYS AS (
                        CASE
                            WHEN is_def_sru THEN 'def:' || COALESCE(sru_repeat_count, '')
                            WHEN is_undef_sru THEN 'undef'
                            ELSE 'none'
                        END
                    ) STORED,
                    member_count INTEGER NOT NULL,
                    members_json TEXT,
                    members_hash TEXT NOT NULL,
                    UNIQUE(inchikey_first, identity_key_strict, sru_key)
                );

                PRAGMA foreign_keys=ON;

                CREATE TABLE cluster_members(
                    cluster_id INTEGER NOT NULL,
                    member_curie TEXT NOT NULL,
                    PRIMARY KEY (cluster_id, member_curie),
                    FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id) ON DELETE CASCADE
                );

                CREATE TABLE relationships(
                    cluster_a       INTEGER NOT NULL,
                    cluster_b       INTEGER NOT NULL,
                    cluster_a_members TEXT NOT NULL,
                    cluster_b_members TEXT NOT NULL,
                    cluster_a_size  INTEGER NOT NULL,
                    cluster_b_size  INTEGER NOT NULL,
                    classification  TEXT    NOT NULL,
                    classification_term_id TEXT,
                    score           REAL,
                    score_details   TEXT,
                    extra_info      TEXT,
                    version_tag     TEXT    NOT NULL,
                    PRIMARY KEY (cluster_a, cluster_b, version_tag),
                    FOREIGN KEY (cluster_a) REFERENCES clusters(cluster_id) ON DELETE CASCADE,
                    FOREIGN KEY (cluster_b) REFERENCES clusters(cluster_id) ON DELETE CASCADE
                );

                CREATE TABLE relationship_members(
                    cluster_a   INTEGER NOT NULL,
                    cluster_b   INTEGER NOT NULL,
                    version_tag TEXT NOT NULL,
                    side        TEXT NOT NULL CHECK (side IN ('A', 'B')),
                    member_curie TEXT NOT NULL,
                    PRIMARY KEY (cluster_a, cluster_b, version_tag, side, member_curie),
                    FOREIGN KEY (cluster_a, cluster_b, version_tag)
                        REFERENCES relationships(cluster_a, cluster_b, version_tag)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_cpr_version ON relationships(version_tag);
                CREATE INDEX IF NOT EXISTS idx_cpr_version_ab ON relationships(version_tag, cluster_a, cluster_b);
                CREATE INDEX IF NOT EXISTS idx_rel_members ON relationships(cluster_a_members, cluster_b_members);

                CREATE UNIQUE INDEX IF NOT EXISTS ux_clusters_ifsmi_disc
                  ON clusters(inchikey_first, identity_key_strict, sru_key);

                CREATE INDEX IF NOT EXISTS idx_ic_inchikey ON clusters(inchikey_first);
                CREATE INDEX IF NOT EXISTS idx_ic_undef_sru ON clusters(is_undef_sru);
                CREATE INDEX IF NOT EXISTS idx_ic_def_sru ON clusters(is_def_sru, sru_repeat_count);

                CREATE INDEX IF NOT EXISTS idx_cluster_members_member ON cluster_members(member_curie);
                CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster ON cluster_members(cluster_id);

                CREATE INDEX IF NOT EXISTS idx_rel_members_member ON relationship_members(member_curie);
                CREATE INDEX IF NOT EXISTS idx_rel_members_pair ON relationship_members(cluster_a, cluster_b, version_tag);

            """
            )

        # Ensure normalized tables exist for older databases
        cur.executescript(
            """
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS cluster_members(
                cluster_id INTEGER NOT NULL,
                member_curie TEXT NOT NULL,
                PRIMARY KEY (cluster_id, member_curie),
                FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS relationship_members(
                cluster_a   INTEGER NOT NULL,
                cluster_b   INTEGER NOT NULL,
                version_tag TEXT NOT NULL,
                side        TEXT NOT NULL CHECK (side IN ('A', 'B')),
                member_curie TEXT NOT NULL,
                PRIMARY KEY (cluster_a, cluster_b, version_tag, side, member_curie),
                FOREIGN KEY (cluster_a, cluster_b, version_tag)
                    REFERENCES relationships(cluster_a, cluster_b, version_tag)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_cluster_members_member ON cluster_members(member_curie);
            CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster ON cluster_members(cluster_id);

            CREATE INDEX IF NOT EXISTS idx_rel_members_member ON relationship_members(member_curie);
            CREATE INDEX IF NOT EXISTS idx_rel_members_pair ON relationship_members(cluster_a, cluster_b, version_tag);
            """
        )

        _backfill_cluster_members(cur)
        _backfill_relationship_members(cur)

    return con
