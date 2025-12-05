"""Functionality to create the output database to store the results of the pipeline"""
import sqlite3

def results_schema(con: sqlite3.Connection) -> sqlite3.Connection:
    """ Create the results database schema if it does not exist. """
    with con:
        cur = con.cursor()

        # Which of the two tables exist?
        have = {
            row[0] for row in cur.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ('clusters','relationships')
            """).fetchall()
        }

        need = {"clusters", "relationships"}

        if have != need:
            # Drop whatever partial state exists and recreate cleanly
            cur.executescript("""
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

                CREATE TABLE relationships(
                    cluster_a       INTEGER NOT NULL,
                    cluster_b       INTEGER NOT NULL,
                    cluster_a_members TEXT NOT NULL,
                    cluster_b_members TEXT NOT NULL,
                    cluster_a_size  INTEGER NOT NULL,
                    cluster_b_size  INTEGER NOT NULL,
                    classification  TEXT    NOT NULL,
                    score           REAL,
                    score_details   TEXT,
                    extra_info      TEXT,
                    version_tag     TEXT    NOT NULL,
                    PRIMARY KEY (cluster_a, cluster_b, version_tag)
                );

                CREATE INDEX IF NOT EXISTS idx_cpr_version ON relationships(version_tag);
                CREATE INDEX IF NOT EXISTS idx_cpr_version_ab ON relationships(version_tag, cluster_a, cluster_b);
                CREATE INDEX IF NOT EXISTS idx_rel_members ON relationships(cluster_a_members, cluster_b_members);

                CREATE UNIQUE INDEX IF NOT EXISTS ux_clusters_ifsmi_disc
                  ON clusters(inchikey_first, identity_key_strict, sru_key);

                CREATE INDEX IF NOT EXISTS idx_ic_inchikey ON clusters(inchikey_first);
                CREATE INDEX IF NOT EXISTS idx_ic_undef_sru ON clusters(is_undef_sru);
                CREATE INDEX IF NOT EXISTS idx_ic_def_sru ON clusters(is_def_sru, sru_repeat_count);

            """)
            return
