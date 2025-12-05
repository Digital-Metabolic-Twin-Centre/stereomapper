"""Functionality to create the cache database"""
import sqlite3

def create_cache(con) -> sqlite3.Connection:
    """ Open or create the stuctures/features cache database with the required schema. """
    con.executescript("""
    -- Structures/features cache only
    CREATE TABLE IF NOT EXISTS structures (
        molecule_id      INTEGER PRIMARY KEY,
        namespace        TEXT NOT NULL DEFAULT 'default',
        molecule_key     TEXT NOT NULL,
        smiles           TEXT,
        inchikey_first   TEXT,
        is_undef_sru     INTEGER NOT NULL DEFAULT 0 CHECK (is_undef_sru IN (0,1)),
        is_def_sru       INTEGER NOT NULL DEFAULT 0 CHECK (is_def_sru   IN (0,1)),
        sru_repeat_count INTEGER,
        error            TEXT,
        feature_version  INTEGER NOT NULL DEFAULT 1,
        feature_blob     BLOB,
        created_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(namespace, molecule_key)
    );

    -- Provenance of how a molecule entered the cache
    CREATE TABLE IF NOT EXISTS sources (
        source_id   TEXT PRIMARY KEY,            -- use TEXT (UUID/ULID) or switch to INTEGER PRIMARY KEY
        molecule_id INTEGER NOT NULL,
        source_kind TEXT NOT NULL,               -- 'file'
        source_ref  TEXT NOT NULL,               -- path or external ID
        accession_curie TEXT,                     -- accesion id inferred from molfile basename passed / id in SDF if present
        file_hash   TEXT,
        created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (molecule_id) REFERENCES structures(molecule_id)
    );

    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_structures_ns_key       ON structures(namespace, molecule_key);
    CREATE INDEX IF NOT EXISTS idx_structures_inchikey     ON structures(inchikey_first);
    CREATE INDEX IF NOT EXISTS idx_structures_smiles         ON structures(smiles);
    CREATE INDEX IF NOT EXISTS idx_sources_molid           ON sources(molecule_id);
    """)

    return con
