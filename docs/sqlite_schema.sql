PRAGMA foreign_keys=ON;

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

CREATE UNIQUE INDEX IF NOT EXISTS ux_clusters_ifsmi_disc
    ON clusters(inchikey_first, identity_key_strict, sru_key);

CREATE INDEX IF NOT EXISTS idx_ic_inchikey ON clusters(inchikey_first);
CREATE INDEX IF NOT EXISTS idx_ic_undef_sru ON clusters(is_undef_sru);
CREATE INDEX IF NOT EXISTS idx_ic_def_sru ON clusters(is_def_sru, sru_repeat_count);

CREATE TABLE cluster_members(
    cluster_id INTEGER NOT NULL,
    member_curie TEXT NOT NULL,
    PRIMARY KEY (cluster_id, member_curie),
    FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cluster_members_member ON cluster_members(member_curie);
CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster ON cluster_members(cluster_id);

CREATE TABLE relationships(
    cluster_a INTEGER NOT NULL,
    cluster_b INTEGER NOT NULL,
    cluster_a_members TEXT NOT NULL,
    cluster_b_members TEXT NOT NULL,
    cluster_a_size INTEGER NOT NULL,
    cluster_b_size INTEGER NOT NULL,
    classification TEXT NOT NULL,
    classification_term_id TEXT,
    score REAL,
    score_details TEXT,
    extra_info TEXT,
    version_tag TEXT NOT NULL,
    PRIMARY KEY (cluster_a, cluster_b, version_tag),
    FOREIGN KEY (cluster_a) REFERENCES clusters(cluster_id) ON DELETE CASCADE,
    FOREIGN KEY (cluster_b) REFERENCES clusters(cluster_id) ON DELETE CASCADE
);

CREATE TABLE relationship_members(
    cluster_a INTEGER NOT NULL,
    cluster_b INTEGER NOT NULL,
    version_tag TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('A', 'B')),
    member_curie TEXT NOT NULL,
    PRIMARY KEY (cluster_a, cluster_b, version_tag, side, member_curie),
    FOREIGN KEY (cluster_a, cluster_b, version_tag)
        REFERENCES relationships(cluster_a, cluster_b, version_tag)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cpr_version ON relationships(version_tag);
CREATE INDEX IF NOT EXISTS idx_cpr_version_ab ON relationships(version_tag, cluster_a, cluster_b);
CREATE INDEX IF NOT EXISTS idx_rel_members ON relationships(cluster_a_members, cluster_b_members);
CREATE INDEX IF NOT EXISTS idx_rel_members_member ON relationship_members(member_curie);
CREATE INDEX IF NOT EXISTS idx_rel_members_pair ON relationship_members(cluster_a, cluster_b, version_tag);
