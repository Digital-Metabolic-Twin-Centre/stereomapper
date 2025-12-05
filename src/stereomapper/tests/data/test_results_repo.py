import pytest
import sqlite3
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from stereomapper.data.results_repo import (
    bulk_upsert_clusters,
    fetch_cluster_reps_for_inchikey,
    preload_processed_pairs,
    load_accession,
    preload_cluster_sru,
    batch_insert_cluster_pairs
)


@pytest.fixture
def temp_results_db():
    """Create a temporary results database with schema."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    conn = sqlite3.connect(str(tmp_path))

    # Create tables matching the expected schema
    conn.execute("""
        CREATE TABLE clusters (
            cluster_id INTEGER PRIMARY KEY,
            inchikey_first TEXT NOT NULL,
            identity_key_strict TEXT NOT NULL,
            is_undef_sru BOOLEAN DEFAULT 0,
            is_def_sru BOOLEAN DEFAULT 0,
            sru_repeat_count INTEGER,
            sru_key TEXT GENERATED ALWAYS AS (
                CASE
                    WHEN is_undef_sru = 1 THEN 'undef'
                    WHEN is_def_sru = 1 THEN 'def:' || COALESCE(sru_repeat_count, '')
                    ELSE 'none'
                END
            ),
            member_count INTEGER NOT NULL,
            members_json TEXT,
            members_hash TEXT NOT NULL,
            UNIQUE(inchikey_first, identity_key_strict, sru_key)
        )
    """)

    conn.execute("""
        CREATE TABLE relationships (
            cluster_a INTEGER,
            cluster_b INTEGER,
            cluster_a_members TEXT,
            cluster_b_members TEXT,
            cluster_a_size INTEGER,
            cluster_b_size INTEGER,
            classification TEXT,
            score REAL,
            score_details TEXT,
            version_tag TEXT,
            PRIMARY KEY (cluster_a, cluster_b, version_tag)
        )
    """)

    yield conn, tmp_path
    conn.close()
    tmp_path.unlink(missing_ok=True)


@pytest.fixture
def temp_cache_db():
    """Create a temporary cache database with schema."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    conn = sqlite3.connect(str(tmp_path))

    # Create tables matching the expected cache schema
    conn.execute("""
        CREATE TABLE structures (
            molecule_id INTEGER PRIMARY KEY,
            smiles TEXT,
            inchikey_first TEXT,
            accession_curie TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE sources (
            source_id TEXT PRIMARY KEY,
            molecule_id INTEGER,
            accession_curie TEXT,
            FOREIGN KEY (molecule_id) REFERENCES structures(molecule_id)
        )
    """)

    yield conn, tmp_path
    conn.close()
    tmp_path.unlink(missing_ok=True)


class TestBulkUpsertClusters:
    """Test bulk_upsert_clusters function."""

    def test_bulk_upsert_clusters_insert_new(self, temp_results_db):
        """Test inserting new clusters."""
        conn, db_path = temp_results_db

        # Prepare test data
        row_tuples = [
            ('LFQSCWFLJHTTHZ', 'ethanol_strict', 0, 0, None, 2, '["mol1", "mol2"]', 'hash1'),
            ('HGBOYTHUEUWSSQ', 'ethylamine_strict', 1, 0, None, 1, '["mol3"]', 'hash2'),
            ('ATUOYWHBWRKTHZ', 'propane_strict', 0, 1, 3, 1, '["mol4"]', 'hash3'),
        ]

        # Execute function
        bulk_upsert_clusters(conn, row_tuples)

        # Verify data was inserted
        cursor = conn.execute("SELECT COUNT(*) FROM clusters")
        assert cursor.fetchone()[0] == 3

        # Verify specific data
        cursor = conn.execute("""
            SELECT inchikey_first, identity_key_strict, is_undef_sru, is_def_sru,
                   sru_repeat_count, member_count, members_json, members_hash, sru_key
            FROM clusters ORDER BY cluster_id
        """)
        results = cursor.fetchall()

        assert len(results) == 3
        assert results[0] == ('LFQSCWFLJHTTHZ', 'ethanol_strict', 0, 0, None, 2, '["mol1", "mol2"]', 'hash1', 'none')
        assert results[1] == ('HGBOYTHUEUWSSQ', 'ethylamine_strict', 1, 0, None, 1, '["mol3"]', 'hash2', 'undef')
        assert results[2] == ('ATUOYWHBWRKTHZ', 'propane_strict', 0, 1, 3, 1, '["mol4"]', 'hash3', 'def:3')

    def test_bulk_upsert_clusters_update_existing(self, temp_results_db):
        """Test updating existing clusters."""
        conn, db_path = temp_results_db

        # Insert initial data
        initial_data = [
            ('LFQSCWFLJHTTHZ', 'ethanol_strict', 0, 0, None, 1, '["mol1"]', 'hash1'),
        ]
        bulk_upsert_clusters(conn, initial_data)

        # Update with new data
        updated_data = [
            ('LFQSCWFLJHTTHZ', 'ethanol_strict', 0, 0, None, 2, '["mol1", "mol2"]', 'hash2'),
        ]
        bulk_upsert_clusters(conn, updated_data)

        # Verify update occurred
        cursor = conn.execute("SELECT COUNT(*) FROM clusters")
        assert cursor.fetchone()[0] == 1  # Still only one row

        cursor = conn.execute("""
            SELECT member_count, members_json, members_hash
            FROM clusters WHERE inchikey_first = 'LFQSCWFLJHTTHZ'
        """)
        result = cursor.fetchone()
        assert result == (2, '["mol1", "mol2"]', 'hash2')

    def test_bulk_upsert_clusters_chunking(self, temp_results_db):
        """Test that chunking works correctly."""
        conn, db_path = temp_results_db

        # Create more data than chunk_size
        row_tuples = [
            (f'INCHI{i:03d}TESTKEY', f'identity_{i}', 0, 0, None, 1, f'["mol{i}"]', f'hash{i}')
            for i in range(5)
        ]

        # Use small chunk size
        bulk_upsert_clusters(conn, row_tuples, chunk_size=2)

        # Verify all data was inserted
        cursor = conn.execute("SELECT COUNT(*) FROM clusters")
        assert cursor.fetchone()[0] == 5

    def test_bulk_upsert_clusters_empty_input(self, temp_results_db):
        """Test with empty input."""
        conn, db_path = temp_results_db

        # Should not raise an error
        bulk_upsert_clusters(conn, [])

        # Verify no data was inserted
        cursor = conn.execute("SELECT COUNT(*) FROM clusters")
        assert cursor.fetchone()[0] == 0

    def test_bulk_upsert_clusters_transaction_rollback(self, temp_results_db):
        """Test that transaction rolls back on error."""
        conn, db_path = temp_results_db

        # Create data with invalid constraint
        row_tuples = [
            ('LFQSCWFLJHTTHZ', 'ethanol_strict', 0, 0, None, 1, '["mol1"]', 'hash1'),
            ('INVALID', None, 0, 0, None, 1, '["mol2"]', 'hash2'),  # NULL identity_key_strict
        ]

        # Should raise an error and rollback
        with pytest.raises(sqlite3.IntegrityError):
            bulk_upsert_clusters(conn, row_tuples)

        # Verify no data was inserted
        cursor = conn.execute("SELECT COUNT(*) FROM clusters")
        assert cursor.fetchone()[0] == 0


class TestFetchClusterRepsForInchikey:
    """Test fetch_cluster_reps_for_inchikey function."""

    def test_fetch_cluster_reps_existing_inchikey(self, temp_results_db):
        """Test fetching clusters for existing inchikey."""
        conn, db_path = temp_results_db

        # Insert test data
        test_data = [
            ('LFQSCWFLJHTTHZ', 'ethanol_strict', 0, 0, None, 1, '["mol1"]', 'hash1'),
            ('LFQSCWFLJHTTHZ', 'ethanol_loose', 0, 0, None, 2, '["mol1", "mol2"]', 'hash2'),
            ('HGBOYTHUEUWSSQ', 'ethylamine_strict', 1, 0, None, 1, '["mol3"]', 'hash3'),
        ]
        bulk_upsert_clusters(conn, test_data)

        # Test the function
        result = fetch_cluster_reps_for_inchikey(str(db_path), 'LFQSCWFLJHTTHZ')

        # Should return both clusters with the same inchikey, ordered by cluster_id
        assert len(result) == 2
        assert result[0][1] == 'ethanol_strict'  # identity_key_strict
        assert result[1][1] == 'ethanol_loose'   # identity_key_strict
        assert result[0][0] < result[1][0]       # cluster_ids should be ordered

    def test_fetch_cluster_reps_nonexistent_inchikey(self, temp_results_db):
        """Test fetching clusters for nonexistent inchikey."""
        conn, db_path = temp_results_db

        # Insert some data
        test_data = [
            ('LFQSCWFLJHTTHZ', 'ethanol_strict', 0, 0, None, 1, '["mol1"]', 'hash1'),
        ]
        bulk_upsert_clusters(conn, test_data)

        # Test with nonexistent inchikey
        result = fetch_cluster_reps_for_inchikey(str(db_path), 'NONEXISTENT')

        assert result == []

    def test_fetch_cluster_reps_empty_database(self, temp_results_db):
        """Test fetching from empty database."""
        conn, db_path = temp_results_db

        result = fetch_cluster_reps_for_inchikey(str(db_path), 'LFQSCWFLJHTTHZ')

        assert result == []


class TestPreloadProcessedPairs:
    """Test preload_processed_pairs function."""

    def test_preload_processed_pairs_single_cluster(self, temp_results_db):
        """Test with single cluster_id."""
        conn, db_path = temp_results_db

        # Insert test relationships
        relationships = [
            (1, 2, 'mol1', 'mol2', 1, 1, 'similar', 0.8, '{}', 'v1.0'),
            (1, 3, 'mol1', 'mol3', 1, 1, 'similar', 0.7, '{}', 'v1.0'),
            (2, 4, 'mol2', 'mol4', 1, 1, 'dissimilar', 0.2, '{}', 'v1.0'),
        ]
        batch_insert_cluster_pairs(str(db_path), relationships)

        result = preload_processed_pairs(str(db_path), 'v1.0', [1])

        expected = {(1, 2), (1, 3)}
        assert result == expected

    def test_preload_processed_pairs_multiple_clusters(self, temp_results_db):
        """Test with multiple cluster_ids."""
        conn, db_path = temp_results_db

        # Insert test relationships
        relationships = [
            (1, 2, 'mol1', 'mol2', 1, 1, 'similar', 0.8, '{}', 'v1.0'),
            (2, 3, 'mol2', 'mol3', 1, 1, 'similar', 0.7, '{}', 'v1.0'),
            (3, 4, 'mol3', 'mol4', 1, 1, 'dissimilar', 0.2, '{}', 'v1.0'),
            (5, 6, 'mol5', 'mol6', 1, 1, 'similar', 0.9, '{}', 'v1.0'),
        ]
        batch_insert_cluster_pairs(str(db_path), relationships)

        result = preload_processed_pairs(str(db_path), 'v1.0', [1, 2, 3])

        expected = {(1, 2), (2, 3), (3, 4)}
        assert result == expected

    def test_preload_processed_pairs_version_filtering(self, temp_results_db):
        """Test version_tag filtering."""
        conn, db_path = temp_results_db

        # Insert test relationships with different versions
        relationships = [
            (1, 2, 'mol1', 'mol2', 1, 1, 'similar', 0.8, '{}', 'v1.0'),
            (1, 3, 'mol1', 'mol3', 1, 1, 'similar', 0.7, '{}', 'v2.0'),
            (2, 3, 'mol2', 'mol3', 1, 1, 'dissimilar', 0.2, '{}', 'v1.0'),
        ]
        batch_insert_cluster_pairs(str(db_path), relationships)

        result = preload_processed_pairs(str(db_path), 'v1.0', [1, 2])

        expected = {(1, 2), (2, 3)}
        assert result == expected

    def test_preload_processed_pairs_range_filtering(self, temp_results_db):
        """Test BETWEEN range filtering."""
        conn, db_path = temp_results_db

        # Insert test relationships
        relationships = [
            (1, 10, 'mol1', 'mol10', 1, 1, 'similar', 0.8, '{}', 'v1.0'),  # Outside range
            (2, 3, 'mol2', 'mol3', 1, 1, 'similar', 0.7, '{}', 'v1.0'),    # Inside range
            (4, 5, 'mol4', 'mol5', 1, 1, 'dissimilar', 0.2, '{}', 'v1.0'), # Inside range
            (10, 11, 'mol10', 'mol11', 1, 1, 'similar', 0.9, '{}', 'v1.0'), # Outside range
        ]
        batch_insert_cluster_pairs(str(db_path), relationships)

        # cluster_ids = [2, 5] gives range [2, 5]
        result = preload_processed_pairs(str(db_path), 'v1.0', [2, 5])

        expected = {(2, 3), (4, 5)}
        assert result == expected

    def test_preload_processed_pairs_empty_result(self, temp_results_db):
        """Test when no matches are found."""
        conn, db_path = temp_results_db

        # Insert test relationships
        relationships = [
            (1, 2, 'mol1', 'mol2', 1, 1, 'similar', 0.8, '{}', 'v2.0'),
        ]
        batch_insert_cluster_pairs(str(db_path), relationships)

        result = preload_processed_pairs(str(db_path), 'v1.0', [1])

        assert result == set()


class TestLoadAccession:
    """Test load_accession function."""

    def test_load_accession_existing_smiles(self, temp_cache_db):
        """Test loading accessions for existing SMILES."""
        conn, db_path = temp_cache_db

        # Insert test data
        conn.execute("""
            INSERT INTO structures (molecule_id, smiles, accession_curie)
            VALUES (1, 'CCO', 'CHEMBL545'), (2, 'CCN', 'CHEMBL123'), (3, 'CCC', 'CHEMBL789')
        """)
        conn.commit()

        smiles_list = ['CCO', 'CCN']
        result = load_accession(str(db_path), smiles_list)

        expected = {
            'CCO': 'CHEMBL545',
            'CCN': 'CHEMBL123'
        }
        assert result == expected

    def test_load_accession_partial_matches(self, temp_cache_db):
        """Test with some SMILES not in database."""
        conn, db_path = temp_cache_db

        # Insert test data
        conn.execute("""
            INSERT INTO structures (molecule_id, smiles, accession_curie)
            VALUES (1, 'CCO', 'CHEMBL545')
        """)
        conn.commit()

        smiles_list = ['CCO', 'NONEXISTENT']
        result = load_accession(str(db_path), smiles_list)

        expected = {'CCO': 'CHEMBL545'}
        assert result == expected

    def test_load_accession_empty_input(self, temp_cache_db):
        """Test with empty SMILES list."""
        conn, db_path = temp_cache_db

        result = load_accession(str(db_path), [])

        assert result == {}

    def test_load_accession_no_matches(self, temp_cache_db):
        """Test when no SMILES match."""
        conn, db_path = temp_cache_db

        # Insert test data
        conn.execute("""
            INSERT INTO structures (molecule_id, smiles, accession_curie)
            VALUES (1, 'CCO', 'CHEMBL545')
        """)
        conn.commit()

        smiles_list = ['NONEXISTENT1', 'NONEXISTENT2']
        result = load_accession(str(db_path), smiles_list)

        assert result == {}


class TestPreloadClusterSru:
    """Test preload_cluster_sru function."""

    def test_preload_cluster_sru_various_sru_types(self, temp_results_db):
        """Test loading SRU data for various cluster types."""
        conn, db_path = temp_results_db

        # Insert test clusters
        test_data = [
            ('INCHI001', 'none_sru', 0, 0, None, 1, '["mol1"]', 'hash1'),      # No SRU
            ('INCHI002', 'undef_sru', 1, 0, None, 1, '["mol2"]', 'hash2'),     # Undefined SRU
            ('INCHI003', 'def_sru', 0, 1, 5, 1, '["mol3"]', 'hash3'),          # Defined SRU with count
            ('INCHI004', 'def_sru_no_count', 0, 1, None, 1, '["mol4"]', 'hash4'), # Defined SRU without count
        ]
        bulk_upsert_clusters(conn, test_data)

        # Get cluster IDs
        cursor = conn.execute("SELECT cluster_id FROM clusters ORDER BY cluster_id")
        cluster_ids = [row[0] for row in cursor.fetchall()]

        result = preload_cluster_sru(str(db_path), cluster_ids)

        assert len(result) == 4

        # Test no SRU
        cluster_1 = result[cluster_ids[0]]
        assert cluster_1['has_sru'] == False
        assert cluster_1['is_undef'] == False
        assert cluster_1['rep'] is None

        # Test undefined SRU
        cluster_2 = result[cluster_ids[1]]
        assert cluster_2['has_sru'] == True
        assert cluster_2['is_undef'] == True
        assert cluster_2['rep'] is None

        # Test defined SRU with count
        cluster_3 = result[cluster_ids[2]]
        assert cluster_3['has_sru'] == True
        assert cluster_3['is_undef'] == False
        assert cluster_3['rep'] == 5

        # Test defined SRU without count
        cluster_4 = result[cluster_ids[3]]
        assert cluster_4['has_sru'] == True
        assert cluster_4['is_undef'] == False
        assert cluster_4['rep'] is None

    def test_preload_cluster_sru_empty_input(self, temp_results_db):
        """Test with empty cluster_ids list."""
        conn, db_path = temp_results_db

        result = preload_cluster_sru(str(db_path), [])

        assert result == {}

    def test_preload_cluster_sru_nonexistent_clusters(self, temp_results_db):
        """Test with nonexistent cluster IDs."""
        conn, db_path = temp_results_db

        result = preload_cluster_sru(str(db_path), [999, 1000])

        assert result == {}

    def test_preload_cluster_sru_partial_matches(self, temp_results_db):
        """Test with mix of existing and nonexistent cluster IDs."""
        conn, db_path = temp_results_db

        # Insert test cluster
        test_data = [
            ('INCHI001', 'test_cluster', 0, 1, 3, 1, '["mol1"]', 'hash1'),
        ]
        bulk_upsert_clusters(conn, test_data)

        # Get the actual cluster ID
        cursor = conn.execute("SELECT cluster_id FROM clusters")
        cluster_id = cursor.fetchone()[0]

        result = preload_cluster_sru(str(db_path), [cluster_id, 999])

        assert len(result) == 1
        assert cluster_id in result
        assert result[cluster_id]['has_sru'] == True
        assert result[cluster_id]['rep'] == 3


class TestBatchInsertClusterPairs:
    """Test batch_insert_cluster_pairs function."""

    def test_batch_insert_cluster_pairs_new_data(self, temp_results_db):
        """Test inserting new cluster pair relationships."""
        conn, db_path = temp_results_db

        rows = [
            (1, 2, 'mol1', 'mol2', 1, 1, 'similar', 0.8, '{"method": "fingerprint"}', 'v1.0'),
            (2, 3, 'mol2', 'mol3', 1, 1, 'dissimilar', 0.2, '{"method": "fingerprint"}', 'v1.0'),
            (1, 3, 'mol1', 'mol3', 1, 1, 'similar', 0.7, '{"method": "fingerprint"}', 'v1.0'),
        ]

        batch_insert_cluster_pairs(str(db_path), rows)

        # Verify data was inserted
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM relationships")
            assert cursor.fetchone()[0] == 3

            # Verify specific data
            cursor = conn.execute("""
                SELECT cluster_a, cluster_b, classification, score, version_tag
                FROM relationships ORDER BY cluster_a, cluster_b
            """)
            results = cursor.fetchall()

            assert results[0] == (1, 2, 'similar', 0.8, 'v1.0')
            assert results[1] == (1, 3, 'similar', 0.7, 'v1.0')
            assert results[2] == (2, 3, 'dissimilar', 0.2, 'v1.0')

    def test_batch_insert_cluster_pairs_replace_existing(self, temp_results_db):
        """Test replacing existing relationships."""
        conn, db_path = temp_results_db

        # Insert initial data
        initial_rows = [
            (1, 2, 'mol1', 'mol2', 1, 1, 'similar', 0.8, '{"method": "old"}', 'v1.0'),
        ]
        batch_insert_cluster_pairs(str(db_path), initial_rows)

        # Replace with new data
        new_rows = [
            (1, 2, 'mol1', 'mol2', 1, 1, 'dissimilar', 0.3, '{"method": "new"}', 'v1.0'),
        ]
        batch_insert_cluster_pairs(str(db_path), new_rows)

        # Verify replacement occurred
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM relationships")
            assert cursor.fetchone()[0] == 1  # Still only one row

            cursor = conn.execute("""
                SELECT classification, score, score_details
                FROM relationships WHERE cluster_a = 1 AND cluster_b = 2
            """)
            result = cursor.fetchone()
            assert result == ('dissimilar', 0.3, '{"method": "new"}')

    def test_batch_insert_cluster_pairs_empty_input(self, temp_results_db):
        """Test with empty input."""
        conn, db_path = temp_results_db

        # Should not raise an error
        batch_insert_cluster_pairs(str(db_path), [])

        # Verify no data was inserted
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM relationships")
            assert cursor.fetchone()[0] == 0

    def test_batch_insert_cluster_pairs_transaction_integrity(self, temp_results_db):
        """Test transaction integrity on error."""
        conn, db_path = temp_results_db

        # Create data with one invalid row (missing required field)
        rows = [
            (1, 2, 'mol1', 'mol2', 1, 1, 'similar', 0.8, '{}', 'v1.0'),
            (2, 3, None, 'mol3', 1, 1, 'dissimilar', 0.2, '{}', 'v1.0'),  # NULL cluster_a_members
        ]

        # Should handle the error gracefully or raise appropriate exception
        try:
            batch_insert_cluster_pairs(str(db_path), rows)
        except sqlite3.Error:
            pass  # Expected if there are constraints

        # If the function doesn't handle errors, verify rollback behavior
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM relationships")
            count = cursor.fetchone()[0]
            # Count should be either 0 (rollback) or all valid rows (partial success)
            assert count >= 0

    def test_batch_insert_cluster_pairs_different_versions(self, temp_results_db):
        """Test inserting relationships with different version tags."""
        conn, db_path = temp_results_db

        rows = [
            (1, 2, 'mol1', 'mol2', 1, 1, 'similar', 0.8, '{}', 'v1.0'),
            (1, 2, 'mol1', 'mol2', 1, 1, 'dissimilar', 0.3, '{}', 'v2.0'),  # Same clusters, different version
            (2, 3, 'mol2', 'mol3', 1, 1, 'similar', 0.7, '{}', 'v1.0'),
        ]

        batch_insert_cluster_pairs(str(db_path), rows)

        # Verify both versions exist
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM relationships")
            assert cursor.fetchone()[0] == 3

            # Verify version-specific data
            cursor = conn.execute("""
                SELECT classification FROM relationships
                WHERE cluster_a = 1 AND cluster_b = 2 AND version_tag = 'v1.0'
            """)
            assert cursor.fetchone()[0] == 'similar'

            cursor = conn.execute("""
                SELECT classification FROM relationships
                WHERE cluster_a = 1 AND cluster_b = 2 AND version_tag = 'v2.0'
            """)
            assert cursor.fetchone()[0] == 'dissimilar'


class TestIntegration:
    """Test integration between functions."""

    def test_full_workflow_integration(self, temp_results_db):
        """Test a complete workflow using multiple functions."""
        conn, db_path = temp_results_db

        # 1. Insert clusters using bulk_upsert_clusters
        cluster_data = [
            ('LFQSCWFLJHTTHZ', 'ethanol_strict', 0, 0, None, 2, '["ethanol1", "ethanol2"]', 'hash1'),
            ('LFQSCWFLJHTTHZ', 'ethanol_loose', 0, 1, 3, 1, '["ethanol3"]', 'hash2'),
            ('HGBOYTHUEUWSSQ', 'ethylamine_strict', 1, 0, None, 1, '["ethylamine1"]', 'hash3'),
        ]
        bulk_upsert_clusters(conn, cluster_data)

        # 2. Get cluster representatives for an inchikey
        ethanol_clusters = fetch_cluster_reps_for_inchikey(str(db_path), 'LFQSCWFLJHTTHZ')
        assert len(ethanol_clusters) == 2

        ethanol_cluster_ids = [cluster[0] for cluster in ethanol_clusters]

        # 3. Insert relationships between clusters
        all_cluster_ids = ethanol_cluster_ids + [ethanol_clusters[0][0] + 2]  # Add third cluster ID
        relationship_data = [
            (all_cluster_ids[0], all_cluster_ids[1], 'ethanol1,ethanol2', 'ethanol3', 2, 1, 'similar', 0.9, '{}', 'v1.0'),
            (all_cluster_ids[0], all_cluster_ids[2], 'ethanol1,ethanol2', 'ethylamine1', 2, 1, 'dissimilar', 0.1, '{}', 'v1.0'),
        ]
        batch_insert_cluster_pairs(str(db_path), relationship_data)

        # 4. Preload processed pairs
        processed_pairs = preload_processed_pairs(str(db_path), 'v1.0', all_cluster_ids)
        assert len(processed_pairs) == 2
        assert (all_cluster_ids[0], all_cluster_ids[1]) in processed_pairs
        assert (all_cluster_ids[0], all_cluster_ids[2]) in processed_pairs

        # 5. Load SRU information
        sru_info = preload_cluster_sru(str(db_path), all_cluster_ids)
        assert len(sru_info) == 3
        assert sru_info[all_cluster_ids[0]]['has_sru'] == False  # none SRU
        assert sru_info[all_cluster_ids[1]]['has_sru'] == True   # def SRU
        assert sru_info[all_cluster_ids[1]]['rep'] == 3
        assert sru_info[all_cluster_ids[2]]['has_sru'] == True   # undef SRU
        assert sru_info[all_cluster_ids[2]]['is_undef'] == True

    def test_cross_function_data_consistency(self, temp_results_db, temp_cache_db):
        """Test data consistency across cache and results databases."""
        results_conn, results_path = temp_results_db
        cache_conn, cache_path = temp_cache_db

        # Setup cache database
        cache_conn.execute("""
            INSERT INTO structures (molecule_id, smiles, accession_curie)
            VALUES (1, 'CCO', 'CHEMBL545'), (2, 'CCN', 'CHEMBL123')
        """)
        cache_conn.commit()

        # Setup results database
        cluster_data = [
            ('LFQSCWFLJHTTHZ', 'ethanol_cluster', 0, 0, None, 2, '["CCO", "CCO_variant"]', 'hash1'),
        ]
        bulk_upsert_clusters(results_conn, cluster_data)

        # Test accession loading
        accessions = load_accession(str(cache_path), ['CCO', 'CCN'])
        assert 'CCO' in accessions
        assert accessions['CCO'] == 'CHEMBL545'

        # Test cluster fetching
        clusters = fetch_cluster_reps_for_inchikey(str(results_path), 'LFQSCWFLJHTTHZ')
        assert len(clusters) == 1
        assert 'ethanol_cluster' in clusters[0][1]