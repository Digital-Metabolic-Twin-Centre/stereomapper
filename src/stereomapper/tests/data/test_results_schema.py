import pytest
import sqlite3
import tempfile
from pathlib import Path

from stereomapper.data.results_schema import results_schema


@pytest.fixture
def memory_db():
    """Create an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def temp_file_db():
    """Create a temporary file-based SQLite database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    conn = sqlite3.connect(str(tmp_path))
    yield conn, tmp_path
    conn.close()
    tmp_path.unlink(missing_ok=True)

class TestResultsSchema:
    """Test results_schema function."""
    
    def test_results_schema_execution(self, memory_db):
        """Test that results_schema executes without error."""
        # The function may not return the connection, but it should execute successfully
        result = results_schema(memory_db)
        
        # Verify that the schema was applied by checking for tables
        cursor = memory_db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('clusters', 'relationships')
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'clusters' in tables
        assert 'relationships' in tables
    
    def test_results_schema_with_file_db(self, temp_file_db):
        """Test results_schema works with file-based database."""
        conn, tmp_path = temp_file_db
        result = results_schema(conn)
        
        assert tmp_path.exists()
        assert tmp_path.stat().st_size > 0  # File should have content
        
        # Verify schema was applied
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('clusters', 'relationships')
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'clusters' in tables
        assert 'relationships' in tables
    
    def test_results_schema_idempotent(self, memory_db):
        """Test that calling results_schema multiple times is safe."""
        # First call
        result1 = results_schema(memory_db)
        
        # Second call should not raise an error
        result2 = results_schema(memory_db)
        
        # Verify both tables still exist after multiple calls
        cursor = memory_db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('clusters', 'relationships')
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'clusters' in tables
        assert 'relationships' in tables


class TestTableCreation:
    """Test that all required tables are created."""
    
    def test_clusters_table_created(self, memory_db):
        """Test that clusters table is created with correct structure."""
        results_schema(memory_db)
        
        # Check table exists
        cursor = memory_db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='clusters'
        """)
        assert cursor.fetchone() is not None
        
        # Check table structure
        cursor = memory_db.execute("PRAGMA table_xinfo(clusters)")
        columns = cursor.fetchall()
        
        expected_column_names = [
            'cluster_id', 'inchikey_first', 'identity_key_strict', 'is_undef_sru',
            'is_def_sru', 'sru_repeat_count', 'sru_key', 'member_count',
            'members_json', 'members_hash'
        ]
        
        actual_column_names = [col[1] for col in columns]
        assert actual_column_names == expected_column_names
        
        # Check primary key
        primary_key_columns = [col[1] for col in columns if col[5] == 1]
        assert primary_key_columns == ['cluster_id']
        
        # Check specific column types
        column_types = {col[1]: col[2] for col in columns}
        assert column_types['cluster_id'] == 'INTEGER'
        assert column_types['inchikey_first'] == 'TEXT'
        assert column_types['identity_key_strict'] == 'TEXT'
        assert column_types['is_undef_sru'] == 'BOOLEAN'
        assert column_types['is_def_sru'] == 'BOOLEAN'
        assert column_types['sru_repeat_count'] == 'INTEGER'
        assert column_types['member_count'] == 'INTEGER'
        assert column_types['members_json'] == 'TEXT'
        assert column_types['members_hash'] == 'TEXT'
    
    def test_relationships_table_created(self, memory_db):
        """Test that relationships table is created with correct structure."""
        results_schema(memory_db)
        
        cursor = memory_db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='relationships'
        """)
        assert cursor.fetchone() is not None
        
        cursor = memory_db.execute("PRAGMA table_info(relationships)")
        columns = cursor.fetchall()
        
        expected_column_names = [
            'cluster_a', 'cluster_b', 'cluster_a_members', 'cluster_b_members',
            'cluster_a_size', 'cluster_b_size', 'classification', 'score',
            'score_details', 'version_tag'
        ]
        
        actual_column_names = [col[1] for col in columns]
        assert actual_column_names == expected_column_names
        
        # Check composite primary key
        primary_key_columns = [col[1] for col in columns if col[5] > 0]
        assert set(primary_key_columns) == {'cluster_a', 'cluster_b', 'version_tag'}
        
        # Check column types
        column_types = {col[1]: col[2] for col in columns}
        assert column_types['cluster_a'] == 'INTEGER'
        assert column_types['cluster_b'] == 'INTEGER'
        assert column_types['classification'] == 'TEXT'
        assert column_types['score'] == 'REAL'
        assert column_types['version_tag'] == 'TEXT'


class TestGeneratedColumns:
    """Test generated columns functionality."""
    
    def test_sru_key_generated_column(self, memory_db):
        """Test that sru_key is properly generated based on SRU flags."""
        results_schema(memory_db)
        
        # Insert test data with different SRU combinations
        test_cases = [
            # (is_undef_sru, is_def_sru, sru_repeat_count, expected_sru_key)
            (0, 0, None, 'none'),
            (1, 0, None, 'undef'),
            (0, 1, 5, 'def:5'),
            (0, 1, None, 'def:'),  # def with no count
        ]
        
        for i, (is_undef, is_def, repeat_count, expected_key) in enumerate(test_cases):
            memory_db.execute("""
                INSERT INTO clusters (
                    inchikey_first, identity_key_strict, is_undef_sru, 
                    is_def_sru, sru_repeat_count, member_count, members_hash
                )
                VALUES (?, ?, ?, ?, ?, 1, ?)
            """, (f'TEST{i:03d}INCHIKEY', f'identity_{i}', is_undef, is_def, repeat_count, f'hash_{i}'))
        
        # Verify generated sru_key values
        cursor = memory_db.execute("""
            SELECT is_undef_sru, is_def_sru, sru_repeat_count, sru_key
            FROM clusters ORDER BY cluster_id
        """)
        
        results = cursor.fetchall()
        assert len(results) == len(test_cases)
        
        for i, (is_undef, is_def, repeat_count, expected_key) in enumerate(test_cases):
            row = results[i]
            assert row[0] == is_undef
            assert row[1] == is_def
            assert row[2] == repeat_count
            assert row[3] == expected_key


class TestIndexCreation:
    """Test that all required indexes are created."""
    
    def test_all_indexes_created(self, memory_db):
        """Test that all expected indexes are created."""
        results_schema(memory_db)
        
        # Get all indexes
        cursor = memory_db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
        """)
        indexes = [row[0] for row in cursor.fetchall()]
        
        expected_indexes = [
            'idx_cpr_version',
            'idx_cpr_version_ab',
            'idx_rel_members',
            'ux_clusters_ifsmi_disc',
            'idx_ic_inchikey',
            'idx_ic_undef_sru',
            'idx_ic_def_sru'
        ]
        
        for expected_index in expected_indexes:
            assert expected_index in indexes, f"Missing index: {expected_index}"
    
    def test_relationships_indexes(self, memory_db):
        """Test relationships table indexes."""
        results_schema(memory_db)
        
        # Test version index
        cursor = memory_db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='index' AND name='idx_cpr_version'
        """)
        index_sql = cursor.fetchone()[0]
        assert 'relationships(version_tag)' in index_sql
        
        # Test composite version index
        cursor = memory_db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='index' AND name='idx_cpr_version_ab'
        """)
        index_sql = cursor.fetchone()[0]
        assert 'relationships(version_tag, cluster_a, cluster_b)' in index_sql
        
        # Test members index
        cursor = memory_db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='index' AND name='idx_rel_members'
        """)
        index_sql = cursor.fetchone()[0]
        assert 'relationships(cluster_a_members, cluster_b_members)' in index_sql
    
    def test_clusters_indexes(self, memory_db):
        """Test clusters table indexes."""
        results_schema(memory_db)
        
        # Test unique index
        cursor = memory_db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='index' AND name='ux_clusters_ifsmi_disc'
        """)
        index_sql = cursor.fetchone()[0]
        assert 'UNIQUE' in index_sql.upper()
        assert 'clusters(inchikey_first, identity_key_strict, sru_key)' in index_sql
        
        # Test inchikey index
        cursor = memory_db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='index' AND name='idx_ic_inchikey'
        """)
        index_sql = cursor.fetchone()[0]
        assert 'clusters(inchikey_first)' in index_sql
        
        # Test SRU indexes
        cursor = memory_db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='index' AND name='idx_ic_undef_sru'
        """)
        index_sql = cursor.fetchone()[0]
        assert 'clusters(is_undef_sru)' in index_sql
        
        cursor = memory_db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='index' AND name='idx_ic_def_sru'
        """)
        index_sql = cursor.fetchone()[0]
        assert 'clusters(is_def_sru, sru_repeat_count)' in index_sql


class TestConstraints:
    """Test database constraints and validation."""
    
    def test_clusters_unique_constraint(self, memory_db):
        """Test that clusters has unique constraint."""
        results_schema(memory_db)
        
        # Insert first record
        memory_db.execute("""
            INSERT INTO clusters (
                inchikey_first, identity_key_strict, is_undef_sru,
                is_def_sru, member_count, members_hash
            )
            VALUES ('TEST001INCHIKEY', 'identity_1', 0, 0, 1, 'hash_1')
        """)
        
        # Try to insert duplicate - should fail due to unique constraint
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            memory_db.execute("""
                INSERT INTO clusters (
                    inchikey_first, identity_key_strict, is_undef_sru,
                    is_def_sru, member_count, members_hash
                )
                VALUES ('TEST001INCHIKEY', 'identity_1', 0, 0, 2, 'hash_2')
            """)
        
        assert "UNIQUE constraint failed" in str(exc_info.value)
    
    def test_clusters_not_null_constraints(self, memory_db):
        """Test that NOT NULL constraints work on clusters table."""
        results_schema(memory_db)
        
        # Try to insert without required fields
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            memory_db.execute("""
                INSERT INTO clusters (inchikey_first) VALUES ('TEST001INCHIKEY')
            """)
        
        assert "NOT NULL constraint failed" in str(exc_info.value)
    
    def test_relationships_primary_key_constraint(self, memory_db):
        """Test relationships table primary key constraint."""
        results_schema(memory_db)
        
        # Insert first relationship
        memory_db.execute("""
            INSERT INTO relationships (
                cluster_a, cluster_b, cluster_a_members, cluster_b_members,
                cluster_a_size, cluster_b_size, classification, version_tag
            )
            VALUES (1, 2, 'mol1', 'mol2', 1, 1, 'similar', 'v1.0')
        """)
        
        # Try to insert duplicate primary key - should fail
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            memory_db.execute("""
                INSERT INTO relationships (
                    cluster_a, cluster_b, cluster_a_members, cluster_b_members,
                    cluster_a_size, cluster_b_size, classification, version_tag
                )
                VALUES (1, 2, 'mol3', 'mol4', 2, 2, 'different', 'v1.0')
            """)
        
        assert "UNIQUE constraint failed" in str(exc_info.value)
    
    def test_clusters_default_values(self, memory_db):
        """Test that clusters table has correct default values."""
        results_schema(memory_db)
        
        # Insert minimal record
        cursor = memory_db.execute("""
            INSERT INTO clusters (
                inchikey_first, identity_key_strict, member_count, members_hash
            )
            VALUES ('TEST001INCHIKEY', 'identity_1', 1, 'hash_1')
            RETURNING *
        """)
        row = cursor.fetchone()
        
        # Check defaults
        assert row[3] == 0  # is_undef_sru default
        assert row[4] == 0  # is_def_sru default
        assert row[6] == 'none'  # sru_key generated from defaults


class TestSchemaRecreation:
    """Test schema recreation when tables exist."""
    
    def test_partial_schema_recreation(self, memory_db):
        """Test that partial schema is dropped and recreated."""
        # Create only clusters table manually
        memory_db.execute("""
            CREATE TABLE clusters (
                cluster_id INTEGER PRIMARY KEY,
                old_column TEXT
            )
        """)
        
        # Insert some data
        memory_db.execute("INSERT INTO clusters (old_column) VALUES ('test')")
        
        # Apply schema - should drop and recreate
        results_schema(memory_db)
        
        # Verify old data is gone and new structure exists
        cursor = memory_db.execute("PRAGMA table_info(clusters)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        assert 'old_column' not in column_names
        assert 'inchikey_first' in column_names
        assert 'identity_key_strict' in column_names
        
        # Verify table is empty
        cursor = memory_db.execute("SELECT COUNT(*) FROM clusters")
        assert cursor.fetchone()[0] == 0
    
    def test_complete_schema_preservation(self, memory_db):
        """Test that complete schema is preserved."""
        # Apply schema first time
        results_schema(memory_db)
        
        # Insert test data
        memory_db.execute("""
            INSERT INTO clusters (
                inchikey_first, identity_key_strict, member_count, members_hash
            )
            VALUES ('TEST001INCHIKEY', 'identity_1', 1, 'hash_1')
        """)
        
        memory_db.execute("""
            INSERT INTO relationships (
                cluster_a, cluster_b, cluster_a_members, cluster_b_members,
                cluster_a_size, cluster_b_size, classification, version_tag
            )
            VALUES (1, 2, 'mol1', 'mol2', 1, 1, 'similar', 'v1.0')
        """)
        
        # Apply schema again - should not recreate since both tables exist
        results_schema(memory_db)
        
        # Verify data is preserved
        cursor = memory_db.execute("SELECT COUNT(*) FROM clusters")
        assert cursor.fetchone()[0] == 1
        
        cursor = memory_db.execute("SELECT COUNT(*) FROM relationships")
        assert cursor.fetchone()[0] == 1


class TestFunctionalOperations:
    """Test that the created schema supports expected operations."""
    
    def test_complete_workflow(self, memory_db):
        """Test a complete workflow using both tables."""
        results_schema(memory_db)
        
        # 1. Insert clusters
        cursor = memory_db.execute("""
            INSERT INTO clusters (
                inchikey_first, identity_key_strict, is_undef_sru, is_def_sru,
                sru_repeat_count, member_count, members_json, members_hash
            )
            VALUES ('LFQSCWFLJHTTHZ', 'ethanol_strict', 0, 0, NULL, 2, 
                    '["mol1", "mol2"]', 'cluster_hash_1')
            RETURNING cluster_id
        """)
        cluster_a_id = cursor.fetchone()[0]
        
        cursor = memory_db.execute("""
            INSERT INTO clusters (
                inchikey_first, identity_key_strict, is_undef_sru, is_def_sru,
                sru_repeat_count, member_count, members_json, members_hash
            )
            VALUES ('HGBOYTHUEUWSSQ', 'ethylamine_strict', 1, 0, NULL, 1,
                    '["mol3"]', 'cluster_hash_2')
            RETURNING cluster_id
        """)
        cluster_b_id = cursor.fetchone()[0]
        
        # 2. Insert relationship
        memory_db.execute("""
            INSERT INTO relationships (
                cluster_a, cluster_b, cluster_a_members, cluster_b_members,
                cluster_a_size, cluster_b_size, classification, score,
                score_details, version_tag
            )
            VALUES (?, ?, 'mol1,mol2', 'mol3', 2, 1, 'dissimilar', 0.1,
                    '{"tanimoto": 0.1, "method": "fingerprint"}', 'v1.0')
        """, (cluster_a_id, cluster_b_id))
        
        # 3. Verify complete query
        cursor = memory_db.execute("""
            SELECT 
                c1.inchikey_first as cluster_a_inchi,
                c1.sru_key as cluster_a_sru,
                c2.inchikey_first as cluster_b_inchi,
                c2.sru_key as cluster_b_sru,
                r.classification,
                r.score
            FROM relationships r
            JOIN clusters c1 ON r.cluster_a = c1.cluster_id
            JOIN clusters c2 ON r.cluster_b = c2.cluster_id
            WHERE r.version_tag = 'v1.0'
        """)
        
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 'LFQSCWFLJHTTHZ'  # cluster_a_inchi
        assert row[1] == 'none'            # cluster_a_sru
        assert row[2] == 'HGBOYTHUEUWSSQ'  # cluster_b_inchi
        assert row[3] == 'undef'           # cluster_b_sru
        assert row[4] == 'dissimilar'      # classification
        assert row[5] == 0.1               # score
    
    def test_index_performance_queries(self, memory_db):
        """Test that indexes work for performance queries."""
        results_schema(memory_db)
        
        # Insert test data
        for i in range(10):
            memory_db.execute("""
                INSERT INTO clusters (
                    inchikey_first, identity_key_strict, member_count, members_hash
                )
                VALUES (?, ?, 1, ?)
            """, (f'INCHI{i:03d}TESTKEY', f'identity_{i}', f'hash_{i}'))
        
        # Test inchikey index
        cursor = memory_db.execute("""
            SELECT cluster_id FROM clusters 
            WHERE inchikey_first = 'INCHI005TESTKEY'
        """)
        assert cursor.fetchone() is not None
        
        # Test unique constraint index
        cursor = memory_db.execute("""
            SELECT cluster_id FROM clusters 
            WHERE inchikey_first = 'INCHI003TESTKEY' 
            AND identity_key_strict = 'identity_3'
            AND sru_key = 'none'
        """)
        assert cursor.fetchone() is not None
        
        # Test SRU index
        cursor = memory_db.execute("""
            SELECT COUNT(*) FROM clusters WHERE is_undef_sru = 0
        """)
        assert cursor.fetchone()[0] == 10