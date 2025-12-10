import pytest
import sqlite3
import tempfile
from pathlib import Path

from stereomapper.data.cache_schema import create_cache


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


class TestCreateCache:
    """Test create_cache function."""
    
    def test_create_cache_returns_connection(self, memory_db):
        """Test that create_cache returns the same connection object."""
        result = create_cache(memory_db)
        assert result is memory_db
        assert isinstance(result, sqlite3.Connection)
    
    def test_create_cache_with_file_db(self, temp_file_db):
        """Test create_cache works with file-based database."""
        conn, tmp_path = temp_file_db
        result = create_cache(conn)
        
        assert result is conn
        assert tmp_path.exists()
        assert tmp_path.stat().st_size > 0  # File should have content
    
    def test_create_cache_idempotent(self, memory_db):
        """Test that calling create_cache multiple times is safe."""
        # First call
        result1 = create_cache(memory_db)
        
        # Second call should not raise an error
        result2 = create_cache(memory_db)
        
        assert result1 is result2
        assert result1 is memory_db


class TestTableCreation:
    """Test that all required tables are created."""
    
    def test_meta_table_created(self, memory_db):
        """Test that meta table is created with correct structure."""
        create_cache(memory_db)
        
        # Check table exists
        cursor = memory_db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='meta'
        """)
        assert cursor.fetchone() is not None
        
        # Check table structure
        cursor = memory_db.execute("PRAGMA table_info(meta)")
        columns = cursor.fetchall()
        
        expected_columns = [
            ('key', 'TEXT', True),    # (name, type, primary_key)
            ('value', 'TEXT', False)
        ]
        
        assert len(columns) == 2
        assert columns[0][1] == 'key'      # column name
        assert columns[0][2] == 'TEXT'     # column type
        assert columns[0][5] == 1          # primary key
        assert columns[1][1] == 'value'
        assert columns[1][2] == 'TEXT'
        assert columns[1][5] == 0          # not primary key
    
    def test_structures_table_created(self, memory_db):
        """Test that structures table is created with correct structure."""
        create_cache(memory_db)
        
        # Check table exists
        cursor = memory_db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='structures'
        """)
        assert cursor.fetchone() is not None
        
        # Check table structure
        cursor = memory_db.execute("PRAGMA table_info(structures)")
        columns = cursor.fetchall()
        
        expected_column_names = [
            'molecule_id', 'namespace', 'molecule_key', 'smiles', 'inchikey_first',
            'is_undef_sru', 'is_def_sru', 'sru_repeat_count', 'error', 
            'feature_version', 'feature_blob', 'created_at', 'updated_at'
        ]
        
        actual_column_names = [col[1] for col in columns]
        assert actual_column_names == expected_column_names
        
        # Check primary key
        primary_key_columns = [col[1] for col in columns if col[5] == 1]
        assert primary_key_columns == ['molecule_id']
        
        # Check specific column types
        column_types = {col[1]: col[2] for col in columns}
        assert column_types['molecule_id'] == 'INTEGER'
        assert column_types['namespace'] == 'TEXT'
        assert column_types['smiles'] == 'TEXT'
        assert column_types['feature_blob'] == 'BLOB'
    
    def test_sessions_table_created(self, memory_db):
        """Test that sessions table is created with correct structure."""
        create_cache(memory_db)
        
        cursor = memory_db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='sessions'
        """)
        assert cursor.fetchone() is not None
        
        cursor = memory_db.execute("PRAGMA table_info(sessions)")
        columns = cursor.fetchall()
        
        expected_column_names = ['session_id', 'namespace', 'started_at', 'args_json']
        actual_column_names = [col[1] for col in columns]
        assert actual_column_names == expected_column_names
        
        # Check primary key
        primary_key_columns = [col[1] for col in columns if col[5] == 1]
        assert primary_key_columns == ['session_id']
    
    def test_session_members_table_created(self, memory_db):
        """Test that session_members table is created with correct structure."""
        create_cache(memory_db)
        
        cursor = memory_db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='session_members'
        """)
        assert cursor.fetchone() is not None
        
        cursor = memory_db.execute("PRAGMA table_info(session_members)")
        columns = cursor.fetchall()
        
        expected_column_names = ['session_id', 'molecule_id']
        actual_column_names = [col[1] for col in columns]
        assert actual_column_names == expected_column_names
        
        # Check composite primary key
        primary_key_columns = [col[1] for col in columns if col[5] > 0]
        assert set(primary_key_columns) == {'session_id', 'molecule_id'}
    
    def test_sources_table_created(self, memory_db):
        """Test that sources table is created with correct structure."""
        create_cache(memory_db)
        
        cursor = memory_db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='sources'
        """)
        assert cursor.fetchone() is not None
        
        cursor = memory_db.execute("PRAGMA table_info(sources)")
        columns = cursor.fetchall()
        
        expected_column_names = [
            'source_id', 'molecule_id', 'source_kind', 'source_ref', 
            'accession_curie', 'file_hash', 'created_at'
        ]
        actual_column_names = [col[1] for col in columns]
        assert actual_column_names == expected_column_names
        
        # Check primary key
        primary_key_columns = [col[1] for col in columns if col[5] == 1]
        assert primary_key_columns == ['source_id']


class TestIndexCreation:
    """Test that all required indexes are created."""
    
    def test_all_indexes_created(self, memory_db):
        """Test that all expected indexes are created."""
        create_cache(memory_db)
        
        # Get all indexes
        cursor = memory_db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
        """)
        indexes = [row[0] for row in cursor.fetchall()]
        
        expected_indexes = [
            'idx_structures_ns_key',
            'idx_structures_inchikey',
            'idx_structures_smiles',
            'idx_session_members',
            'idx_sources_molid'
        ]
        
        for expected_index in expected_indexes:
            assert expected_index in indexes, f"Missing index: {expected_index}"
    
    def test_structures_namespace_key_index(self, memory_db):
        """Test the structures namespace+key index."""
        create_cache(memory_db)
        
        cursor = memory_db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='index' AND name='idx_structures_ns_key'
        """)
        index_sql = cursor.fetchone()[0]
        
        assert 'structures(namespace, molecule_key)' in index_sql
    
    def test_structures_inchikey_index(self, memory_db):
        """Test the structures inchikey index."""
        create_cache(memory_db)
        
        cursor = memory_db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='index' AND name='idx_structures_inchikey'
        """)
        index_sql = cursor.fetchone()[0]
        
        assert 'structures(inchikey_first)' in index_sql
    
    def test_structures_smiles_index(self, memory_db):
        """Test the structures smiles index."""
        create_cache(memory_db)
        
        cursor = memory_db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='index' AND name='idx_structures_smiles'
        """)
        index_sql = cursor.fetchone()[0]
        
        assert 'structures(smiles)' in index_sql
    
    def test_session_members_index(self, memory_db):
        """Test the session_members index."""
        create_cache(memory_db)
        
        cursor = memory_db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='index' AND name='idx_session_members'
        """)
        index_sql = cursor.fetchone()[0]
        
        assert 'session_members(session_id, molecule_id)' in index_sql
    
    def test_sources_molecule_id_index(self, memory_db):
        """Test the sources molecule_id index."""
        create_cache(memory_db)
        
        cursor = memory_db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='index' AND name='idx_sources_molid'
        """)
        index_sql = cursor.fetchone()[0]
        
        assert 'sources(molecule_id)' in index_sql


class TestConstraints:
    """Test database constraints and defaults."""
    
    def test_structures_unique_constraint(self, memory_db):
        """Test that structures has unique constraint on namespace+molecule_key."""
        create_cache(memory_db)
        
        # Insert first record
        memory_db.execute("""
            INSERT INTO structures (namespace, molecule_key, smiles)
            VALUES ('test', 'mol1', 'CCO')
        """)
        
        # Try to insert duplicate - should fail
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            memory_db.execute("""
                INSERT INTO structures (namespace, molecule_key, smiles)
                VALUES ('test', 'mol1', 'CCC')
            """)
        
        assert "UNIQUE constraint failed" in str(exc_info.value)
    
    def test_structures_default_values(self, memory_db):
        """Test that structures table has correct default values."""
        create_cache(memory_db)
        
        # Insert minimal record
        cursor = memory_db.execute("""
            INSERT INTO structures (molecule_key)
            VALUES ('test_mol')
            RETURNING *
        """)
        row = cursor.fetchone()
        
        # Check defaults
        assert row[1] == 'default'  # namespace default
        assert row[5] == 0          # is_undef_sru default
        assert row[6] == 0          # is_def_sru default
        assert row[9] == 1          # feature_version default
        assert row[11] is not None  # created_at should be set
        assert row[12] is not None  # updated_at should be set
    
    def test_sru_check_constraints(self, memory_db):
        """Test that SRU fields have proper check constraints."""
        create_cache(memory_db)
        
        # Valid values should work
        memory_db.execute("""
            INSERT INTO structures (molecule_key, is_undef_sru, is_def_sru)
            VALUES ('test1', 0, 1)
        """)
        
        memory_db.execute("""
            INSERT INTO structures (molecule_key, is_undef_sru, is_def_sru)
            VALUES ('test2', 1, 0)
        """)
        
        # Invalid values should fail
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            memory_db.execute("""
                INSERT INTO structures (molecule_key, is_undef_sru)
                VALUES ('test3', 2)
            """)
        assert "CHECK constraint failed" in str(exc_info.value)
        
        with pytest.raises(sqlite3.IntegrityError):
            memory_db.execute("""
                INSERT INTO structures (molecule_key, is_def_sru)
                VALUES ('test4', -1)
            """)
    
    def test_foreign_key_constraints(self, memory_db):
        """Test foreign key constraints are properly defined."""
        create_cache(memory_db)
        
        # Enable foreign key enforcement
        memory_db.execute("PRAGMA foreign_keys = ON")
        
        # Insert a structure first
        cursor = memory_db.execute("""
            INSERT INTO structures (molecule_key) VALUES ('test_mol')
            RETURNING molecule_id
        """)
        molecule_id = cursor.fetchone()[0]
        
        # Insert a session
        memory_db.execute("""
            INSERT INTO sessions (session_id, namespace) VALUES ('sess1', 'test')
        """)
        
        # Valid foreign key references should work
        memory_db.execute("""
            INSERT INTO sources (source_id, molecule_id, source_kind, source_ref, accession_curie)
            VALUES ('src1', ?, 'file', '/path/test.mol', 'test')
        """, (molecule_id,))
        
        memory_db.execute("""
            INSERT INTO session_members (session_id, molecule_id)
            VALUES ('sess1', ?)
        """, (molecule_id,))
        
        # Invalid foreign key should fail
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            memory_db.execute("""
                INSERT INTO sources (source_id, molecule_id, source_kind, source_ref, accession_curie)
                VALUES ('src2', 99999, 'file', '/path/test.mol', 'test')
            """)
        assert "FOREIGN KEY constraint failed" in str(exc_info.value)


class TestFunctionalOperations:
    """Test that the created schema supports expected operations."""
    
    def test_complete_workflow(self, memory_db):
        """Test a complete workflow using all tables."""
        create_cache(memory_db)
        
        # 1. Insert a structure
        cursor = memory_db.execute("""
            INSERT INTO structures (
                namespace, molecule_key, smiles, inchikey_first, 
                is_undef_sru, is_def_sru, feature_version
            )
            VALUES ('test', 'ethanol', 'CCO', 'LFQSCWFLJHTTHZ', 0, 0, 1)
            RETURNING molecule_id
        """)
        molecule_id = cursor.fetchone()[0]
        
        # 2. Create a session
        memory_db.execute("""
            INSERT INTO sessions (session_id, namespace, args_json)
            VALUES ('session_001', 'test', '{"input_dir": "/path/to/mols"}')
        """)
        
        # 3. Add source information
        memory_db.execute("""
            INSERT INTO sources (
                source_id, molecule_id, source_kind, source_ref, 
                accession_curie, file_hash
            )
            VALUES ('src_001', ?, 'file', '/path/to/ethanol.mol', 'ethanol', 'abc123')
        """, (molecule_id,))
        
        # 4. Link molecule to session
        memory_db.execute("""
            INSERT INTO session_members (session_id, molecule_id)
            VALUES ('session_001', ?)
        """, (molecule_id,))
        
        # 5. Add metadata
        memory_db.execute("""
            INSERT INTO meta (key, value) VALUES ('schema_version', '1')
        """)
        
        # 6. Verify the complete record
        cursor = memory_db.execute("""
            SELECT 
                s.molecule_key, s.smiles, s.inchikey_first,
                src.source_kind, src.source_ref, src.file_hash,
                sess.session_id, sess.namespace,
                m.value as schema_version
            FROM structures s
            JOIN sources src ON s.molecule_id = src.molecule_id
            JOIN session_members sm ON s.molecule_id = sm.molecule_id
            JOIN sessions sess ON sm.session_id = sess.session_id
            JOIN meta m ON m.key = 'schema_version'
            WHERE s.molecule_key = 'ethanol'
        """)
        
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 'ethanol'      # molecule_key
        assert row[1] == 'CCO'          # smiles
        assert row[2] == 'LFQSCWFLJHTTHZ'  # inchikey_first
        assert row[3] == 'file'         # source_kind
        assert row[4] == '/path/to/ethanol.mol'  # source_ref
        assert row[5] == 'abc123'       # file_hash
        assert row[6] == 'session_001'  # session_id
        assert row[7] == 'test'         # namespace
        assert row[8] == '1'            # schema_version
    
    def test_index_performance_query(self, memory_db):
        """Test that indexes work for performance queries."""
        create_cache(memory_db)
        
        # Insert test data
        test_data = [
            ('mol1', 'CCO', 'LFQSCWFLJHTTHZ'),
            ('mol2', 'CCN', 'HGBOYTHUEUWSSQ'),
            ('mol3', 'CCC', 'ATUOYWHBWRKTHZ'),
        ]
        
        for mol_key, smiles, inchikey in test_data:
            memory_db.execute("""
                INSERT INTO structures (molecule_key, smiles, inchikey_first)
                VALUES (?, ?, ?)
            """, (mol_key, smiles, inchikey))
        
        # Test index-based queries
        
        # Query by inchikey (should use idx_structures_inchikey)
        cursor = memory_db.execute("""
            SELECT molecule_key FROM structures 
            WHERE inchikey_first = 'LFQSCWFLJHTTHZ'
        """)
        assert cursor.fetchone()[0] == 'mol1'
        
        # Query by namespace+key (should use idx_structures_ns_key)
        cursor = memory_db.execute("""
            SELECT smiles FROM structures 
            WHERE namespace = 'default' AND molecule_key = 'mol2'
        """)
        assert cursor.fetchone()[0] == 'CCN'
        
        # Query by smiles (should use idx_structures_smiles)
        cursor = memory_db.execute("""
            SELECT molecule_key FROM structures 
            WHERE smiles = 'CCC'
        """)
        assert cursor.fetchone()[0] == 'mol3'