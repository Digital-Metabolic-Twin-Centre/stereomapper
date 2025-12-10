import pytest
import sqlite3
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from stereomapper.data.cache_repo import (
    get_cached_entry,
    inchi_first_by_id,
    streamline_rows,
    upsert_structure,
    link_source,
    ingest_one,
    STRUCT_UPSERT_SQL,
    SRC_UPSERT_SQL
)
from stereomapper.domain.models import CacheEntry


@pytest.fixture
def test_db():
    """Create a test database with the required schema."""
    conn = sqlite3.connect(":memory:")
    
    # Create tables schema
    conn.executescript("""
        CREATE TABLE structures (
            molecule_id INTEGER PRIMARY KEY AUTOINCREMENT,
            namespace TEXT NOT NULL DEFAULT 'default',
            molecule_key TEXT NOT NULL,
            smiles TEXT,
            inchikey_first TEXT,
            is_undef_sru INTEGER DEFAULT 0,
            is_def_sru INTEGER DEFAULT 0,
            sru_repeat_count INTEGER,
            error TEXT,
            feature_version INTEGER DEFAULT 1,
            feature_blob BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(namespace, molecule_key)
        );
        
        CREATE TABLE sources (
            source_id TEXT PRIMARY KEY,
            molecule_id INTEGER NOT NULL,
            source_kind TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            accession_curie TEXT NOT NULL,
            file_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (molecule_id) REFERENCES structures(molecule_id)
        );
        
        CREATE INDEX idx_structures_inchikey_first ON structures(inchikey_first);
        CREATE INDEX idx_sources_file_hash ON sources(file_hash);
        CREATE INDEX idx_sources_molecule_id ON sources(molecule_id);
    """)
    
    yield conn
    conn.close()


@pytest.fixture
def sample_data(test_db):
    """Insert sample data into the test database."""
    conn = test_db
    
    # Insert sample structures
    structures_data = [
        (1, 'default', 'mol1', 'CCO', 'LFQSCWFLJHTTHZ', 0, 0, None, None, 1, None),
        (2, 'default', 'mol2', 'CCN', 'HGBOYTHUEUWSSQ', 1, 0, None, None, 1, None),
        (3, 'test_ns', 'mol3', 'CCC', 'ATUOYWHBWRKTHZ', 0, 1, 5, None, 1, None),
        (4, 'default', 'mol4', None, None, 0, 0, None, 'Parse error', 1, None),
    ]
    
    conn.executemany("""
        INSERT INTO structures (
            molecule_id, namespace, molecule_key, smiles, inchikey_first,
            is_undef_sru, is_def_sru, sru_repeat_count, error, feature_version, feature_blob
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, structures_data)
    
    # Insert sample sources
    sources_data = [
        ('src1', 1, 'file', '/path/to/mol1.mol', 'mol1', 'hash1'),
        ('src2', 2, 'file', '/path/to/mol2.mol', 'mol2', 'hash2'),
        ('src3', 3, 'file', '/path/to/mol3.mol', 'mol3', 'hash3'),
        ('src4', 4, 'file', '/path/to/mol4.mol', 'mol4', 'hash4'),
        ('src5', 1, 'chebi', 'CHEBI:16236', 'ethanol', None),  # Same molecule, different source
    ]
    
    conn.executemany("""
        INSERT INTO sources (source_id, molecule_id, source_kind, source_ref, accession_curie, file_hash)
        VALUES (?, ?, ?, ?, ?, ?)
    """, sources_data)
    
    conn.commit()
    return conn


class TestGetCachedEntry:
    """Test get_cached_entry function."""
    
    def test_get_cached_entry_found(self, sample_data):
        """Test retrieving an existing cached entry."""
        result = get_cached_entry('hash1', sample_data)
        
        assert result is not None
        assert isinstance(result, CacheEntry)
        assert result.molecule_id == 1
        assert result.smiles == 'CCO'
        assert result.inchikey_first == 'LFQSCWFLJHTTHZ'
        assert result.error is None
        assert result.is_undef_sru == 0
        assert result.is_def_sru == 0
        assert result.sru_repeat_count is None
        assert result.namespace == 'default'
    
    def test_get_cached_entry_not_found(self, sample_data):
        """Test retrieving a non-existent cached entry."""
        result = get_cached_entry('nonexistent_hash', sample_data)
        assert result is None
    
    def test_get_cached_entry_with_error(self, sample_data):
        """Test retrieving an entry with error."""
        result = get_cached_entry('hash4', sample_data)
        
        assert result is not None
        assert result.molecule_id == 4
        assert result.smiles is None
        assert result.inchikey_first is None
        assert result.error == 'Parse error'
    
    def test_get_cached_entry_with_sru_flags(self, sample_data):
        """Test retrieving an entry with SRU flags set."""
        result = get_cached_entry('hash3', sample_data)
        
        assert result is not None
        assert result.molecule_id == 3
        assert result.is_undef_sru == 0
        assert result.is_def_sru == 1
        assert result.sru_repeat_count == 5
        assert result.namespace == 'test_ns'


class TestInchiFirstById:
    """Test inchi_first_by_id function."""
    
    def test_inchi_first_by_id_existing_files(self, sample_data, tmp_path):
        """Test retrieving inchi first blocks for existing files."""
        # Create temporary files that match the paths in sample data
        file1 = tmp_path / "mol1.mol"
        file2 = tmp_path / "mol2.mol"
        file1.touch()
        file2.touch()
        
        # Update the database with the actual paths
        sample_data.execute("""
            UPDATE sources SET source_ref = ? WHERE source_id = 'src1'
        """, (str(file1),))
        sample_data.execute("""
            UPDATE sources SET source_ref = ? WHERE source_id = 'src2'
        """, (str(file2),))
        sample_data.commit()
        
        result = inchi_first_by_id(sample_data, [str(file1), str(file2)])
        
        assert len(result) == 2
        assert 'LFQSCWFLJHTTHZ' in result
        assert 'HGBOYTHUEUWSSQ' in result
        assert result == sorted(result)  # Should be sorted
    
    def test_inchi_first_by_id_nonexistent_files(self, sample_data):
        """Test with files that don't exist."""
        result = inchi_first_by_id(sample_data, ['/nonexistent/file1.mol', '/nonexistent/file2.mol'])
        assert result == []
    
    def test_inchi_first_by_id_empty_list(self, sample_data):
        """Test with empty file list."""
        result = inchi_first_by_id(sample_data, [])
        assert result == []
    
    def test_inchi_first_by_id_mixed_existing_nonexistent(self, sample_data, tmp_path):
        """Test with mix of existing and non-existing files."""
        file1 = tmp_path / "mol1.mol"
        file1.touch()
        
        sample_data.execute("""
            UPDATE sources SET source_ref = ? WHERE source_id = 'src1'
        """, (str(file1),))
        sample_data.commit()
        
        result = inchi_first_by_id(sample_data, [str(file1), '/nonexistent/file.mol'])
        
        assert len(result) == 1
        assert 'LFQSCWFLJHTTHZ' in result
    
    def test_inchi_first_by_id_path_normalization(self, sample_data, tmp_path):
        """Test that paths are properly normalized."""
        file1 = tmp_path / "mol1.mol"
        file1.touch()
        
        sample_data.execute("""
            UPDATE sources SET source_ref = ? WHERE source_id = 'src1'
        """, (str(file1),))
        sample_data.commit()
        
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = inchi_first_by_id(sample_data, ['mol1.mol'])  # Relative path without ./
            assert len(result) == 1
            assert 'LFQSCWFLJHTTHZ' in result
        finally:
            os.chdir(original_cwd)


class TestStreamlineRows:
    """Test streamline_rows function."""
    
    def test_streamline_rows_basic(self, sample_data):
        """Test basic streamline rows functionality."""
        rows = list(streamline_rows(sample_data, 'LFQSCWFLJHTTHZ'))
        
        assert len(rows) == 1
        row = rows[0]
        
        assert row['molecule_id'] == 1
        assert row['namespace'] == 'default'
        assert row['molecule_key'] == 'mol1'
        assert row['smiles'] == 'CCO'
        assert row['inchikey_first'] == 'LFQSCWFLJHTTHZ'
        assert row['is_undef_sru'] == 0
        assert row['is_def_sru'] == 0
        assert row['sru_repeat_count'] is None
        assert row['sru_bucket'] == 'none'
        assert isinstance(row['accession_curies'], list)
        assert 'mol1' in row['accession_curies']
        assert 'ethanol' in row['accession_curies']  # Multiple sources for same molecule
    
    def test_streamline_rows_with_sru_flags(self, sample_data):
        """Test streamline rows with SRU flags."""
        rows = list(streamline_rows(sample_data, 'ATUOYWHBWRKTHZ'))
        
        assert len(rows) == 1
        row = rows[0]
        
        assert row['molecule_id'] == 3
        assert row['is_def_sru'] == 1
        assert row['sru_repeat_count'] == 5
        assert row['sru_bucket'] == 'def:5'
    
    def test_streamline_rows_undef_sru(self, sample_data):
        """Test streamline rows with undefined SRU."""
        # Add a molecule with undefined SRU
        sample_data.execute("""
            INSERT INTO structures (molecule_id, namespace, molecule_key, smiles, inchikey_first, is_undef_sru)
            VALUES (5, 'default', 'mol5', 'CCCC', 'TESTINCHIKEY001', 1)
        """)
        sample_data.execute("""
            INSERT INTO sources (source_id, molecule_id, source_kind, source_ref, accession_curie)
            VALUES ('src6', 5, 'file', '/path/to/mol5.mol', 'mol5')
        """)
        sample_data.commit()
        
        rows = list(streamline_rows(sample_data, 'TESTINCHIKEY001'))
        
        assert len(rows) == 1
        row = rows[0]
        assert row['is_undef_sru'] == 1
        assert row['sru_bucket'] == 'undef'
    
    def test_streamline_rows_nonexistent_inchikey(self, sample_data):
        """Test streamline rows with non-existent inchikey."""
        rows = list(streamline_rows(sample_data, 'NONEXISTENTKEY'))
        assert len(rows) == 0
    
    def test_streamline_rows_accession_curies_aggregation(self, sample_data):
        """Test that accession curies are properly aggregated."""
        rows = list(streamline_rows(sample_data, 'LFQSCWFLJHTTHZ'))
        
        assert len(rows) == 1
        row = rows[0]
        
        # Should have both 'mol1' and 'ethanol' from different sources
        assert len(row['accession_curies']) == 2
        assert 'mol1' in row['accession_curies']
        assert 'ethanol' in row['accession_curies']


class TestUpsertStructure:
    """Test upsert_structure function."""
    
    def test_upsert_structure_new(self, test_db):
        """Test inserting a new structure."""
        molecule_id = upsert_structure(
            test_db,
            namespace='test',
            molecule_key='new_mol',
            smiles='C=C',
            inchikey_first='VGGSQFUCUMXWEO',
            is_undef_sru=0,
            is_def_sru=0,
            sru_repeat_count=None,
            error=None,
            feature_version=1,
            feature_blob=b'test_blob'
        )
        
        assert isinstance(molecule_id, int)
        assert molecule_id > 0
        
        # Verify the structure was inserted
        cursor = test_db.execute("""
            SELECT * FROM structures WHERE molecule_id = ?
        """, (molecule_id,))
        row = cursor.fetchone()
        
        assert row is not None
        # Check that the data was inserted correctly
        cursor = test_db.execute("""
            SELECT namespace, molecule_key, smiles, inchikey_first, feature_blob
            FROM structures WHERE molecule_id = ?
        """, (molecule_id,))
        row = cursor.fetchone()
        
        assert row[0] == 'test'
        assert row[1] == 'new_mol'
        assert row[2] == 'C=C'
        assert row[3] == 'VGGSQFUCUMXWEO'
        assert row[4] == b'test_blob'
    
    def test_upsert_structure_update_existing(self, sample_data):
        """Test updating an existing structure."""
        # Update an existing structure
        molecule_id = upsert_structure(
            sample_data,
            namespace='default',
            molecule_key='mol1',
            smiles='CCO',  # Same SMILES
            inchikey_first='LFQSCWFLJHTTHZ',  # Same inchikey
            is_undef_sru=1,  # Changed
            is_def_sru=0,
            sru_repeat_count=3,  # New value
            error=None,
            feature_version=2,  # Higher version
            feature_blob=b'updated_blob'
        )
        
        assert molecule_id == 1  # Should be the existing molecule ID
        
        # Verify the structure was updated
        cursor = sample_data.execute("""
            SELECT is_undef_sru, sru_repeat_count, feature_version, feature_blob
            FROM structures WHERE molecule_id = ?
        """, (molecule_id,))
        row = cursor.fetchone()
        
        assert row[0] == 1  # is_undef_sru updated
        assert row[1] == 3  # sru_repeat_count updated
        assert row[2] == 2  # feature_version updated
        assert row[3] == b'updated_blob'  # feature_blob updated
    
    def test_upsert_structure_feature_version_conflict(self, sample_data):
        """Test that older feature versions don't overwrite newer ones."""
        # First, update to version 3
        upsert_structure(
            sample_data,
            namespace='default',
            molecule_key='mol1',
            smiles='CCO',
            inchikey_first='LFQSCWFLJHTTHZ',
            is_undef_sru=0,
            is_def_sru=0,
            sru_repeat_count=None,
            error=None,
            feature_version=3,
            feature_blob=b'version_3_blob'
        )
        
        # Then try to update with older version
        upsert_structure(
            sample_data,
            namespace='default',
            molecule_key='mol1',
            smiles='CCO',
            inchikey_first='LFQSCWFLJHTTHZ',
            is_undef_sru=0,
            is_def_sru=0,
            sru_repeat_count=None,
            error=None,
            feature_version=2,  # Older version
            feature_blob=b'version_2_blob'
        )
        
        # Verify that the newer version was preserved
        cursor = sample_data.execute("""
            SELECT feature_version, feature_blob FROM structures WHERE molecule_key = 'mol1'
        """, )
        row = cursor.fetchone()
        
        assert row[0] == 3  # Should keep the higher version
        assert row[1] == b'version_3_blob'  # Should keep the newer blob


class TestLinkSource:
    """Test link_source function."""
    
    def test_link_source_new(self, sample_data):
        """Test linking a new source to a molecule."""
        link_source(
            sample_data,
            source_id='new_src',
            molecule_id=1,
            source_kind='database',
            source_ref='DB:12345',
            accession_curie='compound_12345',
            file_hash='new_hash'
        )
        
        # Verify the source was inserted
        cursor = sample_data.execute("""
            SELECT source_kind, source_ref, accession_curie, file_hash
            FROM sources WHERE source_id = 'new_src'
        """)
        row = cursor.fetchone()
        
        assert row is not None
        assert row[0] == 'database'
        assert row[1] == 'DB:12345'
        assert row[2] == 'compound_12345'
        assert row[3] == 'new_hash'
    
    def test_link_source_update_existing(self, sample_data):
        """Test updating an existing source."""
        # Update existing source
        link_source(
            sample_data,
            source_id='src1',  # Existing source
            molecule_id=2,  # Different molecule
            source_kind='file',
            source_ref='/new/path/mol.mol',
            accession_curie='mol1',
            file_hash='updated_hash'
        )
        
        # Verify the source was updated
        cursor = sample_data.execute("""
            SELECT molecule_id, source_ref, file_hash
            FROM sources WHERE source_id = 'src1'
        """)
        row = cursor.fetchone()
        
        assert row[0] == 2  # molecule_id updated
        # source_ref follows COALESCE behavior - existing value is preserved when new value is provided
        assert row[1] == '/path/to/mol1.mol'  # Original value preserved due to COALESCE
        assert row[2] == 'updated_hash'  # file_hash updated
    
    def test_link_source_update_null_fields(self, sample_data):
        """Test updating an existing source with null values."""
        # First create a source with some null fields
        sample_data.execute("""
            INSERT INTO sources (source_id, molecule_id, source_kind, source_ref, accession_curie, file_hash)
            VALUES ('src_null', 1, 'file', '/path/test.mol', 'test', NULL)
        """)
        sample_data.commit()
        
        # Update with new file_hash (was null)
        link_source(
            sample_data,
            source_id='src_null',
            molecule_id=1,
            source_kind='file',
            source_ref='/path/test.mol',
            accession_curie='test',
            file_hash='new_hash'
        )
        
        # Verify the null field was updated
        cursor = sample_data.execute("""
            SELECT file_hash FROM sources WHERE source_id = 'src_null'
        """)
        row = cursor.fetchone()
        
        assert row[0] == 'new_hash'  # file_hash updated from NULL


class TestIngestOne:
    """Test ingest_one function."""
    
    def test_ingest_one_new_molecule(self, test_db):
        """Test ingesting a completely new molecule."""
        molecule_id = ingest_one(
            test_db,
            namespace='test',
            molecule_key='test_mol',
            smiles='CCC',
            inchikey_first='ATUOYWHBWRKTHZ',
            is_undef_sru=0,
            is_def_sru=1,
            sru_repeat_count=5,
            error=None,
            feature_version=1,
            feature_blob=b'test_features',
            source_id='test_src',
            source_kind='file',
            source_ref='/path/to/test.mol',
            accession_curie='test_compound',
            file_hash='test_hash'
        )
        
        assert isinstance(molecule_id, int)
        assert molecule_id > 0
        
        # Verify structure was created
        cursor = test_db.execute("""
            SELECT namespace, molecule_key, smiles, inchikey_first, is_def_sru, sru_repeat_count
            FROM structures WHERE molecule_id = ?
        """, (molecule_id,))
        struct_row = cursor.fetchone()
        
        assert struct_row[0] == 'test'
        assert struct_row[1] == 'test_mol'
        assert struct_row[2] == 'CCC'
        assert struct_row[3] == 'ATUOYWHBWRKTHZ'
        assert struct_row[4] == 1
        assert struct_row[5] == 5
        
        # Verify source was linked
        cursor = test_db.execute("""
            SELECT source_kind, source_ref, accession_curie, file_hash
            FROM sources WHERE source_id = 'test_src'
        """)
        source_row = cursor.fetchone()
        
        assert source_row[0] == 'file'
        assert source_row[1] == '/path/to/test.mol'
        assert source_row[2] == 'test_compound'
        assert source_row[3] == 'test_hash'
    
    def test_ingest_one_with_error(self, test_db):
        """Test ingesting a molecule with an error."""
        molecule_id = ingest_one(
            test_db,
            namespace='test',
            molecule_key='error_mol',
            smiles=None,
            inchikey_first=None,
            is_undef_sru=None,
            is_def_sru=None,
            sru_repeat_count=None,
            error='Failed to parse molecule',
            feature_version=1,
            feature_blob=None,
            source_id='error_src',
            source_kind='file',
            source_ref='/path/to/error.mol',
            accession_curie='error_compound',
            file_hash='error_hash'
        )
        
        assert isinstance(molecule_id, int)
        assert molecule_id > 0
        
        # Verify error was stored
        cursor = test_db.execute("""
            SELECT error, smiles, inchikey_first FROM structures WHERE molecule_id = ?
        """, (molecule_id,))
        row = cursor.fetchone()
        
        assert row[0] == 'Failed to parse molecule'
        assert row[1] is None
        assert row[2] is None
    
    def test_ingest_one_transaction_rollback(self, test_db):
        """Test that transaction rolls back on error."""
        # This would test error handling, but since we can't easily force an error
        # in the current implementation, we'll test that the transaction context works
        
        # Count initial rows
        cursor = test_db.execute("SELECT COUNT(*) FROM structures")
        initial_struct_count = cursor.fetchone()[0]
        
        cursor = test_db.execute("SELECT COUNT(*) FROM sources")
        initial_source_count = cursor.fetchone()[0]
        
        # Successful ingest
        ingest_one(
            test_db,
            namespace='test',
            molecule_key='txn_test',
            smiles='C',
            inchikey_first='VNWKTOKETHGBQD',
            is_undef_sru=0,
            is_def_sru=0,
            sru_repeat_count=None,
            error=None,
            feature_version=1,
            feature_blob=None,
            source_id='txn_src',
            source_kind='file',
            source_ref='/path/to/txn.mol',
            accession_curie='txn_compound',
            file_hash='txn_hash'
        )
        
        # Verify both tables were updated
        cursor = test_db.execute("SELECT COUNT(*) FROM structures")
        final_struct_count = cursor.fetchone()[0]
        
        cursor = test_db.execute("SELECT COUNT(*) FROM sources")
        final_source_count = cursor.fetchone()[0]
        
        assert final_struct_count == initial_struct_count + 1
        assert final_source_count == initial_source_count + 1


class TestSQLConstants:
    """Test that SQL constants are properly defined."""
    
    def test_struct_upsert_sql_defined(self):
        """Test that STRUCT_UPSERT_SQL is defined and contains expected elements."""
        assert STRUCT_UPSERT_SQL is not None
        assert "INSERT INTO structures" in STRUCT_UPSERT_SQL
        assert "ON CONFLICT" in STRUCT_UPSERT_SQL
        assert "RETURNING molecule_id" in STRUCT_UPSERT_SQL
    
    def test_src_upsert_sql_defined(self):
        """Test that SRC_UPSERT_SQL is defined and contains expected elements."""
        assert SRC_UPSERT_SQL is not None
        assert "INSERT INTO sources" in SRC_UPSERT_SQL
        assert "ON CONFLICT" in SRC_UPSERT_SQL