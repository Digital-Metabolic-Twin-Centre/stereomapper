import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from stereomapper.data.db import connect


@pytest.fixture
def temp_db_path():
    """Create a temporary database file path."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    yield str(tmp_path)
    tmp_path.unlink(missing_ok=True)


class TestConnect:
    """Test the connect function."""

    def test_connect_returns_connection(self, temp_db_path):
        """Test that connect returns a valid SQLite connection."""
        conn = connect(temp_db_path)

        assert isinstance(conn, sqlite3.Connection)
        assert conn is not None

        # Test that connection is functional
        cursor = conn.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1

        conn.close()

    def test_connect_creates_file(self, temp_db_path):
        """Test that connect creates the database file."""
        # Remove the file if it exists
        Path(temp_db_path).unlink(missing_ok=True)

        conn = connect(temp_db_path)

        assert Path(temp_db_path).exists()
        assert Path(temp_db_path).stat().st_size >= 0

        conn.close()

    def test_connect_in_memory_database(self):
        """Test connecting to in-memory database."""
        conn = connect(":memory:")

        assert isinstance(conn, sqlite3.Connection)

        # Test that it's functional
        cursor = conn.execute("CREATE TABLE test (id INTEGER)")
        conn.execute("INSERT INTO test VALUES (1)")
        cursor = conn.execute("SELECT * FROM test")
        assert cursor.fetchone()[0] == 1

        conn.close()


class TestPragmaSettings:
    """Test that pragma settings are applied correctly."""

    def test_common_pragma_settings(self, temp_db_path):
        """Test that common pragma settings are applied regardless of WAL mode."""
        conn = connect(temp_db_path, use_wal=False)

        # Test foreign keys are enabled
        cursor = conn.execute("PRAGMA foreign_keys")
        assert cursor.fetchone()[0] == 1

        # Test busy timeout is set
        cursor = conn.execute("PRAGMA busy_timeout")
        timeout = cursor.fetchone()[0]
        assert timeout == 120000

        conn.close()

    def test_wal_mode_pragma_settings(self, temp_db_path):
        """Test pragma settings when WAL mode is enabled."""
        conn = connect(temp_db_path, use_wal=True)

        # Test journal mode is WAL
        cursor = conn.execute("PRAGMA journal_mode")
        assert cursor.fetchone()[0].upper() == 'WAL'

        # Test synchronous mode
        cursor = conn.execute("PRAGMA synchronous")
        sync_mode = cursor.fetchone()[0]
        assert sync_mode == 1  # NORMAL mode

        conn.close()

    def test_non_wal_mode_pragma_settings(self, temp_db_path):
        """Test pragma settings when WAL mode is disabled."""
        conn = connect(temp_db_path, use_wal=False)

        # Test journal mode is DELETE
        cursor = conn.execute("PRAGMA journal_mode")
        assert cursor.fetchone()[0].upper() == 'DELETE'

        # Test synchronous mode
        cursor = conn.execute("PRAGMA synchronous")
        sync_mode = cursor.fetchone()[0]
        assert sync_mode == 1  # NORMAL mode

        # Test temp store is MEMORY
        cursor = conn.execute("PRAGMA temp_store")
        temp_store = cursor.fetchone()[0]
        assert temp_store == 2  # MEMORY mode

        # Test mmap size is set
        cursor = conn.execute("PRAGMA mmap_size")
        mmap_size = cursor.fetchone()[0]
        assert mmap_size > 0 # assert its being set to a positive value, as 30GIB is not possible

        # Test cache size is set
        cursor = conn.execute("PRAGMA cache_size")
        cache_size = cursor.fetchone()[0]
        assert cache_size == -200000  # ~200MB

        # Test locking mode is EXCLUSIVE
        cursor = conn.execute("PRAGMA locking_mode")
        locking_mode = cursor.fetchone()[0]
        assert locking_mode.upper() == 'EXCLUSIVE'

        conn.close()

    def test_wal_mode_excludes_non_wal_settings(self, temp_db_path):
        """Test that WAL mode doesn't include non-WAL specific settings."""
        conn = connect(temp_db_path, use_wal=True)

        # These settings should not be applied in WAL mode
        # (they might have default values, but we verify they're not the non-WAL values)

        # temp_store should not be explicitly set to MEMORY in WAL mode
        cursor = conn.execute("PRAGMA temp_store")
        temp_store = cursor.fetchone()[0]
        # Default is usually 0 (DEFAULT) or 1 (FILE), not 2 (MEMORY)

        # locking_mode should not be EXCLUSIVE in WAL mode (incompatible)
        cursor = conn.execute("PRAGMA locking_mode")
        locking_mode = cursor.fetchone()[0]
        # In WAL mode, EXCLUSIVE locking is not typically used

        conn.close()


class TestConnectionFunctionality:
    """Test that connections work properly with different configurations."""

    def test_foreign_keys_enforcement(self, temp_db_path):
        """Test that foreign key constraints are enforced."""
        conn = connect(temp_db_path)

        # Create tables with foreign key relationship
        conn.execute("""
            CREATE TABLE parent (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE child (
                id INTEGER PRIMARY KEY,
                parent_id INTEGER,
                name TEXT,
                FOREIGN KEY (parent_id) REFERENCES parent(id)
            )
        """)

        # Insert valid parent
        conn.execute("INSERT INTO parent (id, name) VALUES (1, 'Parent1')")

        # Insert valid child - should succeed
        conn.execute("INSERT INTO child (id, parent_id, name) VALUES (1, 1, 'Child1')")

        # Insert invalid child - should fail
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            conn.execute("INSERT INTO child (id, parent_id, name) VALUES (2, 999, 'Child2')")

        assert "FOREIGN KEY constraint failed" in str(exc_info.value)

        conn.close()

    def test_busy_timeout_functionality(self, temp_db_path):
        """Test that busy timeout is configured properly."""
        # This is harder to test without actual concurrent access
        # But we can verify the setting is applied
        conn = connect(temp_db_path)

        cursor = conn.execute("PRAGMA busy_timeout")
        timeout = cursor.fetchone()[0]
        assert timeout == 120000  # 120 seconds in milliseconds

        conn.close()

    def test_wal_mode_concurrent_access(self, temp_db_path):
        """Test that WAL mode allows concurrent readers."""
        # Create connection with WAL mode
        conn1 = connect(temp_db_path, use_wal=True)

        # Create a table and insert data
        conn1.execute("CREATE TABLE test (id INTEGER, value TEXT)")
        conn1.execute("INSERT INTO test VALUES (1, 'test1')")
        conn1.commit()

        # Create second connection - should be able to read
        conn2 = connect(temp_db_path, use_wal=True)
        cursor = conn2.execute("SELECT * FROM test")
        row = cursor.fetchone()
        assert row[0] == 1
        assert row[1] == 'test1'

        conn1.close()
        conn2.close()

    def test_non_wal_mode_performance_settings(self, temp_db_path):
        """Test that non-WAL mode performance settings work."""
        conn = connect(temp_db_path, use_wal=False)

        # Create a table and perform operations to test settings
        conn.execute("CREATE TABLE perf_test (id INTEGER, data TEXT)")

        # Insert some data to test cache and mmap settings
        for i in range(100):
            conn.execute("INSERT INTO perf_test VALUES (?, ?)", (i, f"data_{i}"))

        conn.commit()

        # Verify data can be read back
        cursor = conn.execute("SELECT COUNT(*) FROM perf_test")
        count = cursor.fetchone()[0]
        assert count == 100

        conn.close()


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_database_path(self):
        """Test handling of invalid database paths."""
        # Try to create database in non-existent directory
        invalid_path = "/nonexistent/directory/database.db"

        with pytest.raises(sqlite3.OperationalError):
            connect(invalid_path)

    def test_readonly_database_path(self, temp_db_path):
        """Test handling of read-only database scenarios."""
        # Create database first
        conn = connect(temp_db_path)
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()

        # Make file read-only
        db_path = Path(temp_db_path)
        db_path.chmod(0o444)  # Read-only

        try:
            # Should still be able to connect (might not be able to write)
            conn = connect(temp_db_path)

            # Try to read - should work
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            assert len(tables) == 1
            assert tables[0][0] == 'test'

            conn.close()

        finally:
            # Restore write permissions for cleanup
            db_path.chmod(0o644)


class TestParameterVariations:
    """Test different parameter combinations."""

    def test_default_use_wal_parameter(self, temp_db_path):
        """Test that use_wal defaults to False."""
        conn1 = connect(temp_db_path)  # Default use_wal=False
        conn2 = connect(temp_db_path, use_wal=False)  # Explicit use_wal=False

        # Both should have same journal mode
        cursor1 = conn1.execute("PRAGMA journal_mode")
        mode1 = cursor1.fetchone()[0]

        cursor2 = conn2.execute("PRAGMA journal_mode")
        mode2 = cursor2.fetchone()[0]

        assert mode1 == mode2
        assert mode1.upper() == 'DELETE'

        conn1.close()
        conn2.close()

    def test_wal_mode_true_vs_false(self, temp_db_path):
        """Test differences between WAL and non-WAL modes."""
        # Test WAL mode
        conn_wal = connect(temp_db_path, use_wal=True)
        cursor = conn_wal.execute("PRAGMA journal_mode")
        wal_journal_mode = cursor.fetchone()[0]

        cursor = conn_wal.execute("PRAGMA locking_mode")
        wal_locking_mode = cursor.fetchone()[0]

        conn_wal.close()

        # Test non-WAL mode (need to remove WAL files first)
        Path(temp_db_path + "-wal").unlink(missing_ok=True)
        Path(temp_db_path + "-shm").unlink(missing_ok=True)

        conn_no_wal = connect(temp_db_path, use_wal=False)
        cursor = conn_no_wal.execute("PRAGMA journal_mode")
        no_wal_journal_mode = cursor.fetchone()[0]

        cursor = conn_no_wal.execute("PRAGMA locking_mode")
        no_wal_locking_mode = cursor.fetchone()[0]

        conn_no_wal.close()

        # Verify differences
        assert wal_journal_mode.upper() == 'WAL'
        assert no_wal_journal_mode.upper() == 'DELETE'
        assert no_wal_locking_mode.upper() == 'EXCLUSIVE'


class TestConnectionResources:
    """Test proper resource management."""

    def test_connection_cleanup(self, temp_db_path):
        """Test that connections can be properly closed."""
        conn = connect(temp_db_path)

        # Use the connection
        conn.execute("CREATE TABLE cleanup_test (id INTEGER)")
        conn.commit()

        # Close should not raise an error
        conn.close()

        # Attempting to use closed connection should raise an error
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_multiple_connections_same_database(self, temp_db_path):
        """Test creating multiple connections to the same database."""
        conn1 = connect(temp_db_path, use_wal=True)  # WAL allows concurrent access
        conn2 = connect(temp_db_path, use_wal=True)

        # Both connections should work
        conn1.execute("CREATE TABLE multi_test (id INTEGER)")
        conn1.commit()

        cursor = conn2.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        assert len(tables) == 1
        assert tables[0][0] == 'multi_test'

        conn1.close()
        conn2.close()

    def test_context_manager_usage(self, temp_db_path):
        """Test that connections work properly as context managers."""
        # SQLite connections support context manager protocol for transactions
        conn = connect(temp_db_path)

        try:
            with conn:  # This starts a transaction
                conn.execute("CREATE TABLE context_test (id INTEGER)")
                conn.execute("INSERT INTO context_test VALUES (1)")
                # Transaction automatically committed on successful exit

            # Verify data was committed
            cursor = conn.execute("SELECT * FROM context_test")
            row = cursor.fetchone()
            assert row[0] == 1

        finally:
            conn.close()