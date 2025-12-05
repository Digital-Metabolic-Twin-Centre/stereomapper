import pytest
import tempfile
import sqlite3
import json
import os
import time
import psutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from dataclasses import dataclass
from typing import List, Optional
from stereomapper.utils.itertools import chunked  # removed: no longer used


from stereomapper.runners.pipeline import (
    PipelineConfig,
    PipelineResult,
    stereomapperPipeline,
    run_mol_distance_2D_batch
)
# Import exceptions from the correct modules based on the __init__.py
from stereomapper.domain.exceptions.processing import ProcessingError, CacheError
from stereomapper.domain.exceptions.validation import ValidationError
from stereomapper.domain.exceptions.base import ConfigurationError
from stereomapper.domain.models import ProcessingResult


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""
    
    def test_pipeline_config_creation(self):
        """Test basic pipeline config creation."""
        config = PipelineConfig(
            input=["test.mol"],
            sqlite_output_path="output.db"
        )
        assert config.input == ["test.mol"]
        assert config.sqlite_output_path == "output.db"
        assert config.input_dir is None
        assert config.recursive is False
        assert config.namespace == "default"
    
    def test_pipeline_config_defaults(self):
        """Test default values are set correctly."""
        config = PipelineConfig(
            input=["test.mol"],
            sqlite_output_path="output.db"
        )
        assert config.extensions == (".mol", ".sdf")
        assert config.fresh_cache is False
        assert config.relate_with_cache is False


class TestPipelineResult:
    """Test PipelineResult dataclass."""
    
    def test_pipeline_result_creation(self):
        """Test pipeline result creation."""
        result = PipelineResult(
            n_inputs=100,
            n_session_pairs=50,
            n_cross_pairs=25,
            processing_time=120.5
        )
        assert result.n_inputs == 100
        assert result.n_session_pairs == 50
        assert result.n_cross_pairs == 25
        assert result.processing_time == 120.5
        assert result.output_path is None
        assert result.cache_hits == 0


class TeststereomapperPipeline:
    """Test stereomapperPipeline class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def basic_config(self, temp_dir):
        """Create basic valid configuration."""
        return PipelineConfig(
            input=["test1.mol", "test2.mol"],
            sqlite_output_path=str(temp_dir / "output.db"),
            cache_path=str(temp_dir / "cache.db")
        )
    
    @pytest.fixture
    def mock_molfiles(self, temp_dir):
        """Create mock molfiles for testing."""
        molfiles = []
        for i in range(3):
            molfile = temp_dir / f"test{i}.mol"
            molfile.write_text("mock mol content")
            molfiles.append(str(molfile))
        return molfiles
    
    @patch('stereomapper.runners.pipeline.psutil')
    def test_pipeline_initialization(self, mock_psutil, basic_config):
        """Test pipeline initialization."""
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        
        pipeline = stereomapperPipeline(basic_config)
        assert pipeline.config == basic_config
        assert pipeline.start_time is None
        assert pipeline.cache_conn is None
        assert pipeline.version_tag == "v1.0"
        assert pipeline.metrics['files_processed'] == 0
    
    @patch('stereomapper.runners.pipeline.psutil')
    def test_validate_config_valid(self, mock_psutil, basic_config):
        """Test configuration validation with valid config."""
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        
        pipeline = stereomapperPipeline(basic_config)
        # Should not raise any exception
        pipeline._validate_config()
    
    @patch('stereomapper.runners.pipeline.psutil')
    def test_validate_config_no_input(self, mock_psutil, temp_dir):
        """Test configuration validation fails with no input."""
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        
        config = PipelineConfig(
            input=None,
            input_dir=None,
            sqlite_output_path=str(temp_dir / "output.db")
        )
        with pytest.raises(ConfigurationError) as exc_info:
            stereomapperPipeline(config)

        # message assertions (substring check still works)
        assert "Either input files or input directory must be specified" in str(exc_info.value)

    @patch('stereomapper.runners.pipeline.psutil')
    def test_validate_config_both_inputs(self, mock_psutil, temp_dir):
        """Test configuration validation fails with both input types."""
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        
        config = PipelineConfig(
            input=["test.mol"],
            input_dir="/some/dir",
            sqlite_output_path=str(temp_dir / "output.db")
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            stereomapperPipeline(config)
        assert "Cannot specify both input files and input directory" in str(exc_info.value)
    
    @patch('stereomapper.runners.pipeline.psutil')
    def test_validate_config_no_output_path(self, mock_psutil):
        """Test configuration validation fails with no output path."""
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        
        config = PipelineConfig(
            input=["test.mol"],
            sqlite_output_path=None
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            stereomapperPipeline(config)
        assert "SQLite output path is required" in str(exc_info.value)
    
    @patch('stereomapper.runners.pipeline.psutil')
    @patch('stereomapper.runners.pipeline._resolve_inputs_from_cfg')
    @patch('stereomapper.runners.pipeline.InputValidator')
    def test_initialize_and_validate(self, mock_validator, mock_resolve, mock_psutil, basic_config, mock_molfiles):
        """Test pipeline initialization and validation."""
        # Setup mocks
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        mock_resolve.return_value = mock_molfiles
        validator = mock_validator.return_value
        validator.validate_molfile_paths.return_value = (mock_molfiles, [])
        validator.validate_batch_parameters.return_value = {"chunk_size": 1000}
        
        pipeline = stereomapperPipeline(basic_config)
        result = pipeline._initialize_and_validate()
        
        assert result == mock_molfiles
        assert pipeline.batch_processor is not None
        mock_resolve.assert_called_once_with(basic_config)
        validator.validate_molfile_paths.assert_called_once_with(mock_molfiles)

    @patch('stereomapper.runners.pipeline.psutil')
    @patch('stereomapper.runners.pipeline._resolve_cache_path')
    def test_setup_cache_database_no_path(self, mock_resolve, mock_psutil, basic_config):
        """Test cache database setup fails when path cannot be resolved."""
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        mock_resolve.return_value = None
        
        pipeline = stereomapperPipeline(basic_config)
        
        with pytest.raises(CacheError) as exc_info:
            pipeline._setup_cache_database()
        assert "Cache database path could not be resolved" in str(exc_info.value)
    
    @patch('stereomapper.runners.pipeline.psutil')
    @patch('stereomapper.runners.pipeline.chunked')
    def test_process_molecules_empty_list(self, mock_chunked, mock_psutil, basic_config):
        """Test processing molecules with empty list raises ValidationError."""
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        
        pipeline = stereomapperPipeline(basic_config)
        
        with pytest.raises(ValidationError) as exc_info:
            pipeline._process_molecules([])
        assert "No valid molfiles to process" in str(exc_info.value)
    
    @patch('stereomapper.runners.pipeline.psutil')
    @patch('stereomapper.runners.pipeline.chunked')
    def test_process_molecules_success(self, mock_chunked, mock_psutil, basic_config, mock_molfiles):
        """Test successful molecule processing."""
        # Setup mocks
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        
        mock_batch_processor = Mock()
        # Create a mock ProcessingResult with the correct attributes
        mock_result = Mock()
        mock_result.filename = "test.mol"
        mock_result.error = None
        mock_result.inchikey_first = "ABCD"
        mock_batch_processor.process_batch.return_value = [mock_result]
        
        pipeline = stereomapperPipeline(basic_config)
        pipeline.batch_processor = mock_batch_processor
        pipeline.cache_conn = Mock()
        
        # Mock chunked to return single chunk
        mock_chunked.return_value = [mock_molfiles[:1]]
        
        pipeline._process_molecules(mock_molfiles[:1])
        
        assert pipeline.metrics['files_succeeded'] == 1
        assert pipeline.metrics['files_failed'] == 0
        assert pipeline.processed_molfiles == mock_molfiles[:1]
    
    @patch('stereomapper.runners.pipeline.psutil')
    def test_cleanup(self, mock_psutil, basic_config):
        """Test resource cleanup."""
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        
        pipeline = stereomapperPipeline(basic_config)
        mock_conn = Mock()
        pipeline.cache_conn = mock_conn
        
        pipeline._cleanup()
        
        mock_conn.close.assert_called_once()
        assert pipeline.cache_conn is None
    
    @patch('stereomapper.runners.pipeline.psutil')
    @patch('stereomapper.runners.pipeline.sqlite3.connect')
    def test_update_relationship_metrics(self, mock_connect, mock_psutil, basic_config, temp_dir):
        mock_psutil.Process.return_value = MagicMock()

        # Context-managed connection
        mock_conn = MagicMock()
        mock_ctx = MagicMock()
        mock_connect.return_value = mock_ctx
        mock_ctx.__enter__.return_value = mock_conn
        mock_ctx.__exit__.return_value = None

        # Cursors/iterables returned by conn.execute(...)
        cur1 = MagicMock()
        cur1.__iter__.return_value = iter([("similar", 10), ("different", 5)])
        cur1.fetchall.return_value = [("similar", 10), ("different", 5)]

        cur2 = MagicMock()
        cur2.__iter__.return_value = iter([
            ('{"pair_origin": "session"}',),
            ('{"pair_origin": "cross"}',),
            (None,)
        ])
        cur2.fetchall.return_value = [
            ('{"pair_origin": "session"}',),
            ('{"pair_origin": "cross"}',),
            (None,)
        ]

        # First execute -> cur1, second execute -> cur2
        mock_conn.execute.side_effect = [cur1, cur2]

        pipeline = stereomapperPipeline(basic_config)
        pipeline.config.sqlite_output_path = str(temp_dir / "test.db")
        (temp_dir / "test.db").touch()

        pipeline._update_relationship_metrics()

        assert pipeline.metrics['relationship_totals'] == 15
        assert pipeline.metrics['relationship_class_counts'] == {"similar": 10, "different": 5}

        
    @patch('stereomapper.runners.pipeline.psutil')
    def test_memory_report(self, mock_psutil, basic_config):
        """Test memory reporting."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 1000000000  # 1GB in bytes
        mock_psutil.Process.return_value = mock_process
        
        pipeline = stereomapperPipeline(basic_config)
        pipeline.process = mock_process
        
        # Should not raise exception
        pipeline._memory_report("test")
    
    @patch('stereomapper.runners.pipeline.psutil')
    def test_get_chunk_size_default(self, mock_psutil, basic_config):
        """Test default chunk size when config not available."""
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        
        pipeline = stereomapperPipeline(basic_config)
        chunk_size = pipeline._get_chunk_size()
        assert chunk_size == 2000  # Default chunk size
        
class TestPipelineIntegration:
    """Integration tests for the pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def integration_config(self, temp_dir):
        """Create configuration for integration tests."""
        return PipelineConfig(
            input=["test.mol"],
            sqlite_output_path=str(temp_dir / "output.db"),
            cache_path=str(temp_dir / "cache.db")
        )

    from unittest.mock import Mock, patch
    import os

    @patch('stereomapper.runners.pipeline.psutil')
    @patch('stereomapper.runners.pipeline._resolve_inputs_from_cfg')
    @patch('stereomapper.runners.pipeline.InputValidator')
    @patch('stereomapper.runners.pipeline._resolve_cache_path')
    @patch('stereomapper.runners.pipeline.db')
    @patch('stereomapper.runners.pipeline.cache_schema')
    @patch('stereomapper.runners.pipeline.results_schema')
    @patch('stereomapper.runners.pipeline.cache_repo')
    @patch('stereomapper.runners.pipeline.chunked')
    @patch('stereomapper.runners.pipeline.BatchProcessor')  # <-- CRITICAL
    def test_run_pipeline_success(self, mock_BatchProcessor, mock_chunked, mock_cache_repo,
                                mock_results_schema, mock_cache_schema, mock_db,
                                mock_resolve_cache, mock_validator, mock_resolve_inputs,
                                mock_psutil, integration_config):

        # psutil
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process

        # inputs + validator
        mock_resolve_inputs.return_value = ["test.mol"]
        v = mock_validator.return_value
        v.validate_molfile_paths.return_value = (["test.mol"], [])
        v.validate_batch_parameters.return_value = {"chunk_size": 1000}

        # cache DB setup
        mock_resolve_cache.return_value = "/tmp/cache.db"
        mock_conn = Mock()
        mock_db.connect.return_value = mock_conn
        mock_cache_schema.create_cache.return_value = None

        # results schema
        mock_results_schema.results_schema.return_value = None

        # inchikeys for results phase
        mock_cache_repo.inchi_first_by_id.return_value = ["ABCD"]

        # chunking
        mock_chunked.return_value = [["test.mol"]]

        # BatchProcessor instance used inside _initialize_and_validate()
        bp = mock_BatchProcessor.return_value

        # successful processing result
        mock_result = Mock()
        mock_result.filename = "test.mol"
        mock_result.error = None           # success flag
        mock_result.from_cache = False
        mock_result.inchikey_first = "ABCD"

        bp.process_batch.return_value = [mock_result]

        # run
        from stereomapper.runners.pipeline import stereomapperPipeline, PipelineResult
        pipeline = stereomapperPipeline(integration_config)
        with patch.object(pipeline, '_process_inchikey_group', return_value="processed"):
            with patch.dict(os.environ, {'NO_PROGRESS': '1'}):
                result = pipeline.run()

        assert isinstance(result, PipelineResult)
        assert result.n_inputs == 1
        assert pipeline.metrics['files_succeeded'] == 1
        assert result.processing_time >= 0

    
    @patch('stereomapper.runners.pipeline.psutil')
    @patch('stereomapper.runners.pipeline._resolve_inputs_from_cfg')
    def test_run_pipeline_validation_error(self, mock_resolve_inputs, mock_psutil, integration_config):
        """Test pipeline handles validation errors correctly."""
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        
        mock_resolve_inputs.side_effect = ValidationError("Invalid input")
        
        pipeline = stereomapperPipeline(integration_config)
        
        with pytest.raises(ValidationError):
            with patch.dict(os.environ, {'NO_PROGRESS': '1'}):
                pipeline.run()


class TestLegacyFunction:
    """Test the legacy run_mol_distance_2D_batch function."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_legacy_function_calls_pipeline(self, temp_dir):
        """Test that legacy function correctly calls the new pipeline."""
        config = PipelineConfig(
            input=["test.mol"],
            sqlite_output_path=str(temp_dir / "output.db")
        )
        
        with patch('stereomapper.runners.pipeline.stereomapperPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_result = PipelineResult(n_inputs=1, n_session_pairs=0, n_cross_pairs=0)
            mock_pipeline.run.return_value = mock_result
            mock_pipeline_class.return_value = mock_pipeline
            
            result = run_mol_distance_2D_batch(config)
            
            mock_pipeline_class.assert_called_once_with(config)
            mock_pipeline.run.assert_called_once()
            assert result == mock_result


class TestErrorHandling:
    """Test error handling throughout the pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @patch('stereomapper.runners.pipeline.psutil')
    def test_processing_error_propagation(self, mock_psutil, temp_dir):
        """Test that processing errors are properly propagated."""
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        
        config = PipelineConfig(
            input=["nonexistent.mol"],
            sqlite_output_path=str(temp_dir / "output.db")
        )
        
        with patch('stereomapper.runners.pipeline._resolve_inputs_from_cfg') as mock_resolve:
            mock_resolve.side_effect = ProcessingError("Processing failed")
            
            pipeline = stereomapperPipeline(config)
            
            with pytest.raises(ProcessingError) as exc_info:
                with patch.dict(os.environ, {'NO_PROGRESS': '1'}):
                    pipeline.run()
            
            assert "Processing failed" in str(exc_info.value)
    
    @patch('stereomapper.runners.pipeline._resolve_inputs_from_cfg', return_value=["test.mol"])
    def test_cache_error_propagation(mock_resolve_inputs, temp_dir):
        config = PipelineConfig(
            input=["test.mol"],
            sqlite_output_path=str(temp_dir / "output.db")
        )
        pipeline = stereomapperPipeline(config)

        # Skip validator & batch processor creation entirely
        with patch.object(pipeline, '_initialize_and_validate', return_value=["test.mol"]):
            # Force the cache step to fail
            with patch.object(pipeline, '_setup_cache_database', side_effect=CacheError("Cache failed")):
                with patch.dict(os.environ, {'NO_PROGRESS': '1'}):
                    with pytest.raises(CacheError) as exc_info:
                        pipeline.run()

        assert "Cache failed" in str(exc_info.value)
        
    @patch('stereomapper.runners.pipeline.psutil')
    def test_unexpected_error_handling(self, mock_psutil, temp_dir):
        """Test handling of unexpected errors."""
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        
        config = PipelineConfig(
            input=["test.mol"],
            sqlite_output_path=str(temp_dir / "output.db")
        )
        
        pipeline = stereomapperPipeline(config)
        
        with patch.object(pipeline, '_initialize_and_validate') as mock_init:
            mock_init.side_effect = RuntimeError("Unexpected error")
            
            with pytest.raises(ProcessingError) as exc_info:
                with patch.dict(os.environ, {'NO_PROGRESS': '1'}):
                    pipeline.run()
            
            assert "Unexpected pipeline error" in str(exc_info.value)


# Add this test to debug the import issue
def test_exception_import_debug():
    """Debug what's actually being imported."""
    from stereomapper.domain.exceptions import ConfigurationError
    print(f"ConfigurationError type: {type(ConfigurationError)}")
    print(f"ConfigurationError MRO: {ConfigurationError.__mro__}")
    print(f"Is subclass of Exception: {issubclass(ConfigurationError, Exception)}")
    
    # Try to instantiate
    try:
        exc = ConfigurationError("test")
        print(f"Successfully created: {exc}")
        print(f"Instance type: {type(exc)}")
    except Exception as e:
        print(f"Failed to create: {e}")


if __name__ == "__main__":
    pytest.main([__file__])