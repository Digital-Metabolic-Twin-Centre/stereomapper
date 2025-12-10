import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from stereomapper.config.loader import ConfigurationLoader, configure_from_cli
from stereomapper.config.settings import LogLevel, OutputFormat
from stereomapper.domain.exceptions import ConfigurationError
from argparse import Namespace


class TestConfigurationLoader:
    
    def test_load_defaults(self):
        """Test that load_defaults returns correct default settings."""
        loader = ConfigurationLoader()
        settings = loader.load_defaults()
        
        # Test processing defaults
        assert settings.processing.chunk_size == 2000
        assert settings.processing.timeout_seconds == 300
        assert settings.processing.retry_attempts == 3
        assert settings.processing.enable_cache is True
        
        # Test database defaults
        assert settings.database.cache_path is None
        assert settings.database.output_path is None
        assert settings.database.fresh_cache is False
        assert "journal_mode" in settings.database.pragma_settings
        
        # Test chemistry defaults
        assert settings.chemistry.enable_stereochemistry is True
        assert settings.chemistry.canonicalization_tool == "openbabel"
        assert settings.chemistry.rmsd_threshold == 0.5
        
        # Test output defaults
        assert settings.output.format == OutputFormat.SQLITE
        assert settings.output.include_errors is True
        assert settings.output.verbose_errors is False
        
        # Test logging defaults
        assert settings.logging.level == LogLevel.INFO
        assert settings.logging.console_output is True
        
        # Test general defaults
        assert settings.namespace == "default"
        assert settings.debug_mode is False
        assert settings.dry_run is False

    def test_load_from_cli_args_empty_args(self):
        loader = ConfigurationLoader()
        args = Namespace(config=None, namespace="default")
        
        settings = loader.load_from_cli_args(args)
        
        assert settings.processing.chunk_size == 2000
        assert settings.namespace == "default"

    def test_load_from_cli_args_processing_settings(self):
        """Test CLI args for processing settings."""
        loader = ConfigurationLoader()
        args = MagicMock()
        args.chunk_size = 50000
        args.max_workers = 8
        args.timeout = 600
        
        settings = loader.load_from_cli_args(args)
        
        assert settings.processing.chunk_size == 50000
        assert settings.processing.max_workers == 8
        assert settings.processing.timeout_seconds == 600

    def test_load_from_cli_args_database_settings(self):
        """Test CLI args for database settings."""
        loader = ConfigurationLoader()
        args = MagicMock()
        args.cache_path = "/tmp/cache"
        args.sqlite_output = "/tmp/output.db"
        args.fresh_cache = True
        
        settings = loader.load_from_cli_args(args)
        
        assert settings.database.cache_path == Path("/tmp/cache")
        assert settings.database.output_path == Path("/tmp/output.db")
        assert settings.database.fresh_cache is True

    def test_load_from_cli_args_output_settings(self):
        """Test CLI args for output settings."""
        loader = ConfigurationLoader()
        args = MagicMock()
        args.no_errors = True
        args.verbose_errors = True
        
        settings = loader.load_from_cli_args(args)
        
        assert settings.output.include_errors is False
        assert settings.output.verbose_errors is True

    def test_load_from_cli_args_logging_settings(self):
        """Test CLI args for logging settings."""
        loader = ConfigurationLoader()
        args = MagicMock()
        args.log_file = "/tmp/app.log"
        args.debug = True
        
        settings = loader.load_from_cli_args(args)
        
        assert settings.logging.file_path == Path("/tmp/app.log")
        assert settings.logging.level == LogLevel.DEBUG

    def test_load_from_cli_args_input_files(self):
        """Test CLI args for input files."""
        loader = ConfigurationLoader()
        args = MagicMock()
        args.input = ["file1.sdf", "file2.sdf"]
        
        settings = loader.load_from_cli_args(args)
        
        assert len(settings.input_files) == 2
        assert settings.input_files[0] == Path("file1.sdf")
        assert settings.input_files[1] == Path("file2.sdf")

    def test_load_from_cli_args_input_directory(self):
        loader = ConfigurationLoader()
        args = Namespace(input_dir="/data/molecules")  # match loader's field
        settings = loader.load_from_cli_args(args)
        assert str(settings.input_directory) == "/data/molecules"  # or Path("/data/molecules")

    def test_load_from_cli_args_runtime_settings(self):
        """Test CLI args for runtime settings."""
        loader = ConfigurationLoader()
        args = MagicMock()
        args.namespace = "test_ns"
        args.relate_with_cache = True
        args.recursive = True
        args.debug = True
        args.dry_run = True
        
        settings = loader.load_from_cli_args(args)
        
        assert settings.namespace == "test_ns"
        assert settings.relate_with_cache is True
        assert settings.recursive is True
        assert settings.debug_mode is True
        assert settings.dry_run is True

    def test_load_from_cli_args_missing_attributes(self):
        """Test that missing attributes don't cause errors."""
        loader = ConfigurationLoader()
        args = Mock(spec=[])  # Empty spec means no attributes
        
        settings = loader.load_from_cli_args(args)
        
        # Should use defaults when attributes are missing
        assert settings.namespace == "default"
        assert settings.relate_with_cache is False

    def test_load_from_cli_args_none_values(self):
        """Test that None values don't override defaults."""
        loader = ConfigurationLoader()
        args = MagicMock()
        args.chunk_size = None
        args.cache_path = None
        
        settings = loader.load_from_cli_args(args)
        
        # Should use defaults when values are None
        assert settings.processing.chunk_size == 2000

    def test_load_from_cli_args_configuration_error_passthrough(self):
        """Test that ConfigurationError is re-raised."""
        loader = ConfigurationLoader()
        args = Mock()
        
        with patch.object(loader, 'load_defaults', side_effect=ConfigurationError("Test error")):
            with pytest.raises(ConfigurationError, match="Test error"):
                loader.load_from_cli_args(args)

    def test_load_from_cli_args_generic_exception_handling(self):
        """Test that generic exceptions are wrapped in ConfigurationError."""
        loader = ConfigurationLoader()
        args = MagicMock()
        
        with patch.object(loader, 'load_defaults', side_effect=ValueError("Test error")):
            with pytest.raises(ConfigurationError, match="Failed to load configuration from CLI arguments"):
                loader.load_from_cli_args(args)

    @patch('os.cpu_count', return_value=8)
    def test_load_defaults_cpu_count(self, mock_cpu_count):
        """Test that load_defaults uses os.cpu_count for max_workers."""
        loader = ConfigurationLoader()
        settings = loader.load_defaults()
        
        assert settings.processing.max_workers == 8

    @patch('os.cpu_count', return_value=None)
    def test_load_defaults_cpu_count_fallback(self, mock_cpu_count):
        """Test that load_defaults falls back to 4 when cpu_count is None."""
        loader = ConfigurationLoader()
        settings = loader.load_defaults()
        
        assert settings.processing.max_workers == 4


class TestConfigureFromCli:
    
    def test_configure_from_cli_success(self):
        """Test successful configuration from CLI."""
        args = Mock()
        args.chunk_size = 50000
        
        with patch('stereomapper.config.loader.ConfigurationLoader.load_from_cli_args') as mock_load:
            mock_settings = Mock()
            mock_load.return_value = mock_settings
            
            result = configure_from_cli(args)
            
            mock_load.assert_called_once_with(args)
            mock_settings.validate.assert_called_once()
            assert result == mock_settings

    def test_configure_from_cli_validation_error(self):
        """Test that validation errors are propagated."""
        args = Mock()
        
        with patch('stereomapper.config.loader.ConfigurationLoader.load_from_cli_args') as mock_load:
            mock_settings = Mock()
            mock_settings.validate.side_effect = ConfigurationError("Validation failed")
            mock_load.return_value = mock_settings
            
            with pytest.raises(ConfigurationError, match="Validation failed"):
                configure_from_cli(args)