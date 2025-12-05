import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from stereomapper.config.settings import (
    Settings,
    ProcessingSettings,
    DatabaseSettings,
    ChemistrySettings,
    OutputSettings,
    LoggingSettings,
    LogLevel,
    OutputFormat,
    get_settings,
    set_settings,
)
import stereomapper.config.settings as settings_module
from stereomapper.domain.exceptions import PipelineConfigurationError as ConfigurationError


class TestLogLevel:
    """Test LogLevel enum."""

    def test_log_level_values(self):
        """Test that LogLevel enum has correct values."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestOutputFormat:
    """Test OutputFormat enum."""

    def test_output_format_values(self):
        """Test that OutputFormat enum has correct values."""
        assert OutputFormat.SQLITE.value == "sqlite"


class TestProcessingSettings:
    """Test ProcessingSettings dataclass and validation."""

    def test_default_values(self):
        """Test default values are set correctly."""
        settings = ProcessingSettings()
        assert settings.chunk_size == 2000
        assert settings.max_workers == (os.cpu_count() or 4)
        assert settings.timeout_seconds == 300
        assert settings.retry_attempts == 3
        assert settings.enable_cache is True

    def test_custom_values(self):
        """Test custom values are set correctly."""
        settings = ProcessingSettings(
            chunk_size=50_000,
            max_workers=8,
            timeout_seconds=600,
            retry_attempts=5,
            enable_cache=False
        )
        assert settings.chunk_size == 50_000
        assert settings.max_workers == 8
        assert settings.timeout_seconds == 600
        assert settings.retry_attempts == 5
        assert settings.enable_cache is False

    def test_validate_positive_chunk_size(self):
        """Test validation fails for non-positive chunk_size."""
        settings = ProcessingSettings(chunk_size=0)
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate()
        assert "chunk_size must be positive" in str(exc_info.value)
        assert exc_info.value.context.get('config_field') == "processing.chunk_size"

        settings = ProcessingSettings(chunk_size=-1)
        with pytest.raises(ConfigurationError):
            settings.validate()

    def test_validate_positive_max_workers(self):
        """Test validation fails for non-positive max_workers."""
        settings = ProcessingSettings(max_workers=0)
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate()
        assert "max_workers must be positive" in str(exc_info.value)
        assert exc_info.value.context.get('config_field') == "processing.max_workers"

    def test_validate_success(self):
        """Test validation succeeds for valid settings."""
        settings = ProcessingSettings()
        settings.validate()  # Should not raise


class TestDatabaseSettings:
    """Test DatabaseSettings dataclass and validation."""

    def test_default_values(self):
        """Test default values are set correctly."""
        settings = DatabaseSettings()
        assert settings.cache_path is None
        assert settings.output_path is None
        assert settings.fresh_cache is False
        assert settings.pragma_settings == {
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "cache_size": -64000,
            "temp_store": "MEMORY"
        }

    def test_validate_cache_path_parent_exists(self, tmp_path):
        """Test validation succeeds when cache_path parent exists."""
        cache_file = tmp_path / "cache.db"
        settings = DatabaseSettings(cache_path=cache_file)
        settings.validate()  # Should not raise

    def test_validate_cache_path_parent_missing(self, tmp_path):
        """Test validation fails when cache_path parent doesn't exist."""
        cache_file = tmp_path / "missing" / "cache.db"
        settings = DatabaseSettings(cache_path=cache_file)
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate()
        assert "Cache directory does not exist" in str(exc_info.value)
        assert exc_info.value.context.get('config_field') == "database.cache_path"

    def test_validate_output_path_parent_exists(self, tmp_path):
        """Test validation succeeds when output_path parent exists."""
        output_file = tmp_path / "output.db"
        settings = DatabaseSettings(output_path=output_file)
        settings.validate()  # Should not raise

    def test_validate_output_path_parent_missing(self, tmp_path):
        """Test validation fails when output_path parent doesn't exist."""
        output_file = tmp_path / "missing" / "output.db"
        settings = DatabaseSettings(output_path=output_file)
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate()
        assert "Output directory does not exist" in str(exc_info.value)
        assert exc_info.value.context.get('config_field') == "database.output_path"


class TestChemistrySettings:
    """Test ChemistrySettings dataclass and validation."""

    def test_default_values(self):
        """Test default values are set correctly."""
        settings = ChemistrySettings()
        assert settings.enable_stereochemistry is True
        assert settings.sanitize_molecules is True
        assert settings.remove_hydrogens is True
        assert settings.standardize_molecules is True
        assert settings.canonicalization_tool == "openbabel"
        assert settings.alignment_timeout == 30
        assert settings.rmsd_threshold == 0.5

    def test_validate_valid_canonicalization_tool(self):
        """Test validation succeeds for valid canonicalization tools."""
        for tool in ["openbabel", "rdkit"]:
            settings = ChemistrySettings(canonicalization_tool=tool)
            settings.validate()  # Should not raise

    def test_validate_invalid_canonicalization_tool(self):
        """Test validation fails for invalid canonicalization tool."""
        settings = ChemistrySettings(canonicalization_tool="invalid")
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate()
        assert "Invalid canonicalization tool" in str(exc_info.value)
        assert exc_info.value.context.get('config_field') == "chemistry.canonicalization_tool"

    def test_validate_negative_rmsd_threshold(self):
        """Test validation fails for negative RMSD threshold."""
        settings = ChemistrySettings(rmsd_threshold=-1.0)
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate()
        assert "RMSD threshold must be non-negative" in str(exc_info.value)
        assert exc_info.value.context.get('config_field') == "chemistry.rmsd_threshold"

    def test_validate_zero_rmsd_threshold(self):
        """Test validation succeeds for zero RMSD threshold."""
        settings = ChemistrySettings(rmsd_threshold=0.0)
        settings.validate()  # Should not raise


class TestOutputSettings:
    """Test OutputSettings dataclass and validation."""

    def test_default_values(self):
        """Test default values are set correctly."""
        settings = OutputSettings()
        assert settings.format == OutputFormat.SQLITE
        assert settings.include_errors is True
        assert settings.include_metadata is True
        assert settings.verbose_errors is False

    def test_validate_always_succeeds(self):
        """Test validation always succeeds since we only support SQLite."""
        settings = OutputSettings()
        settings.validate()  # Should not raise


class TestLoggingSettings:
    """Test LoggingSettings dataclass and validation."""

    def test_default_values(self):
        """Test default values are set correctly."""
        settings = LoggingSettings()
        assert settings.level == LogLevel.INFO
        assert settings.file_path is None
        assert settings.console_output is True
        assert settings.include_timestamps is True
        assert settings.format_string == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def test_validate_file_path_parent_exists(self, tmp_path):
        """Test validation succeeds when log file parent exists."""
        log_file = tmp_path / "app.log"
        settings = LoggingSettings(file_path=log_file)
        settings.validate()  # Should not raise

    def test_validate_file_path_parent_missing(self, tmp_path):
        """Test validation fails when log file parent doesn't exist."""
        log_file = tmp_path / "missing" / "app.log"
        settings = LoggingSettings(file_path=log_file)
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate()
        assert "Log directory does not exist" in str(exc_info.value)
        assert exc_info.value.context.get('config_field') == "logging.file_path"


class TestSettings:
    """Test main Settings dataclass and validation."""

    def test_default_values(self):
        """Test default values are set correctly."""
        settings = Settings()
        assert isinstance(settings.processing, ProcessingSettings)
        assert isinstance(settings.database, DatabaseSettings)
        assert isinstance(settings.chemistry, ChemistrySettings)
        assert isinstance(settings.output, OutputSettings)
        assert isinstance(settings.logging, LoggingSettings)
        assert settings.input_files == []
        assert settings.input_directory is None
        assert settings.namespace == "default"
        assert settings.source_kind == "file"
        assert settings.std_version == 1
        assert settings.relate_with_cache is False
        assert settings.recursive is False
        assert settings.debug_mode is False
        assert settings.profile_performance is False
        assert settings.dry_run is False

    def test_validate_input_sources_none_specified(self):
        """Test validation fails when no input sources are specified."""
        settings = Settings()
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate()
        assert "Either input_files or input_directory must be specified" in str(exc_info.value)
        assert exc_info.value.context.get('config_field') == "input_sources"

    def test_validate_input_sources_both_specified(self, tmp_path):
        """Test validation fails when both input sources are specified."""
        settings = Settings(
            input_files=[Path("file1.mol")],
            input_directory=tmp_path,
            database=DatabaseSettings(output_path=tmp_path / "output.db")
        )
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate()
        assert "Cannot specify both input_files and input_directory" in str(exc_info.value)
        assert exc_info.value.context.get('config_field') == "input_sources"

    def test_validate_input_directory_missing(self, tmp_path):
        """Test validation fails when input directory doesn't exist."""
        missing_dir = tmp_path / "missing"
        settings = Settings(
            input_directory=missing_dir,
            database=DatabaseSettings(output_path=tmp_path / "output.db")
        )
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate()
        assert "Input directory does not exist" in str(exc_info.value)
        assert exc_info.value.context.get('config_field') == "input_directory"

    def test_validate_input_directory_not_directory(self, tmp_path):
        """Test validation fails when input directory path is not a directory."""
        not_dir = tmp_path / "file.txt"
        not_dir.touch()
        settings = Settings(
            input_directory=not_dir,
            database=DatabaseSettings(output_path=tmp_path / "output.db")
        )
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate()
        assert "Input directory path is not a directory" in str(exc_info.value)
        assert exc_info.value.context.get('config_field') == "input_directory"

    def test_validate_output_compatibility_missing_db_path(self, tmp_path):
        """Test validation fails when SQLite output format but no database path."""
        settings = Settings(
            input_files=[Path("file1.mol")]
        )
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate()
        assert "SQLite output format requires database.output_path" in str(exc_info.value)
        assert exc_info.value.context.get('config_field') == "database.output_path"

    def test_validate_success_with_input_files(self, tmp_path):
        """Test validation succeeds with valid input files configuration."""
        settings = Settings(
            input_files=[Path("file1.mol")],
            database=DatabaseSettings(output_path=tmp_path / "output.db")
        )
        settings.validate()  # Should not raise

    def test_validate_success_with_input_directory(self, tmp_path):
        """Test validation succeeds with valid input directory configuration."""
        settings = Settings(
            input_directory=tmp_path,
            database=DatabaseSettings(output_path=tmp_path / "output.db")
        )
        settings.validate()  # Should not raise

    def test_validate_propagates_subsetting_errors(self):
        """Test that validation errors from sub-settings are propagated."""
        settings = Settings(
            input_files=[Path("file1.mol")],
            database=DatabaseSettings(output_path=Path("output.db")),
            processing=ProcessingSettings(chunk_size=0)  # Invalid
        )
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate()
        assert "chunk_size must be positive" in str(exc_info.value)

    def test_validate_catches_unexpected_exceptions(self):
        """Test that unexpected exceptions during validation are wrapped."""
        settings = Settings(
            input_files=[Path("file1.mol")],
            database=DatabaseSettings(output_path=Path("output.db"))
        )

        # Mock a sub-validate method to raise unexpected exception
        with patch.object(settings.processing, 'validate', side_effect=RuntimeError("Unexpected")):
            with pytest.raises(ConfigurationError) as exc_info:
                settings.validate()
            assert "Configuration validation failed" in str(exc_info.value)

    def test_to_dict(self, tmp_path):
        """Test conversion to dictionary."""
        cache_path = tmp_path / "cache.db"
        output_path = tmp_path / "output.db"

        settings = Settings(
            processing=ProcessingSettings(chunk_size=50000),
            database=DatabaseSettings(cache_path=cache_path, output_path=output_path, fresh_cache=True),
            namespace="test",
            debug_mode=True,
            dry_run=True
        )

        result = settings.to_dict()

        assert result['processing']['chunk_size'] == 50000
        assert result['database']['cache_path'] == str(cache_path)
        assert result['database']['output_path'] == str(output_path)
        assert result['database']['fresh_cache'] is True
        assert result['output']['format'] == "sqlite"
        assert result['runtime']['namespace'] == "test"
        assert result['runtime']['debug_mode'] is True
        assert result['runtime']['dry_run'] is True

    def test_to_dict_with_none_paths(self):
        """Test conversion to dictionary with None paths."""
        settings = Settings()
        result = settings.to_dict()

        assert result['database']['cache_path'] is None
        assert result['database']['output_path'] is None


class TestGlobalSettings:
    """Test global settings management functions."""

    def setup_method(self):
        """Reset global settings before each test."""
        # Access the module's _settings directly
        settings_module._settings = None

    def test_get_settings_not_initialized(self):
        """Test get_settings raises error when not initialized."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_settings()
        assert "Settings not initialized" in str(exc_info.value)

    def test_set_and_get_settings_success(self, tmp_path):
        """Test setting and getting settings successfully."""
        settings = Settings(
            input_files=[Path("file1.mol")],
            database=DatabaseSettings(output_path=tmp_path / "output.db")
        )

        with patch('stereomapper.config.settings.logger') as mock_logger:
            set_settings(settings)
            mock_logger.info.assert_called_once_with("Configuration loaded and validated successfully")

        retrieved = get_settings()
        assert retrieved is settings

    def test_set_settings_validates_before_setting(self):
        """Test that set_settings validates settings before setting them."""
        # Ensure we start with clean state
        settings_module._settings = None

        invalid_settings = Settings()  # No input sources, will fail validation

        # First, test that the settings validation itself raises an exception
        with pytest.raises(ConfigurationError):
            invalid_settings.validate()

        # Now test the set_settings function
        with pytest.raises(ConfigurationError):
            set_settings(invalid_settings)

        # Settings should not be set after validation failure
        with pytest.raises(ConfigurationError, match="Settings not initialized"):
            get_settings()

    def test_set_settings_overwrites_previous(self, tmp_path):
        """Test that set_settings overwrites previous settings."""
        settings1 = Settings(
            input_files=[Path("file1.mol")],
            database=DatabaseSettings(output_path=tmp_path / "output1.db"),
            namespace="first"
        )

        settings2 = Settings(
            input_files=[Path("file2.mol")],
            database=DatabaseSettings(output_path=tmp_path / "output2.db"),
            namespace="second"
        )

        set_settings(settings1)
        assert get_settings().namespace == "first"

        set_settings(settings2)
        assert get_settings().namespace == "second"