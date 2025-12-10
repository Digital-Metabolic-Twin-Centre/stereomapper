import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import FrozenInstanceError

from stereomapper.config.integration import (
    PipelineConfig,
    get_pipeline_config,
    get_chunk_size,
    get_max_workers,
    get_cache_settings
)
from stereomapper.config.settings import Settings
from stereomapper.domain.exceptions import ConfigurationError


class TestPipelineConfig:
    """Test the PipelineConfig dataclass."""
    
    def test_pipeline_config_creation(self):
        """Test basic PipelineConfig creation."""
        config = PipelineConfig(
            input=["file1.mol", "file2.sdf"],
            input_dir="/test/dir",
            recursive=True,
            sqlite_output_path="/output/results.db",
            cache_path="/cache/cache.db",
            fresh_cache=True,
            namespace="test_namespace",
            relate_with_cache=True
        )
        
        assert config.input == ["file1.mol", "file2.sdf"]
        assert config.input_dir == "/test/dir"
        assert config.recursive is True
        assert config.extensions == (".mol", ".sdf")
        assert config.sqlite_output_path == "/output/results.db"
        assert config.cache_path == "/cache/cache.db"
        assert config.fresh_cache is True
        assert config.namespace == "test_namespace"
        assert config.relate_with_cache is True
    
    def test_pipeline_config_defaults(self):
        """Test PipelineConfig with default values."""
        config = PipelineConfig(input=None)
        
        assert config.input is None
        assert config.input_dir is None
        assert config.recursive is False
        assert config.extensions == (".mol", ".sdf")
        assert config.sqlite_output_path is None
        assert config.cache_path is None
        assert config.fresh_cache is False
        assert config.namespace == "default"
        assert config.relate_with_cache is False
    
    def test_from_settings_with_inputs(self):
        """Test from_settings with explicit inputs."""
        # Mock Settings object
        mock_settings = Mock()
        mock_settings.input_files = [Path("/path/file1.mol"), Path("/path/file2.sdf")]
        mock_settings.input_directory = Path("/input/dir")
        mock_settings.database.output_path = Path("/output/results.db")
        mock_settings.database.cache_path = Path("/cache/cache.db")
        mock_settings.database.fresh_cache = True
        mock_settings.namespace = "test_ns"
        
        # Mock the settings object attributes directly instead of patching getattr
        mock_settings.recursive = True
        mock_settings.relate_with_cache = True
        
        config = PipelineConfig.from_settings(
            mock_settings,
            inputs=["custom1.mol", "custom2.sdf"],
            output_path="/custom/output.db"
        )
        
        assert config.input == ["custom1.mol", "custom2.sdf"]
        assert config.input_dir is None  # Should be None when inputs provided
        assert config.recursive is True
        assert config.sqlite_output_path == "/custom/output.db"
        assert config.cache_path == "/cache/cache.db"
        assert config.fresh_cache is True
        assert config.namespace == "test_ns"
        assert config.relate_with_cache is True
    
    def test_from_settings_without_inputs(self):
        """Test from_settings without explicit inputs."""
        mock_settings = Mock()
        mock_settings.input_files = [Path("/settings/file1.mol")]
        mock_settings.input_directory = Path("/settings/input")
        mock_settings.database.output_path = Path("/settings/output.db")
        mock_settings.database.cache_path = None
        mock_settings.database.fresh_cache = False
        mock_settings.namespace = "settings_ns"
        
        # Mock attributes directly
        mock_settings.recursive = False
        mock_settings.relate_with_cache = False
        
        config = PipelineConfig.from_settings(mock_settings)
        
        assert config.input == ["/settings/file1.mol"]
        assert config.input_dir == "/settings/input"
        assert config.recursive is False
        assert config.sqlite_output_path == "/settings/output.db"
        assert config.cache_path is None
        assert config.fresh_cache is False
        assert config.namespace == "settings_ns"
        assert config.relate_with_cache is False
    
    def test_from_settings_no_input_files(self):
        """Test from_settings when input_files is None."""
        mock_settings = Mock()
        mock_settings.input_files = None
        mock_settings.input_directory = Path("/dir")
        mock_settings.database.output_path = None
        mock_settings.database.cache_path = Path("/cache")
        mock_settings.database.fresh_cache = True
        mock_settings.namespace = "no_files"
        
        # Mock missing attributes to return defaults
        mock_settings.recursive = None
        mock_settings.relate_with_cache = None
        
        config = PipelineConfig.from_settings(mock_settings)
        
        assert config.input is None
        assert config.input_dir == "/dir"
        assert config.sqlite_output_path is None
        assert config.cache_path == "/cache"
    
    def test_from_settings_none_paths(self):
        """Test from_settings when paths are None."""
        mock_settings = Mock()
        mock_settings.input_files = None
        mock_settings.input_directory = None
        mock_settings.database.output_path = None
        mock_settings.database.cache_path = None
        mock_settings.database.fresh_cache = False
        mock_settings.namespace = "none_paths"
        
        # Mock missing attributes
        mock_settings.recursive = None
        mock_settings.relate_with_cache = None
        
        config = PipelineConfig.from_settings(mock_settings)
        
        assert config.input is None
        assert config.input_dir is None
        assert config.sqlite_output_path is None
        assert config.cache_path is None


class TestGetPipelineConfig:
    """Test the get_pipeline_config function."""
    
    @patch('stereomapper.config.integration.get_settings')
    @patch('stereomapper.config.integration.PipelineConfig.from_settings')
    def test_get_pipeline_config(self, mock_from_settings, mock_get_settings):
        """Test get_pipeline_config calls correct methods."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        
        expected_config = Mock()
        mock_from_settings.return_value = expected_config
        
        result = get_pipeline_config()
        
        mock_get_settings.assert_called_once()
        mock_from_settings.assert_called_once_with(mock_settings)
        assert result == expected_config


class TestConfigurationHelpers:
    """Test the configuration helper functions."""
    
    @patch('stereomapper.config.integration.get_settings')
    def test_get_chunk_size(self, mock_get_settings):
        """Test get_chunk_size function."""
        mock_settings = Mock()
        mock_settings.processing.chunk_size = 1000
        mock_get_settings.return_value = mock_settings
        
        result = get_chunk_size()
        
        assert result == 1000
        mock_get_settings.assert_called_once()

    @patch('stereomapper.config.integration.get_settings')
    def test_get_max_workers(self, mock_get_settings):
        """Test get_max_workers function."""
        mock_settings = Mock()
        mock_settings.processing.max_workers = 8
        mock_get_settings.return_value = mock_settings
        
        result = get_max_workers()
        
        assert result == 8
        mock_get_settings.assert_called_once()
    
    @patch('stereomapper.config.integration.get_settings')
    def test_get_cache_settings(self, mock_get_settings):
        """Test get_cache_settings function."""
        mock_settings = Mock()
        mock_settings.database.cache_path = Path("/cache/db.sqlite")
        mock_settings.database.output_path = Path("/output/results.db")
        mock_settings.database.fresh_cache = True
        mock_settings.database.pragma_settings = {"journal_mode": "WAL"}
        mock_get_settings.return_value = mock_settings
        
        result = get_cache_settings()
        
        expected = {
            'cache_path': Path("/cache/db.sqlite"),
            'output_path': Path("/output/results.db"),
            'fresh_cache': True,
            'pragma_settings': {"journal_mode": "WAL"},
        }
        
        assert result == expected
        mock_get_settings.assert_called_once()
    
    @patch('stereomapper.config.integration.get_settings')
    def test_get_cache_settings_none_values(self, mock_get_settings):
        """Test get_cache_settings with None values."""
        mock_settings = Mock()
        mock_settings.database.cache_path = None
        mock_settings.database.output_path = None
        mock_settings.database.fresh_cache = False
        mock_settings.database.pragma_settings = {}
        mock_get_settings.return_value = mock_settings
        
        result = get_cache_settings()
        
        expected = {
            'cache_path': None,
            'output_path': None,
            'fresh_cache': False,
            'pragma_settings': {},
        }
        
        assert result == expected


class TestPipelineConfigEdgeCases:
    """Test edge cases for PipelineConfig."""
    
    def test_pipeline_config_extensions_is_tuple(self):
        """Test that extensions is a tuple (but dataclass fields are mutable by default)."""
        config = PipelineConfig(input=None)
        
        # This should work - accessing the tuple
        assert config.extensions == (".mol", ".sdf")
        assert isinstance(config.extensions, tuple)
        
        # Dataclasses are mutable by default, so we can modify the field
        # (though the tuple itself is immutable)
        config.extensions = (".txt", ".csv")
        assert config.extensions == (".txt", ".csv")
    
    def test_from_settings_path_conversion(self):
        """Test that Path objects are properly converted to strings."""
        mock_settings = Mock()
        mock_settings.input_files = [Path("/test/file.mol")]
        mock_settings.input_directory = Path("/test/dir")
        mock_settings.database.output_path = Path("/test/output.db")
        mock_settings.database.cache_path = Path("/test/cache.db")
        mock_settings.database.fresh_cache = False
        mock_settings.namespace = "path_test"
        
        # Mock missing attributes
        mock_settings.recursive = None
        mock_settings.relate_with_cache = None
        
        config = PipelineConfig.from_settings(mock_settings)
        
        # Verify strings, not Path objects
        assert isinstance(config.input[0], str)
        assert config.input[0] == "/test/file.mol"
        assert isinstance(config.input_dir, str)
        assert config.input_dir == "/test/dir"
        assert isinstance(config.sqlite_output_path, str)
        assert config.sqlite_output_path == "/test/output.db"
        assert isinstance(config.cache_path, str)
        assert config.cache_path == "/test/cache.db"
    
    def test_from_settings_missing_attributes(self):
        """Test from_settings when settings object lacks optional attributes."""
        mock_settings = Mock()
        mock_settings.input_files = [Path("/test/file.mol")]
        mock_settings.input_directory = Path("/test/dir")
        mock_settings.database.output_path = Path("/test/output.db")
        mock_settings.database.cache_path = Path("/test/cache.db")
        mock_settings.database.fresh_cache = False
        mock_settings.namespace = "missing_attrs"
        
        # Don't set recursive or relate_with_cache attributes
        # This will test the getattr calls with defaults
        del mock_settings.recursive
        del mock_settings.relate_with_cache
        
        # Mock spec to control which attributes exist
        mock_settings = Mock(spec=['input_files', 'input_directory', 'database', 'namespace'])
        mock_settings.input_files = [Path("/test/file.mol")]
        mock_settings.input_directory = Path("/test/dir")
        mock_settings.database.output_path = Path("/test/output.db")
        mock_settings.database.cache_path = Path("/test/cache.db")
        mock_settings.database.fresh_cache = False
        mock_settings.namespace = "missing_attrs"
        
        config = PipelineConfig.from_settings(mock_settings)
        
        # Should use defaults when attributes are missing
        assert config.recursive is False  # getattr default
        assert config.relate_with_cache is False  # getattr default


@pytest.fixture
def sample_settings():
    """Fixture providing a sample Settings object."""
    mock_settings = Mock()
    mock_settings.input_files = [Path("/sample/file1.mol"), Path("/sample/file2.sdf")]
    mock_settings.input_directory = Path("/sample/input")
    mock_settings.database.output_path = Path("/sample/output.db")
    mock_settings.database.cache_path = Path("/sample/cache.db")
    mock_settings.database.fresh_cache = True
    mock_settings.database.pragma_settings = {"synchronous": "NORMAL"}
    mock_settings.namespace = "sample"
    mock_settings.processing.chunk_size = 100
    mock_settings.processing.batch_size = 50
    mock_settings.processing.max_workers = 4
    return mock_settings


@pytest.fixture
def sample_pipeline_config():
    """Fixture providing a sample PipelineConfig."""
    return PipelineConfig(
        input=["test1.mol", "test2.sdf"],
        input_dir="/test/input",
        recursive=True,
        sqlite_output_path="/test/output.db",
        cache_path="/test/cache.db",
        fresh_cache=True,
        namespace="test",
        relate_with_cache=True
    )


class TestWithFixtures:
    """Test integration functions using fixtures."""
    
    def test_pipeline_config_with_fixture(self, sample_pipeline_config):
        """Test PipelineConfig using fixture."""
        config = sample_pipeline_config
        
        assert config.input == ["test1.mol", "test2.sdf"]
        assert config.namespace == "test"
        assert config.recursive is True
    
    @patch('stereomapper.config.integration.get_settings')
    def test_integration_functions_with_fixture(self, mock_get_settings, sample_settings):
        """Test all integration functions with fixture."""
        mock_get_settings.return_value = sample_settings
        
        # Test all helper functions
        assert get_chunk_size() == 100
        assert get_max_workers() == 4
        
        cache_settings = get_cache_settings()
        assert cache_settings['fresh_cache'] is True
        assert cache_settings['pragma_settings'] == {"synchronous": "NORMAL"}