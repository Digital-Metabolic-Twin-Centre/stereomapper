import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from stereomapper.config.resolvers import _resolve_cache_path, default_cache_path


class TestResolveCachePath:
    
    def test_both_flags_true_raises_error(self):
        """Test that setting both relate_with_cache and fresh_cache raises ValueError."""
        with pytest.raises(ValueError, match="Cannot set both relate_with_cache=True and fresh_cache=True"):
            _resolve_cache_path(relate_with_cache=True, fresh_cache=True, cache_path=None)
    
    def test_relate_with_cache_existing_default_path(self, tmp_path):
        """Test relate_with_cache=True with existing default cache file."""
        cache_file = tmp_path / "test_cache.sqlite"
        cache_file.touch()
        
        with patch('stereomapper.config.resolvers.default_cache_path', return_value=cache_file):
            result = _resolve_cache_path(relate_with_cache=True, fresh_cache=False, cache_path=None)
            assert result == cache_file
    
    def test_relate_with_cache_missing_default_path(self, tmp_path):
        """Test relate_with_cache=True with missing default cache file raises FileNotFoundError."""
        cache_file = tmp_path / "missing_cache.sqlite"
        
        with patch('stereomapper.config.resolvers.default_cache_path', return_value=cache_file):
            with pytest.raises(FileNotFoundError, match=f"relate_with_cache=True but cache DB not found: {cache_file}"):
                _resolve_cache_path(relate_with_cache=True, fresh_cache=False, cache_path=None)
    
    def test_relate_with_cache_existing_custom_path(self, tmp_path):
        """Test relate_with_cache=True with existing custom cache file."""
        cache_file = tmp_path / "custom_cache.sqlite"
        cache_file.touch()
        
        result = _resolve_cache_path(relate_with_cache=True, fresh_cache=False, cache_path=str(cache_file))
        assert result == cache_file
    
    def test_relate_with_cache_missing_custom_path(self, tmp_path):
        """Test relate_with_cache=True with missing custom cache file raises FileNotFoundError."""
        cache_file = tmp_path / "missing_custom.sqlite"
        
        with pytest.raises(FileNotFoundError, match=f"relate_with_cache=True but cache DB not found: {cache_file}"):
            _resolve_cache_path(relate_with_cache=True, fresh_cache=False, cache_path=str(cache_file))
    
    def test_fresh_cache_with_existing_default_file(self, tmp_path):
        """Test fresh_cache=True with existing default file - should delete and return path."""
        cache_file = tmp_path / "existing_cache.sqlite"
        cache_file.touch()
        assert cache_file.exists()
        
        with patch('stereomapper.config.resolvers.default_cache_path', return_value=cache_file):
            result = _resolve_cache_path(relate_with_cache=False, fresh_cache=True, cache_path=None)
            assert result == cache_file
            assert not cache_file.exists()  # Should be deleted
    
    def test_fresh_cache_with_missing_default_file(self, tmp_path):
        """Test fresh_cache=True with missing default file - should return path."""
        cache_file = tmp_path / "new_cache.sqlite"
        
        with patch('stereomapper.config.resolvers.default_cache_path', return_value=cache_file):
            result = _resolve_cache_path(relate_with_cache=False, fresh_cache=True, cache_path=None)
            assert result == cache_file
    
    def test_fresh_cache_with_existing_custom_file(self, tmp_path):
        """Test fresh_cache=True with existing custom file - should delete and return path."""
        cache_file = tmp_path / "custom_existing.sqlite"
        cache_file.touch()
        assert cache_file.exists()
        
        result = _resolve_cache_path(relate_with_cache=False, fresh_cache=True, cache_path=str(cache_file))
        assert result == cache_file
        assert not cache_file.exists()  # Should be deleted
    
    def test_fresh_cache_with_missing_custom_file(self, tmp_path):
        """Test fresh_cache=True with missing custom file - should return path."""
        cache_file = tmp_path / "custom_new.sqlite"
        
        result = _resolve_cache_path(relate_with_cache=False, fresh_cache=True, cache_path=str(cache_file))
        assert result == cache_file
    
    def test_neither_flag_with_default_path(self, tmp_path):
        """Test neither flag set - should return default path."""
        cache_file = tmp_path / "default_cache.sqlite"
        
        with patch('stereomapper.config.resolvers.default_cache_path', return_value=cache_file):
            result = _resolve_cache_path(relate_with_cache=False, fresh_cache=False, cache_path=None)
            assert result == cache_file
    
    def test_neither_flag_with_custom_path(self, tmp_path):
        """Test neither flag set with custom path - should return custom path."""
        cache_file = tmp_path / "custom_cache.sqlite"
        
        result = _resolve_cache_path(relate_with_cache=False, fresh_cache=False, cache_path=str(cache_file))
        assert result == cache_file