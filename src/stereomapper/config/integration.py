"""Integration helpers for existing pipeline code."""
from typing import List, Optional
from dataclasses import dataclass
from stereomapper.config.settings import Settings, get_settings

@dataclass
class PipelineConfig:
    """Legacy pipeline config for backward compatibility."""
    input: Optional[List[str]]
    input_dir: Optional[str] = None
    recursive: bool = False
    extensions = (".mol", ".sdf")
    sqlite_output_path: str = None
    cache_path: Optional[str] = None
    fresh_cache: bool = False
    namespace: str = "default"
    relate_with_cache: bool = False

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        inputs: Optional[List[str]] = None,
        output_path: Optional[str] = None) -> 'PipelineConfig':
        """Create PipelineConfig from Settings for backward compatibility."""
        return cls(
            input=inputs or
            ([str(f) for f in settings.input_files] if settings.input_files else None),
            input_dir=(
                str(settings.input_directory) if settings.input_directory and not inputs else None
                ),
            recursive=getattr(settings, 'recursive', False),
            sqlite_output_path=output_path or (
                str(settings.database.output_path) if settings.database.output_path else None
                ),
            cache_path=(
                str(settings.database.cache_path) if settings.database.cache_path else None
                ),
            fresh_cache=settings.database.fresh_cache,
            namespace=settings.namespace,
            relate_with_cache=getattr(settings, 'relate_with_cache', False),
        )

def get_pipeline_config() -> PipelineConfig:
    """Get PipelineConfig from current settings."""
    settings = get_settings()
    return PipelineConfig.from_settings(settings)

def get_chunk_size() -> int:
    """Get chunk size from configuration."""
    return get_settings().processing.chunk_size

def get_max_workers() -> int:
    """Get max workers from configuration."""
    return get_settings().processing.max_workers

def get_cache_settings() -> dict:
    """Get cache/database settings."""
    settings = get_settings()
    return {
        'cache_path': settings.database.cache_path,
        'output_path': settings.database.output_path,
        'fresh_cache': settings.database.fresh_cache,
        'pragma_settings': settings.database.pragma_settings,
    }
