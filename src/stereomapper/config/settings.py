"""Core configuration settings for stereomapper."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from stereomapper.domain.exceptions import PipelineConfigurationError as ConfigurationError

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Supported logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class OutputFormat(Enum):
    """Supported output formats."""
    SQLITE = "sqlite"

@dataclass
class ProcessingSettings:
    """Processing-related configuration."""
    chunk_size: int = 2000
    max_workers: int = field(default_factory=lambda: os.cpu_count() or 4)
    timeout_seconds: int = 300
    retry_attempts: int = 3
    enable_cache: bool = True

    def validate(self) -> None:
        """Validate processing settings."""
        if self.chunk_size <= 0:
            raise ConfigurationError(
                "chunk_size must be positive",
                config_field="processing.chunk_size"
            )

        if self.max_workers <= 0:
            raise ConfigurationError(
                "max_workers must be positive",
                config_field="processing.max_workers"
            )

@dataclass
class DatabaseSettings:
    """Database-related configuration."""
    cache_path: Optional[Path] = None
    output_path: Optional[Path] = None
    fresh_cache: bool = False
    pragma_settings: Dict[str, Any] = field(default_factory=lambda: {
        "journal_mode": "WAL",
        "synchronous": "NORMAL",
        "cache_size": -64000,  # 64MB
        "temp_store": "MEMORY"
    })

    def validate(self) -> None:
        """Validate database settings."""
        if self.cache_path and not self.cache_path.parent.exists():
            raise ConfigurationError(
                f"Cache directory does not exist: {self.cache_path.parent}",
                config_field="database.cache_path"
            ).add_suggestion("Create the directory or use a different path")

        if self.output_path and not self.output_path.parent.exists():
            raise ConfigurationError(
                f"Output directory does not exist: {self.output_path.parent}",
                config_field="database.output_path"
            ).add_suggestion("Create the directory or use a different path")

@dataclass
class ChemistrySettings:
    """Chemistry-related configuration."""
    enable_stereochemistry: bool = True
    sanitize_molecules: bool = True
    remove_hydrogens: bool = True
    standardize_molecules: bool = True
    canonicalization_tool: str = "openbabel"  # "openbabel" or "rdkit"
    alignment_timeout: int = 30
    rmsd_threshold: float = 0.5

    def validate(self) -> None:
        """Validate chemistry settings."""
        valid_tools = {"openbabel", "rdkit"}
        if self.canonicalization_tool not in valid_tools:
            raise ConfigurationError(
                f"Invalid canonicalization tool: {self.canonicalization_tool}",
                config_field="chemistry.canonicalization_tool"
            ).add_suggestion(f"Use one of: {valid_tools}")

        if self.rmsd_threshold < 0:
            raise ConfigurationError(
                "RMSD threshold must be non-negative",
                config_field="chemistry.rmsd_threshold"
            )

@dataclass
class OutputSettings:
    """Output-related configuration."""
    format: OutputFormat = OutputFormat.SQLITE  # Keep this field, always SQLite
    include_errors: bool = True
    include_metadata: bool = True
    verbose_errors: bool = False

    def validate(self) -> None:
        """Validate output settings."""
        # Always SQLite, no validation needed
        pass

@dataclass
class LoggingSettings:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    file_path: Optional[Path] = None
    console_output: bool = True
    include_timestamps: bool = True
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def validate(self) -> None:
        """Validate logging settings."""
        if self.file_path and not self.file_path.parent.exists():
            raise ConfigurationError(
                f"Log directory does not exist: {self.file_path.parent}",
                config_field="logging.file_path"
            ).add_suggestion("Create the directory or use console logging only")

@dataclass
class Settings:
    """Main configuration settings for stereomapper."""

    # Core settings
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    chemistry: ChemistrySettings = field(default_factory=ChemistrySettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    # Runtime settings
    input_files: List[Path] = field(default_factory=list)
    input_directory: Optional[Path] = None
    namespace: str = "default"
    source_kind: str = "file"
    std_version: int = 1

    # Command-specific settings (not in base dataclass)
    relate_with_cache: bool = False
    recursive: bool = False

    # Debug/development settings
    debug_mode: bool = False
    profile_performance: bool = False
    dry_run: bool = False

    def validate(self) -> None:
        """Validate all configuration settings."""
        try:
            self.processing.validate()
            self.database.validate()
            self.chemistry.validate()
            self.output.validate()
            self.logging.validate()

            # Cross-validation
            self._validate_input_sources()
            self._validate_output_compatibility()

        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                f"Configuration validation failed: {str(e)}"
            ) from e

    def _validate_input_sources(self) -> None:
        """Validate input file/directory configuration."""
        has_files = bool(self.input_files)
        has_directory = self.input_directory is not None

        if not has_files and not has_directory:
            raise ConfigurationError(
                "Either input_files or input_directory must be specified",
                config_field="input_sources"
            ).add_suggestion("Provide --input-dir or specific file paths")

        if has_files and has_directory:
            raise ConfigurationError(
                "Cannot specify both input_files and input_directory",
                config_field="input_sources"
            ).add_suggestion("Use either --input-dir OR specific file paths, not both")

        if has_directory and not self.input_directory.exists():
            raise ConfigurationError(
                f"Input directory does not exist: {self.input_directory}",
                config_field="input_directory"
            )

        if has_directory and not self.input_directory.is_dir():
            raise ConfigurationError(
                f"Input directory path is not a directory: {self.input_directory}",
                config_field="input_directory"
            )

    def _validate_output_compatibility(self) -> None:
        """Validate output format compatibility."""
        # Since we only support SQLite, just check that database path is provided
        if not self.database.output_path:
            raise ConfigurationError(
                "SQLite output format requires database.output_path",
                config_field="database.output_path"
            ).add_suggestion("Provide --sqlite-output path")

    def to_dict(self) -> dict:
        """Convert settings to dictionary for debugging."""
        return {
            'processing': {
                'chunk_size': self.processing.chunk_size,
                'max_workers': self.processing.max_workers,
                'timeout_seconds': self.processing.timeout_seconds,
            },
            'database': {
                'cache_path': str(self.database.cache_path) if self.database.cache_path else None,
                'output_path': (
                    str(self.database.output_path) if self.database.output_path else None
                    ),
                'fresh_cache': self.database.fresh_cache,
            },
            'output': {
                'format': self.output.format.value,
                'include_errors': self.output.include_errors,
                'verbose_errors': self.output.verbose_errors,
            },
            'runtime': {
                'namespace': self.namespace,
                'debug_mode': self.debug_mode,
                'dry_run': self.dry_run,
            }
        }

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get the current global settings instance."""
    global _settings
    if _settings is None:
        raise ConfigurationError(
            "Settings not initialized. Call set_settings() first."
        ).add_suggestion("Initialize settings in your application startup")
    return _settings

def set_settings(settings: Settings) -> None:
    """Set the global settings instance."""
    global _settings
    settings.validate()  # Validate before setting
    _settings = settings
    logger.info("Configuration loaded and validated successfully")
    