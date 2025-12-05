"""Configuration loading from CLI and programmatic sources."""

import os
import logging
from pathlib import Path
from dataclasses import replace
from stereomapper.config.settings import (
    Settings, ProcessingSettings, DatabaseSettings,
    ChemistrySettings, OutputSettings, LoggingSettings,
    LogLevel, OutputFormat
)
from stereomapper.domain.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class ConfigurationLoader:
    """Loads configuration from CLI args and system defaults."""

    def load_from_cli_args(self, args) -> Settings:
        """Load configuration from CLI arguments."""
        try:
            # Start with system defaults
            settings = self.load_defaults()
            # Processing settings updates
            processing_updates = {}
            if hasattr(args, 'chunk_size') and args.chunk_size:
                processing_updates['chunk_size'] = args.chunk_size
            if hasattr(args, 'max_workers') and args.max_workers:
                processing_updates['max_workers'] = args.max_workers
            if hasattr(args, 'timeout') and args.timeout:
                processing_updates['timeout_seconds'] = args.timeout

            # Database settings updates
            database_updates = {}
            if hasattr(args, 'cache_path') and args.cache_path:
                database_updates['cache_path'] = Path(args.cache_path)
            if hasattr(args, 'sqlite_output') and args.sqlite_output:
                database_updates['output_path'] = Path(args.sqlite_output)
            if hasattr(args, 'fresh_cache'):
                database_updates['fresh_cache'] = args.fresh_cache

           # Output settings updates (simplified)
            output_updates = {}
            # REMOVED: format handling since we only support SQLite
            if hasattr(args, 'no_errors') and args.no_errors:
                output_updates['include_errors'] = False
            if hasattr(args, 'verbose_errors') and args.verbose_errors:
                output_updates['verbose_errors'] = True

            # Logging settings updates
            logging_updates = {}
            if hasattr(args, 'log_file') and args.log_file:
                logging_updates['file_path'] = Path(args.log_file)
            if hasattr(args, 'debug') and args.debug:
                logging_updates['level'] = LogLevel.DEBUG

            # Input handling
            input_files = []
            input_directory = None

            if hasattr(args, 'input') and args.input:
                input_files = [Path(f) for f in args.input]
            elif hasattr(args, 'input_dir') and args.input_dir:
                input_directory = Path(args.input_dir)

            # Runtime settings
            namespace = getattr(args, 'namespace', 'default')
            relate_with_cache = getattr(args, 'relate_with_cache', False)
            recursive = getattr(args, 'recursive', False)
            debug_mode = getattr(args, 'debug', False)
            dry_run = getattr(args, 'dry_run', False)

            # Apply all updates
            updated_settings = replace(
                settings,
                processing=replace(settings.processing, **processing_updates),
                database=replace(settings.database, **database_updates),
                # REMOVED: chemistry=replace(settings.chemistry, **chemistry_updates),
                output=replace(settings.output, **output_updates),
                logging=replace(settings.logging, **logging_updates),
                input_files=input_files,
                input_directory=input_directory,
                namespace=namespace,
                debug_mode=debug_mode,
                dry_run=dry_run,
            )

            # Add custom fields that aren't in the base Settings dataclass
            updated_settings.relate_with_cache = relate_with_cache
            updated_settings.recursive = recursive

            return updated_settings

        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration from CLI arguments: {str(e)}"
            ) from e

    def load_defaults(self) -> Settings:
        """Load default configuration settings."""
        return Settings(
            processing=ProcessingSettings(
                chunk_size=2000,
                max_workers=os.cpu_count() or 4,
                timeout_seconds=300,
                retry_attempts=3,
                enable_cache=True,
            ),
            database=DatabaseSettings(
                cache_path=None,
                output_path=None,
                fresh_cache=False,
                pragma_settings={
                    "journal_mode": "DELETE",
                    "synchronous": "NORMAL",
                    "cache_size": -64000,
                    "temp_store": "MEMORY"
                }
            ),
            chemistry=ChemistrySettings(
                enable_stereochemistry=True,
                sanitize_molecules=True,
                remove_hydrogens=True,
                standardize_molecules=True,
                canonicalization_tool="openbabel",
                alignment_timeout=30,
                rmsd_threshold=0.5,
            ),
            output=OutputSettings(
                format=OutputFormat.SQLITE,  # Always SQLite
                include_errors=True,
                include_metadata=True,
                verbose_errors=False,
            ),

            logging=LoggingSettings(
                level=LogLevel.INFO,
                file_path=None,
                console_output=True,
                include_timestamps=True,
                format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            ),
            namespace="default",
            source_kind="file",
            std_version=1,
            debug_mode=False,
            profile_performance=False,
            dry_run=False,
        )

def configure_from_cli(args) -> Settings:
    """Main entry point to configure settings from CLI args."""
    loader = ConfigurationLoader()
    settings = loader.load_from_cli_args(args)
    settings.validate()
    return settings
