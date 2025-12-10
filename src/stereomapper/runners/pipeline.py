import os
import time
import sqlite3
import logging
import json
import psutil
from tqdm import tqdm
from pathlib import Path
from itertools import tee, islice
from typing import List, Optional, Dict
from dataclasses import dataclass
from stereomapper.utils.logging import setup_logging
from stereomapper.utils.timing import timeit
from stereomapper.utils.itertools import chunked  # removed: no longer used
from stereomapper.config.resolvers import _resolve_inputs_from_cfg, _resolve_cache_path
from stereomapper.config.integration import PipelineConfig
from stereomapper.data import db, cache_repo, results_repo, cache_schema, results_schema
from stereomapper.domain.models import ProcessingResult
from stereomapper.comparison.compare import compare_cluster_relationships
from stereomapper.results import assemblers
from stereomapper.processing import BatchProcessor, InputValidator
# from stereomapper.comparison.compare import compare_cluster_relationships  # removed: duplicate
# from stereomapper.utils.timing import section_timer  # removed: unused

# Add these imports at the top
from stereomapper.domain.exceptions import (
    ProcessingError,
    CacheError,
    ValidationError,
    ConfigurationError
)

# Configuration integration
try:
    from stereomapper.config.settings import get_settings
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False

logger, summary_logger = setup_logging(
    console=True,
    level="INFO",           # Detailed logging to files
    quiet_console=True,     # Minimal console output during progress bar
    console_level="ERROR"   # Only errors to console
)

@dataclass
class PipelineResult:
    """Pipeline execution result."""
    n_inputs: int
    n_session_pairs: int
    n_cross_pairs: int
    output_path: Optional[str] = None
    processing_time: float = 0.0
    cache_hits: int = 0
    processing_errors: int = 0

class stereomapperPipeline:
    """
    Modular stereomapper 2D batch processing pipeline.

    Benefits:
    - Each step is testable in isolation
    - Configuration is injected once
    - State is tracked throughout execution
    - Easy to extend with new processing steps
    - Better error handling and recovery
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.start_time = None
        self.cache_conn = None
        self.process = psutil.Process(os.getpid())
        self.processed_molfiles = []

        # Validate configuration early
        self._validate_config()

        # Get settings from configuration system if available
        self.chunk_size = self._get_chunk_size()
        self.pragma_settings = self._get_pragma_settings()
        self.use_wal = self._should_use_wal()

        if self.chunk_size > 2000:
            logger.warning(f"[init] Large chunk size configured: {self.chunk_size}. Changing to practical limit of 2000.")
            self.chunk_size = 2000

        # Initialize processors
        self.batch_processor = None
        self.version_tag = "v1.0"

        # Metrics tracking
        self.metrics = {
            'files_processed': 0,
            'files_succeeded': 0,
            'files_failed': 0,
            'files_skipped': 0,
            'cache_hits': 0,
            'unique_inchikeys': 0,
            'groups_processed': 0,
            'groups_skipped': 0,
            'groups_failed': 0,
            'processing_time': 0.0,
            'relationship_totals': 0,
            'relationship_class_counts': {},
            'relationship_origin_counts': {},
        }

    def run(self) -> PipelineResult:
        """Execute the complete pipeline."""
        self.start_time = time.time()

        # Initialize main pipeline progress bar
        pipeline_pbar = None
        show_progress = not os.getenv('NO_PROGRESS', '').lower() in ['1', 'true', 'yes']

        try:
            if show_progress:
                pipeline_pbar = tqdm(
                    total=100,  # 100% completion
                    desc="Pipeline",
                    unit="%",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix} [{elapsed}<{remaining}]",
                    ncols=100,  # Shorter width to avoid wrapping
                    position=0,  # Keep at top
                    leave=True   # Leave the bar when done
                )
                pipeline_pbar.set_postfix_str("Initializing...")

            # Temporarily suppress logging during progress bar display
            if show_progress:
                # Increase log level to reduce noise during progress display
                original_level = logger.level
                logger.setLevel(logging.CRITICAL)  # Only show critical errors

                # Also suppress warnings from other modules
                import warnings
                warnings.filterwarnings("ignore")

            # Step 1: Initialize and validate (5% of pipeline)
            molfiles = self._initialize_and_validate()
            if pipeline_pbar:
                pipeline_pbar.update(5)
                pipeline_pbar.set_postfix_str("Setting up databases...")

            # Step 2: Setup databases (5% of pipeline)
            self._setup_cache_database()
            if pipeline_pbar:
                pipeline_pbar.update(5)
                pipeline_pbar.set_postfix_str("Processing molecules...")

            # Step 3: Process molecules (70% of pipeline - usually the longest)
            self._process_molecules(molfiles, pipeline_pbar, 70)
            if pipeline_pbar:
                pipeline_pbar.set_postfix_str("Calculating relationships...")

            # Step 4: Generate results (20% of pipeline)
            self._generate_results(pipeline_pbar, 20)
            if pipeline_pbar:
                pipeline_pbar.set_postfix_str("Finalizing...")

            # Restore logging level
            if show_progress:
                logger.setLevel(original_level)
                warnings.resetwarnings()

            # Step 5: Cleanup and return results
            return self._finalize(pipeline_pbar)

        except (ProcessingError, CacheError, ValidationError, ConfigurationError):
            if pipeline_pbar:
                pipeline_pbar.close()
            # Restore logging level on error
            if show_progress:
                logger.setLevel(original_level)
                warnings.resetwarnings()
            self._cleanup()
            raise
        except Exception as e:
            if pipeline_pbar:
                pipeline_pbar.close()
            # Restore logging level on error
            if show_progress:
                logger.setLevel(original_level)
                warnings.resetwarnings()
            self._cleanup()
            exc = ProcessingError(
                f"Unexpected pipeline error: {str(e)}",
                stage="pipeline_execution"
            )
            exc.add_context('elapsed_time', time.time() - self.start_time)
            raise exc from e

    def _validate_config(self) -> None:
        """Validate pipeline configuration."""
        if not self.config.input and not self.config.input_dir:
            raise ConfigurationError(
                "Either input files or input directory must be specified",
                config_field="input_sources"
            ).add_suggestion("Provide either config.input or config.input_dir")

        if self.config.input and self.config.input_dir:
            raise ConfigurationError(
                "Cannot specify both input files and input directory",
                config_field="input_sources"
            ).add_suggestion("Use either config.input OR config.input_dir, not both")

        if not self.config.sqlite_output_path:
            raise ConfigurationError(
                "SQLite output path is required",
                config_field="sqlite_output_path"
            ).add_suggestion("Set config.sqlite_output_path")

        # Validate output directory is writable
        output_path = Path(self.config.sqlite_output_path)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ConfigurationError(
                f"Cannot create output directory: {output_path.parent}",
                config_field="sqlite_output_path"
            ).add_context('error', str(e))

        logger.debug(f"Configuration validated successfully")

    def _initialize_and_validate(self) -> List[str]:
        """Initialize pipeline and validate inputs."""
        summary_logger.info(f"[startup] Starting stereomapper 2D batch processing")

        # Resolve input files using your existing logic
        molfiles = _resolve_inputs_from_cfg(self.config)
        summary_logger.info(f"[startup] Found {len(molfiles)} input molfiles")

        validator = InputValidator()
        # Validate molfile paths using your existing logic
        valid_molfiles, invalid_molfiles = validator.validate_molfile_paths(molfiles)
        if invalid_molfiles:
            self.metrics['files_skipped'] += len(invalid_molfiles)
            logger.warning(f"[input-check] Found {len(invalid_molfiles)} invalid molfile paths; they will be skipped.")

        # Initialize batch processor
        batch_params = validator.validate_batch_parameters(
            batch_size=self.chunk_size,
            total_files=len(valid_molfiles),
            chunk_size=self.chunk_size
        )
        self.batch_processor = BatchProcessor(chunk_size=batch_params["chunk_size"])

        logger.info(f"[init] Pipeline initialized: {len(valid_molfiles)} valid files, chunk_size={self.chunk_size}")
        return valid_molfiles

    def _setup_cache_database(self) -> None:
        """Setup and connect to cache database."""
        try:
            cache_db_path = _resolve_cache_path(
                relate_with_cache=self.config.relate_with_cache,
                fresh_cache=self.config.fresh_cache,
                cache_path=self.config.cache_path,
            )

            if cache_db_path is None:
                raise CacheError(
                    "Cache database path could not be resolved",
                    operation="cache_path_resolution"
                ).add_suggestion("Check cache_path configuration or file permissions")

            logger.info(f"[db-setup] Using cache database at: {cache_db_path}")
            self.cache_conn = db.connect(cache_db_path, use_wal=self.use_wal)
            cache_schema.create_cache(self.cache_conn)

            logger.info(f"[db-setup] Cache database ready")

        except CacheError:
            raise
        except Exception as e:
            raise CacheError(
                f"Failed to setup cache database: {str(e)}",
                operation="database_setup"
            ).add_context('cache_path', cache_db_path) from e

    def _process_molecules(self, molfiles: List[str], pipeline_pbar=None, allocated_percent=70) -> None:
        """Process molecules in chunks with progress tracking."""
        summary_logger.info(f"[processing] Starting processing of {len(molfiles)} molfiles")
        self.processed_molfiles = molfiles
        if not molfiles:
            raise ValidationError(
                "No valid molfiles to process",
                field_name="molfiles"
            ).add_suggestion("Check input files exist and are readable")

        total = len(molfiles)
        done = 0
        consecutive_failures = 0
        max_consecutive_failures = 5

        # Track last exception for context-safe access
        last_chunk_exception = None
        last_error_type = None

        # index-based batching enables dynamic chunk_size changes to take effect
        idx = 0
        batch_index = 0

        try:
            while idx < total:
                batch_index += 1
                batch_size = min(self.chunk_size, total - idx)
                batch = molfiles[idx: idx + batch_size]

                batch_start = time.perf_counter()
                batch_succeeded = 0
                batch_failed = 0
                batch_cache_hits = 0
                exception_in_batch = False

                try:
                    results: List[ProcessingResult] = self.batch_processor.process_batch(
                        batch,
                        self.cache_conn,
                        namespace=self.config.namespace,
                    )
                    consecutive_failures = 0
                    last_chunk_exception = None
                    last_error_type = None
                except Exception as chunk_error:
                    last_chunk_exception = chunk_error
                    last_error_type = type(chunk_error).__name__
                    logger.error(
                        f"[chunk-{batch_index}] batch failed ({last_error_type}): {chunk_error}. "
                        f"chunk_size={self.chunk_size} files_in_batch={batch_size}"
                    )
                    for bad_sample in batch[:5]:
                        logger.debug(f"[chunk-{batch_index}] sample path: {bad_sample}")
                    logger.debug("Exception details", exc_info=True)
                    exception_in_batch = True
                    consecutive_failures += 1

                    # Adaptive fallback for subsequent batches
                    if consecutive_failures == 1:
                        new_size = max(500, self.chunk_size // 20)
                        if new_size < self.chunk_size:
                            self.chunk_size = new_size
                            logger.warning(f"[chunk-{batch_index}] Reducing chunk_size to {self.chunk_size} after first failure")
                    elif consecutive_failures == 2:
                        self.chunk_size = max(200, self.chunk_size // 5)
                        logger.warning(f"[chunk-{batch_index}] Further reducing chunk_size to {self.chunk_size}")
                    results = []

                if consecutive_failures >= max_consecutive_failures:
                    pe = ProcessingError(
                        f"Too many consecutive chunk failures ({consecutive_failures})",
                        stage="chunk_processing"
                    ).add_context('failed_chunk', batch_index)\
                     .add_context('batch_size', batch_size)\
                     .add_context('chunk_size', self.chunk_size)
                    if last_chunk_exception:
                        pe.add_context('last_error_type', last_error_type)\
                          .add_context('last_error_message', str(last_chunk_exception))
                    raise pe

                # Tally outcomes
                for result in results:
                    if result.error is None:
                        batch_succeeded += 1
                        self.metrics['files_succeeded'] += 1
                        if getattr(result, 'from_cache', False):
                            batch_cache_hits += 1
                            self.metrics['cache_hits'] += 1
                    else:
                        batch_failed += 1
                        self.metrics['files_failed'] += 1

                if exception_in_batch:
                    # Count whole batch as failed for metrics consistency
                    batch_failed = batch_size
                    self.metrics['files_failed'] += batch_size

                self.metrics['files_processed'] = self.metrics['files_succeeded'] + self.metrics['files_failed']
                done += batch_size
                idx += batch_size  # always advance; we do not retry same window

                if pipeline_pbar:
                    progress_percent = (done / total) * allocated_percent
                    current_progress = pipeline_pbar.n
                    target_progress = 10 + progress_percent
                    update_amount = min(target_progress - current_progress, allocated_percent)
                    if update_amount > 0:
                        pipeline_pbar.update(update_amount)
                    cache_rate = (self.metrics['cache_hits'] / self.metrics['files_succeeded'] * 100) if self.metrics['files_succeeded'] > 0 else 0
                    pipeline_pbar.set_postfix_str(
                        f"Files: {done}/{total} | OK: {self.metrics['files_succeeded']} | Cache: {cache_rate:.0f}%"
                    )
                else:
                    if (batch_index % 10 == 0) or (done >= total):
                        batch_time = time.perf_counter() - batch_start
                        speed = (batch_size / batch_time) if batch_time > 0 else float("inf")
                        summary_logger.info(
                            f"[chunk-{batch_index}] Processed {batch_size} files in {batch_time:.1f}s "
                            f"({speed:.1f}/s); batch: ok={batch_succeeded}, err={batch_failed}, cache={batch_cache_hits}; "
                            f"total: {done}/{total} (ok={self.metrics['files_succeeded']}, err={self.metrics['files_failed']})"
                        )

        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(
                f"Molecule processing failed: {str(e)}",
                stage="molecule_processing"
            ).add_context('files_processed', done)\
             .add_context('total_files', total)\
             .add_context('chunk_size', self.chunk_size)\
             .add_context('last_error_type', last_error_type or "n/a") from e

        if self.metrics['files_succeeded'] == 0:
            raise ProcessingError(
                "No files were processed successfully",
                stage="molecule_processing"
            ).add_context('total_files', total)\
             .add_context('failed_files', self.metrics['files_failed'])\
             .add_context('chunk_size_final', self.chunk_size)

    def _generate_results(self, pipeline_pbar=None, allocated_percent=20) -> None:
        """Generate similarity results from processed molecules."""
        if not pipeline_pbar:
            logger.info("[results] Starting similarity calculation")

        unique_inchikeys = list(dict.fromkeys(
            ik for ik in cache_repo.inchi_first_by_id(self.cache_conn, self.processed_molfiles, logger=logger) if ik
        ))

        if not unique_inchikeys:
            logger.error("[results] No valid inchikey_first found in cache database")
            return

        self.metrics['unique_inchikeys'] = len(unique_inchikeys)
        if not pipeline_pbar:
            logger.info(f"[results] Processing {len(unique_inchikeys)} unique inchikey_first values")

        with sqlite3.connect(self.config.sqlite_output_path, timeout=120.0) as res_con:
            # Apply pragma settings
            for pragma, value in self.pragma_settings.items():
                res_con.execute(f"PRAGMA {pragma}={value};")

            results_schema.results_schema(res_con)

            for i, inchikey_first in enumerate(unique_inchikeys, 1):
                try:
                    status = self._process_inchikey_group(res_con, inchikey_first, i, len(unique_inchikeys))
                except Exception:
                    self.metrics['groups_failed'] += 1
                    raise

                if status == "processed":
                    self.metrics['groups_processed'] += 1
                elif status == "skipped":
                    self.metrics['groups_skipped'] += 1
                else:
                    self.metrics['groups_failed'] += 1

                if pipeline_pbar:
                    progress_percent = (i / len(unique_inchikeys)) * allocated_percent
                    current_progress = pipeline_pbar.n
                    target_progress = 80 + progress_percent
                    update_amount = min(target_progress - current_progress, allocated_percent)
                    if update_amount > 0:
                        pipeline_pbar.update(update_amount)

                    pipeline_pbar.set_postfix_str(
                        f"Groups: {self.metrics['groups_processed']}/{len(unique_inchikeys)} ok "
                        f"| skipped {self.metrics['groups_skipped']} | failed {self.metrics['groups_failed']}"
                    )

        if not pipeline_pbar:
            logger.info("[results] Similarity calculation completed")


    def _process_inchikey_group(self, res_con: sqlite3.Connection, inchikey_first: str,
                               group_num: int, total_groups: int) -> str:
        """Process a single inchikey group. Returns processing status."""
        rows_iter = cache_repo.streamline_rows(self.cache_conn, inchikey_first)

        # Peek to check if data exists
        rows_iter, preview_iter = tee(rows_iter)
        preview = list(islice(preview_iter, 3))
        if not preview:
            logger.warning(f"[results] No data found for inchikey_first: {inchikey_first}")
            return "skipped"

        # Build clusters and upsert
        tuples_iter = assemblers.cluster_rows(rows_iter)
        results_repo.bulk_upsert_clusters(res_con, tuples_iter, chunk_size=5000)

        # Calculate relationships
        compare_cluster_relationships(
            results_db_path=self.config.sqlite_output_path,
            inchikey_first=inchikey_first,
            version_tag=self.version_tag,
            logger=logger,
        )

        if group_num % 10 == 0 or group_num == total_groups:
            logger.info(f"[results] Processed {group_num}/{total_groups} inchikey groups")

        return "processed"

    def _finalize(self, pipeline_pbar=None) -> PipelineResult:
        """Finalize pipeline and return results."""
        self._cleanup()

        elapsed_time = time.time() - self.start_time
        self.metrics['processing_time'] = elapsed_time
        self._update_relationship_metrics()

        attempted = self.metrics['files_processed']
        skipped = self.metrics['files_skipped']
        successes = self.metrics['files_succeeded']
        failures = self.metrics['files_failed']
        cache_rate = (self.metrics['cache_hits'] / successes * 100) if successes > 0 else 0.0
        relationship_total = self.metrics['relationship_totals']
        groups_processed = self.metrics['groups_processed']
        groups_skipped = self.metrics['groups_skipped']
        groups_failed = self.metrics['groups_failed']
        unique_groups = self.metrics['unique_inchikeys']

        # Complete the progress bar
        if pipeline_pbar:
            # Ensure we reach 100%
            remaining = 100 - pipeline_pbar.n
            if remaining > 0:
                pipeline_pbar.update(remaining)
            pipeline_pbar.set_postfix_str("Complete!")

            # Give a moment to see completion before closing
            time.sleep(0.5)
            pipeline_pbar.close()

            # Print final summary after progress bar is closed
            print(f"\n✅ Pipeline completed in {elapsed_time:.1f}s")
            print(f"📦 Inputs attempted: {attempted:,} (skipped {skipped:,})")
            print(f"📊 Successes: {successes:,} | Failures: {failures:,}")
            if any((groups_processed, groups_skipped, groups_failed)):
                print(
                    f"🔗 Inchikey groups — processed {groups_processed:,}, "
                    f"skipped {groups_skipped:,}, failed {groups_failed:,}"
                )
            print(f"🧮 Relationship rows: {relationship_total:,}")
            print(f"🧾 Unique inchikeys observed: {unique_groups:,}")
            print(f"💾 Cache hit rate: {cache_rate:.1f}%")
        else:
            summary_logger.info(f"[shutdown] Pipeline completed in {elapsed_time:.2f} seconds")
            self._log_final_metrics()

        origin_counts = self.metrics.get('relationship_origin_counts') or {}
        session_pairs = origin_counts.get('session', 0)
        total_pairs = relationship_total
        cross_pairs = max(total_pairs - session_pairs, 0)

        return PipelineResult(
            n_inputs=attempted,
            n_session_pairs=session_pairs,
            n_cross_pairs=cross_pairs,
            output_path=self.config.sqlite_output_path,
            processing_time=elapsed_time,
            cache_hits=self.metrics.get('cache_hits', 0),
            processing_errors=failures
        )

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.cache_conn:
            self.cache_conn.close()
            self.cache_conn = None

    def _update_relationship_metrics(self) -> None:
        """Summarize relationship rows from the results database."""
        output_path = self.config.sqlite_output_path
        if not output_path:
            return

        output_file = Path(output_path)
        if not output_file.exists():
            return

        class_counts: Dict[str, int] = {}
        origin_counts: Dict[str, int] = {}
        total = 0

        try:
            with sqlite3.connect(output_path, timeout=30.0) as con:
                for classification, count in con.execute(
                    """
                    SELECT classification, COUNT(*)
                    FROM relationships
                    WHERE version_tag = ?
                    GROUP BY classification
                    """,
                    (self.version_tag,),
                ):
                    label = classification or "unspecified"
                    class_counts[label] = count
                    total += count

                extra_rows = con.execute(
                    """
                    SELECT extra_info
                    FROM relationships
                    WHERE version_tag = ?
                    """,
                    (self.version_tag,),
                ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("[results] Could not summarise relationships: %s", exc)
            return

        for (payload,) in extra_rows:
            origin_label = "unspecified"
            if payload:
                parsed = None
                if isinstance(payload, str):
                    try:
                        parsed = json.loads(payload)
                    except Exception:
                        parsed = None
                if isinstance(parsed, dict):
                    origin_label = str(parsed.get("pair_origin") or "unspecified")
                else:
                    origin_label = "note"
            origin_counts[origin_label] = origin_counts.get(origin_label, 0) + 1

        self.metrics['relationship_totals'] = total
        self.metrics['relationship_class_counts'] = class_counts
        self.metrics['relationship_origin_counts'] = origin_counts

    def _memory_report(self, label: str, detailed: bool = False) -> None:
        """Report memory usage with optional detailed breakdown."""
        try:
            rss = self.process.memory_info().rss / 1e6  # MB

            # Get memory threshold from config
            memory_threshold = 2000  # Default 2GB
            if _CONFIG_AVAILABLE:
                try:
                    settings = get_settings()
                    memory_threshold = getattr(settings.processing, 'memory_threshold_mb', 2000)
                except:
                    pass

            if detailed:
                memory_info = self.process.memory_info()
                vms = memory_info.vms / 1e6  # MB
                percent = self.process.memory_percent()
                logger.debug(f"[mem] {label} RSS={rss:.1f}MB, VMS={vms:.1f}MB, %={percent:.1f}%")
            else:
                logger.debug(f"[mem] {label} RSS={rss:.1f}MB")

            # Warn if memory usage is high
            if rss > memory_threshold:
                logger.warning(f"[mem] High memory usage: {rss:.1f}MB (threshold: {memory_threshold}MB)")

        except Exception as e:
            logger.debug(f"[mem] Could not get memory info: {e}")

    def _log_final_metrics(self) -> None:
        """Log final pipeline metrics with performance analysis."""
        logger.info("="*60)
        logger.info("PIPELINE METRICS")
        logger.info("="*60)
        logger.info(f"Files processed:     {self.metrics['files_processed']:,}")
        logger.info(f"Files succeeded:     {self.metrics['files_succeeded']:,}")
        logger.info(f"Files failed:        {self.metrics['files_failed']:,}")
        logger.info(f"Files skipped:       {self.metrics['files_skipped']:,}")
        logger.info(f"Cache hits:          {self.metrics['cache_hits']:,}")
        logger.info(f"Processing time:     {self.metrics['processing_time']:.2f}s")
        logger.info(f"Groups processed:    {self.metrics['groups_processed']:,}")
        logger.info(f"Groups skipped:      {self.metrics['groups_skipped']:,}")
        logger.info(f"Groups failed:       {self.metrics['groups_failed']:,}")
        logger.info(f"Relationship rows:   {self.metrics['relationship_totals']:,}")

        # Performance analysis
        if self.metrics['processing_time'] > 0:
            rate = self.metrics['files_processed'] / self.metrics['processing_time']
            logger.info(f"Processing rate:     {rate:.1f} files/second")

        if self.metrics['files_processed'] > 0:
            success_rate = (self.metrics['files_succeeded'] / self.metrics['files_processed']) * 100
            cache_rate = (self.metrics['cache_hits'] / self.metrics['files_succeeded']) * 100 if self.metrics['files_succeeded'] > 0 else 0
            logger.info(f"Success rate:        {success_rate:.1f}%")
            logger.info(f"Cache hit rate:      {cache_rate:.1f}%")

        class_counts = self.metrics.get('relationship_class_counts') or {}
        if class_counts:
            logger.info("Relationship counts by classification:")
            for cls, count in sorted(class_counts.items()):
                logger.info(f"  {cls}: {count:,}")

        origin_counts = self.metrics.get('relationship_origin_counts') or {}
        if origin_counts:
            logger.info("Relationship counts by origin:")
            for origin, count in sorted(origin_counts.items()):
                logger.info(f"  {origin}: {count:,}")

        # Resource usage
        self._memory_report("Final memory usage", detailed=True)

        logger.info("="*60)

    # Configuration helper methods
    def _get_chunk_size(self) -> int:
        """Get chunk size from configuration."""
        if _CONFIG_AVAILABLE:
            try:
                settings = get_settings()
                return settings.processing.chunk_size
            except:
                pass
        return 2_000

    def _should_use_wal(self) -> bool:
        """Get WAL setting from configuration."""
        if _CONFIG_AVAILABLE:
            try:
                settings = get_settings()
                return settings.database.pragma_settings.get("journal_mode", "DELETE") == "WAL"
            except:
                pass
        return False

    def _get_pragma_settings(self) -> dict:
        """Get database pragma settings from configuration."""
        if _CONFIG_AVAILABLE:
            try:
                settings = get_settings()
                return settings.database.pragma_settings
            except:
                pass
        return {
            "journal_mode": "DELETE",
            "busy_timeout": 120000,
            "foreign_keys": "ON",
            "synchronous": "NORMAL"
        }

# Backward compatibility function
@timeit(logger, "run_mol_distance_2D_batch")
def run_mol_distance_2D_batch(cfg: PipelineConfig) -> PipelineResult:
    """
    Legacy entry point for backward compatibility.
    Now uses the class-based pipeline internally.
    """
    pipeline = stereomapperPipeline(cfg)
    return pipeline.run()
