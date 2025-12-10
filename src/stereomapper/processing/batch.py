"""Batch processing operations for molecular structures"""
import logging
import time
from typing import List, Optional, Dict
import sqlite3
from pathlib import Path

from stereomapper.domain.models import ProcessingResult, CacheEntry
from stereomapper.processing.processor import BulkMoleculeProcessor
from stereomapper.domain.exceptions import (
    BatchProcessingError,
    CacheError,
    ValidationError,
    ParameterValidationError,
    FileNotFoundError,
    ProcessingError
)
from stereomapper.data.cache_repo import get_cached_entry
from stereomapper.results import assemblers
from stereomapper.utils.timing import section_timer
from stereomapper.utils.suppress import setup_clean_logging
from stereomapper.utils.logging import setup_logging

setup_clean_logging()

logger, summary_logger = setup_logging(
    console=True,
    level="INFO",
    quiet_console=False,
    console_level="INFO"
)

class BatchProcessor:
    def __init__(self, chunk_size: int = 100_000):
        if chunk_size <= 0:
            raise ParameterValidationError(
                "chunk_size must be positive",
                parameter_name="chunk_size",
                parameter_value=chunk_size,
                expected_type="positive integer"
            )
        
        if chunk_size > 1_000_000:
            raise ParameterValidationError(
                "chunk_size too large, may cause memory issues",
                parameter_name="chunk_size",
                parameter_value=chunk_size
            ).add_suggestion("Use chunk_size <= 1,000,000")
        
        self.chunk_size = chunk_size
        self.bulk_processor = BulkMoleculeProcessor()

    def process_batch(
            self,
            molfile_list: List[str],
            conn: sqlite3.Connection,
            *,
            std_version: int = 1,
            namespace: str = "default",
            source_kind: str = "file",
    ) -> List[ProcessingResult]:
        """
        Batch process a list of structures in molfiles.
        Processing will involve canonical smiles generation, and structural feature extraction
        """
        # Validate inputs
        self._validate_batch_inputs(molfile_list, conn, std_version, namespace, source_kind)
        
        results: List[ProcessingResult] = []
        
        try:
            # Validate and resolve file paths
            files = self._validate_and_resolve_paths(molfile_list)
            to_process: List[str] = []
            meta_by_path: Dict[str, Dict[str, str]] = {}

            with section_timer(f"Batch processing {len(files)} files", logger):
                # Check cache and prepare metadata
                to_process, meta_by_path = self._prepare_batch_metadata(files, conn, results)

            if not to_process:
                logger.info("All files were cached, no processing needed")
                return results

            with section_timer(f"Processing {len(to_process)} new files in chunks of {self.chunk_size}", logger):
                self._process_uncached_files(
                    to_process,
                    conn,
                    meta_by_path,
                    results,
                    namespace=namespace,
                    source_kind=source_kind,
                    std_version=std_version,
                )
            return results
            
        except (ValidationError, BatchProcessingError, CacheError):
            # Re-raise our custom exceptions
            raise
        except sqlite3.Error as e:
            raise CacheError(
                f"Database error during batch processing: {str(e)}",
                operation="batch_processing"
            ).add_context('batch_size', len(molfile_list)) from e
        except Exception as e:
            raise BatchProcessingError(
                f"Unexpected error during batch processing: {str(e)}",
                batch_size=len(molfile_list)
            ) from e
    
    def _validate_batch_inputs(
        self,
        molfile_list: List[str],
        conn: sqlite3.Connection,
        std_version: int,
        namespace: str,
        source_kind: str
    ) -> None:
        """Validate all batch processing inputs."""
        if not molfile_list:
            raise ValidationError(
                "Cannot process empty molecule list",
                field_name="molfile_list"
            ).add_suggestion("Provide at least one molfile path")
        
        if len(molfile_list) > 100_000:
            raise ValidationError(
                f"Batch size too large: {len(molfile_list)}",
                field_name="batch_size",
                field_value=str(len(molfile_list))
            ).add_suggestion("Process in smaller batches (< 100,000 files)")
        
        if conn is None:
            raise ValidationError(
                "Database connection is required",
                field_name="conn"
            ).add_suggestion("Provide a valid SQLite connection")
        
        if not isinstance(std_version, int) or std_version < 1:
            raise ParameterValidationError(
                "std_version must be a positive integer",
                parameter_name="std_version",
                parameter_value=std_version,
                expected_type="positive integer"
            )
        
        if not namespace or not isinstance(namespace, str):
            raise ParameterValidationError(
                "namespace must be a non-empty string",
                parameter_name="namespace",
                parameter_value=namespace,
                expected_type="non-empty string"
            )
    
    def _validate_and_resolve_paths(self, molfile_list: List[str]) -> List[str]:
        """Validate and resolve file paths, raising exceptions for invalid files."""
        files = []
        invalid_files = []
        
        for file_path in molfile_list:
            try:
                resolved_path = str(Path(file_path).expanduser().resolve())
                
                if not Path(resolved_path).exists():
                    invalid_files.append(file_path)
                    continue
                
                if not Path(resolved_path).is_file():
                    invalid_files.append(file_path)
                    continue
                
                files.append(resolved_path)
                
            except Exception as e:
                logger.warning(f"Error resolving path {file_path}: {e}")
                invalid_files.append(file_path)
        
        if not files:
            raise ValidationError(
                f"No valid files found out of {len(molfile_list)} provided",
                field_name="molfile_list"
            ).add_context('invalid_files', invalid_files)
        
        if invalid_files:
            logger.warning(f"Found {len(invalid_files)} invalid files, processing {len(files)} valid files")
            # Log a few examples
            example_invalid = invalid_files[:3]
            logger.warning(f"Examples of invalid files: {example_invalid}")
        
        return files
    
    def _prepare_batch_metadata(
        self,
        files: List[str],
        conn: sqlite3.Connection,
        results: List[ProcessingResult]
    ) -> tuple[List[str], Dict[str, Dict[str, str]]]:
        """Prepare metadata and check cache for files."""
        to_process = []
        meta_by_path = {}
        cache_hits = 0
        
        try:
            for path in files:
                try:
                    fhash = assemblers.hash_file(path)
                    meta_by_path[path] = {
                        "file_hash": fhash,
                        "source_id": fhash,
                        "source_ref": path
                    }

                    cached_entry: Optional[CacheEntry] = get_cached_entry(fhash, conn)

                    if cached_entry is not None and cached_entry.is_valid:
                        cache_hits += 1
                        result = ProcessingResult(
                            molecule_id=cached_entry.molecule_id,
                            smiles=cached_entry.smiles,
                            error=cached_entry.error,
                            file_path=path,
                        )
                        results.append(result)
                    else:
                        to_process.append(path)
                        
                except Exception as e:
                    logger.error(f"Error preparing metadata for {path}: {e}")
                    # Create error result for this file
                    error_result = ProcessingResult(
                        molecule_id=None,
                        smiles=None,
                        error=f"Metadata preparation failed: {str(e)}",
                        file_path=path,
                    )
                    results.append(error_result)
            
            logger.info(f"Cache hits: {cache_hits}/{len(files)} files")
            return to_process, meta_by_path
            
        except sqlite3.Error as e:
            raise CacheError(
                f"Database error during cache lookup: {str(e)}",
                operation="cache_lookup"
            ).add_context('files_checked', len(files)) from e
    
    def _process_uncached_files(
            self,
            to_process: List[str],
            conn: sqlite3.Connection,
            meta_by_path: Dict[str, Dict[str, str]],
            results: List[ProcessingResult],
            *,
            std_version: int = 1,
            namespace: str = "default",
            source_kind: str = "file",
    ) -> None:
        """Process files not already in the cache, in chunks."""
        if not to_process:
            return
        
        with section_timer(f"Processing {len(to_process)} uncached files", logger):
            try:
                conn.execute("BEGIN")
                
                try:
                    processing_results = self.bulk_processor.process_and_cache_molecules(
                        to_process,
                        conn,
                        namespace=namespace,
                        source_kind=source_kind,
                    )
                    results.extend(processing_results)
                    
                except Exception as e:
                    logger.exception(f"Error during bulk processing: {e}")
                    
                    # Create error results for all failed files
                    for path in to_process:
                        error_result = ProcessingResult(
                            molecule_id=None,
                            smiles=None,
                            error=f"Bulk processing failed: {str(e)}",
                            file_path=path,
                        )
                        results.append(error_result)
                    
                    # Re-raise as BatchProcessingError
                    raise BatchProcessingError(
                        f"Bulk processing failed for {len(to_process)} files",
                        batch_size=len(to_process),
                        failed_count=len(to_process)
                    ) from e

                conn.commit()
                logger.info(f"Successfully processed {len(to_process)} files")
                
            except sqlite3.Error as e:
                logger.exception(f"Database transaction failed, rolling back: {e}")
                try:
                    conn.rollback()
                except sqlite3.Error as rollback_error:
                    logger.error(f"Rollback also failed: {rollback_error}")
                
                raise CacheError(
                    f"Database transaction failed during processing: {str(e)}",
                    operation="transaction"
                ).add_context('files_being_processed', len(to_process)) from e
            except BatchProcessingError:
                # Re-raise our custom exception
                try:
                    conn.rollback()
                except sqlite3.Error as rollback_error:
                    logger.error(f"Rollback failed after processing error: {rollback_error}")
                raise
            except Exception as e:
                logger.exception(f"Unexpected error during uncached file processing: {e}")
                try:
                    conn.rollback()
                except sqlite3.Error as rollback_error:
                    logger.error(f"Rollback failed after unexpected error: {rollback_error}")
                
                raise ProcessingError(
                    f"Unexpected error processing uncached files: {str(e)}",
                    stage="uncached_processing"
                ).add_context('files_count', len(to_process)) from e