"""OpenBabel operations for molecule standardization."""

import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Dict
from stereomapper.domain.exceptions.validation import (
    CanonicalizationError,
    PipelineFileNotFoundError
)
from stereomapper.domain.exceptions.base import ExternalToolError
logger = logging.getLogger(__name__)

class OpenBabelOperations:
    """Handles OpenBabel operations for molecule standardization."""

    @staticmethod
    def canonicalise_molfile(molfile_path: str) -> Optional[str]:
        """
        Canonicalize a molfile to canonical SMILES using OpenBabel.
        Returns None if canonicalization fails (instead of raising exceptions for processing errors).
        Only raises exceptions for fatal errors (file not found, tool not available).
        """
        if not isinstance(molfile_path, str):
            logger.error(f"Invalid input type for molfile_path: {type(molfile_path)}")
            return None

        if not os.path.isfile(molfile_path):
            # File not found is a fatal error - raise exception
            raise PipelineFileNotFoundError(molfile_path)

        if not OpenBabelOperations.is_obabel_available():
            # Tool not available is a fatal error - raise exception
            ex = ExternalToolError(
                "OpenBabel is not available in the system",
                tool_name="OpenBabel",
                command="obabel"
            ).add_suggestion("Install OpenBabel and ensure 'obabel' command is in PATH")
            raise ex

        try:
            # Run OpenBabel command - keeping your exact subprocess call
            result = subprocess.run(['obabel', molfile_path, '-osmi', '-xI', '-xN'],
                                capture_output=True, text=True, check=True)

            if result.returncode != 0:
                # Canonicalization failure is a processing error - log and return None
                error_msg = result.stderr.strip() if result.stderr else "Unknown OpenBabel error"
                logger.warning(f"OpenBabel canonicalization failed for {molfile_path}: {error_msg}")
                return None

            # Extract SMILES from output
            output = result.stdout.strip()
            if not output:
                logger.warning(f"OpenBabel returned empty output for {molfile_path}")
                return None

            # Better parsing - handle different output formats
            lines = output.split('\n')
            if not lines:
                logger.warning(f"OpenBabel returned no valid lines for {molfile_path}")
                return None

            # Take the first non-empty line and extract SMILES
            smiles_line = lines[0].strip()
            if not smiles_line:
                logger.warning(f"OpenBabel returned empty first line for {molfile_path}")
                return None

            # Extract SMILES (first part before any whitespace or tab)
            smiles_parts = smiles_line.split()
            if not smiles_parts:
                logger.warning(f"Could not extract SMILES from OpenBabel output for {molfile_path}: {output}")
                return None

            smiles = smiles_parts[0]

            # Validate the SMILES doesn't look like a path or error message
            if smiles.startswith('/') or 'tmp' in smiles or len(smiles) < 2:
                logger.warning(f"OpenBabel returned invalid SMILES for {molfile_path}: {smiles}")
                return None

            return smiles

        except subprocess.TimeoutExpired:
            # Timeout is a processing error - log and return None
            logger.warning(f"OpenBabel canonicalization timed out for {molfile_path}")
            return None

        except subprocess.CalledProcessError as e:
            # Process failure is a processing error - log and return None
            logger.warning(f"OpenBabel process failed for {molfile_path} with exit code {e.returncode}")
            return None

        except Exception as e:
            # Unexpected errors are processing errors - log and return None
            logger.warning(f"Unexpected error during OpenBabel canonicalization for {molfile_path}: {str(e)}")
            return None

    @staticmethod
    def canonicalise_molfiles_batch(molfile_paths: List[str]) -> Dict[str, Optional[str]]:
        """Batch canonicalize multiple molfiles using OpenBabel.
        
        Args:
            molfile_paths: List of paths to molfiles. Paths will be normalized internally.
            
        Returns:
            Dictionary mapping normalized paths to canonical SMILES (or None if failed).
        """
        if not molfile_paths:
            return {}

        results: Dict[str, Optional[str]] = {}
        valid_paths: List[str] = []

        # --- 1. Validate and normalize paths safely ---
        for path in molfile_paths:
            try:
                # Normalize path to absolute resolved path for consistency
                # Even if input is already normalized, this is idempotent
                normalized_path = str(Path(path).expanduser().resolve())
                
                if not os.path.isfile(normalized_path):
                    # Store with normalized path for consistency
                    results[normalized_path] = None
                    logger.warning(f"File not found in batch: {path}")
                else:
                    valid_paths.append(normalized_path)
            except Exception as e:
                # On error, still try to normalize for consistent key
                try:
                    normalized_path = str(Path(path).expanduser().resolve())
                    results[normalized_path] = None
                except:
                    # If even normalization fails, use original path
                    results[path] = None
                logger.error(f"Error validating path {path}: {e}")

        if not valid_paths:
            raise CanonicalizationError(
                "No valid files found in batch processing",
            ).add_context('total_files', len(molfile_paths))

        # --- 2. Ensure OpenBabel is available before continuing ---
        if not OpenBabelOperations.is_obabel_available():
            raise ExternalToolError(
                "OpenBabel (obabel) is not available for batch processing",
                tool_name="OpenBabel",
                command="obabel"
            ).add_suggestion("Install OpenBabel and ensure 'obabel' command is in PATH")

        MAX_BATCH = 2000  # keep below practical CLI/memory limits
        # --- 3. Process in sub-batches using @filelist to avoid ARG_MAX ---
        for start in range(0, len(valid_paths), MAX_BATCH):
            subpaths = valid_paths[start:start + MAX_BATCH]
            list_path = None
            try:
                with tempfile.NamedTemporaryFile("w", delete=False) as tf:
                    tf.write("\n".join(subpaths))
                    list_path = tf.name

                # Use @filelist; drop -imol to allow mixed formats (.mol/.sdf) by extension
                cmd = ["obabel", f"@{list_path}", "-osmi", "-xI", "-xN"]
                logger.debug(f"[obabel] running: {' '.join(cmd)} (n={len(subpaths)})")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result.returncode != 0:
                    error_msg = result.stderr.strip() or "Unknown batch error"
                    logger.warning(f"OpenBabel batch processing failed (returncode={result.returncode}): {error_msg}")
                    logger.debug(f"Failed batch size: {len(subpaths)}, first file: {subpaths[0] if subpaths else 'N/A'}")
                    # Fall back to per-file canonicalization
                    individual_results = OpenBabelOperations._fallback_individual_processing(subpaths)
                    results.update(individual_results)
                    continue

                # Parse output lines -> map 1:1 to inputs (best effort)
                output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                logger.debug(f"OpenBabel produced {len(output_lines)} lines for {len(subpaths)} files.")

                for path, line in zip(subpaths, output_lines):
                    parts = line.split('\t') if '\t' in line else line.split()
                    smiles = parts[0].strip() if parts else None
                    # Store with normalized path for consistent lookup
                    results[path] = smiles or None

                # Any missing outputs -> None
                for path in subpaths[len(output_lines):]:
                    results[path] = None
                    logger.debug(f"No output line for {path}")

            except subprocess.TimeoutExpired:
                logger.error("OpenBabel batch processing timed out; falling back to individual processing.")
                individual_results = OpenBabelOperations._fallback_individual_processing(subpaths)
                results.update(individual_results)
            except (ExternalToolError, CanonicalizationError):
                raise
            except Exception as e:
                raise CanonicalizationError(
                    f"Unexpected error during batch canonicalization: {e}",
                ).add_context('batch_size', len(subpaths)) from e
            finally:
                if list_path and os.path.exists(list_path):
                    try:
                        os.unlink(list_path)
                    except OSError as e:
                        logger.warning(f"Could not remove temp file {list_path}: {e}")

        logger.debug(f"Canonicalization complete: {len(results)} total entries.")
        return results

    @staticmethod
    def _fallback_individual_processing(molfile_paths: List[str]) -> Dict[str, Optional[str]]:
        """Fallback to individual processing if batch fails."""
        logger.info("Falling back to individual OpenBabel processing")
        results = {}

        for path in molfile_paths:
            try:
                results[path] = OpenBabelOperations.canonicalise_molfile(path)
            except Exception as e:
                logger.error(f"Individual processing failed for {path}: {e}")
                results[path] = None

        return results

    @staticmethod
    def is_obabel_available() -> bool:
        """Check if OpenBabel is available in the system."""
        try:
            # Keeping your exact subprocess call
            result = subprocess.run(
                ["obabel", "-V"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def get_obabel_version() -> Optional[str]:
        """Get OpenBabel version string."""
        try:
            # Keeping your exact subprocess call
            result = subprocess.run(
                ["obabel", "-V"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception:
            return None
        