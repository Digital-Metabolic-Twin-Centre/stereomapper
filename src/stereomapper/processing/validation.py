"""Input validation for processing pipeline."""

import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import os

from stereomapper.domain.chemistry import ChemistryOperations, OpenBabelOperations
from stereomapper.utils.suppress import setup_clean_logging

setup_clean_logging()

logger = logging.getLogger(__name__)

class InputValidator:
    """Validates inputs for the molecular processing pipeline."""
    
    VALID_EXTENSIONS = {'.mol', '.sdf', '.molfile', '.mol2'}
    
    @staticmethod
    def validate_molfile_paths(file_paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate that molfile paths exist and are readable.
        
        Returns:
            (valid_paths, invalid_paths)
        """
        valid_paths = []
        invalid_paths = []
        
        for path_str in file_paths:
            try:
                path = Path(path_str).expanduser().resolve()
                
                if not path.exists():
                    logger.warning(f"File does not exist: {path_str}")
                    invalid_paths.append(path_str)
                elif not path.is_file():
                    logger.warning(f"Path is not a file: {path_str}")
                    invalid_paths.append(path_str)
                elif not InputValidator._has_valid_extension(path):
                    logger.warning(f"File does not have valid molecular extension: {path_str}")
                    invalid_paths.append(path_str)
                elif not InputValidator._is_readable(path):
                    logger.warning(f"File is not readable: {path_str}")
                    invalid_paths.append(path_str)
                else:
                    valid_paths.append(str(path))
                    
            except Exception as e:
                logger.warning(f"Error validating path {path_str}: {e}")
                invalid_paths.append(path_str)
        
        return valid_paths, invalid_paths
    
    @staticmethod
    def _has_valid_extension(path: Path) -> bool:
        """Check if file has a valid molecular file extension."""
        return path.suffix.lower() in InputValidator.VALID_EXTENSIONS
    
    @staticmethod
    def _is_readable(path: Path) -> bool:
        """Check if file is readable."""
        try:
            return os.access(path, os.R_OK)
        except Exception:
            return False
    
    @staticmethod
    def validate_molecule_structure(mol_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a molecule file can be read by RDKit.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            mol = ChemistryOperations.mol_from_molfile(mol_path)
            if mol is None:
                return False, "Could not parse molecule with RDKit"
            
            # Additional validation checks
            num_atoms = mol.GetNumAtoms()
            if num_atoms == 0:
                return False, "Molecule has no atoms"
            
            return True, None
            
        except Exception as e:
            return False, f"RDKit parsing error: {str(e)}"
    
    @staticmethod
    def validate_batch_parameters(
        batch_size: int, 
        total_files: int,
        chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate and adjust batch processing parameters.
        
        Returns:
            Dictionary with validated parameters
        """
        validated = {}
        
        # Validate batch size
        if batch_size <= 0:
            logger.warning("Invalid batch size, using default of 1000")
            validated['batch_size'] = 1000
        elif batch_size > total_files:
            logger.info(f"Batch size ({batch_size}) larger than total files ({total_files}), using {total_files}")
            validated['batch_size'] = total_files
        else:
            validated['batch_size'] = batch_size
        
        # Validate chunk size
        if chunk_size is not None:
            if chunk_size <= 0:
                logger.warning("Invalid chunk size, using batch size")
                validated['chunk_size'] = validated['batch_size']
            else:
                validated['chunk_size'] = min(chunk_size, validated['batch_size'])
        else:
            validated['chunk_size'] = validated['batch_size']
        
        validated['total_files'] = total_files
        
        return validated
    
    @staticmethod
    def validate_database_connection(conn) -> Tuple[bool, Optional[str]]:
        """
        Validate database connection is working.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            cursor = conn.execute("SELECT 1")
            cursor.fetchone()
            return True, None
        except Exception as e:
            return False, f"Database connection error: {str(e)}"
    
    @staticmethod
    def validate_namespace(namespace: str) -> Tuple[bool, Optional[str]]:
        """
        Validate namespace parameter.
        
        Returns:
            (is_valid, error_message)
        """
        if not namespace:
            return False, "Namespace cannot be empty"
        
        if not isinstance(namespace, str):
            return False, "Namespace must be a string"
        
        # Check for invalid characters
        invalid_chars = {'/', '\\', ':', '*', '?', '"', '<', '>', '|'}
        if any(char in namespace for char in invalid_chars):
            return False, f"Namespace contains invalid characters: {invalid_chars}"
        
        return True, None
    
    class CacheValidator:
        """Handles validation for cache operations."""
    
        @staticmethod
        def validate_environment() -> None:
            """Validate that required tools are available."""
            if not OpenBabelOperations.is_obabel_available():
                raise EnvironmentError(
                    "OpenBabel (obabel) is not available in the system PATH. "
                    "Please install OpenBabel to use this function."
                )
            
            obabel_version = OpenBabelOperations.get_obabel_version()
            if obabel_version is None or obabel_version < "2.1.0":
                raise EnvironmentError(
                    f"OpenBabel version 2.1.0 or higher is required. "
                    f"Detected version: {obabel_version if obabel_version else 'not found'}"
                )
        
        @staticmethod
        def validate_and_normalize_files(molfile_list: List[str]) -> List[str]:
            """
            Validate and normalize file paths.
            
            Returns:
                List of valid, normalized file paths.
            """
            import os
            
            valid_files = []
            for path in molfile_list:
                normalized = str(Path(path).expanduser().resolve())
                if os.path.exists(normalized):
                    valid_files.append(normalized)
                else:
                    logger.warning(f"File not found, skipping: {path}")
            
            return valid_files