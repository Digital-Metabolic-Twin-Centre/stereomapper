"""Chemistry utility operations."""

import logging
import tempfile
import os
import math
from typing import Optional, Tuple, Dict, Any, Union
from rdkit import Chem
from .core import ChemistryOperations
from .validation import ChemistryValidator
from .openbabel import OpenBabelOperations


logger = logging.getLogger(__name__)

class ChemistryUtils:
    """Chemistry utility operations and helper functions."""

    @staticmethod
    def generate_formula_from_file(molfile_path: str) -> Optional[str]:
        """Generate molecular formula from molfile."""
        try:
            mol = ChemistryOperations.mol_from_molfile(molfile_path)
            return ChemistryOperations.generate_molecular_formula(mol)
        except Exception:
            logger.exception("Formula generation from file failed for: %s", molfile_path)
            return None

    @staticmethod
    def detect_charge(molfile_path: str) -> int:
        """
        Detect charge from molfile with fallback.
        Combines RDKit detection with manual parsing fallback.
        """
        # Try RDKit first
        charge = ChemistryValidator.detect_charge_from_molfile(molfile_path)
        if charge is not None:
            return charge

        # Fallback to manual parsing
        logger.debug("RDKit charge detection failed, using fallback for: %s", molfile_path)
        return ChemistryValidator.fallback_charge_from_molfile(molfile_path)

    @staticmethod
    def mol_charge(mol: Chem.Mol) -> int:
        """Get formal charge from RDKit Mol object."""
        return ChemistryOperations.get_formal_charge(mol)

    @staticmethod
    def is_wildcard(molfile_path: str) -> bool:
        """Check if molfile contains wildcard atoms."""
        return ChemistryValidator.is_wildcard_molfile(molfile_path)

    @staticmethod
    def create_temp_molfile(mol: Chem.Mol) -> Optional[str]:
        """Create temporary molfile from RDKit Mol object."""
        try:
            mol_block = Chem.MolToMolBlock(mol)
            if not mol_block:
                return None

            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.mol',
                delete=False
            )
            temp_file.write(mol_block)
            temp_file.close()

            return temp_file.name
        except Exception:
            logger.exception("Failed to create temporary molfile")
            return None

    @staticmethod
    def cleanup_temp_file(file_path: str) -> None:
        """Safely remove temporary file."""
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except OSError:
            logger.warning("Failed to remove temporary file: %s", file_path)

    @staticmethod
    def standardise_molecule_pipeline(molfile_path: str) -> Dict[str, Any]:
        """
        Complete molecule standardization pipeline.

        Returns:
            Dictionary with standardization results
        """
        result = {
            'file_path': molfile_path,
            'smiles': None,
            'inchikey': None,
            'formula': None,
            'charge': None,
            'is_wildcard': False,
            'error': None
        }

        try:
            # Check if file exists
            if not os.path.exists(molfile_path):
                result['error'] = 'file_not_found'
                return result

            # Check for wildcards
            result['is_wildcard'] = ChemistryUtils.is_wildcard(molfile_path)
            if result['is_wildcard']:
                result['error'] = 'contains_wildcards'
                return result

            # Detect charge
            result['charge'] = ChemistryUtils.detect_charge(molfile_path)

            # Generate formula
            result['formula'] = ChemistryUtils.generate_formula_from_file(molfile_path)

            # Generate InChIKey
            result['inchikey'] = ChemistryOperations.generate_inchikey_from_file(molfile_path)

            # Canonicalize to SMILES (this would use OpenBabel)
            result['smiles'] = OpenBabelOperations.canonicalize_molfile(molfile_path)

            if not result['smiles']:
                result['error'] = 'canonicalization_failed'

        except Exception as e:
            logger.exception("Standardization pipeline failed for: %s", molfile_path)
            result['error'] = f'pipeline_error: {str(e)}'

        return result

    @staticmethod
    def validate_molecule_data(
        smiles: Optional[str] = None,
        inchikey: Optional[str] = None,
        formula: Optional[str] = None
    ) -> Dict[str, bool]:
        """Validate molecular data quality."""
        validation = {
            'has_smiles': smiles is not None and len(smiles.strip()) > 0,
            'has_inchikey': inchikey is not None and len(inchikey.strip()) > 0,
            'has_formula': formula is not None and len(formula.strip()) > 0,
            'smiles_valid': False,
            'inchikey_valid': False
        }

        # Validate SMILES
        if validation['has_smiles']:
            try:
                mol = ChemistryOperations.mol_from_smiles(smiles)
                validation['smiles_valid'] = mol is not None
            except Exception:
                validation['smiles_valid'] = False

        # Validate InChIKey format (basic check)
        if validation['has_inchikey']:
            # InChIKey should be 27 characters with a dash at position 14
            validation['inchikey_valid'] = (
                len(inchikey) == 27 and
                inchikey[14] == '-' and
                all(c.isalnum() or c == '-' for c in inchikey)
            )

        return validation

    @staticmethod
    def normalise_rmsd(rmsd_result: Union[float, Dict[str, Any], None]) -> Tuple[Optional[float], Optional[str]]:
        """
        Normalize RMSD values from alignment operations.

        Args:
            rmsd_result: Raw RMSD result from alignment (float, dict, or None)

        Returns:
            Tuple of (normalized_rmsd, error_message)
            If successful: (float_value, None)
            If failed: (None, "error description")
        """
        # dict case
        if isinstance(rmsd_result, dict):
            # bubble up explicit errors if present
            if 'error' in rmsd_result and rmsd_result['error']:
                return None, f"align_molecules error: {rmsd_result['error']}"
            val = rmsd_result.get('rmsd', None)
        else:
            val = rmsd_result

        # missing
        if val is None:
            return None, "RMSD missing"

        # flatten numpy scalars etc.
        try:
            val = float(val)
        except Exception as e:
            return None, f"RMSD not a number ({type(val)}): {e}"

        # reject NaN/Inf
        if math.isnan(val) or math.isinf(val):
            return None, "RMSD invalid (nan/inf)"

        return val, None
