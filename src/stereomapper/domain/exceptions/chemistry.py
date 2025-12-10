"""Chemistry-related exceptions."""

from typing import Optional, Dict, Any, List
from .base import stereomapperError, RetryableError

class ChemistryError(stereomapperError):
    """Base class for chemistry-related errors."""
    
    def __init__(
        self,
        message: str,
        *,
        file_path: Optional[str] = None,
        molecule_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if file_path:
            self.add_context('file_path', file_path)
        if molecule_id:
            self.add_context('molecule_id', molecule_id)


class MoleculeParsingError(ChemistryError):
    """Raised when molecule parsing fails."""
    
    def __init__(
        self,
        message: str,
        *,
        parser: Optional[str] = None,
        file_format: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if parser:
            self.add_context('parser', parser)
        if file_format:
            self.add_context('file_format', file_format)
        
        # Add common suggestions
        self.add_suggestion("Check if the input file is a valid molecular structure file")
        self.add_suggestion("Verify the file format matches the expected format")
        if parser == "RDKit":
            self.add_suggestion("Try using OpenBabel parser as alternative")
    
    def _get_default_error_code(self) -> str:
        return "MOLECULE_PARSE_FAILED"


class CanonicalizationError(RetryableError, ChemistryError):
    """Raised when molecule canonicalization fails."""
    
    def __init__(
        self,
        message: str,
        *,
        tool: Optional[str] = None,
        timeout: bool = False,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if tool:
            self.add_context('canonicalization_tool', tool)
        if timeout:
            self.add_context('timeout_occurred', True)
            self.add_suggestion("Increase canonicalization timeout")
        
        # Add tool-specific suggestions
        if tool == "OpenBabel":
            self.add_suggestion("Try using RDKit canonicalization as fallback")
        elif tool == "RDKit":
            self.add_suggestion("Try using OpenBabel canonicalization as fallback")
        
        self.add_suggestion("Check if molecule contains unusual chemistry")
    
    def _get_default_error_code(self) -> str:
        return "CANONICALIZATION_FAILED"


class StereoAnalysisError(ChemistryError):
    """Raised when stereochemistry analysis fails."""
    
    def __init__(
        self,
        message: str,
        *,
        analysis_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if analysis_type:
            self.add_context('analysis_type', analysis_type)
        
        self.add_suggestion("Check if molecules have defined stereochemistry")
        self.add_suggestion("Verify molecules are properly sanitized")
    
    def _get_default_error_code(self) -> str:
        return "STEREO_ANALYSIS_FAILED"


class MoleculeAlignmentError(ChemistryError):
    """Raised when molecule alignment fails."""
    
    def __init__(
        self,
        message: str,
        *,
        rmsd_attempted: bool = False,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if rmsd_attempted:
            self.add_context('rmsd_calculation_attempted', True)
        
        self.add_suggestion("Check if molecules have similar structures")
        self.add_suggestion("Verify molecules have 2D or 3D coordinates")
    
    def _get_default_error_code(self) -> str:
        return "ALIGNMENT_FAILED"


class InvalidMoleculeError(ChemistryError):
    """Raised when molecule data is invalid."""
    
    def __init__(
        self,
        message: str,
        *,
        validation_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if validation_type:
            self.add_context('validation_type', validation_type)
    
    def _get_default_error_code(self) -> str:
        return "INVALID_MOLECULE"


class WildcardMoleculeError(InvalidMoleculeError):
    """Raised when molecule contains wildcard atoms."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            validation_type="wildcard_detection",
            **kwargs
        )
        self.add_suggestion("Remove or replace wildcard atoms (* or R groups)")
        self.add_suggestion("Use specific atom types instead of wildcards")
    
    def _get_default_error_code(self) -> str:
        return "WILDCARD_ATOMS_DETECTED"