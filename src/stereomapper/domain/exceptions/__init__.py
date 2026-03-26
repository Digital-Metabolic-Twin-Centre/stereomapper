"""Custom exceptions for the stereomapper package."""

# Base exceptions
from .base import (
    ConfigurationError,
    DatabaseError,
    ExternalToolError,
    FileSystemError,
    MemoryError,
    ResourceError,
    RetryableError,
    StereomapperError,
)

# Chemistry exceptions
from .chemistry import (
    CanonicalizationError,
    ChemistryError,
    InvalidMoleculeError,
    MoleculeAlignmentError,
    MoleculeParsingError,
    StereoAnalysisError,
    WildcardMoleculeError,
)

# Processing exceptions
from .processing import (
    BatchProcessingError,
    CacheError,
    PipelineConfigurationError,
    ProcessingError,
)

# Validation exceptions
from .validation import (
    FileNotFoundError,
    FileValidationError,
    InvalidFileFormatError,
    ParameterValidationError,
    ValidationError,
)

__all__ = [
    # Base
    "StereomapperError",
    "RetryableError",
    "ConfigurationError",
    "ResourceError",
    "DatabaseError",
    "FileSystemError",
    "MemoryError",
    "ExternalToolError",
    # Chemistry
    "ChemistryError",
    "MoleculeParsingError",
    "CanonicalizationError",
    "StereoAnalysisError",
    "MoleculeAlignmentError",
    "InvalidMoleculeError",
    "WildcardMoleculeError",
    # Processing
    "ProcessingError",
    "BatchProcessingError",
    "CacheError",
    "PipelineConfigurationError",
    # Validation
    "ValidationError",
    "FileValidationError",
    "FileNotFoundError",
    "InvalidFileFormatError",
    "ParameterValidationError",
]
