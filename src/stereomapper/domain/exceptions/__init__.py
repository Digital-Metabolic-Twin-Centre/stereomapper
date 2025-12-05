"""Custom exceptions for the stereomapper package."""

# Base exceptions
from .base import (
    stereomapperError,
    RetryableError,
    ConfigurationError,
    ResourceError,
    DatabaseError,
    FileSystemError,
    MemoryError,
    ExternalToolError
)

# Chemistry exceptions
from .chemistry import (
    ChemistryError,
    MoleculeParsingError,
    CanonicalizationError,
    StereoAnalysisError,
    MoleculeAlignmentError,
    InvalidMoleculeError,
    WildcardMoleculeError
)

# Processing exceptions
from .processing import (
    ProcessingError,
    BatchProcessingError,
    CacheError,
    PipelineConfigurationError,
)

# Validation exceptions
from .validation import (
    ValidationError,
    FileValidationError,
    PipelineFileNotFoundError,
    InvalidFileFormatError,
    ParameterValidationError
)

__all__ = [
    # Base
    "stereomapperError",
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
    "PipelineFileNotFoundError",
    "InvalidFileFormatError",
    "ParameterValidationError",
]
