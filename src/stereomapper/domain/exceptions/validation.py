"""Input validation exceptions."""

from typing import Optional, List, Any, Dict
from .base import stereomapperError, ConfigurationError

class ValidationError(stereomapperError):
    """Base class for validation errors."""
    
    def __init__(
        self,
        message: str,
        *,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if field_name:
            self.add_context('field_name', field_name)
        if field_value is not None:
            self.add_context('field_value', str(field_value))

    def _get_default_error_code(self) -> str:
        return "VALIDATION_ERROR"


class FileValidationError(ValidationError):
    """Raised when file validation fails."""
    
    def __init__(
        self,
        message: str,
        *,
        file_path: Optional[str] = None,
        validation_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if file_path:
            self.add_context('file_path', file_path)
        if validation_type:
            self.add_context('validation_type', validation_type)
    
    def _get_default_error_code(self) -> str:
        return "FILE_VALIDATION_FAILED"


class FileNotFoundError(FileValidationError):
    """Raised when a required file doesn't exist."""
    def __init__(self, file_path: str, **kwargs):
        message = f"File not found: {file_path}"
        super().__init__(message, file_path=file_path, validation_type="existence_check", **kwargs)
        # provide both keys so callers/tests can use either 'file_path' or 'resource_path'
        self.add_context('resource_path', file_path)
        self.add_suggestion("Check if the file path is correct and accessible")
    def _get_default_error_code(self) -> str:
        return "FILE_NOT_FOUND"


class InvalidFileFormatError(FileValidationError):
    """Raised when file format is invalid."""
    def __init__(
        self,
        file_path: str,
        expected_formats: List[str],
        actual_format: Optional[str] = None,
        **kwargs
    ):
        formats_str = ", ".join(expected_formats)
        message = f"Invalid file format for {file_path}. Expected: {formats_str}"
        super().__init__(message, file_path=file_path, validation_type="format_check", **kwargs)
        self.add_context('expected_formats', expected_formats)
        if actual_format:
            self.add_context('actual_format', actual_format)
        self.add_suggestion(f"Ensure file has one of these extensions: {formats_str}")
    def _get_default_error_code(self) -> str:
        return "INVALID_FILE_FORMAT"


class ParameterValidationError(ValidationError):
    """Raised when parameter validation fails."""
    def __init__(
        self,
        parameter_name: str,
        parameter_value: Any,
        expected_type: Optional[str] = None,
        **kwargs
    ):
        message = f"Invalid parameter '{parameter_name}': {parameter_value}"
        super().__init__(message, field_name=parameter_name, field_value=str(parameter_value), **kwargs)
        if expected_type:
            self.add_context('expected_type', expected_type)
        self.add_suggestion(f"Check the value and type of parameter '{parameter_name}'")
    def _get_default_error_code(self) -> str:
        return "PARAMETER_VALIDATION_FAILED"


class CanonicalizationError(ValidationError):
    """Raised when molecule canonicalization fails."""
    def __init__(
        self,
        message: str,
        *,
        molecule_identifier: Optional[str] = None,
        canonicalization_method: Optional[str] = None,
        total_files: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if molecule_identifier:
            self.add_context('molecule_identifier', molecule_identifier)
        if canonicalization_method:
            self.add_context('canonicalization_method', canonicalization_method)
        if total_files is not None:
            self.add_context('total_files', total_files)
        if batch_size is not None:
            self.add_context('batch_size', batch_size)
        self.add_suggestion("Check the input molecule format and structure")
        self.add_suggestion("Verify that required chemical software is installed")
    def _get_default_error_code(self) -> str:
        return "CANONICALIZATION_FAILED"