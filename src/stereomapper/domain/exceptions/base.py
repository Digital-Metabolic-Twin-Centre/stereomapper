"""Base operations for Exception handling in stereomapper pipeline"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from abc import ABC
import logging

logger = logging.getLogger(__name__)

class stereomapperError(Exception, ABC):
    """
    Base exception for all stereomapper-related errors.

    Attributes:
        message (str): The error message.
        error_code (Optional[str]): A unique error code for the exception.
        context (Optional[Dict[str, Any]]): Additional context about the error.
        suggestions (Optional[List[str]]): Suggestions for resolving the error.
        recoverable (bool): Indicates if the error is recoverable.
        timestamp (datetime): The time the error occurred.
        config_field (Optional[str]): A configuration field related to the error.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        recoverable: bool = False
    ):
        """
        Initialize a stereomapperError instance.

        Args:
            message (str): The error message.
            error_code (Optional[str]): A unique error code for the exception.
            context (Optional[Dict[str, Any]]): Additional context about the error.
            suggestions (Optional[List[str]]): Suggestions for resolving the error.
            recoverable (bool): Indicates if the error is recoverable.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._get_default_error_code()
        self.context: Dict[str, Any] = context or {}
        self.suggestions: List[str] = suggestions or []
        self.recoverable = recoverable
        self.timestamp = datetime.now()
        self.config_field: Optional[str] = None  # to be set when relevant

    def _get_default_error_code(self) -> str:
        """
        Get the default error code for the exception.

        Returns:
            str: The default error code.
        """
        return "stereomapper_ERROR"

    def add_context(self, key: str, value: Any) -> None:
        """
        Add additional context to the error.

        Args:
            key (str): The context key.
            value (Any): The context value.

        Returns:
            None
        """
        if key:
            self.context[key] = value
        return self

    def add_suggestion(self, suggestion: str) -> None:
        """
        Add a suggestion for resolving the error.

        Args:
            suggestion (str): The suggestion text.

        Returns:
            None
        """
        if suggestion:
            self.suggestions.append(suggestion)
        return self

    def __str__(self) -> str:
        """
        Return a string representation of the error.

        Returns:
            str: The error message with suggestions (if any).
        """
        base = self.message or ""
        if self.suggestions:
            return f"{base} -- Suggestions: {'; '.join(self.suggestions)}"
        return base


class RetryableError(stereomapperError):
    """
    Exception for retryable errors in the stereomapper pipeline.
    """

    def _get_default_error_code(self) -> str:
        """
        Get the default error code for retryable errors.

        Returns:
            str: The default error code.
        """
        return "RETRYABLE_ERROR"


class ConfigurationError(stereomapperError):
    """
    Exception for configuration-related errors in the stereomapper pipeline.

    Attributes:
        config_field (Optional[str]): The configuration field related to the error.
    """

    def __init__(self, message: str, *, config_field: str | None = None, **kwargs):
        """
        Initialize a ConfigurationError instance.

        Args:
            message (str): The error message.
            config_field (Optional[str]): The configuration field related to the error.
            **kwargs: Additional arguments for the base class.
        """
        super().__init__(message, **kwargs)
        self.config_field = config_field

    def _get_default_error_code(self) -> str:
        """
        Get the default error code for configuration errors.

        Returns:
            str: The default error code.
        """
        return "CONFIGURATION_ERROR"

    def __str__(self) -> str:
        """
        Return a string representation of the configuration error.

        Returns:
            str: The error message with configuration field and suggestions (if any).
        """
        base = self.message or ""
        if getattr(self, "config_field", None):
            base = f"[{self.config_field}] {base}"
        if self.suggestions:
            return f"{base} -- Suggestions: {'; '.join(self.suggestions)}"
        return base


class ResourceError(stereomapperError):
    """
    Exception for resource-related errors in the stereomapper pipeline.
    """

    def _get_default_error_code(self) -> str:
        """
        Get the default error code for resource errors.

        Returns:
            str: The default error code.
        """
        return "RESOURCE_ERROR"


class DatabaseError(ResourceError):
    """
    Exception for database-related errors in the stereomapper pipeline.
    """

    def _get_default_error_code(self) -> str:
        """
        Get the default error code for database errors.

        Returns:
            str: The default error code.
        """
        return "DATABASE_ERROR"


class FileSystemError(ResourceError):
    """
    Exception for file system-related errors in the stereomapper pipeline.
    """

    def _get_default_error_code(self) -> str:
        """
        Get the default error code for file system errors.

        Returns:
            str: The default error code.
        """
        return "FILE_SYSTEM_ERROR"


class MemoryError(ResourceError):
    """
    Exception for memory-related errors in the stereomapper pipeline.
    """

    def _get_default_error_code(self) -> str:
        """
        Get the default error code for memory errors.

        Returns:
            str: The default error code.
        """
        return "OUT_OF_MEMORY"


class ExternalToolError(ResourceError):
    """
    Exception for errors related to external tools in the stereomapper pipeline.

    Attributes:
        tool_name (Optional[str]): The name of the external tool.
        command (Optional[str]): The command that caused the error.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: Optional[str] = None,
        command: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize an ExternalToolError instance.

        Args:
            message (str): The error message.
            tool_name (Optional[str]): The name of the external tool.
            command (Optional[str]): The command that caused the error.
            **kwargs: Additional arguments for the base class.
        """
        super().__init__(message, **kwargs)
        if tool_name:
            self.add_context('tool_name', tool_name)
        if command:
            self.add_context('command', command)
        # helpful default suggestion to match tests expecting "Install ..."
        if tool_name:
            self.add_suggestion(f"Install {tool_name} or ensure it is available on PATH")

    def _get_default_error_code(self) -> str:
        """
        Get the default error code for external tool errors.

        Returns:
            str: The default error code.
        """
        return "EXTERNAL_TOOL_ERROR"