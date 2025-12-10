from typing import Optional, Dict, Any, List
from datetime import datetime
from abc import ABC
import logging

logger = logging.getLogger(__name__)

class stereomapperError(Exception, ABC):
    """Base exception for all stereomapper-related errors."""
    
    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        recoverable: bool = False
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._get_default_error_code()
        self.context: Dict[str, Any] = context or {}
        self.suggestions: List[str] = suggestions or []
        self.recoverable = recoverable
        self.timestamp = datetime.now()
        self.config_field: Optional[str] = None  # to be set when relevant

    def _get_default_error_code(self) -> str:
        # sensible default; subclasses should override when needed
        return "stereomapper_ERROR"

    def add_context(self, key: str, value: Any) -> None:
        if key:
            self.context[key] = value
        return self

    def add_suggestion(self, suggestion: str) -> None:
        if suggestion:
            self.suggestions.append(suggestion)
        return self

    def __str__(self) -> str:
        # include suggestions in string to help tests that search message text
        base = self.message or ""
        if self.suggestions:
            return f"{base} -- Suggestions: {'; '.join(self.suggestions)}"
        return base


class RetryableError(stereomapperError):
    def _get_default_error_code(self) -> str:
        return "RETRYABLE_ERROR"


class ConfigurationError(stereomapperError):
    def __init__(self, message: str, *, config_field: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.config_field = config_field

    def _get_default_error_code(self) -> str:
        return "CONFIGURATION_ERROR"

    def __str__(self) -> str:
        base = self.message or ""
        if getattr(self, "config_field", None):
            base = f"[{self.config_field}] {base}"
        if self.suggestions:
            return f"{base} -- Suggestions: {'; '.join(self.suggestions)}"
        return base

class ResourceError(stereomapperError):
    def _get_default_error_code(self) -> str:
        return "RESOURCE_ERROR"


class DatabaseError(ResourceError):
    def _get_default_error_code(self) -> str:
        return "DATABASE_ERROR"


class FileSystemError(ResourceError):
    def _get_default_error_code(self) -> str:
        return "FILE_SYSTEM_ERROR"


class MemoryError(ResourceError):
    def _get_default_error_code(self) -> str:
        return "OUT_OF_MEMORY"


class ExternalToolError(ResourceError):
    def __init__(
        self,
        message: str,
        *,
        tool_name: Optional[str] = None,
        command: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if tool_name:
            self.add_context('tool_name', tool_name)
        if command:
            self.add_context('command', command)
        # helpful default suggestion to match tests expecting "Install ..."
        if tool_name:
            self.add_suggestion(f"Install {tool_name} or ensure it is available on PATH")
    def _get_default_error_code(self) -> str:
        return "EXTERNAL_TOOL_ERROR"