"""Processing pipeline exceptions."""

from typing import Optional, Dict, Any, List
from .base import stereomapperError, RetryableError, ResourceError

class ProcessingError(stereomapperError):
    """Base class for processing pipeline errors."""
    
    def __init__(
        self,
        message: str,
        *,
        stage: Optional[str] = None,
        batch_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if stage:
            self.add_context('processing_stage', stage)
        if batch_id:
            self.add_context('batch_id', batch_id)


class BatchProcessingError(ProcessingError):
    """Raised when batch processing fails."""
    
    def __init__(
        self,
        message: str,
        *,
        batch_size: Optional[int] = None,
        failed_count: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, stage="batch_processing", **kwargs)
        if batch_size:
            self.add_context('batch_size', batch_size)
        if failed_count:
            self.add_context('failed_molecules', failed_count)
        
        self.add_suggestion("Try reducing batch size")
        self.add_suggestion("Check system resources (memory, disk space)")
    
    def _get_default_error_code(self) -> str:
        return "BATCH_PROCESSING_FAILED"


class CacheError(RetryableError, ProcessingError):
    """Raised when cache operations fail."""
    
    def __init__(
        self,
        message: str,
        *,
        operation: Optional[str] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, stage="caching", **kwargs)
        if operation:
            self.add_context('cache_operation', operation)
        if cache_key:
            self.add_context('cache_key', cache_key)
        
        self.add_suggestion("Check database connection")
        self.add_suggestion("Verify disk space availability")
    
    def _get_default_error_code(self) -> str:
        return "CACHE_OPERATION_FAILED"


class PipelineConfigurationError(ProcessingError):
    """Raised when pipeline configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        *,
        config_field: Optional[str] = None,
        expected_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, stage="configuration", **kwargs)
        if config_field:
            self.add_context('config_field', config_field)
        if expected_type:
            self.add_context('expected_type', expected_type)
        
        self.add_suggestion("Check pipeline configuration file")
        self.add_suggestion("Verify all required parameters are set")
    
    def _get_default_error_code(self) -> str:
        return "PIPELINE_CONFIG_INVALID"