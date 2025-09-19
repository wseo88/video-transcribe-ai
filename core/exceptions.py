"""
Custom exceptions for the video transcription application.

This module defines a hierarchy of custom exceptions to provide better
error handling and more specific error messages throughout the application.
"""

from pathlib import Path
from typing import Optional, Union


class TranscriptionError(Exception):
    """
    Base exception for all transcription-related errors.
    
    This is the parent class for all custom exceptions in the application,
    allowing for broad exception handling while maintaining specific error types.
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(TranscriptionError):
    """Raised when there are issues with configuration validation or setup."""
    pass


class ModelError(TranscriptionError):
    """Base class for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    
    def __init__(self, model_name: str, device: str, original_error: Optional[Exception] = None):
        message = f"Failed to load model '{model_name}' on device '{device}'"
        super().__init__(message, "MODEL_LOAD_ERROR")
        self.model_name = model_name
        self.device = device
        self.original_error = original_error


class ModelInferenceError(ModelError):
    """Raised when model inference fails."""
    
    def __init__(self, operation: str, original_error: Optional[Exception] = None):
        message = f"Model inference failed during {operation}"
        super().__init__(message, "MODEL_INFERENCE_ERROR")
        self.operation = operation
        self.original_error = original_error


class AudioProcessingError(TranscriptionError):
    """Base class for audio processing errors."""
    pass


class AudioExtractionError(AudioProcessingError):
    """Raised when audio extraction from video fails."""
    
    def __init__(self, video_path: Union[str, Path], original_error: Optional[Exception] = None):
        video_path = Path(video_path)
        message = f"Failed to extract audio from video: {video_path.name}"
        super().__init__(message, "AUDIO_EXTRACTION_ERROR")
        self.video_path = video_path
        self.original_error = original_error


class AudioLoadError(AudioProcessingError):
    """Raised when loading audio data fails."""
    
    def __init__(self, audio_path: Union[str, Path], original_error: Optional[Exception] = None):
        audio_path = Path(audio_path)
        message = f"Failed to load audio data from: {audio_path.name}"
        super().__init__(message, "AUDIO_LOAD_ERROR")
        self.audio_path = audio_path
        self.original_error = original_error


class FileSystemError(TranscriptionError):
    """Base class for file system related errors."""
    pass


class FileNotFoundError(FileSystemError):
    """Raised when a required file is not found."""
    
    def __init__(self, file_path: Union[str, Path], file_type: str = "file"):
        file_path = Path(file_path)
        message = f"{file_type.title()} not found: {file_path}"
        super().__init__(message, "FILE_NOT_FOUND")
        self.file_path = file_path
        self.file_type = file_type


class FileAccessError(FileSystemError):
    """Raised when file access is denied or fails."""
    
    def __init__(self, file_path: Union[str, Path], operation: str, original_error: Optional[Exception] = None):
        file_path = Path(file_path)
        message = f"Failed to {operation} file: {file_path.name}"
        super().__init__(message, "FILE_ACCESS_ERROR")
        self.file_path = file_path
        self.operation = operation
        self.original_error = original_error


class ValidationError(TranscriptionError):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, value: any, reason: str):
        message = f"Validation failed for '{field}': {reason} (value: {value})"
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field
        self.value = value
        self.reason = reason


class ResourceError(TranscriptionError):
    """Raised when system resources are insufficient."""
    pass


class MemoryError(ResourceError):
    """Raised when insufficient memory is available."""
    
    def __init__(self, operation: str, required_mb: Optional[float] = None, available_mb: Optional[float] = None):
        message = f"Insufficient memory for {operation}"
        if required_mb and available_mb:
            message += f" (required: {required_mb:.1f}MB, available: {available_mb:.1f}MB)"
        super().__init__(message, "MEMORY_ERROR")
        self.operation = operation
        self.required_mb = required_mb
        self.available_mb = available_mb


class GPUError(ResourceError):
    """Raised when GPU-related operations fail."""
    
    def __init__(self, operation: str, original_error: Optional[Exception] = None):
        message = f"GPU operation failed: {operation}"
        super().__init__(message, "GPU_ERROR")
        self.operation = operation
        self.original_error = original_error


class NetworkError(TranscriptionError):
    """Raised when network operations fail."""
    
    def __init__(self, operation: str, url: Optional[str] = None, original_error: Optional[Exception] = None):
        message = f"Network operation failed: {operation}"
        if url:
            message += f" (URL: {url})"
        super().__init__(message, "NETWORK_ERROR")
        self.operation = operation
        self.url = url
        self.original_error = original_error


class SubtitleError(TranscriptionError):
    """Raised when subtitle generation or processing fails."""
    
    def __init__(self, operation: str, file_path: Optional[Union[str, Path]] = None, original_error: Optional[Exception] = None):
        message = f"Subtitle {operation} failed"
        if file_path:
            file_path = Path(file_path)
            message += f" for file: {file_path.name}"
        super().__init__(message, "SUBTITLE_ERROR")
        self.operation = operation
        self.file_path = file_path
        self.original_error = original_error


class RetryableError(TranscriptionError):
    """
    Raised for errors that might be retryable.
    
    This exception indicates that the operation might succeed if retried,
    typically used for transient network or resource issues.
    """
    
    def __init__(self, message: str, max_retries: int = 3, retry_delay: float = 1.0):
        super().__init__(message, "RETRYABLE_ERROR")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_count = 0
    
    def should_retry(self) -> bool:
        """Check if the operation should be retried."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """Increment the retry count."""
        self.retry_count += 1