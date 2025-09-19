"""
Retry utilities for handling transient failures.

This module provides decorators and utilities for implementing retry logic
with exponential backoff and configurable retry conditions.
"""

import functools
import time
from typing import Callable, Type, Union, Tuple, Optional

from core.exceptions import RetryableError
from core.logging import get_logger

logger = get_logger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    retryable_exceptions: Tuple[Type[Exception], ...] = (RetryableError,),
    on_retry: Optional[Callable] = None,
):
    """
    Decorator to retry a function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 1.0)
        backoff_multiplier: Multiplier for delay after each retry (default: 2.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        exceptions: Tuple of exception types to retry on (default: all exceptions)
        retryable_exceptions: Tuple of exception types that are always retryable
        on_retry: Optional callback function called before each retry
        
    Example:
        @retry(max_attempts=5, delay=2.0, exceptions=(ConnectionError, TimeoutError))
        def unreliable_function():
            # This function will be retried up to 5 times
            # with delays of 2, 4, 8, 16 seconds
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Check if this is the last attempt
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts. "
                            f"Last error: {e}"
                        )
                        raise e
                    
                    # Check if the exception is retryable
                    is_retryable = (
                        isinstance(e, retryable_exceptions) or
                        (hasattr(e, 'should_retry') and e.should_retry())
                    )
                    
                    if not is_retryable:
                        logger.error(f"Function {func.__name__} failed with non-retryable error: {e}")
                        raise e
                    
                    # Log retry attempt
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {current_delay:.1f} seconds..."
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(attempt + 1, e, current_delay)
                        except Exception as callback_error:
                            logger.warning(f"Retry callback failed: {callback_error}")
                    
                    # Wait before retry
                    time.sleep(current_delay)
                    
                    # Calculate next delay with exponential backoff
                    current_delay = min(current_delay * backoff_multiplier, max_delay)
                    
                    # Increment retry count for RetryableError
                    if isinstance(e, RetryableError):
                        e.increment_retry()
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def retry_on_network_error(
    max_attempts: int = 3,
    delay: float = 2.0,
    backoff_multiplier: float = 2.0,
):
    """
    Convenience decorator for retrying network-related operations.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_multiplier: Multiplier for delay after each retry
    """
    from core.exceptions import NetworkError, RetryableError
    
    return retry(
        max_attempts=max_attempts,
        delay=delay,
        backoff_multiplier=backoff_multiplier,
        exceptions=(NetworkError, ConnectionError, TimeoutError),
        retryable_exceptions=(RetryableError, NetworkError),
    )


def retry_on_resource_error(
    max_attempts: int = 5,
    delay: float = 1.0,
    backoff_multiplier: float = 1.5,
):
    """
    Convenience decorator for retrying resource-related operations.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_multiplier: Multiplier for delay after each retry
    """
    from core.exceptions import MemoryError, GPUError, ResourceError, RetryableError
    
    return retry(
        max_attempts=max_attempts,
        delay=delay,
        backoff_multiplier=backoff_multiplier,
        exceptions=(MemoryError, GPUError, ResourceError),
        retryable_exceptions=(RetryableError,),
    )


class RetryContext:
    """
    Context manager for manual retry logic with more control.
    
    Example:
        with RetryContext(max_attempts=3, delay=1.0) as retry_ctx:
            while retry_ctx.should_retry():
                try:
                    result = risky_operation()
                    break
                except Exception as e:
                    if not retry_ctx.handle_exception(e):
                        raise
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_multiplier: float = 2.0,
        max_delay: float = 60.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff_multiplier = backoff_multiplier
        self.max_delay = max_delay
        self.exceptions = exceptions
        self.attempt = 0
        self.current_delay = delay
        self.last_exception = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and exc_type in self.exceptions:
            return False  # Re-raise the exception
        return True
    
    def should_retry(self) -> bool:
        """Check if we should retry the operation."""
        return self.attempt < self.max_attempts
    
    def handle_exception(self, exception: Exception) -> bool:
        """
        Handle an exception and determine if we should retry.
        
        Returns:
            True if we should retry, False if we should give up
        """
        self.last_exception = exception
        self.attempt += 1
        
        if not isinstance(exception, self.exceptions):
            logger.error(f"Non-retryable exception in {self.__class__.__name__}: {exception}")
            return False
        
        if self.attempt >= self.max_attempts:
            logger.error(f"Max retry attempts ({self.max_attempts}) exceeded. Last error: {exception}")
            return False
        
        logger.warning(
            f"Attempt {self.attempt}/{self.max_attempts} failed: {exception}. "
            f"Retrying in {self.current_delay:.1f} seconds..."
        )
        
        time.sleep(self.current_delay)
        self.current_delay = min(self.current_delay * self.backoff_multiplier, self.max_delay)
        
        return True
