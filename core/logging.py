"""
Centralized logging configuration for video transcription application.
Provides consistent logging setup across all modules.
"""

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class LogLevel(Enum):
    """Log level enumeration for type safety."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggingConfig:
    """Configuration class for logging setup."""

    # Default logging configuration
    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_LEVEL = LogLevel.INFO
    DEFAULT_HANDLERS = ["console"]

    # Log file configuration
    LOG_DIR = Path("logs")
    LOG_FILE_FORMAT = (
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )

    def __init__(
        self,
        level: LogLevel = DEFAULT_LEVEL,
        format_string: str = DEFAULT_FORMAT,
        handlers: list = None,
        log_to_file: bool = False,
        log_file_path: Optional[Path] = None,
    ):
        self.level = level
        self.format_string = format_string
        self.handlers = handlers or self.DEFAULT_HANDLERS.copy()
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path or self.LOG_DIR / "transcribe.log"


def setup_logging(
    is_verbose: bool = False,
    logging_config: Optional[LoggingConfig] = None,
) -> logging.Logger:
    """
    Set up centralized logging configuration.

    Args:
        config: TranscribeConfig instance (optional, for verbose setting)
        logging_config: Custom LoggingConfig instance (optional)

    Returns:
        Configured root logger
    """
    # Determine log level
    if is_verbose:
        level = LogLevel.DEBUG
    elif logging_config:
        level = logging_config.level
    else:
        level = LoggingConfig.DEFAULT_LEVEL

    # Create logging config if not provided
    if not logging_config:
        logging_config = LoggingConfig(level=level)

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set log level
    root_logger.setLevel(getattr(logging, level.value))

    # Create formatter
    formatter = logging.Formatter(logging_config.format_string)

    # Add console handler
    if "console" in logging_config.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if requested
    if logging_config.log_to_file and "file" in logging_config.handlers:
        # Create log directory if it doesn't exist
        logging_config.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(logging_config.log_file_path)
        file_formatter = logging.Formatter(LoggingConfig.LOG_FILE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Configure specific loggers
    _configure_third_party_loggers()

    return root_logger


def _configure_third_party_loggers():
    """Configure logging levels for third-party libraries."""
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("whisperx").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: LogLevel) -> None:
    """
    Dynamically change the logging level.

    Args:
        level: New log level
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.value))

    # Update all handlers
    for handler in root_logger.handlers:
        handler.setLevel(getattr(logging, level.value))


def add_file_logging(
    log_file_path: Path,
    level: LogLevel = LogLevel.DEBUG,
    format_string: Optional[str] = None,
) -> None:
    """
    Add file logging to existing configuration.

    Args:
        log_file_path: Path to log file
        level: Log level for file handler
        format_string: Custom format string (optional)
    """
    root_logger = logging.getLogger()

    # Create log directory if it doesn't exist
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(getattr(logging, level.value))

    # Set formatter
    if format_string:
        formatter = logging.Formatter(format_string)
    else:
        formatter = logging.Formatter(LoggingConfig.LOG_FILE_FORMAT)

    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def get_logging_stats() -> Dict[str, Any]:
    """
    Get statistics about current logging configuration.

    Returns:
        Dictionary with logging statistics
    """
    root_logger = logging.getLogger()

    return {
        "root_level": logging.getLevelName(root_logger.level),
        "handlers_count": len(root_logger.handlers),
        "handler_types": [type(h).__name__ for h in root_logger.handlers],
        "effective_level": logging.getLevelName(root_logger.getEffectiveLevel()),
    }


# Convenience function for quick setup
def quick_setup(verbose: bool = False) -> logging.Logger:
    """
    Quick setup for logging with minimal configuration.

    Args:
        verbose: Enable verbose (DEBUG) logging

    Returns:
        Configured root logger
    """
    level = LogLevel.DEBUG if verbose else LogLevel.INFO
    config = LoggingConfig(level=level)
    return setup_logging(logging_config=config)
