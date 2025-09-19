"""
Input validation utilities for the video transcription application.

This module provides comprehensive validation functions for various inputs
including file paths, video files, configuration values, and system resources.
"""

import os
import psutil
from pathlib import Path
from typing import Union, Optional, List

from core.exceptions import ValidationError, FileNotFoundError, FileAccessError, ResourceError


def validate_file_path(
    file_path: Union[str, Path], 
    must_exist: bool = True,
    file_type: str = "file"
) -> Path:
    """
    Validate a file path and return a Path object.
    
    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist
        file_type: Type of file for error messages
        
    Returns:
        Path: Validated Path object
        
    Raises:
        ValidationError: If path is invalid
        FileNotFoundError: If file doesn't exist and must_exist is True
        FileAccessError: If file cannot be accessed
    """
    if not file_path:
        raise ValidationError("file_path", file_path, "Path cannot be empty")
    
    path = Path(file_path).resolve()
    
    if not path.is_absolute():
        raise ValidationError("file_path", str(file_path), "Path must be absolute")
    
    if must_exist and not path.exists():
        raise FileNotFoundError(path, file_type)
    
    if path.exists():
        if not path.is_file() and file_type == "file":
            raise ValidationError("file_path", str(file_path), "Path is not a file")
        if not path.is_dir() and file_type == "directory":
            raise ValidationError("file_path", str(file_path), "Path is not a directory")
    
    return path


def validate_video_file(video_path: Union[str, Path]) -> Path:
    """
    Validate a video file path and check if it's a valid video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Path: Validated video file path
        
    Raises:
        ValidationError: If file is not a valid video file
        FileNotFoundError: If file doesn't exist
        FileAccessError: If file cannot be accessed
    """
    path = validate_file_path(video_path, must_exist=True, file_type="video file")
    
    # Check file extension
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.3gp', '.ogv', '.ts', '.mts'}
    if path.suffix.lower() not in video_extensions:
        raise ValidationError(
            "video_path", 
            str(path), 
            f"Unsupported video format. Supported: {', '.join(sorted(video_extensions))}"
        )
    
    # Check file size
    try:
        file_size = path.stat().st_size
        if file_size == 0:
            raise ValidationError("video_path", str(path), "Video file is empty")
        
        # 10GB limit
        max_size = 10 * 1024 * 1024 * 1024
        if file_size > max_size:
            raise ValidationError(
                "video_path", 
                str(path), 
                f"Video file too large ({file_size / (1024**3):.1f}GB). Maximum: 10GB"
            )
    except OSError as e:
        raise FileAccessError(path, "access", e) from e
    
    # Check if file is readable
    try:
        with open(path, 'rb') as f:
            f.read(1024)  # Try to read first 1KB
    except (OSError, PermissionError) as e:
        raise FileAccessError(path, "read", e) from e
    
    return path


def validate_directory_path(
    dir_path: Union[str, Path], 
    must_exist: bool = True,
    create_if_missing: bool = False
) -> Path:
    """
    Validate a directory path.
    
    Args:
        dir_path: Path to validate
        must_exist: Whether the directory must exist
        create_if_missing: Whether to create the directory if it doesn't exist
        
    Returns:
        Path: Validated directory path
        
    Raises:
        ValidationError: If path is invalid
        FileNotFoundError: If directory doesn't exist and must_exist is True
        FileAccessError: If directory cannot be accessed or created
    """
    if not dir_path:
        raise ValidationError("dir_path", dir_path, "Directory path cannot be empty")
    
    path = Path(dir_path).resolve()
    
    if not path.exists():
        if create_if_missing:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise FileAccessError(path, "create", e) from e
        elif must_exist:
            raise FileNotFoundError(path, "directory")
    
    if path.exists() and not path.is_dir():
        raise ValidationError("dir_path", str(path), "Path is not a directory")
    
    # Check if directory is writable
    try:
        test_file = path / ".write_test"
        test_file.touch()
        test_file.unlink()
    except OSError as e:
        raise FileAccessError(path, "write", e) from e
    
    return path


def validate_language_code(language: str) -> str:
    """
    Validate a language code.
    
    Args:
        language: Language code to validate
        
    Returns:
        str: Validated language code (lowercase)
        
    Raises:
        ValidationError: If language code is invalid
    """
    if not language:
        raise ValidationError("language", language, "Language code cannot be empty")
    
    language = language.strip().lower()
    
    if language == "auto":
        return language
    
    if len(language) != 2:
        raise ValidationError(
            "language", 
            language, 
            "Language code must be 2 characters (e.g., 'en', 'es') or 'auto'"
        )
    
    if not language.isalpha():
        raise ValidationError(
            "language", 
            language, 
            "Language code must contain only letters"
        )
    
    return language


def validate_model_size(model_size: str) -> str:
    """
    Validate a Whisper model size.
    
    Args:
        model_size: Model size to validate
        
    Returns:
        str: Validated model size (lowercase)
        
    Raises:
        ValidationError: If model size is invalid
    """
    valid_sizes = {"tiny", "base", "small", "medium", "large"}
    model_size = model_size.strip().lower()
    
    if model_size not in valid_sizes:
        raise ValidationError(
            "model_size", 
            model_size, 
            f"Invalid model size. Must be one of: {', '.join(sorted(valid_sizes))}"
        )
    
    return model_size


def validate_device(device: str) -> str:
    """
    Validate a processing device.
    
    Args:
        device: Device to validate
        
    Returns:
        str: Validated device (lowercase)
        
    Raises:
        ValidationError: If device is invalid
    """
    valid_devices = {"auto", "cpu", "cuda"}
    device = device.strip().lower()
    
    if device not in valid_devices:
        raise ValidationError(
            "device", 
            device, 
            f"Invalid device. Must be one of: {', '.join(sorted(valid_devices))}"
        )
    
    return device


def validate_output_format(output_format: str) -> str:
    """
    Validate an output subtitle format.
    
    Args:
        output_format: Format to validate
        
    Returns:
        str: Validated format (lowercase)
        
    Raises:
        ValidationError: If format is invalid
    """
    valid_formats = {"srt", "vtt", "txt"}
    output_format = output_format.strip().lower()
    
    if output_format not in valid_formats:
        raise ValidationError(
            "output_format", 
            output_format, 
            f"Invalid output format. Must be one of: {', '.join(sorted(valid_formats))}"
        )
    
    return output_format


def check_system_resources(
    required_memory_mb: Optional[int] = None,
    required_disk_space_mb: Optional[int] = None
) -> None:
    """
    Check if system has sufficient resources.
    
    Args:
        required_memory_mb: Required memory in MB
        required_disk_space_mb: Required disk space in MB
        
    Raises:
        ResourceError: If insufficient resources are available
    """
    if required_memory_mb:
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        if available_memory < required_memory_mb:
            raise ResourceError(
                f"Insufficient memory: {available_memory:.1f}MB available, "
                f"{required_memory_mb}MB required"
            )
    
    if required_disk_space_mb:
        disk_usage = psutil.disk_usage('/')
        available_space = disk_usage.free / (1024 * 1024)
        if available_space < required_disk_space_mb:
            raise ResourceError(
                f"Insufficient disk space: {available_space:.1f}MB available, "
                f"{required_disk_space_mb}MB required"
            )


def validate_video_files_list(video_files: List[Path]) -> List[Path]:
    """
    Validate a list of video files.
    
    Args:
        video_files: List of video file paths
        
    Returns:
        List[Path]: List of validated video file paths
        
    Raises:
        ValidationError: If no valid video files found
    """
    if not video_files:
        raise ValidationError("video_files", video_files, "No video files provided")
    
    valid_files = []
    for video_file in video_files:
        try:
            validated_file = validate_video_file(video_file)
            valid_files.append(validated_file)
        except (ValidationError, FileNotFoundError, FileAccessError) as e:
            # Log warning but continue with other files
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Skipping invalid video file {video_file}: {e}")
    
    if not valid_files:
        raise ValidationError("video_files", video_files, "No valid video files found")
    
    return valid_files
