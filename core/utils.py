"""Utility functions for video file handling."""

from pathlib import Path
from typing import List, Union
from services.video_file_service import VideoFileService

# Global service instance for convenience
_video_service = VideoFileService()


def get_video_files(input_path: Union[str, Path], recursive: bool = False) -> List[Path]:
    """
    Convenience function to get video files using the VideoFileService.
    
    Args:
        input_path: Path to a video file or directory
        recursive: Whether to search subdirectories recursively
        
    Returns:
        List of Path objects for valid video files
        
    Raises:
        FileNotFoundError: If the input path doesn't exist
        ValueError: If the input path is invalid
    """
    return _video_service.get_video_files(input_path, recursive)


def is_video_file(file_path: Union[str, Path]) -> bool:
    """
    Convenience function to check if a file is a video file.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file has a valid video extension, False otherwise
    """
    return _video_service.is_video_file(file_path)


def get_video_file_info(file_path: Union[str, Path]):
    """
    Convenience function to get video file information.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        VideoFileInfo object or None if file cannot be accessed
    """
    return _video_service.get_video_file_info(file_path)
