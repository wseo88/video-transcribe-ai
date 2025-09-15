"""
Video file management service for video transcription.
Handles video file discovery, validation, and metadata operations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VideoFileInfo:
    """Information about a video file"""

    path: Path
    size: int
    extension: str
    name: str
    stem: str

    @property
    def size_mb(self) -> float:
        """Get file size in megabytes."""
        return round(self.size / (1024 * 1024), 2)

    @property
    def size_gb(self) -> float:
        """Get file size in gigabytes."""
        return round(self.size / (1024 * 1024 * 1024), 2)


class VideoFileService:
    """
    Service for managing video file operations.
    Handles discovery, validation, and metadata extraction.
    """

    # Class constants
    DEFAULT_SUPPORTED_FORMATS: Set[str] = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".flv",
        ".wmv",
        ".webm",
    }

    # Additional formats that might be useful
    EXTENDED_FORMATS: Set[str] = DEFAULT_SUPPORTED_FORMATS | {
        ".m4v",
        ".3gp",
        ".ogv",
        ".ts",
        ".mts",
        ".m2ts",
    }

    # Size constants (in bytes)
    MIN_FILE_SIZE = 1024  # 1KB
    MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB

    def __init__(self, supported_formats: Optional[Set[str]] = None):
        self.supported_formats = supported_formats or self.DEFAULT_SUPPORTED_FORMATS
        logger.info(
            f"VideoFileService initialized with formats: {self.supported_formats}"
        )

    def get_video_files(self, input_path: Union[str, Path]) -> List[Path]:
        """
        Get list of video files from input path.

        Args:
            input_path: Path to a video file or directory containing video files

        Returns:
            List of Path objects for valid video files, sorted by name

        Raises:
            FileNotFoundError: If the input path doesn't exist
            ValueError: If the input path is invalid

        Examples:
            >>> get_video_files("video.mp4")
            [PosixPath('video.mp4')]
            >>> get_video_files("./videos/")
            [PosixPath('./videos/movie1.mp4'), PosixPath('./videos/movie2.avi')]
        """
        if not input_path:
            raise ValueError("Input path cannot be empty")

        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if input_path.is_file():
            return self._validate_video_file(input_path)
        elif input_path.is_dir():
            return self._find_video_files_in_directory(input_path)
        else:
            raise ValueError(
                f"Input path is neither a file nor directory: {input_path}"
            )

    def get_video_file_info(
        self, file_path: Union[str, Path]
    ) -> Optional[VideoFileInfo]:
        """
        Get detailed information about a video file.

        Args:
            file_path: Path to the video file

        Returns:
            VideoFileInfo object or None if file cannot be accessed
        """
        file_path = Path(file_path)

        if not self.is_video_file(file_path):
            logger.warning(f"File is not a supported video format: {file_path}")
            return None

        try:
            stat = file_path.stat()
            return VideoFileInfo(
                path=file_path,
                size=stat.st_size,
                extension=file_path.suffix.lower(),
                name=file_path.name,
                stem=file_path.stem,
            )
        except (OSError, FileNotFoundError) as e:
            logger.error(f"Could not get file info for {file_path}: {e}")
            return None

    def get_video_files_summary(self, video_files: List[Path]) -> Dict[str, Any]:
        """
        Get a summary of video files including total count, sizes, and formats.

        Args:
            video_files: List of video file paths

        Returns:
            Dictionary with summary statistics
        """
        total_size = 0
        format_counts = {}
        file_infos = []

        for file_path in video_files:
            file_info = self.get_video_file_info(file_path)
            if file_info:
                total_size += file_info.size
                format_counts[file_info.extension] = (
                    format_counts.get(file_info.extension, 0) + 1
                )
                file_infos.append(file_info)

        return {
            "total_files": len(file_infos),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "format_distribution": format_counts,
            "files": file_infos,
        }

    def _validate_video_file(self, file_path: Path) -> List[Path]:
        """Validate a single video file and return it if it's a valid video file."""
        if not self.is_video_file(file_path):
            logger.warning(f"Unsupported file format: {file_path.suffix}")
            return []

        # Check if file is actually readable
        try:
            if not file_path.is_file():
                logger.warning(f"Path is not a file: {file_path}")
                return []
            # Try to access file stats to ensure it's readable
            file_path.stat()
        except (OSError, PermissionError) as e:
            logger.error(f"Cannot access file {file_path}: {e}")
            return []

        return [file_path]

    def _find_video_files_in_directory(self, directory: Path) -> List[Path]:
        """Find and validate all video files in a directory"""
        video_files = []

        # Search for both lowercase and uppercase extensions
        for suffix in self.supported_formats:
            video_files.extend(directory.glob(f"*{suffix}"))

        return sorted(video_files)

    def is_video_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file is a valid video file based on its extension.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file has a valid video extension, False otherwise
        """
        return Path(file_path).suffix.lower() in self.supported_formats

    def filter_video_files_by_size(
        self,
        video_files: List[Path],
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> List[Path]:
        """
        Filter video files by size constraints.

        Args:
            video_files: List of video file paths
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes

        Returns:
            List of video files that meet the size criteria
        """
        filtered_files = []

        for file_path in video_files:
            file_info = self.get_video_file_info(file_path)
            if file_info is None:
                continue

            if min_size and file_info.size < min_size:
                continue
            if max_size and file_info.size > max_size:
                continue

            filtered_files.append(file_path)

        return filtered_files

    def get_file_size_mb(self, file_path: Union[str, Path]) -> Optional[float]:
        """
        Get file size in megabytes.

        Args:
            file_path: Path to the file

        Returns:
            File size in MB, or None if file cannot be accessed
        """
        file_info = self.get_video_file_info(file_path)
        if file_info:
            return round(file_info.size / (1024 * 1024), 2)
        return None
