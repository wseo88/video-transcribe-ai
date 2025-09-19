"""
Audio processing service for video transcription.
Handles audio extraction from video files with robust error handling and logging.
"""

from pathlib import Path
from typing import Optional, Union

import ffmpeg
import numpy as np

from core.exceptions import (
    AudioExtractionError,
    AudioLoadError,
    FileNotFoundError,
    FileAccessError,
    ValidationError,
    RetryableError,
)
from core.logging import get_logger
from core.retry import retry_on_resource_error

logger = get_logger(__name__)


class AudioService:
    """
    Service for extracting audio from video files using ffmpeg.

    Provides robust audio extraction with configurable parameters,
    proper error handling, and automatic cleanup.
    """

    # Default audio extraction parameters
    DEFAULT_SAMPLE_RATE = "16k"
    DEFAULT_CHANNELS = 1  # mono
    DEFAULT_AUDIO_FORMAT = "wav"

    def __init__(
        self,
        sample_rate: str = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        audio_format: str = DEFAULT_AUDIO_FORMAT,
    ):
        """
        Initialize AudioService with configurable parameters.

        Args:
            sample_rate: Audio sample rate (e.g., "16k", "44.1k", "48k")
            channels: Number of audio channels (1 for mono, 2 for stereo)
            audio_format: Output audio format ("wav", "mp3", "flac")
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_format = audio_format
        logger.info(
            f"AudioService initialized: {channels}ch, {sample_rate}, {audio_format}"
        )

    def _validate_video_file(self, video_path: Path) -> None:
        """
        Validate video file before processing.
        
        Args:
            video_path: Path to the video file
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValidationError: If video file is invalid
            FileAccessError: If file cannot be accessed
        """
        if not video_path.exists():
            raise FileNotFoundError(video_path, "video file")
        
        if not video_path.is_file():
            raise ValidationError("video_path", str(video_path), "Path is not a file")
        
        # Check file size
        try:
            file_size = video_path.stat().st_size
            if file_size == 0:
                raise ValidationError("video_path", str(video_path), "File is empty")
            if file_size > 10 * 1024 * 1024 * 1024:  # 10GB limit
                raise ValidationError("video_path", str(video_path), "File too large (>10GB)")
        except OSError as e:
            raise FileAccessError(video_path, "access", e) from e
        
        # Check if file is readable
        try:
            with open(video_path, 'rb') as f:
                f.read(1024)  # Try to read first 1KB
        except (OSError, PermissionError) as e:
            raise FileAccessError(video_path, "read", e) from e

    @retry_on_resource_error(max_attempts=3, delay=2.0)
    def extract_audio(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Extract audio from video file using ffmpeg with robust error handling.

        Args:
            video_path: Path to the video file
            output_path: Optional custom output path. If None, uses video filename with audio extension

        Returns:
            Path to the extracted audio file

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValidationError: If video file is invalid
            FileAccessError: If file cannot be accessed
            AudioExtractionError: If ffmpeg extraction fails
        """

        video_path = Path(video_path)
        
        # Validate input file
        self._validate_video_file(video_path)

        # Determine output path
        if output_path is None:
            audio_output = video_path.parent / f"{video_path.stem}.{self.audio_format}"
        else:
            audio_output = Path(output_path)

        logger.info(f"Extracting audio from {video_path.name} to {audio_output.name}")

        try:
            # Extract audio using ffmpeg
            (
                ffmpeg.input(str(video_path))
                .output(
                    str(audio_output),
                    ac=self.channels,
                    ar=self.sample_rate,
                    acodec="pcm_s16le" if self.audio_format == "wav" else None,
                )
                .overwrite_output()  # Overwrite existing files
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )

            # Verify output file was created
            if not audio_output.exists():
                error_msg = "Audio extraction failed: output file not created"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            file_size = audio_output.stat().st_size
            logger.info(
                f"✅ Audio extracted successfully: {audio_output.name} ({file_size:,} bytes)"
            )

            return audio_output

        except ffmpeg.Error as e:
            error_msg = f"FFmpeg error during audio extraction: {e.stderr.decode() if e.stderr else str(e)}"
            logger.error(error_msg)
            # Clean up partial output file if it exists
            if audio_output.exists():
                try:
                    audio_output.unlink()
                    logger.debug("Cleaned up partial audio file")
                except OSError:
                    pass
            raise AudioExtractionError(video_path, e) from e

        except Exception as e:
            error_msg = f"Unexpected error during audio extraction: {str(e)}"
            logger.error(error_msg)
            # Clean up partial output file if it exists
            if audio_output.exists():
                try:
                    audio_output.unlink()
                    logger.debug("Cleaned up partial audio file")
                except OSError:
                    pass
            raise AudioExtractionError(video_path, e) from e

    def cleanup_audio_file(self, audio_path: Union[str, Path]) -> bool:
        """
        Clean up an audio file.

        Args:
            audio_path: Path to the audio file to be deleted

        Returns:
            True if file was successfully deleted, False otherwise
        """

        audio_path = Path(audio_path)

        if not audio_path.exists():
            logger.debug(
                f"Audio file does not exist, nothing to clean up: {audio_path}"
            )
            return True

        try:
            audio_path.unlink()
            logger.debug(f"Cleaned up temporary audio file: {audio_path.name}")
            return True
        except OSError as e:
            logger.warning(f"Failed to clean up audio file {audio_path}: {e}")
            return False

    @retry_on_resource_error(max_attempts=2, delay=1.0)
    def load_audio_data(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Load audio data from file into numpy array for processing.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            numpy array containing audio data
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            AudioLoadError: If audio loading fails
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(audio_path, "audio file")
        
        if not audio_path.is_file():
            raise ValidationError("audio_path", str(audio_path), "Path is not a file")
            
        try:
            # Load audio using ffmpeg
            logger.debug(f"Loading audio data from {audio_path.name}")
            out, _ = (
                ffmpeg
                .input(str(audio_path))
                .output('-', format='wav', ac=1, ar=16000)  # mono, 16kHz
                .overwrite_output()
                .run(capture_stdout=True, quiet=True)
            )
            
            if not out:
                raise AudioLoadError(audio_path, "No audio data returned from ffmpeg")
            
            # Convert bytes to numpy array
            audio_data = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
            
            if len(audio_data) == 0:
                raise AudioLoadError(audio_path, "Empty audio data")
            
            logger.debug(f"Loaded audio data: {len(audio_data)} samples")
            return audio_data
            
        except ffmpeg.Error as e:
            error_msg = f"FFmpeg error loading audio: {e.stderr.decode() if e.stderr else str(e)}"
            logger.error(error_msg)
            raise AudioLoadError(audio_path, e) from e
        except Exception as e:
            error_msg = f"Unexpected error loading audio data: {e}"
            logger.error(error_msg)
            raise AudioLoadError(audio_path, e) from e
