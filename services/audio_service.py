"""
Audio processing service for video transcription.
Handles audio extraction from video files with robust error handling and logging.
"""

from pathlib import Path
from typing import Optional, Union

import ffmpeg
import numpy as np

from core.logging import get_logger

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

    def extract_audio(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """
        Extract audio from video file using ffmpeg.

        Args:
            video_path: Path to the video file
            output_path: Optional custom output path. If None, uses video filename with audio extension

        Returns:
            Path to the extracted audio file, or None if extraction failed

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video file is invalid
            RuntimeError: If ffmpeg extraction fails
        """

        video_path = Path(video_path)

        if not video_path.exists():
            error_message = f"Video file does not exist: {video_path}"
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        if not video_path.is_file():
            error_message = f"Path is not a file: {video_path}"
            logger.error(error_message)
            raise ValueError(error_message)

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
            raise RuntimeError(error_msg) from e

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
            raise RuntimeError(error_msg) from e

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

    def load_audio_data(self, audio_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load audio data from file into numpy array for processing.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            numpy array containing audio data, or None if loading failed
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            logger.error(f"Audio file does not exist: {audio_path}")
            return None
            
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
            
            # Convert bytes to numpy array
            audio_data = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
            logger.debug(f"Loaded audio data: {len(audio_data)} samples")
            return audio_data
            
        except ffmpeg.Error as e:
            logger.error(f"Failed to load audio data from {audio_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading audio data from {audio_path}: {e}")
            return None
