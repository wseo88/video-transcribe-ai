from pathlib import Path
from typing import Optional

import whisperx
from deepmultilingualpunctuation import PunctuationModel

from core.config import TranscribeConfig
from core.exceptions import (
    ModelLoadError,
    ModelInferenceError,
    AudioExtractionError,
    AudioLoadError,
    SubtitleError,
    FileNotFoundError,
    FileAccessError,
    ValidationError,
    GPUError,
    MemoryError,
    TranscriptionError,
)
from core.logging import get_logger
from core.retry import retry_on_resource_error, retry_on_network_error
from services.audio_service import AudioService
from services.model_service import ModelService
from services.subtitle_service import SubtitleService
from services.video_file_service import VideoFileService

logger = get_logger(__name__)


class TranscriptionService:

    def __init__(self, config: TranscribeConfig):
        """
        Initialize the TranscriptionService with configuration and dependencies.

        Args:
            config: Configuration object containing transcription settings including
                   device, model size, language, input/output paths, and format options.

        Attributes:
            config: The transcription configuration
            video_file_service: Service for discovering video files
            model_service: Service for loading and managing AI models
            audio_service: Service for extracting audio from video files
            model: The loaded Whisper model (None until loaded)
            model_alignment: The alignment model for precise timing (None until loaded)
            metadata: Model metadata for alignment (None until loaded)
            punctuation_model: Model for improving punctuation in transcriptions
            subtitle_service: Service for generating subtitle files
        """
        self.config = config
        self.video_file_service = VideoFileService()
        self.model_service = ModelService(self.config.device)
        self.audio_service = AudioService()
        self.model = None
        self.model_alignment = None
        self.metadata = None
        self.punctuation_model = PunctuationModel()
        self.subtitle_service = SubtitleService(config)

    def _get_video_files_to_process(self) -> Optional[list[Path]]:
        """
        Discover and return video files to be processed for transcription.

        Searches the configured input directory for supported video file formats
        and returns a list of valid video file paths. If no video files are found,
        logs an error and returns None.

        Returns:
            List of Path objects pointing to video files, or None if no files found.
            Returns None if the input directory is invalid or contains no supported formats.

        Raises:
            No exceptions are raised; errors are logged and None is returned.
        """
        video_files: list[Path] = self.video_file_service.get_video_files(
            self.config.input
        )
        if not video_files:
            logger.error("No video files found!")
            return None
        return video_files

    @retry_on_resource_error(max_attempts=2, delay=5.0)
    def _load_models(self) -> bool:
        """
        Load the Whisper model and alignment models required for transcription.

        Loads the main Whisper model with the configured size and the alignment
        model for precise timing. Both models are stored as instance attributes
        for use during transcription.

        Returns:
            True if all models loaded successfully, False if any model fails to load.

        Raises:
            ModelLoadError: If model loading fails
            GPUError: If GPU-related errors occur during loading
            MemoryError: If insufficient memory for model loading
        """
        logger.info("Loading models...")
        try:
            self.model = self.model_service.get_model(size=self.config.model_size)
            self.model_alignment, self.metadata = (
                self.model_service.load_alignment_model()
            )
            logger.info("Models loaded successfully.")
            return True
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "cuda" in error_msg or "gpu" in error_msg:
                raise GPUError("model loading", e) from e
            elif "memory" in error_msg or "out of memory" in error_msg:
                raise MemoryError("model loading") from e
            else:
                raise ModelLoadError(self.config.model_size, self.config.device, e) from e
        except Exception as e:
            raise ModelLoadError(self.config.model_size, self.config.device, e) from e

    def transcribe(self) -> int:
        """
        Main transcription method that processes all discovered video files.

        Orchestrates the complete transcription workflow:
        1. Discovers video files in the configured input directory
        2. Loads required AI models (Whisper and alignment)
        3. Processes each video file through the complete pipeline
        4. Generates subtitle files in the configured output format
        5. Cleans up resources and provides a summary

        The method handles errors gracefully, continuing to process remaining
        files even if individual files fail. A summary of successful and failed
        processing is logged at the end.

        Returns:
            Number of videos successfully transcribed

        Raises:
            No exceptions are raised; all errors are handled internally and logged.
        """        
        successful = 0
        failed = 0

        # Get video files
        video_files = self._get_video_files_to_process()
        if video_files is None:
            logger.error("No video files to process")
            return successful

        # Get effective output directory (handled by Pydantic model)
        output_dir = self.config.effective_output_dir

        # Load models
        if not self._load_models():
            logger.error("Failed to load models, cannot proceed")
            return successful

        # Process files
        
        logger.info(f"Processing {len(video_files)} file(s)...")

        for i, video_file in enumerate(video_files, 1):
            logger.info(f"[{i}/{len(video_files)}] Processing {video_file.name}")

            if self.process_video(video_file, output_dir):
                successful += 1
            else:
                failed += 1

        # Summary
        logger.info(
            f"Processing complete! ✅ {successful} successful, ❌ {failed} failed"
        )

        # Cleanup models to free memory (best-effort)
        try:
            self.model_service.cleanup_models()
        except Exception:
            pass

        return successful

    def process_video(
        self,
        video_file: Path,
        output_dir: Path,
    ) -> bool:
        """
        Process a single video file through the complete transcription pipeline.

        Performs the following steps for each video file:
        1. Checks if output already exists (skips if found)
        2. Validates that required models are loaded
        3. Extracts audio from the video file
        4. Transcribes the audio using the Whisper model
        5. Aligns the transcription for precise timing
        6. Generates subtitle files in the configured format
        7. Cleans up temporary audio files

        The method includes comprehensive error handling for each step,
        with specific error messages for different failure modes (file not found,
        GPU memory issues, transcription failures, etc.).

        Args:
            video_file: Path to the video file to process. Must be a valid
                       video file with supported format.
            output_dir: Directory where subtitle files will be written.
                       Subdirectories will be created as needed.

        Returns:
            True if the video was processed successfully and subtitle files
            were generated, False if any step in the pipeline failed.
            Returns True immediately if output already exists (skip mode).

        Raises:
            No exceptions are raised; all errors are caught and logged internally.
            The method handles FileNotFoundError, RuntimeError (GPU issues),
            KeyError (malformed results), and other exceptions gracefully.
        """
        logger.info(f"Processing: {video_file.name}")

        # Skip if output already exists to save work
        expected_output: Path = (
            Path(output_dir)
            / video_file.stem
            / f"{video_file.stem}.{self.config.output_format}"
        )
        if expected_output.exists():
            logger.info(
                f"Skipping {video_file.name} — subtitle already exists at {expected_output}"
            )
            return True

        # Validate that models are loaded
        if self.model is None or self.model_alignment is None or self.metadata is None:
            logger.error("Models not loaded, cannot process video")
            return False

        audio_file: Optional[str] = None
        try:
            # Extract audio from video
            logger.debug("Extracting audio...")
            try:
                audio_file = self.audio_service.extract_audio(str(video_file))
            except (FileNotFoundError, ValidationError, FileAccessError) as e:
                logger.error(f"Audio extraction failed: {e}")
                return False
            except AudioExtractionError as e:
                logger.error(f"Audio extraction failed: {e}")
                return False

            # Load audio data for transcription
            logger.debug("Loading audio data...")
            try:
                audio_data = self.audio_service.load_audio_data(audio_file)
            except (FileNotFoundError, ValidationError) as e:
                logger.error(f"Audio loading failed: {e}")
                return False
            except AudioLoadError as e:
                logger.error(f"Audio loading failed: {e}")
                return False

            # Determine transcription task
            task = "translate"
            transcribe_kwargs = {"task": task}

            if self.config.language != "auto":
                transcribe_kwargs["language"] = self.config.language

            # Transcribe audio
            logger.debug("Transcribing audio...")
            try:
                result: TranscriptionService.TranscribeResult = self.model.transcribe(
                    audio_data, **transcribe_kwargs
                )
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "cuda" in error_msg:
                    logger.error("Transcription failed due to GPU/CUDA error (possible OOM)")
                    raise GPUError("transcription", e) from e
                else:
                    logger.error("Transcription failed with runtime error")
                    raise ModelInferenceError("transcription", e) from e
            except Exception as e:
                logger.error("Transcription failed")
                raise ModelInferenceError("transcription", e) from e

            # Align transcription
            logger.debug("Aligning transcription...")
            try:
                aligned_result: TranscriptionService.AlignedResult = whisperx.align(
                    result["segments"],
                    self.model_alignment,
                    self.metadata,
                    str(audio_file),  # WhisperX expects file path, not numpy array
                    self.model_service.device,
                    return_char_alignments=True,
                )
            except KeyError as e:
                logger.error("Alignment failed: 'segments' missing or malformed in transcription result")
                raise ModelInferenceError("alignment", e) from e
            except Exception as e:
                logger.error("Alignment failed")
                raise ModelInferenceError("alignment", e) from e

            # Generate subtitles
            logger.debug("Generating subtitles...")
            try:
                output_path: Path = self.subtitle_service.write_subtitles(
                    self.punctuation_model, aligned_result, video_file.stem, output_dir
                )
                logger.debug(f"Subtitle written successfully to {output_path}")
            except Exception as e:
                logger.error("Failed to write subtitles")
                raise SubtitleError("generation", video_file, e) from e

            return True

        except (AudioExtractionError, AudioLoadError, ModelInferenceError, SubtitleError) as e:
            logger.error(f"Processing failed: {e}")
            return False
        except (FileNotFoundError, ValidationError, FileAccessError) as e:
            logger.error(f"File error: {e}")
            return False
        except (GPUError, MemoryError) as e:
            logger.error(f"Resource error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error processing {video_file.name}: {e}")
            return False

        finally:
            # Ensure cleanup of temporary audio file when possible
            if audio_file is not None:
                if not self.audio_service.cleanup_audio_file(audio_file):
                    logger.warning(
                        f"Failed to clean up temporary audio file: {audio_file}"
                    )
                else:
                    logger.debug("Cleaned up temporary audio file")
