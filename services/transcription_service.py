from pathlib import Path
from typing import Optional

import whisperx
from deepmultilingualpunctuation import PunctuationModel

from core.config import TranscribeConfig
from core.logging import get_logger
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

    def _load_models(self) -> bool:
        """
        Load the Whisper model and alignment models required for transcription.

        Loads the main Whisper model with the configured size and the alignment
        model for precise timing. Both models are stored as instance attributes
        for use during transcription. If loading fails, logs the error and
        returns False.

        Returns:
            True if all models loaded successfully, False if any model fails to load.
            On failure, the corresponding model attributes remain None.

        Raises:
            No exceptions are raised; errors are logged and False is returned.
        """
        logger.info("Loading models...")
        try:
            self.model = self.model_service.get_model(size=self.config.model_size)
            self.model_alignment, self.metadata = (
                self.model_service.load_alignment_model()
            )
            logger.info("Models loaded successfully.")
            return True
        except Exception:
            logger.exception("Failed to load models")
            return False

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
            except FileNotFoundError:
                logger.exception(
                    f"Input video not found while extracting audio: {video_file}"
                )
                return False
            except Exception:
                logger.exception(f"Failed to extract audio from {video_file.name}")
                return False

            if audio_file is None:
                logger.error(f"Failed to extract audio from {video_file.name}")
                return False

            # Load audio data for transcription
            logger.debug("Loading audio data...")
            audio_data = self.audio_service.load_audio_data(audio_file)
            if audio_data is None:
                logger.error(f"Failed to load audio data from {audio_file}")
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
            except RuntimeError as runtime_error:
                message_lower = str(runtime_error).lower()
                if "out of memory" in message_lower or "cuda" in message_lower:
                    logger.exception(
                        "Transcription failed due to GPU/CUDA error (possible OOM)."
                    )
                else:
                    logger.exception("Transcription failed with runtime error")
                return False
            except Exception:
                logger.exception("Transcription failed")
                return False

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
            except KeyError:
                logger.exception(
                    "Alignment failed: 'segments' missing or malformed in transcription result"
                )
                return False
            except Exception:
                logger.exception("Alignment failed")
                return False

            # Generate subtitles
            logger.debug("Generating subtitles...")
            try:
                output_path: Path = self.subtitle_service.write_subtitles(
                    self.punctuation_model, aligned_result, video_file.stem, output_dir
                )
                logger.debug(f"Subtitle written successfully to {output_path}")
            except Exception:
                logger.exception("Failed to write subtitles")
                return False

            return True

        finally:
            # Ensure cleanup of temporary audio file when possible
            if audio_file is not None:
                if not self.audio_service.cleanup_audio_file(audio_file):
                    logger.warning(
                        f"Failed to clean up temporary audio file: {audio_file}"
                    )
                else:
                    logger.debug("Cleaned up temporary audio file")
