import os
import typing as T
from pathlib import Path

import whisperx
from deepmultilingualpunctuation import PunctuationModel

from core.logging import get_logger
from core.models import TranscribeConfig
from services.audio_service import AudioService
from services.model_service import ModelService
from services.subtitle_service import SubtitleService
from services.video_file_service import VideoFileService

logger = get_logger(__name__)


class TranscriptionService:

    def __init__(self, config: TranscribeConfig):
        self.config = config
        self.vide_file_service = VideoFileService()
        self.model_service = ModelService(self.config.device)
        self.audio_service = AudioService()
        self.model = None
        self.model_alignment = None
        self.metadata = None
        self.punctuation_model = PunctuationModel()
        self.subtitle_service = SubtitleService(config)

    def _get_video_files_to_process(self) -> T.Optional[T.List[Path]]:
        video_files: T.List[Path] = self.vide_file_service.get_video_files(
            self.config.input
        )
        if not video_files:
            logger.error("No video files found!")
            return 1
        return video_files

    def _load_models(self):
        # Load models
        logger.info("Loading models...")
        try:
            self.model = self.model_service.get_model(size=self.config.model_size)
            self.model_alignment, self.metadata = (
                self.model_service.load_alignment_model(language_code="en")
            )
            logger.info("Models loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return 1

    def transcribe(self):
        # Get video files
        video_files = self._get_video_files_to_process()
        # Get effective output directory (handled by Pydantic model)
        output_dir = self.config.effective_output_dir
        # load_models
        self._load_models()

        # Process files
        successful = 0
        failed = 0

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

        return 0 if failed == 0 else 1

    def process_video(
        self,
        video_file: Path,
        output_dir: Path,
    ) -> bool:
        """Process a single video file."""
        logger.info(f"Processing: {video_file.name}")

        try:
            # Extract audio from video
            logger.debug("Extracting audio...")
            audio_file = self.audio_service.extract_audio(str(video_file))
            
            if audio_file is None:
                logger.error(f"Failed to extract audio from {video_file.name}")
                return False

            # Determine transcription task
            task = "translate"
            transcribe_kwargs = {"task": task}

            if self.config.language != "auto":
                transcribe_kwargs["language"] = self.config.language

            # Transcribe audio
            logger.debug("Transcribing audio...")
            result = self.model.transcribe(audio_file, **transcribe_kwargs)

            # Align transcription
            logger.debug("Aligning transcription...")
            aligned_result = whisperx.align(
                result["segments"],
                self.model_alignment,
                self.metadata,
                audio_file,
                self.model_service.device,
                return_char_alignments=True,
            )

            # Generate subtitles
            logger.debug("Generating subtitles...")
            output_path = self.subtitle_service.write_subtitles(
                self.punctuation_model, aligned_result, video_file.stem, output_dir
            )

            # Clean up temporary audio file
            if not self.audio_service.cleanup_audio_file(audio_file):
                logger.warning(f"Failed to clean up temporary audio file: {audio_file}")
            else:
                logger.debug("Cleaned up temporary audio file")

            return True

        except Exception as e:
            logger.error(f"❌ Error processing {video_file.name}: {str(e)}")
            return False
