import typing as T

import torch
import whisperx

from core.logging import get_logger

logger = get_logger(__name__)


class ModelService:
    """
    Model management service for video transcription.
    Handles loading and managing WhisperX models.
    """

    DEFAULT_ALIGNMENT_LANGUAGE = "en"
    VALID_MODEL_SIZE = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, device: str, alignment_language: T.Optional[str] = None):
        self.alignment_language = (
            alignment_language
            if alignment_language
            else self.DEFAULT_ALIGNMENT_LANGUAGE
        )
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device
        # Hold references for later cleanup
        self._whisper_model = None
        self._align_model = None
        self._align_metadata = None

    def get_model(self, size="medium", compute_type="int8_float16"):
        """
        Load WhisperX model for transcription.

        Args:
            size (str): Model size (tiny, base, small, medium, large)
            compute_type (str): Compute type for optimization

        Returns:
            WhisperX model instance

        Raises:
            ValueError: If model size is invalid
            RuntimeError: If model loading fails
        """
        if size not in self.VALID_MODEL_SIZE:
            raise ValueError(
                f"Invalid model size: {size}. Must be one of {self.VALID_MODEL_SIZE}"
            )

        try:
            self._whisper_model = whisperx.load_model(
                size, self.device, compute_type=compute_type
            )
            return self._whisper_model
        except Exception as e:
            logger.error(f"Failed to load model: {size}: {e}")
            raise

    def load_alignment_model(self):
        """
        Load alignment model for word-level timing.

        Returns:
            tuple: (model, metadata) for alignment
        """
        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=self.alignment_language, device=self.device
        )
        return self._align_model, self._align_metadata

    def cleanup_models(self) -> None:
        """Best-effort cleanup to free model memory (CPU/GPU)."""
        try:
            # Drop references
            self._whisper_model = None
            self._align_model = None
            self._align_metadata = None
            # Torch-specific cleanup
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            # Avoid raising on cleanup
            pass
