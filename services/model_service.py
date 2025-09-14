import torch
import whisperx


class ModelService:
    """
    Model management service for video transcription.
    Handles loading and managing WhisperX models.
    """

    def __init__(self, device: str):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def get_model(self, size="medium", device="cuda", compute_type="int8_float16"):
        """
        Load WhisperX model for transcription.
        
        Args:
            size (str): Model size (tiny, base, small, medium, large)
            device (str): Device to run on (cuda/cpu)
            compute_type (str): Compute type for optimization
        
        Returns:
            WhisperX model instance
        """
        return whisperx.load_model(size, device, compute_type=compute_type)

    def load_alignment_mode(self, language_code="en", device=None):
        """
        Load alignment model for word-level timing.
        
        Args:
            language_code (str): Language code for alignment
            device (str): Device to run on (defaults to DEVICE)
        
        Returns:
            tuple: (model, metadata) for alignment
        """
        if device is None:
            device = self.device
        return whisperx.load_align_model(language_code=language_code, device=device)



