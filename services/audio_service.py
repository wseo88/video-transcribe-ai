
"""
Audio processing service for video transcription.
Handles audio extraction from video files.
"""

import ffmpeg
from pathlib import Path


class AudioService:
    def __init__(self):
        pass

    def extract_audio(self, video_path: str):
        """
        Extract audio from video file using ffmpeg.
        
        Args:
            video_path (str): Path to the video file
        
        Returns:
            str: Path to the extracted audio file
        """
        audio_output = f"{Path(video_path).stem}.wav"
        ffmpeg.input(video_path).output(audio_output, ac=1, ar='16k').run()
        return audio_output
