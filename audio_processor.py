"""
Audio processing module for video transcription.
Handles audio extraction from video files.
"""

import ffmpeg
from pathlib import Path


def extract_audio(video_path, audio_output="audio.wav"):
    """
    Extract audio from video file using ffmpeg.
    
    Args:
        video_path (str): Path to the video file
        audio_output (str): Output audio filename (optional)
    
    Returns:
        str: Path to the extracted audio file
    """
    audio_output = f"{Path(video_path).stem}.wav"
    ffmpeg.input(video_path).output(audio_output, ac=1, ar='16k').run()
    return audio_output
