"""
Subtitle formatting module for video transcription.
Handles SRT file generation and subtitle formatting.
"""

import os
from pathlib import Path
from deepmultilingualpunctuation import PunctuationModel
from text_processor import split_at_sentence_end, merge_short_cues


def format_timestamp(seconds):
    """
    Format seconds into SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted timestamp
    """
    if seconds is None:
        return "00:00:00,000"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')


def split_subtitle(text, max_chars=42):
    """
    Split subtitle text into multiple lines if too long.
    
    Args:
        text (str): Subtitle text
        max_chars (int): Maximum characters per line
    
    Returns:
        str: Formatted subtitle with line breaks
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_chars and current_line:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1

    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)


def write_to_srt(punct_model, aligned_result, video_name, output_dir=None):
    """
    Write transcription results to SRT subtitle file.
    
    Args:
        punct_model: Punctuation restoration model
        aligned_result (dict): Aligned transcription results
        video_name (str): Name of the video file (without extension)
        output_dir (str, optional): Output directory path
    
    Returns:
        str: Path to the generated SRT file
    """
    if output_dir is None:
        output_dir = video_name
    
    # Create video-specific subdirectory
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    output_path = os.path.join(video_output_dir, f"{video_name}.srt")

    srt_index = 1
    with open(output_path, "w", encoding="utf-8") as srt_file:
        all_cues = []
        for segment in aligned_result["segments"]:
            text = punct_model.restore_punctuation(segment["text"])
            word_data = segment.get('words', [])
            sentences = split_at_sentence_end(text, word_data)
            all_cues.extend(sentences)

        merged_cues = merge_short_cues(all_cues)

        for cue in merged_cues:
            formatted_text = split_subtitle(cue['text'])
            
            srt_file.write(f"{srt_index}\n")
            srt_file.write(f"{format_timestamp(cue['start'])} --> {format_timestamp(cue['end'])}\n")
            srt_file.write(f"{formatted_text}\n\n")
            
            srt_index += 1
    
    return output_path
