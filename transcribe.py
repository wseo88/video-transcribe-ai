import torch
import ffmpeg
import os
import re
import whisperx
from pathlib import Path
from deepmultilingualpunctuation import PunctuationModel

import whisperx.utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_audio(video_path, audio_output="audio.wav"):
    audio_output = f"{Path(video_path).stem}.wav"
    ffmpeg.input(video_path).output(audio_output, ac=1, ar='16k').run()
    return audio_output


def get_model(size="medium", device="cuda", compute_type="int8_float16"):
    return whisperx.load_model(size, device, compute_type=compute_type)


def format_timestamp(seconds):
    if seconds is None:
        return "00:00:00,000"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def split_subtitle(text, max_chars=42):
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


def extract_words(text):
    return set(re.findall(r'\b[\w\']+\b', text.lower()))


def split_at_sentence_end(text, word_data):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    current_word_index = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            sentence_word_count = len(sentence.split())
            sentence_word_data = word_data[current_word_index:current_word_index + sentence_word_count]
            if sentence_word_data:
                start_time = next((word['start'] for word in sentence_word_data if 'start' in word), None)
                end_time = next((word['end'] for word in reversed(sentence_word_data) if 'end' in word), None)
                if start_time is not None and end_time is not None:
                    result.append({
                        'text': sentence,
                        'start': start_time,
                        'end': end_time
                    })
                else:
                    # If start or end time is missing, use the previous valid timestamp
                    if result:
                        prev_end = result[-1]['end']
                        result.append({
                            'text': sentence,
                            'start': prev_end,
                            'end': prev_end + 1  # Add 1 second as a placeholder duration
                        })
                    else:
                        # If it's the first sentence and times are missing, use 0 as start time
                        result.append({
                            'text': sentence,
                            'start': 0,
                            'end': 1  # Add 1 second as a placeholder duration
                        })
            current_word_index += sentence_word_count
    return result

def merge_short_cues(cues, min_duration=3):
    merged_cues = []
    current_cue = None

    for cue in cues:
        if current_cue is None:
            current_cue = cue
        else:
            duration = cue['end'] - current_cue['start']
            if duration < min_duration:
                current_cue['text'] += ' ' + cue['text']
                current_cue['end'] = cue['end']
            else:
                merged_cues.append(current_cue)
                current_cue = cue

    if current_cue:
        merged_cues.append(current_cue)

    return merged_cues


def write_to_srt(punct_model, aligned_result, video_name):
    os.makedirs(video_name, exist_ok=True)
    output_path = os.path.join(video_name, f"{video_name}.srt")

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

def main():
    # Get all mp4 files (without mp4 extensions)
    current_dir = Path(".")
    file_names = [file.name for file in list(current_dir.glob("*.mp4"))]

    # Load model
    model = get_model()
    model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    punct_model = PunctuationModel()
    for file_name in file_names:
        print(f"Processing {file_name}...")
        
        audio_file = extract_audio(file_name)
        result = model.transcribe(audio_file, task="translate")
        aligned_result = whisperx.align(
            result["segments"], model_a, metadata, audio_file, "cuda", return_char_alignments=True
        )
        #aligned_result["language"] = result["language"]
        output_path = write_to_srt(punct_model, aligned_result, Path(file_name).stem)

        print(f"✅ Subtitles saved to: {output_path}")
        os.remove(audio_file)
main()