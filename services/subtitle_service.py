import os
import re


class SubtitleService:
    """
    Subtitle formatting service for video transcription.
    Handles SRT file generation and subtitle formatting.
    """

    def __init__(self):
        pass

    def _format_timestamp(self, seconds):
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
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")

    def _split_subtitle(self, text, max_chars=42):
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
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    def _split_at_sentence_end(self, text, word_data):
        """
        Split text at sentence boundaries and align with word timing data.

        Args:
            text (str): Input text
            word_data (list): List of word timing data

        Returns:
            list: List of sentence segments with timing
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result = []
        current_word_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence_word_count = len(sentence.split())
                sentence_word_data = word_data[
                    current_word_index : current_word_index + sentence_word_count
                ]

                if sentence_word_data:
                    start_time = next(
                        (
                            word["start"]
                            for word in sentence_word_data
                            if "start" in word
                        ),
                        None,
                    )
                    end_time = next(
                        (
                            word["end"]
                            for word in reversed(sentence_word_data)
                            if "end" in word
                        ),
                        None,
                    )

                    if start_time is not None and end_time is not None:
                        result.append(
                            {"text": sentence, "start": start_time, "end": end_time}
                        )
                    else:
                        # If start or end time is missing, use the previous valid timestamp
                        if result:
                            prev_end = result[-1]["end"]
                            result.append(
                                {
                                    "text": sentence,
                                    "start": prev_end,
                                    "end": prev_end
                                    + 1,  # Add 1 second as a placeholder duration
                                }
                            )
                        else:
                            # If it's the first sentence and times are missing, use 0 as start time
                            result.append(
                                {
                                    "text": sentence,
                                    "start": 0,
                                    "end": 1,  # Add 1 second as a placeholder duration
                                }
                            )
                current_word_index += sentence_word_count

        return result

    def _merge_short_cues(self, cues, min_duration=3):
        """
        Merge subtitle cues that are too short.

        Args:
            cues (list): List of subtitle cues
            min_duration (float): Minimum duration in seconds

        Returns:
            list: List of merged cues
        """
        merged_cues = []
        current_cue = None

        for cue in cues:
            if current_cue is None:
                current_cue = cue
            else:
                duration = cue["end"] - current_cue["start"]
                if duration < min_duration:
                    current_cue["text"] += " " + cue["text"]
                    current_cue["end"] = cue["end"]
                else:
                    merged_cues.append(current_cue)
                    current_cue = cue

        if current_cue:
            merged_cues.append(current_cue)

        return merged_cues

    def write_to_srt(self, punct_model, aligned_result, video_name, output_dir=None):
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
                word_data = segment.get("words", [])
                sentences = self._split_at_sentence_end(text, word_data)
                all_cues.extend(sentences)

            merged_cues = self._merge_short_cues(all_cues)

            for cue in merged_cues:
                formatted_text = self._split_subtitle(cue["text"])

                srt_file.write(f"{srt_index}\n")
                srt_file.write(
                    f"{self._format_timestamp(cue['start'])} --> {self._format_timestamp(cue['end'])}\n"
                )
                srt_file.write(f"{formatted_text}\n\n")

                srt_index += 1

        return output_path
