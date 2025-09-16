import re
from pathlib import Path
from typing import Any, Optional, Union

from core.config import TranscribeConfig
from core.logging import get_logger

logger = get_logger(__name__)


class SubtitleService:
    """
    Subtitle formatting service for video transcription.
    Handles SRT, VTT, and TXT file generation and subtitle formatting.
    """

    # Default configuration constants
    DEFAULT_MAX_CHARS_PER_LINE = 42
    DEFAULT_MIN_CUE_DURATION = 3.0
    DEFAULT_MAX_CUE_DURATION = 7.0
    DEFAULT_MAX_LINES_PER_CUE = 2

    def __init__(self, config: Optional[TranscribeConfig] = None):
        """
        Initialize the SubtitleService.

        Args:
            config: Optional TranscribeConfig for subtitle formatting options
        """
        self.config = config
        self.max_chars_per_line = self.DEFAULT_MAX_CHARS_PER_LINE
        self.min_cue_duration = self.DEFAULT_MIN_CUE_DURATION
        self.max_cue_duration = self.DEFAULT_MAX_CUE_DURATION
        self.max_lines_per_cue = self.DEFAULT_MAX_LINES_PER_CUE

        # Override defaults with config if provided
        if config:
            self._apply_config_settings()

    def _apply_config_settings(self) -> None:
        """
        Apply configuration settings from TranscribeConfig.
        """
        # Future: Add config-based customization options
        # For now, we use the defaults but this provides extensibility
        pass

    def _format_timestamp(
        self, seconds: Optional[float], format_type: str = "srt"
    ) -> str:
        """
        Format seconds into timestamp format (HH:MM:SS,mmm)

        Args:
            seconds: Time in seconds
            format_type: Format type ("srt", "vtt")

        Returns:
            Formatted timestamp string
        """
        if seconds is None:
            return "00:00:00,000" if format_type == "srt" else "00:00:00.000"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60

        if format_type == "srt":
            return f"{hours:02d}:{minutes:02d}:{seconds_remainder:06.3f}".replace(
                ".", ","
            )
        elif format_type == "vtt":
            return f"{hours:02d}:{minutes:02d}:{seconds_remainder:06.3f}"
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def _split_subtitle(self, text: str, max_chars: Optional[int] = None) -> str:
        """
        Split subtitle text into multiple lines if too long.

        Args:
            text: Subtitle text to split
            max_chars: Maximum characters per line (uses default if None)

        Returns:
            Formatted subtitle with line breaks
        """
        if not text or not text.strip():
            return ""

        if max_chars is None:
            max_chars = self.max_chars_per_line

        words = text.split()
        if not words:
            return ""

        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word)
            # Check if adding this word would exceed the limit
            if (
                current_length + word_length + (1 if current_line else 0) > max_chars
                and current_line
            ):
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                current_line.append(word)
                current_length += word_length + (1 if current_line else 0)

        if current_line:
            lines.append(" ".join(current_line))

        # Limit to maximum lines per cue
        if len(lines) > self.max_lines_per_cue:
            lines = lines[: self.max_lines_per_cue]
            # Add ellipsis if truncated
            if len(lines) == self.max_lines_per_cue:
                lines[-1] = lines[-1][: max_chars - 3] + "..."

        return "\n".join(lines)

    def _split_at_sentence_end(
        self, text: str, word_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Split text at sentence boundaries and align with word timing data.

        Args:
            text: Input text to split
            word_data: list of word timing data dictionaries

        Returns:
            list of sentence segments with timing information
        """
        if not text or not text.strip():
            return []

        # Split at sentence boundaries (period, exclamation, question mark)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result = []
        current_word_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_word_count = len(sentence.split())

            # Extract word data for this sentence
            sentence_word_data = word_data[
                current_word_index : current_word_index + sentence_word_count
            ]

            if sentence_word_data:
                # Find start and end times from word data
                start_time = self._extract_start_time(sentence_word_data)
                end_time = self._extract_end_time(sentence_word_data)

                if start_time is not None and end_time is not None:
                    result.append(
                        {"text": sentence, "start": start_time, "end": end_time}
                    )
                else:
                    # Handle missing timing data
                    result.append(self._create_fallback_cue(sentence, result))
            else:
                # No word data available, create fallback cue
                result.append(self._create_fallback_cue(sentence, result))

            current_word_index += sentence_word_count

        return result

    def _extract_start_time(self, word_data: list[dict[str, Any]]) -> Optional[float]:
        """Extract start time from word data."""
        for word in word_data:
            if "start" in word and word["start"] is not None:
                return float(word["start"])
        return None

    def _extract_end_time(self, word_data: list[dict[str, Any]]) -> Optional[float]:
        """Extract end time from word data."""
        for word in reversed(word_data):
            if "end" in word and word["end"] is not None:
                return float(word["end"])
        return None

    def _create_fallback_cue(
        self, text: str, existing_cues: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Create a fallback cue when timing data is missing."""
        if existing_cues:
            prev_end = existing_cues[-1]["end"]
            return {
                "text": text,
                "start": prev_end,
                "end": prev_end + 1.0,  # 1 second placeholder duration
            }
        else:
            return {
                "text": text,
                "start": 0.0,
                "end": 1.0,  # 1 second placeholder duration
            }

    def _merge_short_cues(
        self, cues: list[dict[str, Any]], min_duration: Optional[float] = None
    ) -> list[dict[str, Any]]:
        """
        Merge subtitle cues that are too short and split cues that are too long.

        Args:
            cues: list of subtitle cues
            min_duration: Minimum duration in seconds (uses default if None)

        Returns:
            list of merged and optimized cues
        """
        if not cues:
            return []

        if min_duration is None:
            min_duration = self.min_cue_duration

        merged_cues = []
        current_cue = None

        for cue in cues:
            if current_cue is None:
                current_cue = cue.copy()  # Create a copy to avoid modifying original
            else:
                duration = cue["end"] - current_cue["start"]

                # Check if we should merge with current cue
                if duration < min_duration:
                    # Merge with current cue
                    current_cue["text"] += " " + cue["text"]
                    current_cue["end"] = cue["end"]

                    # Check if merged cue is now too long
                    merged_duration = current_cue["end"] - current_cue["start"]
                    if merged_duration > self.max_cue_duration:
                        # Split the merged cue
                        split_cues = self._split_long_cue(current_cue)
                        merged_cues.extend(split_cues[:-1])  # Add all but the last
                        current_cue = split_cues[-1]  # Keep the last one as current
                else:
                    # Current cue is long enough, add it and start new one
                    merged_cues.append(current_cue)
                    current_cue = cue.copy()

        if current_cue:
            # Check if final cue is too long
            final_duration = current_cue["end"] - current_cue["start"]
            if final_duration > self.max_cue_duration:
                merged_cues.extend(self._split_long_cue(current_cue))
            else:
                merged_cues.append(current_cue)

        return merged_cues

    def _split_long_cue(self, cue: dict[str, Any]) -> list[dict[str, Any]]:
        """Split a cue that is too long into multiple shorter cues."""
        duration = cue["end"] - cue["start"]
        if duration <= self.max_cue_duration:
            return [cue]

        # Calculate how many splits we need
        num_splits = int(duration / self.max_cue_duration) + 1
        split_duration = duration / num_splits

        # Split the text into roughly equal parts
        words = cue["text"].split()
        words_per_split = len(words) // num_splits

        split_cues = []
        current_start = cue["start"]

        for i in range(num_splits):
            if i == num_splits - 1:  # Last split gets remaining words
                split_words = words[i * words_per_split :]
            else:
                split_words = words[i * words_per_split : (i + 1) * words_per_split]

            if split_words:  # Only create cue if there are words
                split_cues.append(
                    {
                        "text": " ".join(split_words),
                        "start": current_start,
                        "end": current_start + split_duration,
                    }
                )
                current_start += split_duration

        return split_cues

    def write_subtitles(
        self,
        punct_model: Any,
        aligned_result: dict[str, Any],
        video_name: str,
        output_dir: Optional[Union[str, Path]] = None,
        output_format: Optional[str] = None,
    ) -> str:
        """
        Write transcription results to subtitle file in specified format.

        Args:
            punct_model: Punctuation restoration model
            aligned_result: Aligned transcription results
            video_name: Name of the video file (without extension)
            output_dir: Output directory path
            output_format: Output format ("srt", "vtt", "txt")

        Returns:
            Path to the generated subtitle file

        Raises:
            ValueError: If output format is not supported
            OSError: If file writing fails
        """
        # Determine output format
        if output_format is None:
            output_format = self.config.output_format if self.config else "srt"

        if output_format not in ["srt", "vtt", "txt"]:
            raise ValueError(f"Unsupported output format: {output_format}")

        # Set up output directory
        if output_dir is None:
            output_dir = video_name

        output_dir = Path(output_dir)

        # Create video-specific subdirectory
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        output_path = video_output_dir / f"{video_name}.{output_format}"

        logger.info(f"Generating {output_format.upper()} subtitles for {video_name}")

        try:
            # Process segments and create cues
            all_cues = self._process_segments(punct_model, aligned_result)

            # Merge and optimize cues
            optimized_cues = self._merge_short_cues(all_cues)

            # Write to file based on format
            if output_format == "srt":
                self._write_srt_file(output_path, optimized_cues)
            elif output_format == "vtt":
                self._write_vtt_file(output_path, optimized_cues)
            elif output_format == "txt":
                self._write_txt_file(output_path, optimized_cues)

            logger.info(f"✅ Subtitles saved to: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"❌ Failed to write subtitles: {e}")
            raise

    def _process_segments(
        self, punct_model: Any, aligned_result: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process transcription segments and create subtitle cues."""
        all_cues = []

        for segment in aligned_result.get("segments", []):
            if not segment.get("text"):
                continue

            # Restore punctuation
            text = punct_model.restore_punctuation(segment["text"])
            word_data = segment.get("words", [])

            # Split into sentences with timing
            sentences = self._split_at_sentence_end(text, word_data)
            all_cues.extend(sentences)

        return all_cues

    def _write_srt_file(self, output_path: Path, cues: list[dict[str, Any]]) -> None:
        """Write cues to SRT format file."""
        with open(output_path, "w", encoding="utf-8") as srt_file:
            for i, cue in enumerate(cues, 1):
                formatted_text = self._split_subtitle(cue["text"])

                srt_file.write(f"{i}\n")
                srt_file.write(
                    f"{self._format_timestamp(cue['start'], 'srt')} --> "
                    f"{self._format_timestamp(cue['end'], 'srt')}\n"
                )
                srt_file.write(f"{formatted_text}\n\n")

    def _write_vtt_file(self, output_path: Path, cues: list[dict[str, Any]]) -> None:
        """Write cues to VTT format file."""
        with open(output_path, "w", encoding="utf-8") as vtt_file:
            vtt_file.write("WEBVTT\n\n")

            for cue in cues:
                formatted_text = self._split_subtitle(cue["text"])

                vtt_file.write(
                    f"{self._format_timestamp(cue['start'], 'vtt')} --> "
                    f"{self._format_timestamp(cue['end'], 'vtt')}\n"
                )
                vtt_file.write(f"{formatted_text}\n\n")

    def _write_txt_file(self, output_path: Path, cues: list[dict[str, Any]]) -> None:
        """Write cues to plain text file."""
        with open(output_path, "w", encoding="utf-8") as txt_file:
            for cue in cues:
                txt_file.write(f"{cue['text']}\n")

    # Backward compatibility method
    def write_to_srt(
        self,
        punct_model: Any,
        aligned_result: dict[str, Any],
        video_name: str,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Write transcription results to SRT subtitle file (backward compatibility).

        This method is kept for backward compatibility with existing code.
        New code should use write_subtitles() instead.
        """
        return self.write_subtitles(
            punct_model, aligned_result, video_name, output_dir, "srt"
        )
