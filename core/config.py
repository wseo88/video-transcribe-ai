from pathlib import Path
from typing import (
    Literal,
    Optional,
)

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class TranscribeConfigModel(BaseModel):
    """Pydantic model of video transcription arguments"""

    # input options
    input: str = Field(
        default=".", description="Input video file or directory containing MP4 files"
    )

    # model options
    model_size: Literal["tiny", "base", "small", "medium", "large"] = Field(
        default="medium", description="Whisper model size"
    )
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto", description="Device to use for processing"
    )

    # Language options
    language: str = Field(
        default="auto", description='Language code or "auto" for detection'
    )
    # no_translate: bool = Field(default=False, description='Transcribe only (no translation to English)')

    # Output options
    output: Optional[str] = Field(
        default=None, description="Output directory for subtitle files"
    )
    output_format: Literal["srt", "vtt", "txt"] = Field(
        default="srt", description="Output subtitle format"
    )
    # keep_audio: bool = Field(default=False, description='Keep extracted audio files')

    # Processing options
    # batch_size: int = Field(default=1, ge=1, description='Number of files to process simultaneously')
    # resume: bool = Field(default=False, description='Skip files that already have subtitle files')

    # Utility options
    verbose: bool = Field(default=False, description="Enable verbose logging")
    # dry_run: bool = Field(default=False, description='Show what would be processed without actually processing')

    @field_validator("input")
    @classmethod
    def validate_input_path(cls, value):
        """Validate input path exists."""
        path = Path(value)
        if not path.exists():
            raise ValueError(f"Input path does not exist: {value}")
        return value

    @field_validator("language")
    @classmethod
    def validate_language(cls, value):
        """Validate language code."""
        if value != "auto" and len(value) != 2:
            raise ValueError(
                'Language code must be 2 characters (e.g., "en", "es") or "auto"'
            )
        return value

    @property
    def effective_output_dir(self) -> Path:
        """Get the effective output directory."""
        if self.output:
            output_dir = Path(self.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir
        else:
            input_path = Path(self.input)
            if input_path.is_file():
                return input_path.parent
            else:
                return input_path
