from pathlib import Path
from typing import (
    Literal,
    Optional,
)

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


# Constants for better type safety and maintainability
class ModelSize:
    """Supported Whisper model sizes."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

    ALL = [TINY, BASE, SMALL, MEDIUM, LARGE]


class Device:
    """Supported processing devices."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"

    ALL = [AUTO, CPU, CUDA]


class OutputFormat:
    """Supported output subtitle formats."""

    SRT = "srt"
    VTT = "vtt"
    TXT = "txt"

    ALL = [SRT, VTT, TXT]


class TranscribeConfig(BaseModel):
    """
    Configuration model for video transcription using OpenAI Whisper.

    This model defines all the parameters needed for transcribing video files,
    including input/output paths, model settings, language options, and processing preferences.

    Examples:
        >>> # Basic usage with defaults
        >>> config = TranscribeConfig(input="video.mp4")

        >>> # Advanced usage with custom settings
        >>> config = TranscribeConfig(
        ...     input="./videos/",
        ...     model_size="large",
        ...     device="cuda",
        ...     language="en",
        ...     output="./subtitles/",
        ...     output_format="srt",
        ...     verbose=True
        ... )
    """

    # Input options
    input: str = Field(
        default=".",
        description="Input video file or directory containing MP4 files. "
        "If a directory is provided, all MP4 files will be processed.",
        examples=["video.mp4", "./videos/", "/path/to/video.mp4"],
    )

    # Model options
    model_size: Literal["tiny", "base", "small", "medium", "large"] = Field(
        default="medium",
        description="Whisper model size. Larger models are more accurate but slower. "
        "Options: tiny (fastest, least accurate) to large (slowest, most accurate).",
        examples=["tiny", "base", "small", "medium", "large"],
    )
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Device to use for processing. 'auto' will automatically detect "
        "the best available device (CUDA if available, otherwise CPU).",
        examples=["auto", "cpu", "cuda"],
    )

    # Language options
    language: str = Field(
        default="auto",
        description='Language code for transcription (e.g., "en", "es", "fr") or "auto" '
        "for automatic language detection. Use ISO 639-1 language codes.",
        examples=["auto", "en", "es", "fr", "de", "ja", "ko"],
    )

    # Output options
    output: Optional[str] = Field(
        default=None,
        description="Output directory for subtitle files. If not specified, "
        "subtitles will be saved in the same directory as the input file(s).",
        examples=[None, "./subtitles/", "/path/to/output/"],
    )
    output_format: Literal["srt", "vtt", "txt"] = Field(
        default="srt",
        description="Output subtitle format. SRT is the most widely supported format.",
        examples=["srt", "vtt", "txt"],
    )

    # Utility options
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging output for debugging and monitoring progress.",
    )

    @field_validator("input")
    @classmethod
    def validate_input_path(cls, value: str) -> str:
        """
        Validate that the input path exists and is accessible.

        Args:
            value: The input path string

        Returns:
            The validated input path

        Raises:
            ValueError: If the path doesn't exist or is not accessible
        """
        if not value or not value.strip():
            raise ValueError("Input path cannot be empty")

        path = Path(value.strip())

        if not path.exists():
            raise ValueError(
                f"Input path does not exist: {value}. "
                f"Please check the path and try again."
            )

        if not path.is_file() and not path.is_dir():
            raise ValueError(
                f"Input path must be a file or directory: {value}. "
                f"Found: {path.stat().st_mode if path.exists() else 'does not exist'}"
            )

        return str(path.resolve())

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        """
        Validate language code format and supported languages.

        Args:
            value: The language code string

        Returns:
            The validated language code

        Raises:
            ValueError: If the language code format is invalid
        """
        if not value or not value.strip():
            raise ValueError("Language code cannot be empty")

        value = value.strip().lower()

        if value == "auto":
            return value

        if len(value) != 2:
            raise ValueError(
                f'Language code must be 2 characters (e.g., "en", "es") or "auto". '
                f'Got: "{value}" (length: {len(value)})'
            )

        # Check if it's a valid ISO 639-1 language code format (letters only)
        if not value.isalpha():
            raise ValueError(f'Language code must contain only letters. Got: "{value}"')

        return value

    @field_validator("output")
    @classmethod
    def validate_output_path(cls, value: Optional[str]) -> Optional[str]:
        """
        Validate output directory path.

        Args:
            value: The output path string or None

        Returns:
            The validated output path or None

        Raises:
            ValueError: If the output path is invalid
        """
        if value is None:
            return None

        if not value.strip():
            raise ValueError("Output path cannot be empty when specified")

        path = Path(value.strip())

        # Check if parent directory exists
        if not path.parent.exists():
            raise ValueError(
                f"Parent directory of output path does not exist: {path.parent}. "
                f"Please create the directory or choose a different path."
            )

        return str(path.resolve())

    @model_validator(mode="after")
    def validate_model_device_compatibility(self) -> "TranscribeConfig":
        """
        Validate model and device compatibility.

        Returns:
            The validated config instance

        Raises:
            ValueError: If model and device combination is invalid
        """
        # Add any cross-field validation logic here
        # For example, checking if CUDA is available when device is set to "cuda"
        return self

    @property
    def effective_output_dir(self) -> Path:
        """
        Get the effective output directory for subtitle files.

        If an output directory is specified, it will be created if it doesn't exist.
        Otherwise, it returns the directory containing the input file(s).

        Returns:
            Path: The directory where subtitle files will be saved
        """
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

    @property
    def input_path(self) -> Path:
        """
        Get the input path as a Path object.

        Returns:
            Path: The resolved input path
        """
        return Path(self.input).resolve()

    @property
    def is_directory_input(self) -> bool:
        """
        Check if the input is a directory.

        Returns:
            bool: True if input is a directory, False if it's a file
        """
        return self.input_path.is_dir()

    @property
    def is_file_input(self) -> bool:
        """
        Check if the input is a single file.

        Returns:
            bool: True if input is a file, False if it's a directory
        """
        return self.input_path.is_file()

    def get_output_filename(self, input_filename: str) -> str:
        """
        Generate output filename for a given input filename.

        Args:
            input_filename: The name of the input file

        Returns:
            str: The output filename with the correct extension
        """
        input_path = Path(input_filename)
        base_name = input_path.stem
        return f"{base_name}.{self.output_format}"

    def __str__(self) -> str:
        """Return a string representation of the configuration."""
        return (
            f"TranscribeConfig("
            f"input='{self.input}', "
            f"model_size='{self.model_size}', "
            f"device='{self.device}', "
            f"language='{self.language}', "
            f"output_format='{self.output_format}', "
            f"verbose={self.verbose}"
            f")"
        )

    def __repr__(self) -> str:
        """Return a detailed string representation of the configuration."""
        return self.__str__()
