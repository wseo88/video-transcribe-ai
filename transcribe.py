import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Literal
import whisperx
from deepmultilingualpunctuation import PunctuationModel
from pydantic import BaseModel, Field, field_validator

# Import our custom modules
from audio_processor import extract_audio
from model_manager import get_model, load_alignment_model, DEVICE
from subtitle_formatter import write_to_srt


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TranscribeConfig(BaseModel):
    """Pydantic model of video transcription arguments"""

    # input options
    input: str = Field(default='.', description='Input video file or directory containing MP4 files')

    # model options
    model: Literal['tiny', 'base','small', 'medium', 'large'] = Field(default="medium", description="Whisper model size")
    device: Literal['cuda', 'cpu', 'auto'] = Field(default='auto', description='Device to use for processing')

    # Language options
    language: str = Field(default='auto', description='Language code or "auto" for detection')
    #no_translate: bool = Field(default=False, description='Transcribe only (no translation to English)')

    # Output options
    output: Optional[str] = Field(default=None, description='Output directory for subtitle files')
    output_format: Literal['srt', 'vtt', 'txt'] = Field(default='srt', description='Output subtitle format')
    #keep_audio: bool = Field(default=False, description='Keep extracted audio files')
    
    # Processing options
    batch_size: int = Field(default=1, ge=1, description='Number of files to process simultaneously')
    #resume: bool = Field(default=False, description='Skip files that already have subtitle files')

    # Utility options
    verbose: bool = Field(default=False, description='Enable verbose logging')
    dry_run: bool = Field(default=False, description='Show what would be processed without actually processing')
    
    
    @field_validator('input')
    @classmethod
    def validate_input_path(cls, value):
        """Validate input path exists."""
        path = Path(value)
        if not path.exists():
            raise ValueError(f'Input path does not exist: {value}')
        return value

    @field_validator('language')
    @classmethod
    def validate_language(cls, value):
        """Validate language code."""
        if value != 'auto' and len(value) != 2:
            raise ValueError('Language code must be 2 characters (e.g., "en", "es") or "auto"')
        return value
    
    @property
    def effective_device(self) -> str:
        """Get the effective device to use."""
        if self.device == 'auto':
            return DEVICE
        return self.device
    
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




def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="🎥 Video Transcribe AI - Transcribe MP4 videos to SRT subtitles using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
%(prog)s                                   # Process all MP4 files in current directory
%(prog)s -i video.mp4                      # Process specific video file
%(prog)s -i ./videos/ -m large             # Process all videos in directory with large model
%(prog)s -i video.mp4 -l auto -o ./output/ # Auto-detect language, save to output directory
%(prog)s -i video.mp4 -m tiny --verbose    # Use tiny model with verbose logging
        """
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '-i', '--input',
        type=str,
        default='.',
        help='Input video file or directory containing MP4 files (default: current directory)'
    )
    
    # Model options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument(
        '-m', '--model',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default='medium',
        help='Whisper model size (default: medium)'
    )
    model_group.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to use for processing (default: auto)'
    )
    
    # Language options
    language_group = parser.add_argument_group('Language Options')
    language_group.add_argument(
        '-l', '--language',
        type=str,
        default='auto',
        help='Language code (e.g., en, es, fr) or "auto" for detection (default: auto)'
    )

    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '-o', '--output',
        type=str,
        help='Output directory for subtitle files (default: same as input)'
    )
    output_group.add_argument(
        '--output-format',
        choices=['srt', 'vtt', 'txt'],
        default='srt',
        help='Output subtitle format (default: srt)'
    )
    output_group.add_argument(
        '--keep-audio',
        action='store_true',
        help='Keep extracted audio files (default: delete after processing)'
    )
    
    # Processing options
    processing_group = parser.add_argument_group('Processing Options')
    processing_group.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Number of files to process simultaneously (default: 1)'
    )
    
    # Utility options
    utility_group = parser.add_argument_group('Utility Options')
    utility_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    utility_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    utility_group.add_argument(
        '--version',
        action='version',
        version='Video Transcribe AI 1.0.0'
    )
    
    return parser


def parse_args_to_config(args: argparse.Namespace) -> TranscribeConfig:
    """Convert argparse Namespace to TranscribeConfig."""
    try:
        # Convert args to dict, handling the no_translate flag
        config_dict = vars(args)
        return TranscribeConfig(**config_dict)
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        sys.exit(1)


def get_video_files(input_path: str) -> List[Path]:
    """Get list of video files from input path."""
    input_path = Path(input_path)
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']:
            return [input_path]
        else:
            logger.error(f"Unsupported file format: {input_path.suffix}")
            return []
    elif input_path.is_dir():
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.glob(ext))
        return sorted(video_files)
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return []


def process_video(
    video_file: Path,
    model,
    model_a,
    metadata,
    punct_model,
    output_dir: Path,
    language: str,

    device: str
) -> bool:
    """Process a single video file."""
    logger.info(f"Processing: {video_file.name}")
    
    try:
        # Extract audio from video
        logger.debug("Extracting audio...")
        audio_file = extract_audio(str(video_file))
        
        # Determine transcription task
        task = "transcribe"
        transcribe_kwargs = {"task": task}
        
        if language != "auto":
            transcribe_kwargs["language"] = language
        
        # Transcribe audio
        logger.debug("Transcribing audio...")
        result = model.transcribe(audio_file, **transcribe_kwargs)
        
        # Align transcription
        logger.debug("Aligning transcription...")
        aligned_result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio_file, 
            device, 
            return_char_alignments=True
        )
        
        # Generate subtitles
        logger.debug("Generating subtitles...")
        output_path = write_to_srt(punct_model, aligned_result, video_file.stem, output_dir)
        logger.info(f"✅ Subtitles saved to: {output_path}")
        
        # Clean up temporary audio file
        os.remove(audio_file)
        logger.debug("Cleaned up temporary audio file")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing {video_file.name}: {str(e)}")
        return False


def main():
    """Main function with CLI interface."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Convert to Pydantic config with validation
    config = parse_args_to_config(args)
    
    # Set logging level
    if config.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get video files
    video_files = get_video_files(config.input)
    
    if not video_files:
        logger.error("No video files found!")
        return 1
    
    # Get effective output directory (handled by Pydantic model)
    output_dir = config.effective_output_dir
    
    # Dry run mode
    if config.dry_run:
        logger.info("DRY RUN - Files that would be processed:")
        for video_file in video_files:
            logger.info(f"  - {video_file}")
        return 0
    
    # Load models
    logger.info("Loading models...")
    try:
        model = get_model(size=config.model, device=config.effective_device)
        model_a, metadata = load_alignment_model(language_code="en", device=config.effective_device)
        punct_model = PunctuationModel()
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return 1
    
    # Process files
    successful = 0
    failed = 0
    
    logger.info(f"Processing {len(video_files)} file(s)...")
    
    for i, video_file in enumerate(video_files, 1):
        logger.info(f"[{i}/{len(video_files)}] Processing {video_file.name}")
        
        if process_video(
            video_file,
            model,
            model_a,
            metadata,
            punct_model,
            output_dir,
            config.language,
            config.effective_device
        ):
            successful += 1
        else:
            failed += 1
    
    # Summary
    logger.info(f"Processing complete! ✅ {successful} successful, ❌ {failed} failed")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())