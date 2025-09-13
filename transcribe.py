import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import whisperx
from deepmultilingualpunctuation import PunctuationModel

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


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="🎥 Video Transcribe AI - Transcribe MP4 videos to SRT subtitles using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Process all MP4 files in current directory
  %(prog)s -i video.mp4                       # Process specific video file
  %(prog)s -i ./videos/ -m large             # Process all videos in directory with large model
  %(prog)s -i video.mp4 -l auto -o ./output/ # Auto-detect language, save to output directory
  %(prog)s -i video.mp4 --no-translate       # Transcribe without translation
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
    language_group.add_argument(
        '--no-translate',
        action='store_true',
        help='Transcribe only (no translation to English)'
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
    processing_group.add_argument(
        '--resume',
        action='store_true',
        help='Skip files that already have subtitle files'
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


def should_skip_file(video_file: Path, output_dir: Path, resume: bool) -> bool:
    """Check if file should be skipped based on resume option."""
    if not resume:
        return False
    
    subtitle_file = output_dir / video_file.stem / f"{video_file.stem}.srt"
    return subtitle_file.exists()


def process_video(
    video_file: Path,
    model,
    model_a,
    metadata,
    punct_model,
    output_dir: Path,
    language: str,
    translate: bool,
    keep_audio: bool,
    device: str
) -> bool:
    """Process a single video file."""
    logger.info(f"Processing: {video_file.name}")
    
    try:
        # Extract audio from video
        logger.debug("Extracting audio...")
        audio_file = extract_audio(str(video_file))
        
        # Determine transcription task
        task = "translate" if translate else "transcribe"
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
        if not keep_audio:
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
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine device
    if args.device == 'auto':
        device = DEVICE
    else:
        device = args.device
    
    # Get video files
    video_files = get_video_files(args.input)
    
    if not video_files:
        logger.error("No video files found!")
        return 1
    
    # Set up output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if Path(args.input).is_file():
            output_dir = Path(args.input).parent
        else:
            output_dir = Path(args.input)
    
    # Filter files based on resume option
    if args.resume:
        original_count = len(video_files)
        video_files = [f for f in video_files if not should_skip_file(f, output_dir, args.resume)]
        skipped_count = original_count - len(video_files)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} files that already have subtitles")
    
    if not video_files:
        logger.info("No files to process!")
        return 0
    
    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN - Files that would be processed:")
        for video_file in video_files:
            logger.info(f"  - {video_file}")
        return 0
    
    # Load models
    logger.info("Loading models...")
    try:
        model = get_model(size=args.model, device=device)
        model_a, metadata = load_alignment_model(language_code="en", device=device)
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
            args.language,
            not args.no_translate,
            args.keep_audio,
            device
        ):
            successful += 1
        else:
            failed += 1
    
    # Summary
    logger.info(f"Processing complete! ✅ {successful} successful, ❌ {failed} failed")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())