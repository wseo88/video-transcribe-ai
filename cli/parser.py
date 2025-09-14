import argparse


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
        '--model-size',
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
    
    return parser