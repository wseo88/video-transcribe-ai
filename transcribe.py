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



from cli.parser import (
    setup_argument_parser,
    parse_args_to_config,
)
from core.logging import (
    setup_logging,
    get_logger,
)


def main():
    """Main function with CLI interface."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Convert to Pydantic config with validation
    config = parse_args_to_config(args)
    
    # Set up centralized logging
    setup_logging(config=config)
    logger = get_logger(__name__)

    
    # Get video files
    video_files = get_video_files(config.input)
    
    if not video_files:
        logger.error("No video files found!")
        return 1
    
    # Get effective output directory (handled by Pydantic model)
    output_dir = config.effective_output_dir
    
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