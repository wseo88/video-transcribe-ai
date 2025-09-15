import sys

from cli.parser import (
    parse_args_to_config,
    setup_argument_parser,
)
from core.logging import setup_logging
from services.transcription_service import TranscriptionService


def main():
    """Main function with CLI interface."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Convert to Pydantic config with validation
    config = parse_args_to_config(args)

    # Set up centralized logging
    setup_logging(is_verbose=config.verbose)

    transcription_service = TranscriptionService(config=config)
    transcription_service.transcribe()


if __name__ == "__main__":
    sys.exit(main())
