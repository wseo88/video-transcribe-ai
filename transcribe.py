import signal
import sys
import time
from pathlib import Path

from cli.parser import (
    parse_args_to_config,
    setup_argument_parser,
)
from core.logging import get_logger, setup_logging
from services.transcription_service import TranscriptionService

# Global flag for graceful shutdown
shutdown_requested = False
logger = get_logger(__name__)


def signal_handler(signum: int, frame) -> None:
    """Handle interrupt signals for graceful shutdown."""
    global shutdown_requested
    logger.info("🛑 Shutdown signal received. Finishing current operation...")
    shutdown_requested = True


def validate_environment() -> bool:
    """Validate that the environment is ready for transcription."""
    try:
        # Check if required directories exist or can be created
        current_dir = Path.cwd()
        if not current_dir.exists():
            logger.error(f"❌ Current directory does not exist: {current_dir}")
            return False

        # Check write permissions
        test_file = current_dir / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            logger.error(f"❌ No write permission in current directory: {current_dir}")
            return False

        return True
    except Exception as e:
        logger.error(f"❌ Environment validation failed: {e}")
        return False


def main() -> int:
    """
    Main function with CLI interface.

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    start_time = time.time()
    exit_code = 0

    try:
        # Parse command line arguments
        parser = setup_argument_parser()
        args = parser.parse_args()

        # Convert to Pydantic config with validation
        config = parse_args_to_config(args)

        # Set up centralized logging
        setup_logging(is_verbose=config.verbose)

        logger.info("🎥 Video Transcribe AI - Starting transcription process")
        logger.info(f"📁 Input: {config.input}")
        logger.info(f"🔧 Model: {config.model_size}, Device: {config.device}")
        logger.info(f"🌐 Language: {config.language}")

        # Validate environment before starting
        if not validate_environment():
            logger.error("❌ Environment validation failed")
            return 1

        # Create and run transcription service
        transcription_service = TranscriptionService(config=config)
        result = transcription_service.transcribe()

        # Handle transcription result
        if result is None:
            logger.error("❌ Transcription service returned None")
            exit_code = 1
        elif isinstance(result, int):
            exit_code = result
        else:
            logger.warning(
                f"⚠️ Unexpected return type from transcription service: {type(result)}"
            )
            exit_code = 1

    except KeyboardInterrupt:
        logger.info("🛑 Operation cancelled by user")
        exit_code = 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        exit_code = 1
    finally:
        # Calculate and log execution time
        execution_time = time.time() - start_time
        if exit_code == 0:
            logger.info(
                f"✅ Transcription completed successfully in {execution_time:.2f} seconds"
            )
        else:
            logger.error(f"❌ Transcription failed after {execution_time:.2f} seconds")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
