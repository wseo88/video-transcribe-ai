class TranscriptionService:

    def __init__(self):
        pass


    def process_video(
        video_file: Path,
        model,
        model_a,
        metadata,
        punct_model,
        output_dir: Path,
        language: str,
        device: str,
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
