import os
import whisperx
from pathlib import Path
from deepmultilingualpunctuation import PunctuationModel

# Import our custom modules
from audio_processor import extract_audio
from model_manager import get_model, load_alignment_model, DEVICE
from subtitle_formatter import write_to_srt

def main():
    """Main function to process all MP4 files in the current directory."""
    # Get all mp4 files
    current_dir = Path(".")
    mp4_files = list(current_dir.glob("*.mp4"))
    
    if not mp4_files:
        print("No MP4 files found in the current directory.")
        return
    
    # Load models once for efficiency
    print("Loading models...")
    model = get_model()
    model_a, metadata = load_alignment_model(language_code="en", device=DEVICE)
    punct_model = PunctuationModel()
    print("Models loaded successfully.")
    
    # Process each file
    for mp4_file in mp4_files:
        file_name = mp4_file.name
        print(f"Processing {file_name}...")
        
        try:
            # Extract audio from video
            audio_file = extract_audio(file_name)
            
            # Transcribe audio
            result = model.transcribe(audio_file, task="translate")
            
            # Align transcription
            aligned_result = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio_file, 
                DEVICE, 
                return_char_alignments=True
            )
            
            # Generate subtitles
            output_path = write_to_srt(punct_model, aligned_result, Path(file_name).stem)
            print(f"✅ Subtitles saved to: {output_path}")
            
            # Clean up temporary audio file
            os.remove(audio_file)
            
        except Exception as e:
            print(f"❌ Error processing {file_name}: {str(e)}")
            continue
    
    print("Processing complete!")


if __name__ == "__main__":
    main()