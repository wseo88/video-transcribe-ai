## 🎥 Video Transcribe AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![WhisperX](https://img.shields.io/badge/Powered%20by-WhisperX-orange.svg)](https://github.com/m-bain/whisperX)

Video Transcribe AI is a CLI tool that batch-transcribes videos and generates professional subtitles using OpenAI Whisper + WhisperX. It extracts audio, restores punctuation, aligns words, and writes clean `.srt`, `.vtt`, or `.txt` files.

### ✨ Highlights
- Professional subtitles: sentence-aware splitting, line wrapping, readable cue durations
- Word-level alignment via WhisperX
- Punctuation restoration
- Batch processing of files/directories
- CUDA acceleration when available
- Flexible output formats: `srt`, `vtt`, `txt`

## 📦 Requirements
- Python 3.8+
- FFmpeg installed and on PATH
- Optional: CUDA-capable GPU + drivers for faster inference

## 🚀 Quick Start
```bash
git clone https://github.com/your-username/video-transcribe-ai.git
cd video-transcribe-ai

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

# Process all MP4 files in the current directory
python transcribe.py
```

## 🧭 Usage
```bash
python transcribe.py [options]
```

### Options
- --input, -i: Input video file or directory (default: current directory)
- --model-size: tiny | base | small | medium | large (default: medium)
- --device: auto | cpu | cuda (default: auto)
- --language, -l: 2-letter code (e.g., en, es) or "auto" (default: auto)
- --output, -o: Output directory (default: alongside input)
- --output-format: srt | vtt | txt (default: srt)
- --verbose, -v: Enable verbose logging
- --dry-run: Show what would be processed without doing work

### Examples
```bash
# Process all videos in a folder
python transcribe.py -i ./videos

# Process a single file with the large model
python transcribe.py -i myvideo.mp4 --model-size large

# Auto-detect language and write VTT files to a custom folder
python transcribe.py -i ./videos -o ./subs --output-format vtt

# Force CPU and show verbose logs
python transcribe.py -i myvideo.mp4 --device cpu --verbose
```

## 🔊 What it does
1. Scans input for supported video files
2. Extracts audio with FFmpeg
3. Runs Whisper (size configurable)
4. Aligns segments/words with WhisperX
5. Restores punctuation
6. Writes subtitles (`.srt`/`.vtt`) with clean timestamps and wrapping

## 📁 Output
For each input video `X.mp4`, subtitles are written to:
```
<output_dir>/<X>/<X>.<srt|vtt|txt>
```
Example:
```
output/
└─ talk/
   └─ talk.srt
```

## ⚙️ Configuration Details
- Model size and device are resolved in `services/model_service.py`
- Output format, language, and paths are validated via `core/models.py` (`TranscribeConfig`)
- Subtitles are created in `services/subtitle_service.py` (supports SRT, VTT, TXT)

## 🧩 Project Structure (key files)
```
video-transcribe-ai/
├─ transcribe.py                 # CLI entrypoint
├─ cli/parser.py                 # CLI options & examples
├─ core/
│  ├─ models.py                  # TranscribeConfig (Pydantic)
│  └─ logging.py                 # Centralized logging
├─ services/
│  ├─ audio_service.py           # FFmpeg audio extraction
│  ├─ model_service.py           # Whisper/WhisperX model loading
│  ├─ transcription_service.py   # Orchestration
│  ├─ subtitle_service.py        # SRT/VTT/TXT generation
│  └─ video_file_service.py      # File discovery & helpers
└─ requirements.txt
```

## 🧪 Tips & Troubleshooting
- Ensure FFmpeg is installed (`ffmpeg -version` should work)
- If CUDA is requested but unavailable, the app falls back to CPU
- Large models require significant RAM/VRAM; start with `medium` or `small`
- Use `--verbose` for detailed logs

## 🤝 Contributing
PRs are welcome! Please open an issue for feature requests or significant changes.

1. Fork the repo
2. Create a branch: `git checkout -b feat/my-change`
3. Commit: `git commit -m "feat: my change"`
4. Push: `git push origin feat/my-change`
5. Open a PR

## 📝 License
MIT — see [LICENSE](LICENSE).

## 🙏 Acknowledgments
- [OpenAI Whisper](https://github.com/openai/whisper)
- [WhisperX](https://github.com/m-bain/whisperX)
- [Deep Multilingual Punctuation](https://github.com/oliverguhr/deepmultilingualpunctuation)
- [FFmpeg](https://ffmpeg.org/)

---
⭐ If this helped you, consider starring the repo!