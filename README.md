# 🎥 Video Transcribe AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![WhisperX](https://img.shields.io/badge/Powered%20by-WhisperX-orange.svg)](https://github.com/m-bain/whisperX)

A powerful command-line tool that automatically transcribes audio from MP4 video files and generates professional-quality SRT subtitle files using OpenAI's Whisper and advanced post-processing techniques.

## ✨ Features

### 🎯 Core Functionality
- **Automatic Transcription**: Extract and transcribe audio from MP4 videos using OpenAI Whisper
- **Professional Subtitles**: Generate high-quality `.srt` (SubRip Subtitle) files
- **Batch Processing**: Process multiple video files in a single run
- **Smart Text Processing**: Automatic punctuation restoration and sentence boundary detection

### 🚀 Advanced Features
- **Word-Level Alignment**: Precise timing alignment using WhisperX
- **Intelligent Cue Merging**: Automatically merges short subtitle segments for better readability
- **Multi-Language Support**: Automatic language detection with translation capabilities
- **GPU Acceleration**: CUDA support for faster processing (when available)
- **Flexible Model Selection**: Support for all Whisper model sizes (`tiny`, `base`, `small`, `medium`, `large`)

### 🎨 Output Quality
- **Professional Formatting**: Properly formatted timestamps and line breaks
- **Sentence-Aware Splitting**: Intelligent text segmentation at natural sentence boundaries
- **Optimized Readability**: Automatic subtitle length optimization for better viewing experience

## 📋 Requirements

- **Python**: 3.8 or higher
- **FFmpeg**: For audio extraction from video files
- **CUDA** (optional): For GPU acceleration

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/video-transcribe-ai.git
cd video-transcribe-ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Tool

Simply place your MP4 files in the project directory and run:

```bash
python transcribe.py
```

The tool will automatically:
1. Find all MP4 files in the current directory
2. Extract audio from each video
3. Transcribe using Whisper
4. Generate professional SRT subtitle files
5. Clean up temporary files

## 📁 Project Structure

```
video-transcribe-ai/
├── transcribe.py           # Main application entry point
├── audio_processor.py      # Audio extraction from videos
├── model_manager.py        # Whisper model loading and management
├── subtitle_formatter.py   # SRT file generation and formatting
├── text_processor.py       # Text processing and optimization
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

### Model Selection

The tool uses the `medium` Whisper model by default. You can modify the model size in `model_manager.py`:

```python
def get_model(size="medium", device="cuda", compute_type="int8_float16"):
    # Available sizes: tiny, base, small, medium, large
    return whisperx.load_model(size, device, compute_type=compute_type)
```

### Language Settings

The tool currently supports English transcription with translation. To modify language settings, update the `transcribe.py` file:

```python
# For transcription only (no translation)
result = model.transcribe(audio_file, task="transcribe", language="en")

# For translation to English
result = model.transcribe(audio_file, task="translate")
```

## 📊 Performance

| Model Size | Speed | Accuracy | Memory Usage |
|------------|-------|----------|--------------|
| tiny       | Fastest | Good     | ~1 GB        |
| base       | Fast    | Better   | ~1 GB        |
| small      | Medium  | Good     | ~2 GB        |
| medium     | Slower  | Better   | ~5 GB        |
| large      | Slowest | Best     | ~10 GB       |

## 🎯 Use Cases

- **Content Creators**: Generate subtitles for YouTube videos, tutorials, and presentations
- **Educators**: Create accessible educational content with accurate subtitles
- **Media Production**: Batch process multiple videos for subtitle generation
- **Accessibility**: Make video content accessible to hearing-impaired audiences
- **Localization**: Translate and subtitle content for international audiences

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - The core transcription engine
- [WhisperX](https://github.com/m-bain/whisperX) - Enhanced Whisper with word-level alignment
- [Deep Multilingual Punctuation](https://github.com/oliverguhr/deepmultilingualpunctuation) - Punctuation restoration
- [FFmpeg](https://ffmpeg.org/) - Audio/video processing

## 📞 Support

If you encounter any issues or have questions, please:

1. Check the [Issues](https://github.com/your-username/video-transcribe-ai/issues) page
2. Create a new issue with detailed information about your problem
3. Include your system specifications and error messages

---

⭐ **Star this repository if you find it helpful!**