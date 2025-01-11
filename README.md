# AskAnAI

## Overview
AskAnAI is a tool that records, transcribes, and queries an AI. This project is designed to leverage AI capabilities to process and interact with audio input.

## Key Features

### Audio Processing
- **Advanced Voice Isolation**: Uses Demucs technology to separate vocals from background noise
- **Multi-Speaker Detection**: Automatically detects and transcribes conversations with multiple speakers
- **Audio Cleanup Pipeline**:
  - Spectral subtraction for noise reduction
  - Bandpass filtering optimized for human voice frequencies
  - Intelligent silence trimming

### Speech Recognition & Processing
- **High-Accuracy Transcription**: Powered by OpenAI's Whisper model
- **Speaker Diarization**: Automatically identifies and separates different speakers in conversations
- **Grammar Correction**: AI automatically corrects grammatical errors in transcribed questions

### AI Interaction
- **Local AI Processing**: Uses Ollama with LLaMA 3.1 for fast, private responses
- **Context-Aware Responses**: Handles both single questions and multi-speaker discussions
- **Real-Time Processing**: Immediate transcription and response generation

### User Interface
- **Clean Qt-Based GUI**: User-friendly interface
- **Multi-Screen Design**: Separate screens for recording, processing, and results
- **Keyboard Shortcuts**: Easy recording toggle with Enter key
- **Progress Tracking**: Visual feedback during processing

## Prerequisites

### Basic Requirements
- Docker
- Audio input device (microphone)

### Optional Hardware
- NVIDIA or AMD GPU (enhances performance but not required)
  - For GPU optimization, visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for specific installation instructions

## Installation & Setup

### Building the Docker Image
```sh
docker build -t askanai .
```

### Enable X11 Display Access
```sh
xhost +local:
```

### Running the Application
```sh
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/snd askanai
```
Note: Initial loading may take a few moments as the system initializes all components.

## Project Structure
```
AskAnAI/
├── main/
│   ├── main.py           # Main application and GUI
│   └── backend_processing.py  # Audio processing pipeline
├── setup/
│   ├── entrypoint.sh     # Docker entry point
│   └── requirements.txt   # Python dependencies
└── Dockerfile            # Container configuration
```

## Technical Details

### Audio Processing Pipeline
1. Voice Isolation (Demucs)
2. Noise Reduction (Spectral Subtraction)
3. Voice Frequency Optimization (Bandpass Filter)
4. Silence Removal
5. Multi-Speaker Detection & Separation

### AI Processing Pipeline
1. Speech-to-Text Conversion (Whisper)
2. Context Analysis
3. Grammar Correction
4. Response Generation (LLaMA 3.1)
5. Result Storage

## Output
All processed conversations are automatically saved with:
- Cleaned audio files
- Original recordings
- Transcriptions
- AI responses
