# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an automated voice memo transcription system that monitors an input directory for audio/video files, transcribes them using OpenAI Whisper, and outputs formatted transcripts. The system includes email notifications, automatic file management, and optional AI-powered transcript formatting using Google's Gemini API.

## Key Commands

### Running the Service
```bash
# Start transcription service manually
python3 transcribe_audio.py

# Install as auto-startup service (macOS)
./install_service.sh

# Uninstall service (macOS)
./uninstall_service.sh

# Install dependencies
pip install -r requirements.txt
```

### Service Management (macOS)
```bash
# Check service status
launchctl list com.voicememo.transcriber

# View logs
tail -f launchd.log
tail -f launchd_error.log

# Manual start/stop
launchctl load ~/Library/LaunchAgents/com.voicememo.transcriber.plist
launchctl unload ~/Library/LaunchAgents/com.voicememo.transcriber.plist
```

### Linux Setup
See `LINUX_SETUP.md` for systemd service configuration instructions.

## Architecture

### Core Components

**AudioTranscriber Class** (`transcribe_audio.py`):
- Main orchestrator handling the entire transcription pipeline
- Manages device detection (CPU/GPU/Apple Silicon MPS)
- Handles SSL configuration for corporate networks
- Implements progress tracking and error notification

**Configuration System** (`config.py`):
- Centralized configuration using environment variables via `.env` file
- Performance optimization settings
- Email configuration for notifications
- Gemini API configuration for AI formatting

### Processing Pipeline

1. **File Detection**: Monitors `INPUT/` directory for supported formats
2. **Audio Extraction**: Converts video files (mp4/mov) to audio using ffmpeg
3. **Audio Preprocessing**: Normalizes audio for Whisper compatibility (mono, 16kHz, WAV)
4. **Transcription**: Uses OpenAI Whisper with device-optimized settings
5. **AI Formatting**: Optional Gemini API integration for structured markdown output
6. **File Management**: Saves transcripts to `OUTPUT/`, moves originals to `PROCESSED/`
7. **Email Delivery**: Sends formatted transcripts via email

### Performance Optimizations

The system includes comprehensive performance optimizations (see `PERFORMANCE_OPTIMIZATIONS.md`):
- GPU acceleration with automatic device detection
- Optimized Whisper model selection (`base` model for speed/quality balance)
- Parallel processing with ThreadPoolExecutor
- Audio preprocessing for optimal Whisper input
- Memory management with garbage collection

## Directory Structure

```
INPUT/          # Drop audio/video files here for processing
OUTPUT/         # Formatted transcript files (.md)
PROCESSED/      # Processed audio files moved here
temp_audio/     # Temporary audio extraction (auto-cleanup)
```

## Configuration

### Environment Variables (.env file)
```bash
GMAIL_ADDRESS=your_email@gmail.com
GMAIL_APP_PASSWORD=your_app_password
RECIPIENT_EMAIL=recipient@gmail.com
GEMINI_API_KEY=your_gemini_api_key
```

### Key Configuration Options (`config.py`)

**Whisper Model Selection**:
- `tiny`, `base` (recommended), `small`, `medium`, `large`
- Trade-off between speed and accuracy

**Performance Settings**:
- `max_workers`: Parallel processing threads (default: 2)
- `enable_parallel`: Enable concurrent file processing
- `memory_cleanup`: Aggressive memory management

**Audio Processing**:
- Supports: mp3, m4a, wav, flac, ogg, aac, wma, mp4, mov
- Auto-converts to optimal format for Whisper (16kHz mono WAV)

### SSL Configuration
For corporate networks with certificate issues:
- Set `SSL_VERIFY = False` in `config.py`
- System automatically configures urllib and SSL contexts

## Email Integration

Two email types:
1. **Error Notifications**: Sent automatically on processing failures
2. **Transcript Delivery**: Formatted transcripts sent as HTML emails with subject line extracted from AI-generated title

## Gemini AI Integration

Optional AI-powered transcript formatting:
- Automatically adds section headings and structure
- Extracts meaningful titles from content
- Configurable via `GEMINI_CONFIG` in `config.py`
- Falls back gracefully if API unavailable

## Device Compatibility

**Automatic Device Detection**:
- CUDA (NVIDIA GPUs)
- Apple Silicon MPS (M1/M2/M3 Macs)  
- CPU fallback with compatibility handling

**Error Handling**:
- MPS backend compatibility issues auto-fallback to CPU
- SSL certificate problems with helpful error messages
- Memory management for GPU processing

## Logging

Comprehensive logging system:
- File: `transcription.log`
- Console output with real-time progress
- Service logs: `launchd.log` and `launchd_error.log` (macOS)
- Email notifications for critical errors

## Common Development Tasks

When modifying the system:
1. Test with various audio formats in `INPUT/` directory
2. Monitor logs for performance and errors
3. Verify email notifications work correctly
4. Test AI formatting with Gemini API integration
5. Check device detection across different hardware