import os
from pathlib import Path

# Directory Configuration
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "INPUT"
OUTPUT_DIR = BASE_DIR / "OUTPUT"
PROCESSED_DIR = INPUT_DIR / "processed"

# Whisper Configuration
WHISPER_MODEL = "large"

# Polling Configuration
POLL_INTERVAL = 10  # seconds

# Supported Audio Formats
SUPPORTED_FORMATS = {'.mp3', '.m4a', '.wav', '.flac', '.ogg', '.aac', '.wma', '.mp4'}

# Email Configuration
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': '',  # Set this to your Gmail address
    'sender_password': '',  # Set this to your Gmail app password
    'recipient_email': 'colehmcconnell@gmail.com',
    'subject': 'Voice Memo Transcription Error'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'filename': BASE_DIR / 'transcription.log'
}