from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory Configuration
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "INPUT"
OUTPUT_DIR = BASE_DIR / "OUTPUT"
PROCESSED_DIR = INPUT_DIR / "processed"

# Whisper Configuration
# WHISPER_MODEL = "tiny"
# WHISPER_MODEL = "base"
# WHISPER_MODEL = "small"
WHISPER_MODEL = "medium"
# WHISPER_MODEL = "large"
# WHISPER_MODEL = "large-v1"
# WHISPER_MODEL = "large-v2"
# WHISPER_MODEL = "large-v3"

# SSL Configuration (for networks with certificate issues)
SSL_VERIFY = False  # Set to False if you have SSL certificate issues

# Polling Configuration
POLL_INTERVAL = 10  # seconds

# Supported Audio Formats
SUPPORTED_FORMATS = {'.mp3', '.m4a', '.wav', '.flac', '.ogg', '.aac', '.wma', '.mp4', '.mov'}

# Video Processing Configuration
VIDEO_FORMATS = {'.mp4', '.mov'}
AUDIO_EXTRACTION_CONFIG = {
    'output_format': 'mp3',  # Format for extracted audio
    'audio_quality': '192k',  # Audio bitrate for extraction
    'temp_dir': BASE_DIR / "temp_audio"  # Temporary directory for extracted audio
}

# Email Configuration
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': os.getenv('GMAIL_ADDRESS', ''),
    'sender_password': os.getenv('GMAIL_APP_PASSWORD', ''),
    'recipient_email': os.getenv('RECIPIENT_EMAIL', 'colehmcconnell@gmail.com'),
    'subject': 'Voice Memo Transcription Error'
}

# Progress Configuration
PROGRESS_CONFIG = {
    'enabled': True,  # Enable progress logging during transcription
    'update_interval': 10,  # Log progress every N percent (1-100)
    'show_time_estimate': True  # Show estimated time remaining
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'filename': BASE_DIR / 'transcription.log'
}