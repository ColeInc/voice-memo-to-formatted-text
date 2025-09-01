from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory Configuration
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "INPUT"
OUTPUT_DIR = BASE_DIR / "OUTPUT"
PROCESSED_DIR = BASE_DIR / "PROCESSED"

# Whisper Configuration
# WHISPER_MODEL = "tiny"     # Fastest, lower quality
WHISPER_MODEL = "base"       # Good balance of speed and quality (recommended)
# WHISPER_MODEL = "small"    # Higher quality, slower
# WHISPER_MODEL = "medium"   # Even higher quality, much slower
# WHISPER_MODEL = "large"    # Highest quality, very slow
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
    'output_format': 'wav',  # WAV is faster to decode for Whisper
    'sample_rate': 16000,    # 16kHz is optimal for Whisper (reduces file size)
    'audio_channels': 1,     # Mono audio (sufficient for speech)
    'audio_quality': '128k', # Lower bitrate for faster processing
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

# Performance Configuration
PERFORMANCE_CONFIG = {
    'max_workers': 2,  # Number of parallel transcription workers (adjust based on GPU memory)
    'enable_parallel': True,  # Enable parallel processing of multiple files
    'memory_cleanup': True,  # Enable aggressive memory cleanup between files
}

# Gemini API Configuration
GEMINI_CONFIG = {
    'api_key': os.getenv('GEMINI_API_KEY', ''),
    'model': 'gemini-1.5-flash',  # Fast and cost-effective model
    'enabled': bool(os.getenv('GEMINI_API_KEY', '')),  # Auto-enable if API key is present
    'prompt': """take the following text which is a series of notes collected by me. i want you to separate it into respective sections and provide meaningful titles for each section you deem to be on a different topic/key point. E.g. there may be numerous snippets of advice from a podcast, and i want you to take the existing text, leave it exactly as it is, but insert headings at the places you deem necessary. Additionally, create a relevant overall title for the entire text content that summarizes the main theme or topic. Your response should start with a main title using # (H1) heading, followed by the formatted content with section headings. Give me back nicely formatted markdown in your response""",
    'timeout': 30,  # seconds
    'max_retries': 3
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'filename': BASE_DIR / 'transcription.log'
}