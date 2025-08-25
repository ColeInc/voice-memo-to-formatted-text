#!/usr/bin/env python3
"""
Voice Memo Transcription System

Monitors INPUT directory for audio files, transcribes them using OpenAI Whisper,
and outputs formatted transcripts to OUTPUT directory.
"""

import logging
import smtplib
import time
import traceback
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import List, Optional
import shutil
import sys

try:
    import whisper
except ImportError:
    print("Error: OpenAI Whisper not installed. Please run: pip install -r requirements.txt")
    sys.exit(1)

from config import (
    INPUT_DIR, OUTPUT_DIR, PROCESSED_DIR,
    WHISPER_MODEL, POLL_INTERVAL, SUPPORTED_FORMATS,
    EMAIL_CONFIG, LOGGING_CONFIG
)


class AudioTranscriber:
    def __init__(self):
        self.setup_logging()
        self.setup_directories()
        self.load_whisper_model()
        
    def setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG['level']),
            format=LOGGING_CONFIG['format'],
            handlers=[
                logging.FileHandler(LOGGING_CONFIG['filename']),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Audio Transcriber initialized")
        
    def setup_directories(self):
        """Ensure all required directories exist"""
        directories = [INPUT_DIR, OUTPUT_DIR, PROCESSED_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Directory ready: {directory}")
            
    def load_whisper_model(self):
        """Load the Whisper model"""
        try:
            self.logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
            self.model = whisper.load_model(WHISPER_MODEL)
            self.logger.info("Whisper model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load Whisper model: {str(e)}"
            self.logger.error(error_msg)
            self.send_error_email("Model Loading Error", error_msg)
            raise
            
    def send_error_email(self, error_type: str, error_details: str):
        """Send error notification email"""
        if not EMAIL_CONFIG['sender_email'] or not EMAIL_CONFIG['sender_password']:
            self.logger.warning("Email credentials not configured, skipping email notification")
            return
            
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_CONFIG['sender_email']
            msg['To'] = EMAIL_CONFIG['recipient_email']
            msg['Subject'] = f"{EMAIL_CONFIG['subject']}: {error_type}"
            
            body = f"""
            Voice Memo Transcription Error Report
            
            Error Type: {error_type}
            Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Error Details:
            {error_details}
            
            Traceback:
            {traceback.format_exc()}
            
            Please check the system and resolve the issue.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            text = msg.as_string()
            server.sendmail(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['recipient_email'], text)
            server.quit()
            
            self.logger.info("Error notification email sent successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to send error email: {str(e)}")
            
    def get_audio_files(self) -> List[Path]:
        """Get list of audio files in INPUT directory"""
        audio_files = []
        try:
            for file_path in INPUT_DIR.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
                    audio_files.append(file_path)
                    
            if audio_files:
                audio_files.sort(key=lambda x: x.stat().st_mtime)
                self.logger.info(f"Found {len(audio_files)} audio file(s) to process")
            
            return audio_files
            
        except Exception as e:
            error_msg = f"Error scanning INPUT directory: {str(e)}"
            self.logger.error(error_msg)
            self.send_error_email("Directory Scan Error", error_msg)
            return []
            
    def transcribe_audio(self, audio_file: Path) -> Optional[str]:
        """Transcribe audio file using Whisper"""
        try:
            self.logger.info(f"Starting transcription of: {audio_file.name}")
            start_time = time.time()
            
            result = self.model.transcribe(str(audio_file))
            transcript = result["text"].strip()
            
            processing_time = time.time() - start_time
            self.logger.info(f"Transcription completed in {processing_time:.2f} seconds")
            self.logger.info(f"Transcript length: {len(transcript)} characters")
            
            return transcript
            
        except Exception as e:
            error_msg = f"Error transcribing {audio_file.name}: {str(e)}"
            self.logger.error(error_msg)
            self.send_error_email("Transcription Error", f"File: {audio_file.name}\n{error_msg}")
            return None
            
    def save_transcript(self, transcript: str, original_filename: str) -> Optional[Path]:
        """Save transcript to OUTPUT directory with timestamp"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = Path(original_filename).stem
            output_filename = f"{timestamp}_{base_name}_transcript.txt"
            output_path = OUTPUT_DIR / output_filename
            
            formatted_transcript = self.format_transcript(transcript, original_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_transcript)
                
            self.logger.info(f"Transcript saved: {output_path.name}")
            return output_path
            
        except Exception as e:
            error_msg = f"Error saving transcript for {original_filename}: {str(e)}"
            self.logger.error(error_msg)
            self.send_error_email("File Save Error", error_msg)
            return None
            
    def format_transcript(self, transcript: str, original_filename: str) -> str:
        """Format transcript with metadata"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        formatted = f"""Voice Memo Transcript
======================

Original File: {original_filename}
Transcription Date: {timestamp}
Model Used: {WHISPER_MODEL}

Transcript:
-----------

{transcript}

---
Generated by Voice Memo Transcription System
"""
        return formatted
        
    def move_to_processed(self, audio_file: Path) -> bool:
        """Move processed audio file to processed directory"""
        try:
            processed_path = PROCESSED_DIR / audio_file.name
            
            # Handle filename conflicts
            counter = 1
            while processed_path.exists():
                stem = audio_file.stem
                suffix = audio_file.suffix
                processed_path = PROCESSED_DIR / f"{stem}_{counter}{suffix}"
                counter += 1
                
            shutil.move(str(audio_file), str(processed_path))
            self.logger.info(f"Moved to processed: {processed_path.name}")
            return True
            
        except Exception as e:
            error_msg = f"Error moving {audio_file.name} to processed: {str(e)}"
            self.logger.error(error_msg)
            self.send_error_email("File Move Error", error_msg)
            return False
            
    def process_file(self, audio_file: Path) -> bool:
        """Process a single audio file"""
        self.logger.info(f"Processing file: {audio_file.name}")
        
        # Transcribe audio
        transcript = self.transcribe_audio(audio_file)
        if not transcript:
            return False
            
        # Save transcript
        output_path = self.save_transcript(transcript, audio_file.name)
        if not output_path:
            return False
            
        # Move original file to processed
        if not self.move_to_processed(audio_file):
            return False
            
        self.logger.info(f"Successfully processed: {audio_file.name}")
        return True
        
    def run(self):
        """Main processing loop"""
        self.logger.info("Starting audio transcription service")
        self.logger.info(f"Monitoring directory: {INPUT_DIR}")
        self.logger.info(f"Output directory: {OUTPUT_DIR}")
        self.logger.info(f"Poll interval: {POLL_INTERVAL} seconds")
        
        try:
            while True:
                audio_files = self.get_audio_files()
                
                if audio_files:
                    # Process files one at a time
                    for audio_file in audio_files:
                        try:
                            success = self.process_file(audio_file)
                            if success:
                                self.logger.info(f"File processed successfully: {audio_file.name}")
                            else:
                                self.logger.error(f"Failed to process file: {audio_file.name}")
                        except Exception as e:
                            error_msg = f"Unexpected error processing {audio_file.name}: {str(e)}"
                            self.logger.error(error_msg)
                            self.send_error_email("Processing Error", error_msg)
                else:
                    self.logger.debug("No audio files found, continuing to monitor...")
                    
                time.sleep(POLL_INTERVAL)
                
        except KeyboardInterrupt:
            self.logger.info("Transcription service stopped by user")
        except Exception as e:
            error_msg = f"Fatal error in main loop: {str(e)}"
            self.logger.critical(error_msg)
            self.send_error_email("System Error", error_msg)
            raise


def main():
    """Entry point for the transcription service"""
    try:
        transcriber = AudioTranscriber()
        transcriber.run()
    except Exception as e:
        print(f"Failed to start transcription service: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()