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
import ssl
import os
import threading
import re

try:
    import whisper
    import ffmpeg
except ImportError as e:
    if "whisper" in str(e):
        print("Error: OpenAI Whisper not installed. Please run: pip install -r requirements.txt")
    elif "ffmpeg" in str(e):
        print("Error: ffmpeg-python not installed. Please run: pip install -r requirements.txt")
    else:
        print(f"Import error: {e}")
    sys.exit(1)

from config import (
    INPUT_DIR, OUTPUT_DIR, PROCESSED_DIR,
    WHISPER_MODEL, POLL_INTERVAL, SUPPORTED_FORMATS,
    EMAIL_CONFIG, LOGGING_CONFIG, SSL_VERIFY, PROGRESS_CONFIG,
    VIDEO_FORMATS, AUDIO_EXTRACTION_CONFIG
)


class AudioTranscriber:
    def __init__(self):
        self.setup_ssl()
        self.setup_logging()
        self.setup_directories()
        self.load_whisper_model()
        self.setup_temp_directory()
        self.last_progress = 0  # Track last reported progress percentage
        self.heartbeat_active = False  # Flag for heartbeat thread
        
    def setup_ssl(self):
        """Configure SSL settings for networks with certificate issues"""
        if not SSL_VERIFY:
            # Print warning early since logger may not be set up yet
            print("SSL verification disabled - this reduces security")
            
            # Set environment variables first (these affect urllib and other libraries)
            os.environ['PYTHONHTTPSVERIFY'] = '0'
            os.environ['CURL_CA_BUNDLE'] = ''
            os.environ['REQUESTS_CA_BUNDLE'] = ''
            
            # Monkey patch urllib to disable SSL verification
            import urllib.request
            import urllib.error
            
            # Create an SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Replace the default HTTPS handler
            https_handler = urllib.request.HTTPSHandler(context=ssl_context)
            opener = urllib.request.build_opener(https_handler)
            urllib.request.install_opener(opener)
        
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
        
    def setup_temp_directory(self):
        """Ensure temporary audio directory exists"""
        AUDIO_EXTRACTION_CONFIG['temp_dir'].mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Temp audio directory ready: {AUDIO_EXTRACTION_CONFIG['temp_dir']}")
        
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
            
            # Handle SSL issues by temporarily disabling verification if needed
            if not SSL_VERIFY:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                
            self.model = whisper.load_model(WHISPER_MODEL)
            self.logger.info("Whisper model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load Whisper model: {str(e)}"
            
            # Provide helpful error message for SSL issues
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                ssl_help = (
                    "\n\nSSL Certificate Error Detected!\n"
                    "This often happens on corporate networks or with proxy settings.\n"
                    "To fix this, edit config.py and set: SSL_VERIFY = False\n"
                    "Warning: This reduces security but allows the script to work."
                )
                error_msg += ssl_help
                self.logger.error("SSL certificate verification failed. Consider setting SSL_VERIFY = False in config.py")
            
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
        """Get list of audio and video files in INPUT directory"""
        media_files = []
        try:
            for file_path in INPUT_DIR.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
                    media_files.append(file_path)
                    
            if media_files:
                media_files.sort(key=lambda x: x.stat().st_mtime)
                self.logger.info(f"Found {len(media_files)} media file(s) to process")
            
            return media_files
            
        except Exception as e:
            error_msg = f"Error scanning INPUT directory: {str(e)}"
            self.logger.error(error_msg)
            self.send_error_email("Directory Scan Error", error_msg)
            return []
    
    def progress_callback(self, segments, segment_size, left_segment, rate):
        """Progress callback for transcription with mad-whisper-progress"""
        if not PROGRESS_CONFIG['enabled']:
            return
            
        # Convert rate to percentage
        progress_percent = int(rate * 100)
        
        # Only log at specified intervals to avoid spam
        if progress_percent >= self.last_progress + PROGRESS_CONFIG['update_interval']:
            self.last_progress = progress_percent
            
            # Log progress with optional time estimate
            if PROGRESS_CONFIG['show_time_estimate'] and rate > 0:
                # Simple time estimate based on current progress
                elapsed_time = time.time() - self.transcription_start_time
                if rate > 0.01:  # Avoid division by very small numbers
                    estimated_total = elapsed_time / rate
                    remaining_time = estimated_total - elapsed_time
                    remaining_min = int(remaining_time // 60)
                    remaining_sec = int(remaining_time % 60)
                    
                    self.logger.info(f"Transcription progress: {progress_percent}% complete "
                                   f"(~{remaining_min}m {remaining_sec}s remaining)")
                else:
                    self.logger.info(f"Transcription progress: {progress_percent}% complete")
            else:
                self.logger.info(f"Transcription progress: {progress_percent}% complete")
    
    def heartbeat_progress(self, filename: str):
        """Fallback progress indicator using threading"""
        if not PROGRESS_CONFIG['enabled']:
            return
            
        start_time = time.time()
        while self.heartbeat_active:
            elapsed_time = time.time() - start_time
            elapsed_min = int(elapsed_time // 60)
            elapsed_sec = int(elapsed_time % 60)
            self.logger.info(f"Transcribing {filename}... ({elapsed_min}m {elapsed_sec}s elapsed)")
            
            # Wait for 30 seconds or until heartbeat is stopped
            for _ in range(30):
                if not self.heartbeat_active:
                    break
                time.sleep(1)
            
    def transcribe_audio(self, audio_file: Path) -> Optional[str]:
        """Transcribe audio file using Whisper"""
        heartbeat_thread = None
        try:
            self.logger.info(f"Starting transcription of: {audio_file.name}")
            start_time = time.time()
            self.transcription_start_time = start_time  # Store for progress callback
            self.last_progress = 0  # Reset progress tracker
            
            # Start heartbeat thread as fallback
            if PROGRESS_CONFIG['enabled']:
                self.heartbeat_active = True
                heartbeat_thread = threading.Thread(
                    target=self.heartbeat_progress, 
                    args=(audio_file.name,)
                )
                heartbeat_thread.daemon = True
                heartbeat_thread.start()
            
            # Try to use progress callback first, fall back to regular transcribe
            try:
                if PROGRESS_CONFIG['enabled']:
                    result = self.model.transcribe(str(audio_file), progress=self.progress_callback)
                else:
                    result = self.model.transcribe(str(audio_file))
            except TypeError as e:
                # If progress parameter is not supported, fall back to regular transcribe
                self.logger.warning(f"Progress callback not supported, using fallback: {e}")
                result = self.model.transcribe(str(audio_file))
            
            # Stop heartbeat thread
            if heartbeat_thread:
                self.heartbeat_active = False
                heartbeat_thread.join(timeout=1)
                
            transcript = result["text"].strip()
            
            processing_time = time.time() - start_time
            self.logger.info(f"Transcription completed in {processing_time:.2f} seconds")
            self.logger.info(f"Transcript length: {len(transcript)} characters")
            
            return transcript
            
        except Exception as e:
            # Stop heartbeat thread if running
            if heartbeat_thread:
                self.heartbeat_active = False
                heartbeat_thread.join(timeout=1)
                
            error_msg = f"Error transcribing {audio_file.name}: {str(e)}"
            self.logger.error(error_msg)
            self.send_error_email("Transcription Error", f"File: {audio_file.name}\n{error_msg}")
            return None
            
    def save_transcript(self, transcript: str, original_filename: str) -> Optional[Path]:
        """Save transcript to OUTPUT directory with timestamp"""
        try:
            # Format: 2025-08-25_4-16-56pm.txt
            now = datetime.now()
            date_part = now.strftime('%Y-%m-%d')
            hour = now.hour
            minute = now.minute
            second = now.second
            
            # Convert to 12-hour format
            if hour == 0:
                hour_12 = 12
                am_pm = 'am'
            elif hour < 12:
                hour_12 = hour
                am_pm = 'am'
            elif hour == 12:
                hour_12 = 12
                am_pm = 'pm'
            else:
                hour_12 = hour - 12
                am_pm = 'pm'
            
            time_part = f"{hour_12}-{minute:02d}-{second:02d}{am_pm}"
            output_filename = f"{date_part}_{time_part}.txt"
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
            
    def add_paragraph_breaks(self, transcript: str) -> str:
        """Add intelligent paragraph breaks to transcript based on speech patterns"""
        if not transcript.strip():
            return transcript
            
        # Clean up the transcript first
        text = transcript.strip()
        
        # Split into sentences (basic approach)
        sentences = re.split(r'[.!?]+', text)
        
        # Remove empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return transcript
            
        paragraphs = []
        current_paragraph = []
        sentence_count = 0
        
        for sentence in sentences:
            # Add the sentence to current paragraph
            current_paragraph.append(sentence)
            sentence_count += 1
            
            # Determine if we should start a new paragraph based on:
            # 1. Length - after 3-5 sentences
            # 2. Topic indicators - words that suggest topic changes
            # 3. Time indicators - words that suggest time progression
            
            should_break = False
            
            # Check for natural paragraph break indicators
            topic_indicators = [
                'however', 'but', 'although', 'meanwhile', 'furthermore',
                'additionally', 'on the other hand', 'in contrast', 
                'speaking of', 'moving on', 'another thing', 'also',
                'next', 'then', 'after that', 'later', 'earlier',
                'first', 'second', 'third', 'finally', 'in conclusion'
            ]
            
            # Check if sentence starts with a topic indicator
            sentence_lower = sentence.lower()
            for indicator in topic_indicators:
                if sentence_lower.startswith(indicator):
                    should_break = True
                    break
            
            # Break after 3-5 sentences regardless
            if sentence_count >= 4:
                should_break = True
            
            # If very long sentence (likely run-on), break after it
            if len(sentence) > 150:
                should_break = True
                
            if should_break and len(current_paragraph) > 1:
                # Join current paragraph and add to paragraphs
                paragraph_text = '. '.join(current_paragraph) + '.'
                paragraphs.append(paragraph_text)
                current_paragraph = []
                sentence_count = 0
        
        # Add remaining sentences as final paragraph
        if current_paragraph:
            paragraph_text = '. '.join(current_paragraph) + '.'
            paragraphs.append(paragraph_text)
        
        # Join paragraphs with double newlines
        return '\n\n'.join(paragraphs)
    
    def format_transcript(self, transcript: str, original_filename: str) -> str:
        """Format transcript with metadata and paragraph breaks"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add intelligent paragraph breaks
        formatted_transcript = self.add_paragraph_breaks(transcript)
        
        formatted = f"""Voice Memo Transcript
======================

Original File: {original_filename}
Transcription Date: {timestamp}
Model Used: {WHISPER_MODEL}

Transcript:
-----------

{formatted_transcript}

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
    
    def is_video_file(self, file_path: Path) -> bool:
        """Check if file is a video format that needs audio extraction"""
        return file_path.suffix.lower() in VIDEO_FORMATS
    
    def extract_audio_from_video(self, video_file: Path) -> Optional[Path]:
        """Extract audio from video file using ffmpeg"""
        try:
            self.logger.info(f"Extracting audio from video: {video_file.name}")
            
            # Generate temporary audio filename
            audio_filename = f"{video_file.stem}_extracted.{AUDIO_EXTRACTION_CONFIG['output_format']}"
            temp_audio_path = AUDIO_EXTRACTION_CONFIG['temp_dir'] / audio_filename
            
            # Remove temp file if it exists
            if temp_audio_path.exists():
                temp_audio_path.unlink()
            
            # Extract audio using ffmpeg
            stream = ffmpeg.input(str(video_file))
            audio_stream = stream.audio
            
            # Apply audio extraction settings
            out = ffmpeg.output(
                audio_stream,
                str(temp_audio_path),
                acodec='mp3' if AUDIO_EXTRACTION_CONFIG['output_format'] == 'mp3' else 'libmp3lame',
                audio_bitrate=AUDIO_EXTRACTION_CONFIG['audio_quality']
            )
            
            # Run the extraction
            ffmpeg.run(out, quiet=True, overwrite_output=True)
            
            if temp_audio_path.exists():
                self.logger.info(f"Audio extracted successfully: {temp_audio_path.name}")
                return temp_audio_path
            else:
                self.logger.error("Audio extraction failed - output file not created")
                return None
                
        except Exception as e:
            error_msg = f"Error extracting audio from {video_file.name}: {str(e)}"
            self.logger.error(error_msg)
            self.send_error_email("Audio Extraction Error", error_msg)
            return None
    
    def cleanup_temp_audio(self, temp_audio_path: Path) -> bool:
        """Remove temporary audio file"""
        try:
            if temp_audio_path.exists():
                temp_audio_path.unlink()
                self.logger.info(f"Cleaned up temporary audio file: {temp_audio_path.name}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temporary audio file {temp_audio_path.name}: {str(e)}")
            return False
            
    def process_file(self, input_file: Path) -> bool:
        """Process a single audio or video file"""
        self.logger.info(f"Processing file: {input_file.name}")
        
        temp_audio_path = None
        audio_file_to_process = input_file
        
        try:
            # Check if this is a video file that needs audio extraction
            if self.is_video_file(input_file):
                self.logger.info(f"Video file detected: {input_file.name}")
                temp_audio_path = self.extract_audio_from_video(input_file)
                if not temp_audio_path:
                    self.logger.error(f"Failed to extract audio from video: {input_file.name}")
                    return False
                audio_file_to_process = temp_audio_path
                self.logger.info(f"Using extracted audio: {temp_audio_path.name}")
            
            # Transcribe audio (from either original audio file or extracted audio)
            transcript = self.transcribe_audio(audio_file_to_process)
            if not transcript:
                return False
                
            # Save transcript using original filename
            output_path = self.save_transcript(transcript, input_file.name)
            if not output_path:
                return False
                
            # Move original file to processed
            if not self.move_to_processed(input_file):
                return False
                
            self.logger.info(f"Successfully processed: {input_file.name}")
            return True
            
        finally:
            # Clean up temporary audio file if it was created
            if temp_audio_path:
                self.cleanup_temp_audio(temp_audio_path)
        
    def run(self):
        """Main processing loop"""
        self.logger.info("Starting audio/video transcription service")
        self.logger.info(f"Monitoring directory: {INPUT_DIR}")
        self.logger.info(f"Output directory: {OUTPUT_DIR}")
        self.logger.info(f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}")
        self.logger.info(f"Video formats (audio extraction): {', '.join(sorted(VIDEO_FORMATS))}")
        self.logger.info(f"Poll interval: {POLL_INTERVAL} seconds")
        
        try:
            while True:
                media_files = self.get_audio_files()
                
                if media_files:
                    # Process files one at a time
                    for media_file in media_files:
                        try:
                            success = self.process_file(media_file)
                            if success:
                                self.logger.info(f"File processed successfully: {media_file.name}")
                            else:
                                self.logger.error(f"Failed to process file: {media_file.name}")
                        except Exception as e:
                            error_msg = f"Unexpected error processing {media_file.name}: {str(e)}"
                            self.logger.error(error_msg)
                            self.send_error_email("Processing Error", error_msg)
                else:
                    self.logger.debug("No media files found, continuing to monitor...")
                    
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