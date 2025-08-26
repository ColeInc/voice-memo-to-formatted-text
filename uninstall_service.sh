#!/bin/bash

# Voice Memo Transcriber Service Uninstallation Script for macOS
# This script removes the voice memo transcription service from startup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Voice Memo Transcriber Service Uninstallation${NC}"
echo "=============================================="

TARGET_PLIST="$HOME/Library/LaunchAgents/com.voicememo.transcriber.plist"

# Check if service is running and stop it
if launchctl list | grep -q "com.voicememo.transcriber"; then
    echo "Stopping service..."
    launchctl unload "$TARGET_PLIST"
    echo -e "${GREEN}✓ Service stopped${NC}"
else
    echo -e "${YELLOW}Service is not currently running${NC}"
fi

# Remove the plist file
if [ -f "$TARGET_PLIST" ]; then
    rm "$TARGET_PLIST"
    echo -e "${GREEN}✓ Service configuration removed${NC}"
else
    echo -e "${YELLOW}Service configuration file not found${NC}"
fi

echo ""
echo -e "${GREEN}Service uninstalled successfully!${NC}"
echo "The voice memo transcriber will no longer start automatically at login."
echo ""
echo "To manually run the transcriber when needed:"
echo "  python3 transcribe_audio.py"