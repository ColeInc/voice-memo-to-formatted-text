# Linux Setup Instructions

For Linux users who want the voice memo transcriber to start automatically at boot.

## systemd Service Setup

### 1. Create the service file

Create a systemd service file (replace `YOUR_USERNAME` and `YOUR_PROJECT_PATH` with your actual values):

```bash
sudo nano /etc/systemd/system/voice-memo-transcriber.service
```

Add the following content:

```ini
[Unit]
Description=Voice Memo Transcription Service
After=multi-user.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USERNAME
Group=YOUR_USERNAME
WorkingDirectory=/path/to/voice-memo-to-formatted-text
ExecStart=/usr/bin/python3 /path/to/voice-memo-to-formatted-text/transcribe_audio.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Environment variables
Environment=PATH=/usr/local/bin:/usr/bin:/bin
Environment=PYTHONPATH=/path/to/voice-memo-to-formatted-text

[Install]
WantedBy=multi-user.target
```

### 2. Enable and start the service

```bash
# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start at boot
sudo systemctl enable voice-memo-transcriber.service

# Start the service now
sudo systemctl start voice-memo-transcriber.service
```

### 3. Service management commands

```bash
# Check service status
sudo systemctl status voice-memo-transcriber.service

# View logs
sudo journalctl -u voice-memo-transcriber.service -f

# Stop the service
sudo systemctl stop voice-memo-transcriber.service

# Disable auto-start
sudo systemctl disable voice-memo-transcriber.service

# Restart the service
sudo systemctl restart voice-memo-transcriber.service
```

## Alternative: User Service (runs only when user is logged in)

If you prefer the service to run only when you're logged in:

### 1. Create user service directory
```bash
mkdir -p ~/.config/systemd/user
```

### 2. Create the user service file
```bash
nano ~/.config/systemd/user/voice-memo-transcriber.service
```

Content (no need to change USER or use sudo):
```ini
[Unit]
Description=Voice Memo Transcription Service
After=default.target

[Service]
Type=simple
WorkingDirectory=%h/path/to/voice-memo-to-formatted-text
ExecStart=/usr/bin/python3 %h/path/to/voice-memo-to-formatted-text/transcribe_audio.py
Restart=always
RestartSec=10

Environment=PYTHONPATH=%h/path/to/voice-memo-to-formatted-text

[Install]
WantedBy=default.target
```

### 3. Enable user service
```bash
# Reload user systemd
systemctl --user daemon-reload

# Enable the service
systemctl --user enable voice-memo-transcriber.service

# Start the service
systemctl --user start voice-memo-transcriber.service

# Enable lingering (so service starts even when not logged in)
sudo loginctl enable-linger $USER
```

### 4. User service management
```bash
# Status
systemctl --user status voice-memo-transcriber.service

# Logs  
journalctl --user -u voice-memo-transcriber.service -f

# Stop
systemctl --user stop voice-memo-transcriber.service
```

## Troubleshooting

### Permission Issues
If you get permission errors:
```bash
# Make sure your user owns the project directory
sudo chown -R $USER:$USER /path/to/voice-memo-to-formatted-text

# Ensure the Python script is executable
chmod +x /path/to/voice-memo-to-formatted-text/transcribe_audio.py
```

### Python Path Issues
If the service can't find Python modules:
```bash
# Find your Python path
which python3

# Update the ExecStart line in your service file with the correct path
ExecStart=/usr/bin/python3 /path/to/your/script.py
```

### Dependencies Not Found
Make sure all dependencies are installed system-wide or in the correct Python environment:
```bash
# Install system-wide
sudo pip3 install -r requirements.txt

# Or use a virtual environment and update the service ExecStart path
```