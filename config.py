"""
Configuration file for Smoke & Fire Detection System
"""
from pathlib import Path

# Model Configuration
MODEL_PATH = Path(__file__).parent / "best_nano_111.pt"
CONFIDENCE = 0.10  # Detection confidence threshold
IOU_THRESHOLD = 0.4
SMOKE_CONFIDENCE = 0.25
FIRE_CONFIDENCE = 0.10

# Detection Settings
TARGET_HEIGHT = 512
DETECTION_INTERVAL = 10  # Process every Nth frame
ALERT_COOLDOWN = 5.0  # Seconds between alerts
MIN_SMOKE_AREA = 100
SMOKE_CONFIRM_FRAMES = 3

# Web Server Configuration
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# Camera Configuration
CAMERA_INDICES = [0, 1]  # Cameras to use
CAMERA_0_ZONE = "smoke_area"  # Allowed zone
CAMERA_1_ZONE = "non_smoke_area"  # Alert zone

# Logging Configuration
ENABLE_LOGGING = True
LOG_DETECTIONS = True
SCREENSHOT_ON_DETECTION = True
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"

# Performance Settings
USE_GPU = True  # Auto-detect if available
MAX_FPS = 30

