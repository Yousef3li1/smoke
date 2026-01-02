# 🔥 Smoke & Fire Detection System

<div align="center">

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.3+-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**AI-powered real-time smoke and fire detection using YOLO deep learning**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-endpoints) • [Configuration](#-configuration)

</div>

---

## 📖 Overview

A comprehensive, production-ready smoke and fire detection system that uses YOLO (You Only Look Once) deep learning models for real-time video analysis. The system provides multiple deployment options including a modern Flask web application with live statistics, standalone detection scripts, and advanced detection algorithms with temporal validation.

### Key Capabilities

- 🔥 **Real-time Fire Detection** - High-sensitivity fire/flame detection
- 💨 **Advanced Smoke Detection** - Temporal validation and shape analysis to reduce false positives
- 📊 **Live Statistics Dashboard** - Real-time detection metrics and history
- 📹 **Multi-Camera Support** - Monitor multiple cameras simultaneously
- 🚨 **Instant Alerts** - Server-Sent Events (SSE) with audio alarms
- 💾 **Detection Logging** - SQLite database for detection history
- ⚙️ **Configurable** - Easy customization via configuration file

---

## ✨ Features

### Core Detection
- **Fire Detection**: Real-time fire/flame detection with configurable sensitivity
- **Smoke Detection**: Advanced smoke detection with temporal validation
- **Dual Zone Monitoring**: Separate monitoring for smoke-allowed and non-smoke areas
- **Temporal Validation**: Reduces false positives using frame history analysis
- **Shape Analysis**: Validates smoke detections using contour analysis

### Web Application
- **Live Video Streams**: MJPEG streaming for real-time camera feeds
- **Real-time Statistics**: Live dashboard with detection counts and metrics
- **Detection History**: View recent detections with timestamps
- **Real-time Alerts**: Server-Sent Events (SSE) for instant notifications
- **Audio Alarms**: Automatic alarm sound on detection
- **Modern UI**: Responsive, gradient-based web interface
- **Health Monitoring**: System health check endpoint

### Data & Analytics
- **SQLite Database**: Persistent storage of all detections
- **Statistics API**: RESTful endpoints for detection data
- **Detection Logging**: Automatic logging with camera, timestamp, and confidence
- **Historical Analysis**: Track detections by hour, camera, and type

### Standalone Scripts
- **Command-line Detection**: Run detection without web interface
- **Video File Support**: Process video files instead of live cameras
- **Screenshot Capture**: Save detection screenshots automatically
- **FPS Monitoring**: Performance tracking and optimization

---

## 📁 Project Structure

```
ttgg/
│
├── app.py                      # Flask web application (main entry point)
├── fire_detector.py            # Advanced Detector class with temporal validation
├── detector.py                 # Simple standalone detection script
├── simple_detect.py            # Basic terminal-only detection
├── Smoke detection.py          # Alternative detection implementation
├── detection_stats.py          # Statistics and logging module
├── config.py                   # Configuration file
│
├── best_nano_111.pt            # Pre-trained YOLO model (required)
├── requirements.txt            # Python dependencies
├── run.bat                     # Quick start script (Windows)
│
├── templates/
│   └── index.html              # Web UI frontend
│
├── static/
│   └── emergency.mp3           # Alarm sound file
│
├── screenshots/                 # Saved detection screenshots (auto-created)
├── detections.db               # SQLite database (auto-created)
│
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Webcam(s) or video file(s)
- CUDA-capable GPU (optional, for faster inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ttgg
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or using Python's module installer:
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Verify model file**
   
   Ensure `best_nano_111.pt` is present in the project directory. This is the pre-trained YOLO model required for detection.

4. **Run the application**
   
   **Option 1: Using the batch script (Windows)**
   ```bash
   run.bat
   ```
   
   **Option 2: Direct Python execution**
   ```bash
   python app.py
   ```

5. **Access the dashboard**
   
   Open your browser and navigate to: **http://127.0.0.1:5000**

---

## 💻 Usage

### Web Application (Recommended)

Start the Flask server:
```bash
python app.py
```

The web interface provides:
- **Live camera feeds** (Camera 0 and Camera 1)
- **Real-time statistics** dashboard
- **Detection history** panel
- **Audio alerts** on detection
- **Test alert** button for manual testing

**Zone Logic:**
- **Camera 0**: Smoke Area (allowed, no alerts)
- **Camera 1**: Non-Smoke Area (triggers alerts on detection)

### Standalone Detection Script

**Basic usage (default camera 0):**
```bash
python detector.py
```

**Specify camera index:**
```bash
python detector.py --camera 1
```

**Use video file:**
```bash
python detector.py --video path/to/video.mp4
```

**Custom model and confidence:**
```bash
python detector.py --model custom_model.pt --confidence 0.20
```

**Controls:**
- Press `q` to quit
- Press `s` to save screenshot

### Simple Terminal Detection

Run basic detection without UI:
```bash
python simple_detect.py
```

---

## 🌐 API Endpoints

### Web Application Routes

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard page |
| `/video_feed/<camera_index>` | GET | MJPEG video stream |
| `/events` | GET | Server-Sent Events stream |
| `/trigger/smoking` | POST | Manually trigger test alert |
| `/api/stats` | GET | Get detection statistics |
| `/api/recent` | GET | Get recent detections (limit query param) |
| `/api/health` | GET | Health check endpoint |

### API Examples

**Get Statistics:**
```bash
curl http://127.0.0.1:5000/api/stats
```

**Get Recent Detections:**
```bash
curl http://127.0.0.1:5000/api/recent?limit=10
```

**Health Check:**
```bash
curl http://127.0.0.1:5000/api/health
```

### Event Format (SSE)

```json
{
  "type": "smoking",
  "demo": "Smoking Detection",
  "level": "critical",
  "message": "Smoke detected! Immediate action required.",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "camera": 1
}
```

---

## ⚙️ Configuration

The system can be configured via `config.py`. Key settings:

```python
# Model Configuration
MODEL_PATH = "best_nano_111.pt"
CONFIDENCE = 0.10  # Detection confidence threshold
SMOKE_CONFIDENCE = 0.25
FIRE_CONFIDENCE = 0.10

# Detection Settings
TARGET_HEIGHT = 512
DETECTION_INTERVAL = 10  # Process every Nth frame
ALERT_COOLDOWN = 5.0  # Seconds between alerts

# Web Server Configuration
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# Camera Configuration
CAMERA_INDICES = [0, 1]
```

For detailed configuration options, see `config.py`.

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Web Browser │  │  Terminal UI  │  │  Video File  │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
└─────────┼─────────────────┼──────────────────┼─────────┘
          │                 │                  │
          ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│              Application Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   app.py     │  │  detector.py │  │ simple_*.py  │ │
│  │  (Flask)     │  │  (Standalone)│  │  (Basic)     │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
└─────────┼─────────────────┼──────────────────┼─────────┘
          │                 │                  │
          └─────────────────┼──────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│            Detection Engine (fire_detector.py)          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  YOLO Model (best_nano_111.pt)                   │  │
│  │  ┌──────────────┐  ┌──────────────┐            │  │
│  │  │ Fire Detect  │  │ Smoke Detect │            │  │
│  │  └──────┬───────┘  └──────┬───────┘            │  │
│  │         │                 │                      │  │
│  │  ┌──────▼─────────────────▼──────┐             │  │
│  │  │ Temporal Validator            │             │  │
│  │  │ Shape Analyzer                │             │  │
│  │  │ Mask Refinement               │             │  │
│  │  └───────────────────────────────┘             │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│         Statistics & Logging (detection_stats.py)      │
│  ┌──────────────┐  ┌──────────────┐                   │
│  │ SQLite DB    │  │ Real-time    │                   │
│  │ (History)    │  │ Statistics   │                   │
│  └──────────────┘  └──────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

### Detection Pipeline

1. **Frame Capture**: Read frame from camera/video source
2. **Preprocessing**: Resize frame to target resolution (512px height)
3. **YOLO Inference**: Run model inference on frame
4. **Detection Processing**:
   - Extract bounding boxes and masks
   - Classify as fire or smoke
   - Apply confidence thresholds
5. **Validation** (for smoke):
   - Temporal validation (frame history)
   - Shape analysis (contour properties)
   - Mask refinement
6. **Visualization**: Draw bounding boxes, labels, status
7. **Alert Generation**: Trigger alerts if detection confirmed
8. **Logging**: Store detection in database
9. **Output**: Display frame and broadcast events

---

## 🔧 Technical Details

### Detection Algorithm

#### Fire Detection
- **Confidence Threshold**: 0.10 (very sensitive)
- **Color**: Red bounding boxes
- **Logic**: Direct YOLO output, no additional validation
- **Priority**: High (immediate alert)

#### Smoke Detection
- **Confidence Threshold**: 0.15-0.25 (depending on mode)
- **Color**: Yellow/Cyan bounding boxes
- **Temporal Validation**: Requires 3+ frames with increasing area
- **Shape Analysis**: 
  - Minimum area: 100 pixels
  - Solidity check (area/hull ratio)
  - Complexity score (perimeter²/area)
- **Mask Refinement**: Morphological operations (open, dilate)

### Model Information

- **Model Type**: YOLO (YOLOv8 Nano variant)
- **Model File**: `best_nano_111.pt`
- **Input Size**: Variable (resized to 512px height)
- **Classes**: Fire, Smoke (and potentially others)
- **Device**: CPU or CUDA (auto-detected)

### Performance

- **FPS**: ~20-30 FPS on CPU, higher on GPU
- **Latency**: ~50-100ms per frame
- **Memory**: ~200-500MB (depending on resolution)
- **Detection Rate**: Every 10 frames in web app (configurable)

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Camera Not Opening

**Problem**: `Cannot open camera 0`

**Solutions**:
- Check camera is connected and not used by another application
- Try different camera index: `python detector.py --camera 1`
- On Linux, check permissions: `sudo usermod -a -G video $USER`
- Verify camera works: `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"`

#### 2. Model Not Found

**Problem**: `Model not found at best_nano_111.pt`

**Solutions**:
- Verify `best_nano_111.pt` exists in project directory
- Check file path in `config.py` matches actual location
- Download model if missing (check project repository)

#### 3. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'ultralytics'`

**Solutions**:
```bash
pip install -r requirements.txt
```

#### 4. Low Detection Accuracy

**Problem**: Too many false positives or missed detections

**Solutions**:
- Adjust confidence threshold in `config.py`
- Improve lighting conditions
- Ensure camera angle covers detection area
- Use temporal validation (already enabled in advanced detector)

#### 5. Web App Not Loading

**Problem**: Browser shows connection error

**Solutions**:
- Check Flask server is running: `python app.py`
- Verify port 5000 is not in use
- Try different port: Change `PORT` in `config.py`
- Check firewall settings

#### 6. Slow Performance

**Problem**: Low FPS or laggy detection

**Solutions**:
- Reduce frame resolution in `config.py`
- Use GPU if available (CUDA)
- Increase detection interval (process fewer frames)
- Close other applications using CPU/GPU

---

## 📊 Detection Zones

The system supports dual-zone monitoring:

### Zone 0: Smoke Area (Allowed)
- **Camera Index**: 0
- **Behavior**: Detections logged but no alerts triggered
- **Use Case**: Areas where smoking is permitted
- **Visual**: Yellow/Orange header in web UI

### Zone 1: Non-Smoke Area (Alerts)
- **Camera Index**: 1
- **Behavior**: Alerts triggered on any detection
- **Use Case**: Areas where smoking is prohibited
- **Visual**: Red/Pink header in web UI

---

## 🔒 Security Considerations

- **Local Network Only**: Web app binds to `0.0.0.0` (all interfaces)
- **No Authentication**: Add authentication for production use
- **HTTPS**: Use HTTPS in production (reverse proxy recommended)
- **Input Validation**: Validate camera indices and file paths
- **Resource Limits**: Monitor CPU/GPU usage to prevent DoS

---

## 🚧 Future Enhancements

Potential improvements:
- [ ] Database logging of detections ✅ (Implemented)
- [ ] Email/SMS notifications
- [ ] Mobile app integration
- [ ] Multi-user support with authentication
- [ ] Detection history and analytics ✅ (Implemented)
- [ ] Custom model training interface
- [ ] Integration with security systems
- [ ] Cloud deployment support
- [ ] REST API for external integrations
- [ ] WebSocket support for lower latency

---

## 📝 License

This project is provided as-is for educational and research purposes. Please ensure compliance with local regulations when deploying in production environments.

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Model accuracy improvements
- Performance optimization
- Additional detection algorithms
- UI/UX enhancements
- Documentation improvements
- Bug fixes and testing

---

## 📞 Support

For issues or questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review code comments in relevant files
3. Verify all dependencies are installed correctly
4. Test with `simple_detect.py` first to isolate issues

---

## 🎓 Learning Resources

To understand this project better:
- **YOLO**: [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- **OpenCV**: [OpenCV Python Tutorials](https://docs.opencv.org/)
- **Flask**: [Flask Documentation](https://flask.palletsprojects.com/)
- **Computer Vision**: Study object detection and image processing

---

## 📸 Screenshots

*Add screenshots of the web interface here*

---

<div align="center">

**Made with ❤️ using Python, Flask, and YOLO**

⭐ Star this repo if you find it useful!

</div>
