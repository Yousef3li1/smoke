"""
Simple Smoke Detection Flask App
================================
Standalone Flask web application for smoke/fire detection.

Usage:
    python app.py

Then open: http://127.0.0.1:5000
"""

import cv2
import numpy as np
import threading
import time
import json
from pathlib import Path
from datetime import datetime
from queue import Queue

from flask import Flask, Response, render_template, jsonify, request

# Import configuration and stats
try:
    import config
    MODEL_PATH = config.MODEL_PATH
    CONFIDENCE = config.CONFIDENCE
    ALERT_COOLDOWN = config.ALERT_COOLDOWN
    DETECTION_INTERVAL = config.DETECTION_INTERVAL
    HOST = config.HOST
    PORT = config.PORT
    DEBUG = config.DEBUG
    CAMERA_INDICES = config.CAMERA_INDICES
except ImportError:
    print("Warning: config.py not found, using defaults")
    MODEL_PATH = Path(__file__).parent / "best_nano_111.pt"
    CONFIDENCE = 0.10
    ALERT_COOLDOWN = 5.0
    DETECTION_INTERVAL = 10
    HOST = "0.0.0.0"
    PORT = 5000
    DEBUG = True
    CAMERA_INDICES = [0, 1]

# Import detector
try:
    from fire_detector import Detector
except ImportError:
    print("Warning: fire_detector.py not found, using built-in detector")
    Detector = None

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    exit(1)

# Import stats
try:
    from detection_stats import stats
    STATS_ENABLED = True
except ImportError:
    print("Warning: detection_stats.py not found, statistics disabled")
    STATS_ENABLED = False
    stats = None


app = Flask(__name__)

# --- Global State ---
detector = None
camera_captures = {}
camera_frames = {}
camera_locks = {}
camera_threads = {}
event_listeners = []
event_lock = threading.Lock()


def init_detector():
    """Initialize the smoke/fire detector."""
    global detector
    if detector is not None:
        return
    
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please ensure best_nano_111.pt is in the project directory")
        return
    
    try:
        print(f"Loading model from: {MODEL_PATH}")
        if Detector:
            detector = Detector(MODEL_PATH, iou_threshold=0.2)
        else:
            # Fallback to simple YOLO
            detector = YOLO(str(MODEL_PATH))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check that the model file is valid and dependencies are installed")
        detector = None


def broadcast_event(event: dict):
    """Broadcast SSE event to all listeners."""
    with event_lock:
        for q in event_listeners:
            try:
                q.put_nowait(event)
            except:
                pass


def add_listener():
    """Add a new SSE listener."""
    q = Queue()
    with event_lock:
        event_listeners.append(q)
    return q


def remove_listener(q):
    """Remove an SSE listener."""
    with event_lock:
        if q in event_listeners:
            event_listeners.remove(q)


def camera_reader_thread(camera_index: int):
    """Background thread that reads frames from camera and runs detection."""
    global detector
    
    # Initialize detector in thread
    init_detector()
    
    frame_counter = 0
    last_alert_time = 0.0
    
    while camera_index in camera_captures:
        cap = camera_captures.get(camera_index)
        if not cap or not cap.isOpened():
            time.sleep(0.1)
            continue
        
        lock = camera_locks.get(camera_index)
        if not lock:
            time.sleep(0.1)
            continue
        
        with lock:
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_counter += 1
                
                # Run detection every 10 frames
                if frame_counter % 10 == 0 and detector is not None:
                    try:
                        if hasattr(detector, 'process_frame'):
                            processed, detection = detector.process_frame(frame.copy())
                            frame = processed
                        else:
                            # Simple YOLO detection
                            results = detector(frame, conf=CONFIDENCE, verbose=False)
                            detection = None
                            for result in results:
                                if result.boxes and len(result.boxes) > 0:
                                    for box in result.boxes:
                                        cls = int(box.cls[0])
                                        name = detector.names.get(cls, "").lower()
                                        if "fire" in name:
                                            detection = "Fire"
                                        elif "smoke" in name:
                                            detection = "Smoke"
                                        
                                        # Draw box
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        conf = float(box.conf[0])
                                        color = (0, 0, 255) if "fire" in name else (128, 128, 128)
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                        cv2.putText(frame, f"{name}: {conf:.2f}", (x1, y1-10),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Trigger alert on ANY fire/smoke detection
                        if detection in ["Fire", "Smoke", "Smoking"]:
                            now = time.time()
                            if now - last_alert_time > ALERT_COOLDOWN:
                                last_alert_time = now
                                print(f"")
                                print(f"="*50)
                                print(f"🚨 ALERT: {detection.upper()} DETECTED!")
                                print(f"   Camera: {camera_index}")
                                print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
                                print(f"="*50)
                                
                                # Log detection to stats
                                if STATS_ENABLED and stats:
                                    zone = "smoke_area" if camera_index == 0 else "non_smoke_area"
                                    stats.log_detection(
                                        detection_type=detection,
                                        camera_index=camera_index,
                                        confidence=0.0,  # Could extract from detection
                                        zone=zone
                                    )
                                
                                # Broadcast SSE event
                                broadcast_event({
                                    "type": "smoking",
                                    "demo": "Smoking Detection",
                                    "level": "critical",
                                    "message": f"{detection} detected! Immediate action required.",
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                    "camera": camera_index
                                })
                    except Exception as e:
                        print(f"Detection error: {e}")
                
                # Add zone label
                zone = "SMOKE AREA" if camera_index == 0 else "NON-SMOKE AREA"
                color = (0, 255, 255) if camera_index == 0 else (0, 255, 0)
                cv2.putText(frame, zone, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                camera_frames[camera_index] = frame
        
        time.sleep(0.03)  # ~30 FPS


def init_camera(camera_index: int) -> bool:
    """Initialize a camera."""
    if camera_index in camera_captures:
        return True
    
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Cannot open camera {camera_index}")
            print("Tip: Try a different camera index or check if camera is in use")
            return False
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        camera_captures[camera_index] = cap
        camera_locks[camera_index] = threading.Lock()
        camera_frames[camera_index] = None
        
        # Start reader thread
        thread = threading.Thread(target=camera_reader_thread, args=(camera_index,), daemon=True)
        thread.start()
        camera_threads[camera_index] = thread
        
        print(f"Camera {camera_index} initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing camera {camera_index}: {e}")
        return False


def generate_stream(camera_index: int):
    """Generate MJPEG stream for camera."""
    init_camera(camera_index)
    
    while True:
        frame = camera_frames.get(camera_index)
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)


def event_stream():
    """SSE event stream generator."""
    q = add_listener()
    try:
        while True:
            event = q.get()
            yield f"data: {json.dumps(event)}\n\n"
    except GeneratorExit:
        remove_listener(q)


# --- Routes ---

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index: int):
    """MJPEG video stream endpoint."""
    return Response(
        generate_stream(camera_index),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/events')
def events():
    """SSE endpoint for real-time alerts."""
    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/trigger/smoking', methods=['POST'])
def trigger_smoking():
    """Manual trigger for testing."""
    broadcast_event({
        "type": "smoking",
        "demo": "Smoking Detection",
        "level": "critical",
        "message": "Smoking activity detected in non-smoking zone!",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "camera": 1
    })
    return jsonify({"status": "ok"})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get detection statistics."""
    if not STATS_ENABLED or not stats:
        return jsonify({"error": "Statistics not available"}), 503
    return jsonify(stats.get_stats())


@app.route('/api/recent', methods=['GET'])
def get_recent():
    """Get recent detections."""
    if not STATS_ENABLED or not stats:
        return jsonify({"error": "Statistics not available"}), 503
    limit = int(request.args.get('limit', 20))
    return jsonify(stats.get_recent_detections(limit=limit))


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    health = {
        "status": "healthy",
        "detector_loaded": detector is not None,
        "cameras_active": len([c for c in camera_captures.values() if c and c.isOpened()]),
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    status_code = 200 if health["detector_loaded"] and health["model_exists"] else 503
    return jsonify(health), status_code


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Simple Smoke Detection - Flask App")
    print("="*50)
    print(f"Model: {MODEL_PATH}")
    print("Open: http://127.0.0.1:5000")
    print("="*50 + "\n")
    
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
