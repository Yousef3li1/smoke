"""
Simple Smoke and Fire Detection
================================
Standalone script for real-time smoke/fire detection using YOLO.
All files are self-contained in this folder.

Usage:
    python detector.py              # Use default camera (0)
    python detector.py --camera 1   # Use camera 1
    python detector.py --video path/to/video.mp4  # Use video file

Requirements:
    pip install ultralytics opencv-python numpy

Press 'q' to quit, 's' to save a screenshot.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Try to import YOLO, provide helpful error if missing
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    exit(1)


class SmokeFireDetector:
    """Simple smoke and fire detector using YOLO."""
    
    def __init__(self, model_path: str = None, confidence: float = 0.15):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLO model (.pt file). If None, uses default.
            confidence: Detection confidence threshold (0-1).
        """
        # Default to model in same directory
        if model_path is None:
            model_path = Path(__file__).parent / "best_nano_111.pt"
        
        self.model_path = Path(model_path)
        self.confidence = confidence
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("Model loaded successfully!")
        
        # Detection history for temporal smoothing
        self.detection_history = []
        self.history_length = 5
        
        # Colors for drawing
        self.colors = {
            "fire": (0, 0, 255),      # Red
            "smoke": (128, 128, 128),  # Gray
            "default": (0, 255, 255)   # Yellow
        }
    
    def detect(self, frame: np.ndarray) -> tuple:
        """
        Run detection on a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            (processed_frame, detection_label, detections_list)
        """
        if frame is None:
            return None, None, []
        
        # Run YOLO inference
        results = self.model(frame, conf=self.confidence, verbose=False)
        
        detection_label = None
        detections = []
        
        # Process results
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Get class name
                    class_name = self.model.names.get(cls, "unknown").lower()
                    
                    detections.append({
                        "class": class_name,
                        "confidence": conf,
                        "box": (x1, y1, x2, y2)
                    })
                    
                    # Determine detection label
                    if "fire" in class_name or "flame" in class_name:
                        detection_label = "Fire"
                    elif "smoke" in class_name:
                        detection_label = "Smoke"
                    
                    # Draw box
                    color = self.colors.get(class_name, self.colors["default"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Temporal smoothing
        self.detection_history.append(detection_label)
        if len(self.detection_history) > self.history_length:
            self.detection_history.pop(0)
        
        # Count recent detections
        fire_count = sum(1 for d in self.detection_history if d == "Fire")
        smoke_count = sum(1 for d in self.detection_history if d == "Smoke")
        
        # Require multiple detections for confirmation
        confirmed_detection = None
        if fire_count >= 2:
            confirmed_detection = "Fire"
        elif smoke_count >= 2:
            confirmed_detection = "Smoke"
        
        return frame, confirmed_detection, detections


def draw_status(frame, detection, fps):
    """Draw status overlay on frame."""
    h, w = frame.shape[:2]
    
    # Status bar background
    cv2.rectangle(frame, (0, h - 40), (w, h), (30, 30, 30), -1)
    
    # Detection status
    if detection:
        status_color = (0, 0, 255) if detection == "Fire" else (0, 165, 255)
        status_text = f"⚠ {detection.upper()} DETECTED!"
        cv2.putText(frame, status_text, (10, h - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    else:
        cv2.putText(frame, "Monitoring...", (10, h - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 100, h - 12), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Controls hint
    cv2.putText(frame, "Press 'q' to quit, 's' to screenshot", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame


def main():
    parser = argparse.ArgumentParser(description="Simple Smoke/Fire Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--video", type=str, help="Video file path (overrides camera)")
    parser.add_argument("--model", type=str, help="Path to YOLO model")
    parser.add_argument("--confidence", type=float, default=0.15, help="Confidence threshold")
    args = parser.parse_args()
    
    # Initialize detector
    try:
        detector = SmokeFireDetector(
            model_path=args.model,
            confidence=args.confidence
        )
    except Exception as e:
        print(f"Error loading detector: {e}")
        return
    
    # Open video source
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Cannot open source: {source}")
        return
    
    print(f"Opening source: {source}")
    print("Press 'q' to quit, 's' to save screenshot")
    
    # FPS tracking
    fps = 0
    frame_count = 0
    start_time = cv2.getTickCount()
    
    # Screenshot directory
    screenshot_dir = Path(__file__).parent / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if args.video:
                # Video ended, restart
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        # Run detection
        processed, detection, detections = detector.detect(frame)
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = cv2.getTickCount()
            fps = 30 / ((current_time - start_time) / cv2.getTickFrequency())
            start_time = current_time
        
        # Draw status
        processed = draw_status(processed, detection, fps)
        
        # Print alert to console
        if detection:
            print(f"[ALERT] {detection} detected!")
        
        # Show frame
        cv2.imshow("Smoke/Fire Detection", processed)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = screenshot_dir / f"detection_{timestamp}.jpg"
            cv2.imwrite(str(filename), processed)
            print(f"Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")


if __name__ == "__main__":
    main()
