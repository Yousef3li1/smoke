"""
Simple Fire/Smoke Detection - Terminal Only
============================================
Runs camera, detects fire/smoke, prints to terminal.

Usage: python simple_detect.py
"""

import cv2
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("Install: pip install ultralytics opencv-python")
    exit(1)

# Load model
MODEL_PATH = Path(__file__).parent / "best_nano_111.pt"
print(f"Loading model: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
print("Model loaded!\n")

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit(1)

print("=" * 50)
print("  SMOKE/FIRE DETECTION RUNNING")
print("  Press 'q' to quit")
print("=" * 50)
print()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model(frame, conf=0.10, verbose=False)
    
    for result in results:
        if result.boxes and len(result.boxes) > 0:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names.get(cls, "unknown")
                
                # Print detection
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] DETECTED: {name.upper()} (confidence: {conf:.2%})")
                
                # Draw on frame
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 0, 255) if "fire" in name.lower() else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name}: {conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Show frame
    cv2.imshow("Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nStopped.")
