import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque

# ===============================
# CONFIG — BALANCED FOR ACCURACY
# ===============================
MODEL_PATH = "best_nano_111.pt"
DEVICE = "cpu"
FRAME_SIZE = 512

CONF_THRES = 0.15             # Reasonable confidence threshold
IOU_THRES = 0.4

MASK_THRESHOLD = 0.25          # Higher for more certain detections
MIN_SMOKE_AREA = 100           # Larger minimum area

SMOKE_CONFIRM_FRAMES = 3
TEMPORAL_WINDOW = 5

# ===============================
# LOAD MODEL
# ===============================
model = YOLO(MODEL_PATH)
model.to(DEVICE)

print("\n✅ Model loaded")
print("📌 Model classes:", model.names)
print("📌 Using device:", DEVICE, "\n")

# ===============================
# TEMPORAL TRACKER
# ===============================
class SmokeTemporalValidator:
    def __init__(self, maxlen=TEMPORAL_WINDOW):
        self.history = deque(maxlen=maxlen)

    def update(self, area):
        self.history.append(area)

    def confirmed(self):
        if len(self.history) < SMOKE_CONFIRM_FRAMES:
            return False
        return self.history[-1] >= self.history[0]
    
    def reset(self):
        self.history.clear()

smoke_tracker = SmokeTemporalValidator()

# ===============================
# SHAPE ANALYSIS
# ===============================
def smoke_shape_score(cnt):
    area = cv2.contourArea(cnt)
    if area < MIN_SMOKE_AREA:
        return 0.0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    perimeter = cv2.arcLength(cnt, True)
    complexity = (perimeter ** 2) / (4 * np.pi * area + 1e-6)

    return (1 - solidity) * 0.6 + min(complexity / 8, 1.0) * 0.4

# ===============================
# MASK REFINEMENT
# ===============================
def refine_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = (mask > MASK_THRESHOLD).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    return mask

# ===============================
# VIDEO INPUT
# ===============================
cap = cv2.VideoCapture(0)
prev_time = time.time()

print("🚀 Starting smoke + fire detection (YOLO-based)...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

    # ===============================
    # YOLO INFERENCE ONLY
    # ===============================
    results = model(
        frame,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=DEVICE,
        verbose=False
    )

    result = results[0]
    smoke_detected = False
    fire_detected = False

    # ===============================
    # PROCESS DETECTIONS
    # ===============================
    if result.boxes is not None:
        boxes = result.boxes.data.cpu().numpy()

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
        else:
            masks = []

        for i, box in enumerate(boxes):
            cls_id = int(box[5])
            conf = box[4]
            cls_name = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box[:4])

            print(f"➡ Detected: {cls_name} | conf={conf:.2f}")

            # ===============================
            # SMOKE LOGIC
            # ===============================
            if "smoke" in cls_name.lower():
                if i < len(masks):
                    raw_mask = masks[i]
                    refined = refine_mask(raw_mask)
                    
                    if refined.shape[:2] != frame.shape[:2]:
                        refined = cv2.resize(refined, (frame.shape[1], frame.shape[0]), 
                                           interpolation=cv2.INTER_NEAREST)

                    contours, _ = cv2.findContours(
                        refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        score = smoke_shape_score(cnt)
                        smoke_tracker.update(area)

                        if score > 0.4 and smoke_tracker.confirmed():
                            smoke_detected = True
                            overlay = frame.copy()
                            overlay[refined == 1] = (0, 255, 255)
                            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                            cv2.putText(frame, f"SMOKE {conf:.2f}", (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    # Fallback: bounding box only
                    if conf > 0.25:
                        smoke_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, f"SMOKE {conf:.2f}", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # ===============================
            # FIRE LOGIC (More Sensitive)
            # ===============================
            if "fire" in cls_name.lower():
                if conf > 0.15:  # Lower threshold for better detection
                    fire_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame, f"FIRE {conf:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2
                    )

    # ===============================
    # ALERT TEXT
    # ===============================
    alert_y = 30
    
    if smoke_detected:
        cv2.putText(frame, "SMOKE DETECTED", (15, alert_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        alert_y += 30
        
    if fire_detected:
        cv2.putText(frame, "FIRE DETECTED", (15, alert_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        alert_y += 30

    # ===============================
    # STATUS BAR
    # ===============================
    status_color = (0, 255, 0)
    status_text = "Safe"
    
    if smoke_detected and fire_detected:
        status_color = (0, 0, 255)
        status_text = "DANGER: Smoke + Fire!"
    elif smoke_detected:
        status_color = (0, 255, 255)
        status_text = "Smoke Warning"
    elif fire_detected:
        status_color = (0, 128, 255)
        status_text = "Fire Warning"
    
    cv2.rectangle(frame, (0, FRAME_SIZE - 35), (FRAME_SIZE, FRAME_SIZE), (30, 30, 30), -1)
    cv2.putText(frame, f"Status: {status_text}", (10, FRAME_SIZE - 12),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # ===============================
    # FPS
    # ===============================
    fps = 1 / (time.time() - prev_time + 1e-6)
    prev_time = time.time()

    cv2.putText(frame, f"FPS: {int(fps)}", (FRAME_SIZE - 70, FRAME_SIZE - 12),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Smoke + Fire Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
