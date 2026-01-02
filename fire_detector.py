import base64
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from collections import deque

# ===============================
# HELPERS FROM Smoke Detection.py
# ===============================

class SmokeTemporalValidator:
    def __init__(self, maxlen=5):
        self.history = deque(maxlen=maxlen)

    def update(self, area):
        self.history.append(area)

    def confirmed(self):
        SMOKE_CONFIRM_FRAMES = 3
        if len(self.history) < SMOKE_CONFIRM_FRAMES:
            return False
        return self.history[-1] >= self.history[0]

def smoke_shape_score(cnt):
    MIN_SMOKE_AREA = 100
    area = cv2.contourArea(cnt)
    if area < MIN_SMOKE_AREA:
        return 0.0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    perimeter = cv2.arcLength(cnt, True)
    complexity = (perimeter ** 2) / (4 * np.pi * area + 1e-6)

    return (1 - solidity) * 0.6 + min(complexity / 8, 1.0) * 0.4

def refine_mask(mask):
    MASK_THRESHOLD = 0.25
    kernel = np.ones((3, 3), np.uint8)
    mask = (mask > MASK_THRESHOLD).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    return mask

class Detector:
    def __init__(
        self,
        model_path: Path,
        target_height: int = 512,
        iou_threshold: float = 0.4,
        min_confidence: float = 0.15,  # Higher for accuracy
        smoke_confidence: float = 0.25
        ):
        """
        Initialize the FireDetector with a YOLO model.
        Uses YOLO-only detection to avoid false positives.
        """
        self.logger = logging.getLogger(__name__)

        try:
            self.model = YOLO(str(model_path))
            self.target_height = target_height
            self.iou_threshold = iou_threshold
            self.min_confidence = min_confidence
            self.smoke_confidence = smoke_confidence
            self.names = self.model.model.names
            
            self.smoke_tracker = SmokeTemporalValidator()

            self.colors = {
                "fire": (0, 0, 255),
                "smoke": (0, 255, 255)
            }

            self.logger.info("Fire detector initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize fire detector: {e}")
            raise

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        new_width = int(self.target_height * aspect_ratio)
        return cv2.resize(frame, (new_width, self.target_height))

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """
        Process a video frame to detect fire and smoke using YOLO only.
        """
        try:
            frame = self.resize_frame(frame)
            results = self.model(
                frame, 
                iou=self.iou_threshold, 
                conf=self.min_confidence,
                verbose=False
            )
            detected_status = None
            
            if not results:
                return frame, None
                
            result = results[0]
            
            has_masks = result.masks is not None
            masks = result.masks.data.cpu().numpy() if has_masks else []
            boxes = result.boxes.data.cpu().numpy() if result.boxes is not None else []
            
            smoke_found = False
            fire_found = False

            for i, box in enumerate(boxes):
                cls_id = int(box[5])
                conf = box[4]
                class_name = self.names[cls_id]
                x1, y1, x2, y2 = map(int, box[:4])

                # ===============================
                # SMOKE LOGIC
                # ===============================
                if "smoke" in class_name.lower():
                    if has_masks and i < len(masks):
                        raw_mask = masks[i]
                        refined = refine_mask(raw_mask)
                        
                        if refined.shape[:2] != frame.shape[:2]:
                            refined = cv2.resize(refined, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                        contours, _ = cv2.findContours(
                            refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )

                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            score = smoke_shape_score(cnt)
                            
                            if score > 0.4:
                                self.smoke_tracker.update(area)
                                
                                if self.smoke_tracker.confirmed():
                                    smoke_found = True
                                    detected_status = "Smoke"
                                    
                                    overlay = frame.copy()
                                    overlay[refined == 1] = (0, 255, 255)
                                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                                    
                                    cv2.putText(frame, f"SMOKE {conf:.2f}", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        if conf > self.smoke_confidence:
                            smoke_found = True
                            detected_status = "Smoke"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cvzone.putTextRect(frame, f"Smoke {conf:.2f}", (x1, y1-10), scale=1, thickness=1, colorR=(0,255,255))

                # ===============================
                # FIRE LOGIC (More Sensitive)
                # ===============================
                elif "fire" in class_name.lower():
                    if conf > 0.10:  # Very sensitive fire detection
                        fire_found = True
                        if detected_status != "Smoke":
                            detected_status = "Fire"
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(
                            frame, f"FIRE {conf:.2f}",
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2
                        )

            # Determine final status
            if smoke_found and fire_found:
                detected_status = "Smoking"

            self._add_frame_info(frame, detected_status)

            return frame, detected_status

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return frame, None

    def _add_frame_info(self, frame: np.ndarray, detection: Optional[str]) -> None:
        height, width = frame.shape[:2]
        
        if detection == "Smoking":
            status_text = "Status: SMOKING DETECTED!"
            color = (0, 0, 255)
        elif detection == "Smoke":
            status_text = "Status: Smoke Warning"
            color = (0, 255, 255)
        elif detection == "Fire":
            status_text = "Status: Fire Warning"
            color = (0, 128, 255)
        else:
            status_text = "Status: Scanning..."
            color = (255, 255, 255)
        
        cvzone.putTextRect(frame, status_text, (10, height - 10), scale=1.5, thickness=2, colorR=(0,0,0), colorT=color)

    def detect_image(self, image: np.ndarray, return_image: bool = True) -> Dict[str, Any]:
        if image is None or image.size == 0:
            raise ValueError("Empty image provided for detection")

        processed_frame, detection = self.process_frame(image)
        response: Dict[str, Any] = {
            "detected": detection is not None,
            "detection": detection
        }

        if return_image:
            success, buffer = cv2.imencode(".jpg", processed_frame)
            if success:
                response["image_base64"] = base64.b64encode(buffer).decode("utf-8")
            else:
                self.logger.warning("Failed to encode processed frame for API response")

        return response

    def detect_image_bytes(self, image_bytes: bytes, return_image: bool = True) -> Dict[str, Any]:
        frame_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Unable to decode image bytes for detection")

        return self.detect_image(frame, return_image)
