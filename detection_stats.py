"""
Detection Statistics and Logging Module
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import threading

class DetectionStats:
    """Track and store detection statistics."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent / "detections.db"
        self.lock = threading.Lock()
        self.stats = {
            "total_detections": 0,
            "fire_detections": 0,
            "smoke_detections": 0,
            "by_camera": defaultdict(int),
            "by_hour": defaultdict(int),
            "recent_detections": []
        }
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for detection logging."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camera_index INTEGER,
                    detection_type TEXT NOT NULL,
                    confidence REAL,
                    zone TEXT,
                    image_path TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: Could not initialize database: {e}")
    
    def log_detection(
        self,
        detection_type: str,
        camera_index: int = 0,
        confidence: float = 0.0,
        zone: str = "unknown",
        image_path: Optional[str] = None
    ):
        """Log a detection event."""
        with self.lock:
            timestamp = datetime.now().isoformat()
            
            # Update in-memory stats
            self.stats["total_detections"] += 1
            if "fire" in detection_type.lower():
                self.stats["fire_detections"] += 1
            if "smoke" in detection_type.lower():
                self.stats["smoke_detections"] += 1
            
            self.stats["by_camera"][camera_index] += 1
            hour = datetime.now().hour
            self.stats["by_hour"][hour] += 1
            
            # Add to recent detections (keep last 50)
            detection_record = {
                "timestamp": timestamp,
                "type": detection_type,
                "camera": camera_index,
                "confidence": confidence,
                "zone": zone
            }
            self.stats["recent_detections"].append(detection_record)
            if len(self.stats["recent_detections"]) > 50:
                self.stats["recent_detections"].pop(0)
            
            # Store in database
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO detections 
                    (timestamp, camera_index, detection_type, confidence, zone, image_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (timestamp, camera_index, detection_type, confidence, zone, image_path))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Warning: Could not log to database: {e}")
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        with self.lock:
            return {
                "total_detections": self.stats["total_detections"],
                "fire_detections": self.stats["fire_detections"],
                "smoke_detections": self.stats["smoke_detections"],
                "by_camera": dict(self.stats["by_camera"]),
                "by_hour": dict(self.stats["by_hour"]),
                "recent_detections": self.stats["recent_detections"][-10:]  # Last 10
            }
    
    def get_recent_detections(self, limit: int = 20) -> List[Dict]:
        """Get recent detections from database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, camera_index, detection_type, confidence, zone
                FROM detections
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "timestamp": row[0],
                    "camera": row[1],
                    "type": row[2],
                    "confidence": row[3],
                    "zone": row[4]
                }
                for row in rows
            ]
        except Exception as e:
            print(f"Warning: Could not fetch detections: {e}")
            return []

# Global stats instance
stats = DetectionStats()

