"""AccidentDetector: Detects vehicle collisions and abrupt decelerations using
spatial IoU overlap and velocity drop heuristics.
"""

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from traffic_ai.config import SNAPSHOTS_DIR, ACCIDENT_OVERLAP_IOU, ACCIDENT_SUDDEN_STOP_SPEED


class AccidentDetector:
    """Heuristic computer-vision engine for detecting collision and breakdown events."""

    def __init__(self, snapshots_dir: Optional[Path] = None):
        self.snapshots_dir = snapshots_dir or SNAPSHOTS_DIR
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        # tracker_id -> list of recent speeds [(timestamp, speed_px_s), ...]
        self.speed_history: Dict[int, List[Tuple[float, float]]] = {}
        # Avoid repeat alerts for same vehicle collision
        self.reported_incidents: Dict[str, float] = {}

    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Compute Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes."""
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter_width = max(0, xB - xA)
        inter_height = max(0, yB - yA)
        inter_area = inter_width * inter_height

        box1_area = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        box2_area = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))

        iou = inter_area / float(box1_area + box2_area - inter_area)
        return float(iou)

    def update_speed(self, tracker_id: int, current_speed: float, timestamp: float) -> None:
        """Record vehicle speed history for abrupt deceleration checks."""
        if tracker_id not in self.speed_history:
            self.speed_history[tracker_id] = []
        hist = self.speed_history[tracker_id]
        hist.append((timestamp, current_speed))
        # Keep last 10 entries
        if len(hist) > 10:
            hist.pop(0)

    def detect_sudden_stop(
        self,
        tracker_id: int,
        vehicle_type: str,
        current_speed: float,
        location: Tuple[int, int],
        frame: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """Detect a dangerous abrupt halt from cruising speed."""
        hist = self.speed_history.get(tracker_id, [])
        if len(hist) < 4:
            return None

        key = f"stop_{tracker_id}"
        if key in self.reported_incidents:
            return None

        # Check speed 3 frames ago vs current
        prev_speed = hist[-3][1]
        if prev_speed > 18.0 and current_speed < ACCIDENT_SUDDEN_STOP_SPEED:
            self.reported_incidents[key] = datetime.utcnow().timestamp()
            snapshot_path = self._save_snapshot(frame, f"abrupt_stop_{tracker_id}")
            return {
                "incident_type": "VEHICLE_BREAKDOWN",
                "severity": "MEDIUM",
                "description": f"Vehicle #{tracker_id} ({vehicle_type}) experienced sudden stop (speed dropped {prev_speed:.1f} -> {current_speed:.1f} px/s).",
                "location_details": f"Coordinates ({location[0]}, {location[1]})",
                "snapshot_path": snapshot_path
            }
        return None

    def detect_collision(
        self,
        detected_vehicles: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """Detect vehicle-vehicle collisions using spatial bounding box overlap."""
        n = len(detected_vehicles)
        if n < 2:
            return None

        for i in range(n):
            v1 = detected_vehicles[i]
            box1 = v1.get("box", [])
            id1 = v1.get("tracker_id", i)

            for j in range(i + 1, n):
                v2 = detected_vehicles[j]
                box2 = v2.get("box", [])
                id2 = v2.get("tracker_id", j)

                if len(box1) == 4 and len(box2) == 4:
                    iou = self._calculate_iou(box1, box2)
                    pair_key = f"col_{min(id1, id2)}_{max(id1, id2)}"

                    if iou >= ACCIDENT_OVERLAP_IOU and pair_key not in self.reported_incidents:
                        self.reported_incidents[pair_key] = datetime.utcnow().timestamp()
                        snapshot_path = self._save_snapshot(frame, f"collision_{id1}_{id2}")
                        return {
                            "incident_type": "ACCIDENT",
                            "severity": "HIGH",
                            "description": f"Vehicle collision detected between #{id1} ({v1.get('vehicle_type', 'veh')}) and #{id2} ({v2.get('vehicle_type', 'veh')}) with IoU overlap {iou:.2f}.",
                            "location_details": f"Intersection Lane Area (IoU: {iou:.2f})",
                            "snapshot_path": snapshot_path
                        }
        return None

    def _save_snapshot(self, frame: Optional[np.ndarray], prefix: str) -> Optional[str]:
        if frame is None or frame.size == 0:
            return None
        try:
            ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:19]
            save_path = self.snapshots_dir / f"{prefix}_{ts_str}.jpg"

            annotated = frame.copy()
            cv2.rectangle(annotated, (10, 10), (520, 45), (0, 0, 220), -1)
            cv2.putText(annotated, "INCIDENT EVIDENCE SNAPSHOT", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imwrite(str(save_path), annotated)
            return str(save_path)
        except Exception:
            return None
