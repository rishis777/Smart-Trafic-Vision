"""ViolationDetector: Detects red-light running, wrong-way driving,
and illegal parking in designated zones, capturing formal police evidence snapshots.
"""

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from traffic_ai.config import SNAPSHOTS_DIR, ILLEGAL_PARKING_TIME_SEC


class ViolationDetector:
    """Identifies traffic infractions using geometric and temporal heuristics."""

    def __init__(self, snapshots_dir: Optional[Path] = None):
        self.snapshots_dir = snapshots_dir or SNAPSHOTS_DIR
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.parked_vehicles: Dict[int, Dict[str, Any]] = {}
        self.flagged_violations: Dict[str, float] = {}

    def _line_intersect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        q1: Tuple[float, float],
        q2: Tuple[float, float]
    ) -> bool:
        """Check if line segment p1-p2 intersects line segment q1-q2."""
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

    def detect_red_light_jump(
        self,
        tracker_id: int,
        vehicle_type: str,
        trajectory: List[Tuple[int, int]],
        stop_line: List[List[int]],
        signal_state: str,
        frame: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """Detect if vehicle crossed stop line when light was RED."""
        if signal_state != "RED" or len(trajectory) < 2 or not stop_line:
            return None

        key = f"red_light_{tracker_id}"
        if key in self.flagged_violations:
            return None

        p1 = trajectory[-2]
        p2 = trajectory[-1]
        q1 = (stop_line[0][0], stop_line[0][1])
        q2 = (stop_line[1][0], stop_line[1][1])

        if self._line_intersect(p1, p2, q1, q2):
            self.flagged_violations[key] = datetime.now().timestamp()
            notes = f"Vehicle #{tracker_id} ({vehicle_type}) crossed stop-line while signal was RED."
            snapshot_path = self._save_evidence_snapshot(
                frame=frame,
                prefix=f"red_light_veh_{tracker_id}",
                infraction_title="RED LIGHT RUNNING VIOLATION",
                details=notes
            )
            return {
                "violation_type": "RED_LIGHT_JUMP",
                "vehicle_id": tracker_id,
                "vehicle_type": vehicle_type,
                "snapshot_path": snapshot_path,
                "notes": notes
            }
        return None

    def detect_wrong_way(
        self,
        tracker_id: int,
        vehicle_type: str,
        trajectory: List[Tuple[int, int]],
        allowed_direction: str,
        frame: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """Detect if vehicle is traveling opposite to designated traffic flow."""
        if len(trajectory) < 8:
            return None

        key = f"wrong_way_{tracker_id}"
        if key in self.flagged_violations:
            return None

        start_pt = trajectory[0]
        end_pt = trajectory[-1]
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]

        distance = np.hypot(dx, dy)
        if distance < 25:
            return None

        is_wrong = False
        if allowed_direction == "DOWN" and dy < -20:
            is_wrong = True
        elif allowed_direction == "UP" and dy > 20:
            is_wrong = True
        elif allowed_direction == "RIGHT" and dx < -20:
            is_wrong = True
        elif allowed_direction == "LEFT" and dx > 20:
            is_wrong = True

        if is_wrong:
            self.flagged_violations[key] = datetime.now().timestamp()
            notes = f"Vehicle #{tracker_id} driving opposite to {allowed_direction} lane flow (vector: dx={dx}, dy={dy})."
            snapshot_path = self._save_evidence_snapshot(
                frame=frame,
                prefix=f"wrong_way_veh_{tracker_id}",
                infraction_title="WRONG-WAY DRIVING INFRACTION",
                details=notes
            )
            return {
                "violation_type": "WRONG_WAY",
                "vehicle_id": tracker_id,
                "vehicle_type": vehicle_type,
                "snapshot_path": snapshot_path,
                "notes": notes
            }
        return None

    def detect_illegal_parking(
        self,
        tracker_id: int,
        vehicle_type: str,
        centroid: Tuple[int, int],
        speed: float,
        no_parking_zones: List[Dict[str, Any]],
        dt: float,
        frame: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """Detect if a vehicle is parked or stationary in a designated no-parking zone."""
        key = f"illegal_park_{tracker_id}"
        if key in self.flagged_violations:
            return None

        point = (int(centroid[0]), int(centroid[1]))
        in_zone = False
        target_zone_name = "Restricted Zone"

        for zone in no_parking_zones:
            poly = np.array(zone.get("polygon", []), dtype=np.int32)
            if len(poly) >= 3:
                if cv2.pointPolygonTest(poly, point, False) >= 0:
                    in_zone = True
                    target_zone_name = zone.get("name", "Restricted Zone")
                    break

        if in_zone and speed < 1.5:
            if tracker_id not in self.parked_vehicles:
                self.parked_vehicles[tracker_id] = {
                    "duration": 0.0,
                    "zone_name": target_zone_name
                }
            self.parked_vehicles[tracker_id]["duration"] += dt

            if self.parked_vehicles[tracker_id]["duration"] >= ILLEGAL_PARKING_TIME_SEC:
                self.flagged_violations[key] = datetime.now().timestamp()
                dur = int(self.parked_vehicles[tracker_id]["duration"])
                notes = f"Vehicle #{tracker_id} stationary in {target_zone_name} for {dur}s (threshold: {int(ILLEGAL_PARKING_TIME_SEC)}s)."
                snapshot_path = self._save_evidence_snapshot(
                    frame=frame,
                    prefix=f"parking_veh_{tracker_id}",
                    infraction_title="ILLEGAL PARKING INFRACTION",
                    details=notes
                )
                return {
                    "violation_type": "ILLEGAL_PARKING",
                    "vehicle_id": tracker_id,
                    "vehicle_type": vehicle_type,
                    "snapshot_path": snapshot_path,
                    "notes": notes
                }
        else:
            if tracker_id in self.parked_vehicles:
                del self.parked_vehicles[tracker_id]

        return None

    def _save_evidence_snapshot(
        self,
        frame: Optional[np.ndarray],
        prefix: str,
        infraction_title: str,
        details: str
    ) -> Optional[str]:
        """Save high-resolution police citation evidence snapshot."""
        if frame is None or frame.size == 0:
            return None
        try:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = f"{prefix}_{ts_str}.jpg"
            save_path = self.snapshots_dir / filename

            annotated = frame.copy()
            h, w = annotated.shape[:2]

            # Top Police Header Banner
            cv2.rectangle(annotated, (0, 0), (w, 60), (15, 23, 42), -1)
            cv2.line(annotated, (0, 60), (w, 60), (220, 38, 38), 3)

            date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header_text = f"TRAFFIC LAW ENFORCEMENT &bull; CITATION EVIDENCE RECORD"
            cv2.putText(annotated, header_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(annotated, f"TIMESTAMP: {date_str} | INTERSECTION: INT_01", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)

            # Bottom Infraction Detail Ribbon
            cv2.rectangle(annotated, (0, h - 55), (w, h), (15, 23, 42), -1)
            cv2.line(annotated, (0, h - 55), (w, h - 55), (220, 38, 38), 2)
            cv2.putText(annotated, f"INFRACTION: {infraction_title}", (20, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (244, 63, 94), 2)
            cv2.putText(annotated, details, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imwrite(str(save_path), annotated)
            return str(save_path)
        except Exception:
            return None
