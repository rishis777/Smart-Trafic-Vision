"""EmergencyDetector: Detects emergency vehicles (ambulance, fire, police)
and orchestrates green corridor preemption across intersections.
"""

from typing import Dict, List, Optional, Any
import time
from datetime import datetime
from traffic_ai.config import DIRECTIONS


class EmergencyDetector:
    """Manages emergency vehicle detection and priority green corridor lifecycle."""

    def __init__(self, auto_clear_timeout: float = 30.0):
        self.auto_clear_timeout = auto_clear_timeout
        self.is_active = False
        self.active_corridor_direction: Optional[str] = None
        self.emergency_vehicle_type: str = "ambulance"
        self.emergency_start_time: float = 0.0
        self.incident_history: List[Dict[str, Any]] = []

    def check_detected_vehicles(self, vehicles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Scan vision detections for emergency vehicle classifications or simulated tags."""
        for v in vehicles:
            v_type = str(v.get("vehicle_type", "")).lower()
            if v_type in ["ambulance", "fire_truck", "police", "emergency"]:
                direction = v.get("direction", "NORTH")
                return self.trigger_emergency(direction=direction, vehicle_type=v_type)
        return None

    def trigger_emergency(self, direction: str, vehicle_type: str = "ambulance") -> Dict[str, Any]:
        """Activate green corridor for specified corridor direction."""
        if direction not in DIRECTIONS:
            direction = "NORTH"

        self.is_active = True
        self.active_corridor_direction = direction
        self.emergency_vehicle_type = vehicle_type
        self.emergency_start_time = time.time()

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "EMERGENCY_CORRIDOR_ACTIVATED",
            "direction": direction,
            "vehicle_type": vehicle_type,
            "priority": "CRITICAL",
            "message": f"PRIORITY GREEN CORRIDOR activated for {vehicle_type.upper()} along {direction} corridor."
        }
        self.incident_history.insert(0, event)
        if len(self.incident_history) > 30:
            self.incident_history.pop()
        return event

    def check_auto_clear(self) -> Optional[Dict[str, Any]]:
        """Automatically release corridor after safety timeout (e.g. 30s) if vehicle passed."""
        if self.is_active:
            elapsed = time.time() - self.emergency_start_time
            if elapsed >= self.auto_clear_timeout:
                return self.clear_emergency()
        return None

    def clear_emergency(self) -> Dict[str, Any]:
        """Release corridor and return intersection to Normal Operation."""
        prev_direction = self.active_corridor_direction
        self.is_active = False
        self.active_corridor_direction = None

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "EMERGENCY_CORRIDOR_CLEARED",
            "direction": prev_direction,
            "message": f"Emergency corridor along {prev_direction} cleared. Resumed Normal Operation."
        }
        self.incident_history.insert(0, event)
        return event

    def get_status(self) -> Dict[str, Any]:
        """Return current emergency priority status."""
        elapsed = time.time() - self.emergency_start_time if self.is_active else 0.0
        remaining = max(0, int(self.auto_clear_timeout - elapsed)) if self.is_active else 0
        return {
            "is_emergency_active": self.is_active,
            "corridor_direction": self.active_corridor_direction,
            "vehicle_type": self.emergency_vehicle_type,
            "remaining_seconds": remaining,
            "elapsed_seconds": int(elapsed)
        }
