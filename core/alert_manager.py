"""AlertManager: Manages operational alerts, incident classifications,
congestion timers, and alert lifecycle.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
from traffic_ai.config import CONGESTION_ALERT_DURATION_SEC, DIRECTIONS
from traffic_ai.db.models import Alert, Incident


class AlertSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    CONGESTION = "CONGESTION"
    EMERGENCY_VEHICLE = "EMERGENCY_VEHICLE"
    ACCIDENT = "ACCIDENT"
    VIOLATION = "VIOLATION"


class IncidentType(str, Enum):
    ACCIDENT = "ACCIDENT"
    VEHICLE_BREAKDOWN = "VEHICLE_BREAKDOWN"
    ROAD_OBSTRUCTION = "ROAD_OBSTRUCTION"


class AlertManager:
    """Detects alert conditions (like persistent congestion) and manages alert lifecycle."""

    def __init__(self, db_session=None):
        self.db_session = db_session
        self.congestion_timers: Dict[str, float] = {d: 0.0 for d in DIRECTIONS}
        self.congestion_alert_sent: Dict[str, bool] = {d: False for d in DIRECTIONS}
        self.active_alerts: List[Dict[str, Any]] = []

    def update_congestion_status(self, direction: str, density_level: str, dt: float) -> Optional[Dict[str, Any]]:
        """Track how long high/severe density has persisted and trigger alert when threshold met."""
        if direction not in self.congestion_timers:
            self.congestion_timers[direction] = 0.0
            self.congestion_alert_sent[direction] = False

        if density_level in ["HIGH", "SEVERE"]:
            self.congestion_timers[direction] += dt
            if self.congestion_timers[direction] >= CONGESTION_ALERT_DURATION_SEC and not self.congestion_alert_sent[direction]:
                self.congestion_alert_sent[direction] = True
                msg = f"Heavy congestion ({density_level}) on {direction} lane persisted for {int(self.congestion_timers[direction])}s"
                alert_dict = self.generate_alert(
                    alert_type=AlertType.CONGESTION.value,
                    severity=AlertSeverity.HIGH.value if density_level == "HIGH" else AlertSeverity.CRITICAL.value,
                    message=msg,
                    direction=direction
                )
                return alert_dict
        else:
            # Congestion cleared
            self.congestion_timers[direction] = 0.0
            self.congestion_alert_sent[direction] = False

        return None

    def generate_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        direction: str = "GENERAL",
        intersection_id: str = "INT_01"
    ) -> Dict[str, Any]:
        """Create and register an operational alert for operator review."""
        alert_payload = {
            "id": int(time.time() * 1000) % 1000000,
            "timestamp": datetime.utcnow().isoformat(),
            "intersection_id": intersection_id,
            "direction": direction,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "is_resolved": False
        }
        self.active_alerts.insert(0, alert_payload)
        # Keep recent 50
        if len(self.active_alerts) > 50:
            self.active_alerts.pop()
        return alert_payload

    def save_alert(self, alert_payload: Dict[str, Any]) -> Optional[Alert]:
        """Persist alert payload into SQLAlchemy database."""
        if not self.db_session:
            return None
        try:
            alert = Alert(
                intersection_id=alert_payload.get("intersection_id", "INT_01"),
                direction=alert_payload.get("direction", "GENERAL"),
                alert_type=alert_payload.get("alert_type", AlertType.CONGESTION.value),
                severity=alert_payload.get("severity", AlertSeverity.MEDIUM.value),
                message=alert_payload.get("message", ""),
                is_resolved=False
            )
            self.db_session.add(alert)
            self.db_session.commit()
            alert_payload["db_id"] = alert.id
            return alert
        except Exception:
            self.db_session.rollback()
            return None

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Return currently unresolved alerts for real-time dashboard."""
        return [a for a in self.active_alerts if not a.get("is_resolved", False)]

    def resolve_alert(self, alert_id: int, resolved_by: str = "operator") -> bool:
        """Mark alert as acknowledged and resolved."""
        found = False
        for a in self.active_alerts:
            if a.get("id") == alert_id or a.get("db_id") == alert_id:
                a["is_resolved"] = True
                a["resolved_by"] = resolved_by
                found = True

        if self.db_session:
            try:
                db_alert = Alert.query.get(alert_id)
                if db_alert:
                    db_alert.is_resolved = True
                    db_alert.resolved_by = resolved_by
                    db_alert.resolved_at = datetime.utcnow()
                    self.db_session.commit()
                    found = True
            except Exception:
                self.db_session.rollback()
        return found
