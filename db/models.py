"""SQLAlchemy models for AI-Based Traffic Management System.

Covers User authentication, Vehicle tracking logs, periodic TrafficData,
Alerts, Incidents, SignalLogs, Violations, and Reports.
"""

from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User account model with Role-Based Access Control (RBAC)."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(32), nullable=False, default="road_user")
    # Roles: 'admin', 'traffic_police', 'control_operator', 'authority', 'road_user'
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def has_role(self, *allowed_roles: str) -> bool:
        return self.role in allowed_roles

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Vehicle(db.Model):
    """Individual vehicle tracking record."""
    __tablename__ = "vehicles"

    id = db.Column(db.Integer, primary_key=True)
    tracker_id = db.Column(db.Integer, nullable=False, index=True)
    vehicle_type = db.Column(db.String(32), nullable=False)  # car, motorcycle, bus, truck, ambulance
    direction = db.Column(db.String(16), default="NORTH")
    first_seen = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    avg_speed = db.Column(db.Float, default=0.0)
    intersection_id = db.Column(db.String(32), default="INT_01")

    def to_dict(self):
        return {
            "id": self.id,
            "tracker_id": self.tracker_id,
            "vehicle_type": self.vehicle_type,
            "direction": self.direction,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "avg_speed": round(self.avg_speed, 1),
            "intersection_id": self.intersection_id
        }


class TrafficData(db.Model):
    """Time-series traffic metrics aggregated periodically (every N seconds)."""
    __tablename__ = "traffic_data"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    intersection_id = db.Column(db.String(32), default="INT_01", index=True)
    direction = db.Column(db.String(16), default="NORTH")  # NORTH, SOUTH, EAST, WEST
    vehicle_count = db.Column(db.Integer, default=0)
    density = db.Column(db.Float, default=0.0)  # 0.0 to 1.0
    density_level = db.Column(db.String(16), default="LOW")  # LOW, MODERATE, HIGH, SEVERE
    avg_speed = db.Column(db.Float, default=0.0)
    queue_length = db.Column(db.Integer, default=0)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "intersection_id": self.intersection_id,
            "direction": self.direction,
            "vehicle_count": self.vehicle_count,
            "density": round(self.density, 3),
            "density_level": self.density_level,
            "avg_speed": round(self.avg_speed, 1),
            "queue_length": self.queue_length
        }


class Alert(db.Model):
    """Real-time operational alerts for human operator verification."""
    __tablename__ = "alerts"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    intersection_id = db.Column(db.String(32), default="INT_01")
    direction = db.Column(db.String(16), default="GENERAL")
    alert_type = db.Column(db.String(32), nullable=False)  # CONGESTION, EMERGENCY_VEHICLE, ACCIDENT, VIOLATION
    severity = db.Column(db.String(16), default="MEDIUM")  # LOW, MEDIUM, HIGH, CRITICAL
    message = db.Column(db.Text, nullable=False)
    is_resolved = db.Column(db.Boolean, default=False)
    resolved_by = db.Column(db.String(64), nullable=True)
    resolved_at = db.Column(db.DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "intersection_id": self.intersection_id,
            "direction": self.direction,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "is_resolved": self.is_resolved,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class Incident(db.Model):
    """Road incidents like accidents, vehicle breakdowns, or roadblocks."""
    __tablename__ = "incidents"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    intersection_id = db.Column(db.String(32), default="INT_01")
    incident_type = db.Column(db.String(32), nullable=False)  # ACCIDENT, VEHICLE_BREAKDOWN, ROAD_OBSTRUCTION
    severity = db.Column(db.String(16), default="HIGH")
    description = db.Column(db.Text, nullable=False)
    location_details = db.Column(db.String(128), default="Main Intersection")
    snapshot_path = db.Column(db.String(256), nullable=True)
    status = db.Column(db.String(32), default="REPORTED")  # REPORTED, VERIFIED, CLEARED, FALSE_ALARM

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "intersection_id": self.intersection_id,
            "incident_type": self.incident_type,
            "severity": self.severity,
            "description": self.description,
            "location_details": self.location_details,
            "snapshot_path": self.snapshot_path,
            "status": self.status
        }


class SignalLog(db.Model):
    """Log of every traffic signal state change and timing decision."""
    __tablename__ = "signal_logs"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    intersection_id = db.Column(db.String(32), default="INT_01")
    mode = db.Column(db.String(24), default="ADAPTIVE")  # ADAPTIVE, FIXED, MANUAL_OVERRIDE, EMERGENCY
    active_direction = db.Column(db.String(16), nullable=False)
    green_time = db.Column(db.Integer, nullable=False)
    yellow_time = db.Column(db.Integer, default=4)
    reason = db.Column(db.String(128), default="Calculated by density")
    operator_user = db.Column(db.String(64), nullable=True)
    avg_wait_time = db.Column(db.Float, default=0.0)
    queue_length = db.Column(db.Integer, default=0)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "intersection_id": self.intersection_id,
            "mode": self.mode,
            "active_direction": self.active_direction,
            "green_time": self.green_time,
            "yellow_time": self.yellow_time,
            "reason": self.reason,
            "operator_user": self.operator_user,
            "avg_wait_time": round(self.avg_wait_time, 1),
            "queue_length": self.queue_length
        }


class Violation(db.Model):
    """Traffic violations detected by AI for human officer verification."""
    __tablename__ = "violations"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    intersection_id = db.Column(db.String(32), default="INT_01")
    violation_type = db.Column(db.String(32), nullable=False)  # RED_LIGHT_JUMP, WRONG_WAY, ILLEGAL_PARKING
    vehicle_id = db.Column(db.Integer, nullable=True)
    vehicle_type = db.Column(db.String(32), default="car")
    snapshot_path = db.Column(db.String(256), nullable=True)
    status = db.Column(db.String(32), default="PENDING_VERIFICATION")  # PENDING_VERIFICATION, CONFIRMED, DISMISSED
    notes = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "intersection_id": self.intersection_id,
            "violation_type": self.violation_type,
            "vehicle_id": self.vehicle_id,
            "vehicle_type": self.vehicle_type,
            "snapshot_path": self.snapshot_path,
            "status": self.status,
            "notes": self.notes
        }


class Report(db.Model):
    """Saved metadata for generated PDF and CSV reports."""
    __tablename__ = "reports"

    id = db.Column(db.Integer, primary_key=True)
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
    title = db.Column(db.String(128), nullable=False)
    report_type = db.Column(db.String(32), nullable=False)
    file_path = db.Column(db.String(256), nullable=False)
    file_format = db.Column(db.String(8), default="PDF")  # PDF, CSV
    generated_by = db.Column(db.String(64), default="system")

    def to_dict(self):
        return {
            "id": self.id,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "title": self.title,
            "report_type": self.report_type,
            "file_path": self.file_path,
            "file_format": self.file_format,
            "generated_by": self.generated_by
        }
