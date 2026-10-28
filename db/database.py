"""Database manager providing init, save_data, retrieve_query, backup, and seeding.
"""

import shutil
from datetime import datetime, timedelta
import random
from pathlib import Path
from typing import Any, List, Optional, Type
from flask import Flask
from traffic_ai.db.models import db, User, Vehicle, TrafficData, Alert, Incident, SignalLog, Violation, Report
from traffic_ai.config import DB_PATH, DATA_DIR, DIRECTIONS


class Database:
    """Encapsulates database operations according to the project UML design."""

    def __init__(self, app: Optional[Flask] = None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask) -> None:
        self.app = app
        db.init_app(app)
        with app.app_context():
            db.create_all()
            self.seed_default_users()

    def save_data(self, record: db.Model) -> db.Model:
        """Persist a single model instance or list of instances."""
        try:
            if isinstance(record, (list, tuple)):
                db.session.add_all(record)
            else:
                db.session.add(record)
            db.session.commit()
            return record
        except Exception as e:
            db.session.rollback()
            raise e

    def retrieve_query(
        self,
        model: Type[db.Model],
        filters: Optional[dict] = None,
        limit: int = 100,
        order_by: Any = None
    ) -> List[Any]:
        """Query models with optional filters, ordering, and limit."""
        query = model.query
        if filters:
            for attr, value in filters.items():
                if hasattr(model, attr):
                    query = query.filter(getattr(model, attr) == value)
        if order_by is not None:
            query = query.order_by(order_by)
        else:
            if hasattr(model, "timestamp"):
                query = query.order_by(model.timestamp.desc())
            elif hasattr(model, "id"):
                query = query.order_by(model.id.desc())
        if limit:
            query = query.limit(limit)
        return query.all()

    def backup(self, destination_path: Optional[str] = None) -> str:
        """Create a backup of the SQLite database file."""
        if not destination_path:
            timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_dir = DATA_DIR / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            destination_path = str(backup_dir / f"traffic_system_backup_{timestamp_str}.db")

        if Path(DB_PATH).exists():
            shutil.copy2(DB_PATH, destination_path)
            return destination_path
        raise FileNotFoundError(f"Database file not found at {DB_PATH}")

    def seed_default_users(self) -> None:
        """Ensure standard role-based accounts exist."""
        default_accounts = [
            ("admin", "admin@traffic.local", "Admin@123", "admin"),
            ("operator", "operator@traffic.local", "Operator@123", "control_operator"),
            ("police", "police@traffic.local", "Police@123", "traffic_police"),
            ("authority", "authority@traffic.local", "Authority@123", "authority"),
            ("citizen", "citizen@traffic.local", "Citizen@123", "road_user"),
        ]

        for username, email, password, role in default_accounts:
            existing = User.query.filter_by(username=username).first()
            if not existing:
                u = User(username=username, email=email, role=role)
                u.set_password(password)
                db.session.add(u)
        db.session.commit()

    def seed_demo_data(self, days: int = 3) -> None:
        """Generate realistic historical traffic data and signal logs for demo."""
        # Check if already seeded
        if TrafficData.query.first():
            return

        now = datetime.utcnow()
        start_time = now - timedelta(days=days)
        current = start_time

        traffic_records = []
        signal_records = []

        density_levels = ["LOW", "MODERATE", "HIGH", "SEVERE"]
        phase_names = ["NORTH", "EAST", "SOUTH", "WEST"]

        # Step by 15-minute intervals
        while current < now:
            hour = current.hour
            # Morning & evening rush hours
            is_rush = (8 <= hour <= 10) or (17 <= hour <= 20)

            for d in DIRECTIONS:
                base_count = random.randint(15, 35) if is_rush else random.randint(3, 14)
                if d in ["NORTH", "SOUTH"]:
                    base_count += random.randint(2, 6)

                max_capacity = 40
                density_val = min(1.0, round(base_count / max_capacity + random.uniform(-0.05, 0.08), 3))
                density_val = max(0.05, density_val)

                if density_val < 0.25:
                    d_level = "LOW"
                    avg_speed = random.uniform(45.0, 60.0)
                elif density_val < 0.55:
                    d_level = "MODERATE"
                    avg_speed = random.uniform(30.0, 45.0)
                elif density_val < 0.80:
                    d_level = "HIGH"
                    avg_speed = random.uniform(15.0, 30.0)
                else:
                    d_level = "SEVERE"
                    avg_speed = random.uniform(5.0, 15.0)

                queue_len = int(base_count * random.uniform(0.3, 0.7))

                td = TrafficData(
                    timestamp=current,
                    intersection_id="INT_01",
                    direction=d,
                    vehicle_count=base_count,
                    density=density_val,
                    density_level=d_level,
                    avg_speed=avg_speed,
                    queue_length=queue_len
                )
                traffic_records.append(td)

            # Signal log entry for one of the phases
            active_p = random.choice(phase_names)
            green_t = random.randint(25, 55) if is_rush else random.randint(15, 30)
            wait_t = random.uniform(18.0, 42.0)
            sig = SignalLog(
                timestamp=current,
                intersection_id="INT_01",
                mode="ADAPTIVE",
                active_direction=active_p,
                green_time=green_t,
                yellow_time=4,
                reason="Adaptive density balancing",
                avg_wait_time=wait_t,
                queue_length=random.randint(4, 18)
            )
            signal_records.append(sig)

            # Occasionally add sample alerts/violations
            if random.random() < 0.05 and is_rush:
                alt = Alert(
                    timestamp=current,
                    intersection_id="INT_01",
                    direction=random.choice(DIRECTIONS),
                    alert_type="CONGESTION",
                    severity="HIGH",
                    message="Heavy queue formation detected exceeding 15 vehicles",
                    is_resolved=True,
                    resolved_by="operator",
                    resolved_at=current + timedelta(minutes=10)
                )
                db.session.add(alt)

            if random.random() < 0.03:
                viol = Violation(
                    timestamp=current,
                    intersection_id="INT_01",
                    violation_type=random.choice(["RED_LIGHT_JUMP", "ILLEGAL_PARKING", "WRONG_WAY"]),
                    vehicle_id=random.randint(100, 999),
                    vehicle_type=random.choice(["car", "motorcycle"]),
                    snapshot_path=None,
                    status="CONFIRMED" if random.random() < 0.7 else "PENDING_VERIFICATION",
                    notes="Automated CV trigger verified by officer"
                )
                db.session.add(viol)

            current += timedelta(minutes=15)

        db.session.add_all(traffic_records)
        db.session.add_all(signal_records)
        db.session.commit()
