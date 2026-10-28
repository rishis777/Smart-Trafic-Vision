"""Flask entry point for AI-Based Traffic Management System.

Runs the asynchronous AI computer-vision pipeline in a dedicated background
thread, manages state synchronization, and serves the REST API & modern web dashboard.
"""

import sys
import os
import time
import threading
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from flask import (
    Flask, render_template, request, jsonify, Response,
    redirect, url_for, flash, send_from_directory, abort
)
from flask_login import login_user, logout_user, login_required, current_user
import cv2

# Add package root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))

from traffic_ai import config
from traffic_ai.db.models import db, User, TrafficData, Alert, Incident, SignalLog, Violation, Report
from traffic_ai.db.database import Database
from traffic_ai.services.auth_service import (
    login_manager, role_required, can_control_signals, can_view_reports, can_manage_users
)
from traffic_ai.services.report_service import ReportService
from traffic_ai.services.prediction_service import PredictionService
from traffic_ai.core.camera import Camera
from traffic_ai.core.traffic_analyzer import TrafficAnalyzer
from traffic_ai.core.traffic_signal import TrafficSignal
from traffic_ai.core.signal_controller import SignalController
from traffic_ai.core.ai_engine import AIEngine
from traffic_ai.core.alert_manager import AlertManager


class PipelineRuntime:
    """Encapsulates singleton state for background AI worker and video streaming."""
    def __init__(self):
        self.camera: Optional[Camera] = None
        self.analyzer: Optional[TrafficAnalyzer] = None
        self.signal: Optional[TrafficSignal] = None
        self.controller: Optional[SignalController] = None
        self.ai_engine: Optional[AIEngine] = None
        self.alert_manager: Optional[AlertManager] = None
        self.report_service: Optional[ReportService] = None
        self.prediction_service: Optional[PredictionService] = None

        self.latest_jpeg_frame: Optional[bytes] = None
        self.frame_lock = threading.Lock()
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None

        self.telemetry_cache: Dict[str, Any] = {
            "total_active_vehicles": 0,
            "total_counted": 0,
            "average_density": 0.0,
            "direction_metrics": {},
            "type_totals": {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
            "signal": {},
            "emergency": {},
            "active_alerts": []
        }
        self.last_db_persist_time = 0.0


runtime = PipelineRuntime()


def init_pipeline(video_source: str | int = None) -> None:
    """Initialize CV engine, state machines, and machine learning components."""
    if video_source is None:
        demo_vid = config.SAMPLE_VIDEOS_DIR / "traffic_demo.mp4"
        real_vid = config.SAMPLE_VIDEOS_DIR / "real_traffic.mp4"
        if demo_vid.exists():
            video_source = str(demo_vid)
        elif real_vid.exists():
            video_source = str(real_vid)
        else:
            video_source = "mock"

    runtime.camera = Camera(source=video_source, width=1280, height=720)
    runtime.camera.start_stream()

    runtime.analyzer = TrafficAnalyzer()
    runtime.signal = TrafficSignal(initial_direction="NORTH")
    runtime.controller = SignalController(runtime.signal, runtime.analyzer)
    runtime.ai_engine = AIEngine(traffic_analyzer=runtime.analyzer)
    runtime.alert_manager = AlertManager()
    runtime.report_service = ReportService()
    runtime.prediction_service = PredictionService()

    try:
        runtime.prediction_service.train_baseline()
    except Exception:
        pass


def traffic_pipeline_worker(app: Flask) -> None:
    """Continuous AI Computer-Vision and Traffic Control pipeline execution loop."""
    print("[AI WORKER] Surveillance pipeline worker thread started.")
    last_loop_time = time.time()

    with app.app_context():
        runtime.alert_manager.db_session = db.session

        while runtime.is_running:
            now = time.time()
            dt = max(0.01, min(0.1, now - last_loop_time))
            last_loop_time = now

            ret, frame = runtime.camera.capture_video()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            transition_log = runtime.signal.update(dt)
            if transition_log:
                try:
                    s_log = SignalLog(
                        intersection_id="INT_01",
                        mode=transition_log.get("mode", "ADAPTIVE"),
                        active_direction=transition_log.get("active_direction", "NORTH"),
                        green_time=transition_log.get("green_time", 25),
                        yellow_time=4,
                        reason=transition_log.get("reason", "Adaptive density allocation")
                    )
                    db.session.add(s_log)
                    db.session.commit()
                except Exception:
                    db.session.rollback()

            emergency_clear_event = runtime.ai_engine.emergency_detector.check_auto_clear()
            if emergency_clear_event:
                runtime.signal.clear_emergency_corridor()

            signal_state = runtime.signal.get_current_state()
            if hasattr(runtime.camera, "set_signals"):
                runtime.camera.set_signals(signal_state["signals"])

            is_intersection = (runtime.camera is not None and getattr(runtime.camera, "use_synthetic", False))
            synth_boxes = None
            if hasattr(runtime.camera, "get_active_vehicle_boxes") and runtime.camera.use_synthetic:
                synth_boxes = runtime.camera.get_active_vehicle_boxes()

            analysis = runtime.ai_engine.analyze_frame(
                frame=frame,
                current_signals=signal_state["signals"],
                dt=dt,
                synthetic_detections=synth_boxes,
                draw_intersection_overlay=is_intersection
            )

            annotated_frame = analysis["annotated_frame"]
            direction_metrics = analysis["direction_metrics"]
            violations = analysis["violations"]
            incidents = analysis["incidents"]
            emergency_ev = analysis["emergency_event"]

            if emergency_ev:
                corridor_dir = emergency_ev.get("direction", "NORTH")
                runtime.signal.set_emergency_corridor(corridor_dir)
                runtime.alert_manager.generate_alert(
                    alert_type="EMERGENCY_VEHICLE",
                    severity="CRITICAL",
                    message=emergency_ev.get("message", "Emergency Vehicle Preemption"),
                    direction=corridor_dir
                )

            runtime.controller.generate_signal_decision(direction_metrics)

            for d, m in direction_metrics.items():
                dens_lvl = m.get("density_level", "LOW")
                cong_alert = runtime.alert_manager.update_congestion_status(d, dens_lvl, dt)
                if cong_alert:
                    runtime.alert_manager.save_alert(cong_alert)

            for viol in violations:
                try:
                    v_record = Violation(
                        intersection_id="INT_01",
                        violation_type=viol["violation_type"],
                        vehicle_id=viol["vehicle_id"],
                        vehicle_type=viol["vehicle_type"],
                        snapshot_path=viol.get("snapshot_path"),
                        status="PENDING_VERIFICATION",
                        notes=viol.get("notes", "")
                    )
                    db.session.add(v_record)
                    db.session.commit()
                except Exception:
                    db.session.rollback()

            for inc in incidents:
                try:
                    i_record = Incident(
                        intersection_id="INT_01",
                        incident_type=inc["incident_type"],
                        severity=inc["severity"],
                        description=inc["description"],
                        location_details=inc.get("location_details", "Lane"),
                        snapshot_path=inc.get("snapshot_path"),
                        status="REPORTED"
                    )
                    db.session.add(i_record)
                    db.session.commit()
                except Exception:
                    db.session.rollback()

            if now - runtime.last_db_persist_time >= 4.0:
                runtime.last_db_persist_time = now
                try:
                    for d, m in direction_metrics.items():
                        td = TrafficData(
                            timestamp=datetime.utcnow(),
                            intersection_id="INT_01",
                            direction=d,
                            vehicle_count=m.get("count", 0),
                            density=m.get("density", 0.0),
                            density_level=m.get("density_level", "LOW"),
                            avg_speed=35.0,
                            queue_length=int(m.get("count", 0) * 0.7)
                        )
                        db.session.add(td)
                    db.session.commit()
                except Exception:
                    db.session.rollback()

            total_active = len(analysis["active_vehicles"])
            total_counted = sum(m.get("cumulative_counted", 0) for m in direction_metrics.values())
            avg_dens = sum(m.get("density", 0.0) for m in direction_metrics.values()) / max(1, len(direction_metrics))

            type_totals = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
            for m in direction_metrics.values():
                b = m.get("type_breakdown", {})
                for k in type_totals:
                    type_totals[k] += b.get(k, 0)

            runtime.telemetry_cache = {
                "total_active_vehicles": total_active,
                "total_counted": total_counted,
                "average_density": round(float(avg_dens), 3),
                "direction_metrics": direction_metrics,
                "type_totals": type_totals,
                "signal": runtime.signal.get_current_state(),
                "emergency": runtime.ai_engine.emergency_detector.get_status(),
                "active_alerts": runtime.alert_manager.get_active_alerts()
            }

            ret_enc, jpeg_bytes = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret_enc:
                with runtime.frame_lock:
                    runtime.latest_jpeg_frame = jpeg_bytes.tobytes()

            time.sleep(0.015)


def generate_mjpeg_stream():
    """Generator yielding multipart JPEG frames for video element."""
    while True:
        with runtime.frame_lock:
            frame_bytes = runtime.latest_jpeg_frame

        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        time.sleep(0.033)


def register_routes(app: Flask) -> None:
    """Register all view and API endpoints on the Flask application."""

    @app.route("/")
    def public_view():
        return render_template("public.html")

    @app.route("/dashboard")
    @login_required
    def dashboard_view():
        return render_template("dashboard.html")

    @app.route("/reports")
    @login_required
    @role_required("admin", "authority", "control_operator", "traffic_police")
    def reports_view():
        report_data = runtime.report_service.generate_report_data(days=3) if runtime.report_service else {}
        hotspots = runtime.prediction_service.get_hotspots() if runtime.prediction_service else []
        return render_template("reports.html", report_data=report_data, hotspots=hotspots)

    @app.route("/violations")
    @login_required
    @role_required("admin", "control_operator", "traffic_police", "authority")
    def violations_view():
        violations = Violation.query.order_by(Violation.timestamp.desc()).limit(100).all()
        return render_template("violations.html", violations=violations)

    @app.route("/admin")
    @login_required
    @role_required("admin")
    def admin_view():
        users = User.query.order_by(User.id.asc()).all()
        return render_template("admin.html", users=users)

    @app.route("/login", methods=["GET", "POST"])
    def login_view():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard_view"))

        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")

            user = User.query.filter_by(username=username).first()
            if user and user.check_password(password):
                login_user(user)
                flash(f"Welcome back, {user.username} ({user.role})!", "success")
                next_page = request.args.get("next")
                return redirect(next_page or url_for("dashboard_view"))
            else:
                flash("Invalid username or password credentials.", "danger")

        return render_template("login.html")

    @app.route("/logout")
    @login_required
    def logout_view():
        logout_user()
        flash("You have been signed out successfully.", "info")
        return redirect(url_for("public_view"))

    @app.route("/admin/create_user", methods=["POST"])
    @login_required
    @role_required("admin")
    def admin_create_user():
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        role = request.form.get("role", "control_operator")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("admin_view"))

        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("User with that username or email already exists.", "warning")
            return redirect(url_for("admin_view"))

        new_user = User(username=username, email=email, role=role)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash(f"User '{username}' provisioned with role '{role}'.", "success")
        return redirect(url_for("admin_view"))

    @app.route("/admin/backup", methods=["POST"])
    @login_required
    @role_required("admin")
    def admin_backup():
        try:
            dest = Database().backup()
            flash(f"Database backup created successfully: {dest}", "success")
        except Exception as e:
            flash(f"Backup failed: {e}", "danger")
        return redirect(url_for("admin_view"))

    @app.route("/snapshot/<path:filename>")
    def serve_snapshot(filename):
        return send_from_directory(str(config.SNAPSHOTS_DIR), filename)

    @app.route("/video_feed")
    def video_feed():
        return Response(
            generate_mjpeg_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @app.route("/api/traffic/live")
    def api_traffic_live():
        return jsonify(runtime.telemetry_cache)

    @app.route("/api/traffic/history")
    @login_required
    def api_traffic_history():
        records = TrafficData.query.order_by(TrafficData.timestamp.desc()).limit(120).all()
        return jsonify([r.to_dict() for r in reversed(records)])

    @app.route("/api/signal/override", methods=["POST"])
    @login_required
    @role_required("admin", "control_operator", "traffic_police")
    def api_signal_override():
        data = request.get_json() or {}
        direction = data.get("direction", "NORTH")
        duration = int(data.get("duration", 30))
        reason = data.get("reason", f"Manual command by {current_user.username}")

        success = runtime.signal.set_manual_override(direction, reason=reason)
        if success:
            return jsonify({"status": "SUCCESS", "override_direction": direction, "duration": duration})
        return jsonify({"error": "Invalid direction or override active"}), 400

    @app.route("/api/signal/clear_override", methods=["POST"])
    @login_required
    @role_required("admin", "control_operator", "traffic_police")
    def api_signal_clear_override():
        runtime.signal.clear_manual_override()
        return jsonify({"status": "CLEARED"})

    @app.route("/api/signal/mode", methods=["POST"])
    @login_required
    @role_required("admin", "control_operator", "traffic_police")
    def api_signal_mode():
        data = request.get_json() or {}
        mode = data.get("mode", "ADAPTIVE")
        success = runtime.signal.set_mode(mode)
        if success:
            return jsonify({"status": "SUCCESS", "mode": mode})
        return jsonify({"error": "Invalid mode specification"}), 400

    @app.route("/api/emergency/trigger", methods=["POST"])
    @login_required
    @role_required("admin", "control_operator", "traffic_police")
    def api_emergency_trigger():
        data = request.get_json() or {}
        direction = data.get("direction", "NORTH")
        v_type = data.get("vehicle_type", "ambulance")

        event = runtime.ai_engine.emergency_detector.trigger_emergency(direction, v_type)
        runtime.signal.set_emergency_corridor(direction)
        return jsonify({"status": "ACTIVATED", "event": event})

    @app.route("/api/emergency/clear", methods=["POST"])
    @login_required
    @role_required("admin", "control_operator", "traffic_police")
    def api_emergency_clear():
        runtime.ai_engine.emergency_detector.clear_emergency()
        runtime.signal.clear_emergency_corridor()
        return jsonify({"status": "CLEARED"})

    @app.route("/api/alerts/resolve/<int:alert_id>", methods=["POST"])
    @login_required
    @role_required("admin", "control_operator", "traffic_police")
    def api_resolve_alert(alert_id):
        success = runtime.alert_manager.resolve_alert(alert_id, resolved_by=current_user.username)
        return jsonify({"status": "RESOLVED" if success else "NOT_FOUND"})

    @app.route("/api/violations/verify/<int:violation_id>", methods=["POST"])
    @login_required
    @role_required("admin", "traffic_police", "control_operator")
    def api_verify_violation(violation_id):
        data = request.get_json() or {}
        new_status = data.get("status", "CONFIRMED")

        v = Violation.query.get(violation_id)
        if not v:
            return jsonify({"error": "Violation not found"}), 404

        v.status = new_status
        v.notes = f"{v.notes or ''} [Audited by {current_user.username} as {new_status}]"
        db.session.commit()
        return jsonify({"status": "UPDATED", "violation": v.to_dict()})

    @app.route("/api/reports/generate", methods=["POST"])
    @login_required
    @role_required("admin", "authority", "control_operator", "traffic_police")
    def api_reports_generate():
        data = request.get_json() or {}
        fmt = data.get("format", "pdf").lower()
        days = int(data.get("days", 3))

        report_data = runtime.report_service.generate_report_data(days=days)

        if fmt == "csv":
            filepath = runtime.report_service.export_csv(report_data)
        else:
            filepath = runtime.report_service.export_pdf(report_data)

        filename = Path(filepath).name
        return jsonify({
            "status": "GENERATED",
            "format": fmt,
            "filename": filename,
            "download_url": url_for("api_download_report", filename=filename)
        })

    @app.route("/api/camera/switch", methods=["POST"])
    @login_required
    def api_camera_switch():
        data = request.get_json() or {}
        cam_id = data.get("camera_id", "cam1")
        if cam_id == "cam1":
            p = config.SAMPLE_VIDEOS_DIR / "real_traffic.mp4"
            video_src = str(p) if p.exists() else str(config.SAMPLE_VIDEOS_DIR / "traffic_demo.mp4")
        elif cam_id == "cam2":
            video_src = str(config.SAMPLE_VIDEOS_DIR / "traffic_demo.mp4")
        else:
            video_src = 0

        with runtime.frame_lock:
            if runtime.camera:
                runtime.camera.stop_stream()
            if cam_id == "cam1":
                p = config.SAMPLE_VIDEOS_DIR / "real_traffic.mp4"
                video_src = str(p) if p.exists() else str(config.SAMPLE_VIDEOS_DIR / "traffic_demo.mp4")
                runtime.camera = Camera(source=video_src, width=1280, height=720)
            elif cam_id == "cam2":
                runtime.camera = Camera(source="mock", width=1280, height=720)
            elif cam_id == "cam3":
                runtime.camera = Camera(source=0, width=1280, height=720)
            else:
                runtime.camera = Camera(source="mock", width=1280, height=720)
            runtime.camera.start_stream()
        return jsonify({"status": "SUCCESS", "active_camera": cam_id})

    @app.route("/api/reports/download/<path:filename>")
    @login_required
    @role_required("admin", "authority", "control_operator", "traffic_police")
    def api_download_report(filename):
        return send_from_directory(str(config.EXPORTS_DIR), filename, as_attachment=True)

    @app.route("/api/prediction")
    def api_prediction():
        hour = int(request.args.get("hour", datetime.utcnow().hour))
        day = int(request.args.get("day", datetime.utcnow().weekday()))
        direction = request.args.get("direction", "NORTH")

        res = runtime.prediction_service.predict_congestion(direction, hour, day)
        return jsonify(res)


def create_app(test_config=None) -> Flask:
    """Application factory for Flask web application."""
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "web" / "templates"),
        static_folder=str(BASE_DIR / "web" / "static")
    )

    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["SQLALCHEMY_DATABASE_URI"] = config.SQLALCHEMY_DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = config.SQLALCHEMY_TRACK_MODIFICATIONS

    if test_config:
        app.config.update(test_config)

    Database(app)
    login_manager.init_app(app)
    register_routes(app)

    return app


app = create_app()


def start_background_pipeline(video_source: str | int = None) -> None:
    """Start the AI background worker thread safely."""
    init_pipeline(video_source)
    runtime.is_running = True
    runtime.worker_thread = threading.Thread(target=traffic_pipeline_worker, args=(app,), daemon=True)
    runtime.worker_thread.start()


def main():
    parser = argparse.ArgumentParser(description="AI-Based Traffic Management System Command Center")
    parser.add_argument("--demo", action="store_true", help="Launch in zero-setup demo mode with bundled video & seeded data")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file or webcam index")
    parser.add_argument("--port", type=int, default=5000, help="Web server listening port (default 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default 127.0.0.1)")
    args = parser.parse_args()

    video_src = args.video
    if args.demo or video_src is None:
        demo_vid = config.SAMPLE_VIDEOS_DIR / "traffic_demo.mp4"
        real_vid = config.SAMPLE_VIDEOS_DIR / "real_traffic.mp4"
        if not demo_vid.exists():
            from traffic_ai.data.generate_sample_video import generate_traffic_demo_video
            generate_traffic_demo_video()
        video_src = str(demo_vid) if demo_vid.exists() else "mock"

    print("=" * 65)
    print(" [TRAFFIC AI] AI-BASED TRAFFIC MANAGEMENT SYSTEM - RUNTIME INITIALIZED")
    print("=" * 65)
    print(f" * Video Input Source: {video_src}")
    print(f" * Database URI:      {config.SQLALCHEMY_DATABASE_URI}")
    print(f" * Dashboard URL:      http://{args.host}:{args.port}/dashboard")
    print(f" * Public Portal:      http://{args.host}:{args.port}/")
    print("=" * 65)

    with app.app_context():
        db_mgr = Database()
        db_mgr.seed_demo_data(days=3)

    start_background_pipeline(video_source=video_src)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
