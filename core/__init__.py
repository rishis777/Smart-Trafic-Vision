"""Core traffic management modules."""
from traffic_ai.core.camera import Camera, Sensor
from traffic_ai.core.traffic_analyzer import TrafficAnalyzer
from traffic_ai.core.traffic_signal import TrafficSignal, SignalState, SignalMode
from traffic_ai.core.signal_controller import SignalController
from traffic_ai.core.ai_engine import AIEngine
from traffic_ai.core.alert_manager import AlertManager, AlertType, AlertSeverity
from traffic_ai.core.violation_detector import ViolationDetector
from traffic_ai.core.accident_detector import AccidentDetector
from traffic_ai.core.emergency_detector import EmergencyDetector

__all__ = [
    "Camera", "Sensor", "TrafficAnalyzer", "TrafficSignal", "SignalState", "SignalMode",
    "SignalController", "AIEngine", "AlertManager", "AlertType", "AlertSeverity",
    "ViolationDetector", "AccidentDetector", "EmergencyDetector"
]
