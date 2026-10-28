"""Services package."""
from traffic_ai.services.auth_service import login_manager, role_required, can_control_signals, can_view_reports, can_manage_users
from traffic_ai.services.report_service import ReportService
from traffic_ai.services.prediction_service import PredictionService

__all__ = [
    "login_manager", "role_required", "can_control_signals", "can_view_reports", "can_manage_users",
    "ReportService", "PredictionService"
]
