"""Authentication and Role-Based Access Control (RBAC) service.

Provides Flask-Login setup, role decorators, and permission checking.
"""

from functools import wraps
from typing import Callable, List
from flask import redirect, url_for, flash, request
from flask_login import LoginManager, current_user
from traffic_ai.db.models import User

login_manager = LoginManager()
login_manager.login_view = "login_view"
login_manager.login_message = "Please sign in to access this page."
login_manager.login_message_category = "warning"


@login_manager.user_loader
def load_user(user_id: str) -> User | None:
    """Load user by database ID."""
    if not user_id:
        return None
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None


def role_required(*allowed_roles: str) -> Callable:
    """Decorator to enforce role permissions on view functions."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                if request.is_json or request.path.startswith("/api/"):
                    return {"error": "Authentication required"}, 401
                return redirect(url_for("login_view", next=request.url))

            user_role = getattr(current_user, "role", "")
            if user_role not in allowed_roles and user_role != "admin":
                if request.is_json or request.path.startswith("/api/"):
                    return {"error": f"Forbidden: role '{user_role}' lacks permission"}, 403
                flash(f"Access denied: Requires one of {allowed_roles}", "danger")
                return redirect(url_for("dashboard_view"))

            return fn(*args, **kwargs)
        return wrapper
    return decorator


def can_control_signals(user: User) -> bool:
    """Check if the given user can alter signal states or override."""
    if not user:
        return False
    return getattr(user, "role", None) in ["admin", "control_operator", "traffic_police"]


def can_view_reports(user: User) -> bool:
    """Check if user has access to analytics and report downloads."""
    if not user:
        return False
    return getattr(user, "role", None) in ["admin", "authority", "control_operator", "traffic_police"]


def can_manage_users(user: User) -> bool:
    """Check if user can edit users/cameras/system settings."""
    if not user:
        return False
    return getattr(user, "role", None) == "admin"
