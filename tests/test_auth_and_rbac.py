"""Unit tests for Authentication and Role-Based Access Control (RBAC).
"""

import pytest
from traffic_ai.app import create_app, init_pipeline, runtime
from traffic_ai.db.models import db, User
from traffic_ai.services.auth_service import can_control_signals, can_view_reports, can_manage_users


@pytest.fixture
def app_instance():
    test_config = {
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "SECRET_KEY": "test-secret-key"
    }
    app = create_app(test_config)
    with app.app_context():
        # Initialize pipeline runtime dummy
        init_pipeline("mock")
        yield app


@pytest.fixture
def client(app_instance):
    return app_instance.test_client()


def test_user_password_hashing(app_instance):
    """Verify password hashing security."""
    with app_instance.app_context():
        u = User(username="test_officer", email="test@traffic.gov", role="traffic_police")
        u.set_password("SecurePass@2026")
        assert u.password_hash != "SecurePass@2026"
        assert u.check_password("SecurePass@2026") is True
        assert u.check_password("WrongPassword") is False


def test_role_permission_checks(app_instance):
    """Verify role capability matrices."""
    with app_instance.app_context():
        admin = User(username="admin_user", role="admin")
        operator = User(username="operator_user", role="control_operator")
        police = User(username="police_user", role="traffic_police")
        authority = User(username="auth_user", role="authority")
        citizen = User(username="citizen_user", role="road_user")

        # can_control_signals: admin, control_operator, traffic_police
        assert can_control_signals(admin) is True
        assert can_control_signals(operator) is True
        assert can_control_signals(police) is True
        assert can_control_signals(authority) is False
        assert can_control_signals(citizen) is False

        # can_manage_users: admin only
        assert can_manage_users(admin) is True
        assert can_manage_users(operator) is False
        assert can_manage_users(citizen) is False


def test_unauthenticated_override_blocked(client):
    """Unauthenticated users must not be able to send signal override commands."""
    res = client.post("/api/signal/override", json={"direction": "NORTH", "duration": 30})
    # API endpoints return 401 or redirect to login
    assert res.status_code in [401, 302]


def test_road_user_override_forbidden(client, app_instance):
    """Road users must receive 403 Forbidden on signal override endpoints."""
    with client.session_transaction() as sess:
        with app_instance.app_context():
            u = User.query.filter_by(username="citizen").first()
            sess["_user_id"] = str(u.id)

    res = client.post("/api/signal/override", json={"direction": "NORTH", "duration": 30})
    assert res.status_code == 403
    assert b"Forbidden" in res.data or b"lacks permission" in res.data


def test_operator_override_permitted(client, app_instance):
    """Control operators are authorized to force signal state."""
    with client.session_transaction() as sess:
        with app_instance.app_context():
            u = User.query.filter_by(username="operator").first()
            sess["_user_id"] = str(u.id)

    res = client.post("/api/signal/override", json={"direction": "EAST", "duration": 30})
    assert res.status_code == 200
    assert res.get_json()["status"] == "SUCCESS"
    assert res.get_json()["override_direction"] == "EAST"
