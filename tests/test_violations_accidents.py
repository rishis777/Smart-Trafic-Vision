"""Unit tests for Violations (Red Light, Wrong Way, Parking) and Accident Heuristics.
"""

import pytest
import numpy as np
from traffic_ai.core.violation_detector import ViolationDetector
from traffic_ai.core.accident_detector import AccidentDetector


@pytest.fixture
def violation_det():
    return ViolationDetector()


@pytest.fixture
def accident_det():
    return AccidentDetector()


def test_red_light_jump_detection(violation_det):
    """Vehicle crossing stop line when light is RED must trigger violation."""
    stop_line = [[300, 500], [600, 500]]

    # Case 1: Light is RED and vehicle trajectory crosses from (450, 480) to (450, 520)
    viol = violation_det.detect_red_light_jump(
        tracker_id=101,
        vehicle_type="car",
        trajectory=[(450, 480), (450, 520)],
        stop_line=stop_line,
        signal_state="RED"
    )
    assert viol is not None
    assert viol["violation_type"] == "RED_LIGHT_JUMP"
    assert viol["vehicle_id"] == 101

    # Case 2: Light is GREEN and vehicle crosses -> NO violation
    viol_green = violation_det.detect_red_light_jump(
        tracker_id=102,
        vehicle_type="car",
        trajectory=[(450, 480), (450, 520)],
        stop_line=stop_line,
        signal_state="GREEN"
    )
    assert viol_green is None


def test_wrong_way_detection(violation_det):
    """Vehicle driving backwards or opposite to flow must trigger violation."""
    # Lane allows "DOWN" (y should increase)
    # Vehicle moving upwards from y=600 to y=400 (dy = -200)
    upward_trajectory = [(400, 600 - i * 20) for i in range(11)]

    viol = violation_det.detect_wrong_way(
        tracker_id=201,
        vehicle_type="motorcycle",
        trajectory=upward_trajectory,
        allowed_direction="DOWN"
    )
    assert viol is not None
    assert viol["violation_type"] == "WRONG_WAY"

    # Moving downward in "DOWN" lane -> normal flow, no violation
    downward_trajectory = [(400, 100 + i * 20) for i in range(11)]
    viol_normal = violation_det.detect_wrong_way(
        tracker_id=202,
        vehicle_type="car",
        trajectory=downward_trajectory,
        allowed_direction="DOWN"
    )
    assert viol_normal is None


def test_illegal_parking_duration(violation_det):
    """Vehicle stationary in no-parking polygon beyond limit triggers violation."""
    zones = [{
        "name": "Bus_Bay_No_Parking",
        "polygon": [[100, 100], [300, 100], [300, 300], [100, 300]]
    }]
    centroid = (200, 200)  # Inside polygon

    # First 5 seconds (below 10s threshold) -> no violation
    res1 = violation_det.detect_illegal_parking(301, "car", centroid, speed=0.2, no_parking_zones=zones, dt=5.0)
    assert res1 is None

    # Cumulative time reaches 11 seconds -> violation triggered
    res2 = violation_det.detect_illegal_parking(301, "car", centroid, speed=0.2, no_parking_zones=zones, dt=6.0)
    assert res2 is not None
    assert res2["violation_type"] == "ILLEGAL_PARKING"


def test_collision_detection_iou(accident_det):
    """Overlapping bounding boxes with high IoU trigger collision incident."""
    # Box 1: [100, 100, 200, 200]
    # Box 2: [120, 110, 210, 210] (high overlap)
    vehicles = [
        {"tracker_id": 401, "vehicle_type": "car", "box": [100, 100, 200, 200]},
        {"tracker_id": 402, "vehicle_type": "truck", "box": [120, 110, 210, 210]}
    ]

    incident = accident_det.detect_collision(vehicles)
    assert incident is not None
    assert incident["incident_type"] == "ACCIDENT"
    assert incident["severity"] == "HIGH"
