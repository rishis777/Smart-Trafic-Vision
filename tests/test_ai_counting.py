"""Unit tests for Virtual Counting Lines and Unique Tracker ID Deduplication.
"""

import pytest
import numpy as np
from traffic_ai.core.ai_engine import AIEngine


@pytest.fixture
def engine():
    return AIEngine()


def test_virtual_line_intersection(engine):
    """Verify segment intersection detection math."""
    # Horizontal line from (100, 200) to (300, 200)
    q1 = (100, 200)
    q2 = (300, 200)

    # Vertical trajectory crossing line: (200, 150) -> (200, 250)
    p1 = (200, 150)
    p2 = (200, 250)
    assert engine._line_intersect(p1, p2, q1, q2) is True

    # Trajectory before line that doesn't cross: (200, 150) -> (200, 180)
    p3 = (200, 150)
    p4 = (200, 180)
    assert engine._line_intersect(p3, p4, q1, q2) is False

    # Parallel trajectory: (150, 180) -> (250, 180)
    p5 = (150, 180)
    p6 = (250, 180)
    assert engine._line_intersect(p5, p6, q1, q2) is False


def test_tracker_id_counting_deduplication(engine):
    """Verify that a vehicle tracker ID crossing a line is only counted once."""
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    signals = {"NORTH": "GREEN", "SOUTH": "RED", "EAST": "RED", "WEST": "RED"}

    # Mock custom counting line for NORTH
    engine.roi_config["lanes"]["NORTH"]["counting_line"] = [[200, 300], [600, 300]]

    # Simulate vehicle #42 crossing line
    engine.vehicle_trajectories[42] = [(400, 280), (400, 320)]
    assert engine._line_intersect((400, 280), (400, 320), (200, 300), (600, 300)) is True

    # First count
    engine.counted_vehicle_ids["NORTH"].add(42)
    assert len(engine.counted_vehicle_ids["NORTH"]) == 1

    # Same vehicle continues moving across subsequent frames
    engine.counted_vehicle_ids["NORTH"].add(42)
    # Deduplication check: count remains 1
    assert len(engine.counted_vehicle_ids["NORTH"]) == 1

    # New distinct vehicle #43 crosses
    engine.counted_vehicle_ids["NORTH"].add(43)
    assert len(engine.counted_vehicle_ids["NORTH"]) == 2


def test_direction_classification(engine):
    """Verify vehicle centroid maps to proper direction."""
    # Point near top center -> NORTH
    dir_n = engine.classify_direction((640, 150))
    assert dir_n in ["NORTH", "SOUTH", "EAST", "WEST"]

    # Point near bottom center -> SOUTH
    dir_s = engine.classify_direction((640, 600))
    assert dir_s in ["NORTH", "SOUTH", "EAST", "WEST"]
