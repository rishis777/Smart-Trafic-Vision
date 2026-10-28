"""Unit tests for TrafficAnalyzer: density classification & green time recommendation.
"""

import pytest
from traffic_ai.core.traffic_analyzer import TrafficAnalyzer
from traffic_ai.config import DENSITY_THRESHOLDS, SIGNAL_TIMINGS


@pytest.fixture
def analyzer():
    return TrafficAnalyzer()


def test_density_classification_levels(analyzer):
    """Verify LOW, MODERATE, HIGH, and SEVERE classifications based on thresholds."""
    # 0 vehicles -> LOW
    d_val, d_lvl = analyzer.calculate_density(vehicle_count=0, max_capacity=20)
    assert d_val == 0.0
    assert d_lvl == "LOW"

    # 4 vehicles out of 20 (20%) -> LOW (< 25%)
    d_val, d_lvl = analyzer.calculate_density(vehicle_count=4, max_capacity=20)
    assert d_lvl == "LOW"

    # 8 vehicles out of 20 (40%) -> MODERATE (25% - 55%)
    d_val, d_lvl = analyzer.calculate_density(vehicle_count=8, max_capacity=20)
    assert d_lvl == "MODERATE"

    # 13 vehicles out of 20 (65%) -> HIGH (55% - 80%)
    d_val, d_lvl = analyzer.calculate_density(vehicle_count=13, max_capacity=20)
    assert d_lvl == "HIGH"

    # 18 vehicles out of 20 (90%) -> SEVERE (>= 80%)
    d_val, d_lvl = analyzer.calculate_density(vehicle_count=18, max_capacity=20)
    assert d_lvl == "SEVERE"


def test_density_with_physical_occupancy(analyzer):
    """Verify combining bounding box area and vehicle count."""
    # 50% physical area occupied and 40% count
    d_val, d_lvl = analyzer.calculate_density(
        vehicle_count=8,
        occupied_area=50000.0,
        roi_total_area=100000.0,
        max_capacity=20
    )
    # Expected: 0.60 * 0.5 + 0.40 * 0.4 = 0.30 + 0.16 = 0.46 -> MODERATE
    assert 0.40 <= d_val <= 0.52
    assert d_lvl == "MODERATE"


def test_recommend_green_time_scaling(analyzer):
    """Verify adaptive green time respects min and max bounds."""
    min_g = SIGNAL_TIMINGS["MIN_GREEN"]
    max_g = SIGNAL_TIMINGS["MAX_GREEN"]

    # Low density (0.05) -> near min_green
    g_low = analyzer.recommend_green_time(density=0.05, vehicle_count=1)
    assert g_low >= min_g
    assert g_low <= min_g + 8

    # High density (0.95) -> near max_green
    g_high = analyzer.recommend_green_time(density=0.95, vehicle_count=20)
    assert g_high >= max_g - 5
    assert g_high <= max_g

    # Heavy vehicle adjustment (buses/trucks provide extra clearance seconds)
    g_with_buses = analyzer.recommend_green_time(density=0.5, vehicle_count=10, heavy_vehicle_count=3)
    g_without_buses = analyzer.recommend_green_time(density=0.5, vehicle_count=10, heavy_vehicle_count=0)
    assert g_with_buses > g_without_buses


def test_analyze_pattern(analyzer):
    """Verify pattern recognition extracts busiest direction and queue trends."""
    records = [
        {"direction": "NORTH", "vehicle_count": 20, "density": 0.8},
        {"direction": "SOUTH", "vehicle_count": 5, "density": 0.2},
        {"direction": "EAST", "vehicle_count": 8, "density": 0.3},
        {"direction": "WEST", "vehicle_count": 2, "density": 0.1}
    ]
    summary = analyzer.analyze_pattern(records)
    assert summary["busiest_direction"] == "NORTH"
    assert summary["total_vehicles"] == 35
    assert summary["peak_density"] == 0.8
