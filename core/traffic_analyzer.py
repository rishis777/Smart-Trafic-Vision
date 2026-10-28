"""TrafficAnalyzer: calculates density, analyzes traffic patterns,
predicts congestion trends, and computes optimal green time allocations.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from traffic_ai.config import DENSITY_THRESHOLDS, SIGNAL_TIMINGS


class TrafficAnalyzer:
    """Core mathematical and analytical engine for traffic metrics."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or DENSITY_THRESHOLDS
        self.min_green = SIGNAL_TIMINGS["MIN_GREEN"]
        self.max_green = SIGNAL_TIMINGS["MAX_GREEN"]
        self.default_green = SIGNAL_TIMINGS["DEFAULT_GREEN"]

    def calculate_density(
        self,
        vehicle_count: int,
        occupied_area: float = 0.0,
        roi_total_area: float = 1.0,
        max_capacity: int = 25
    ) -> Tuple[float, str]:
        """Calculate traffic density (0.0 - 1.0) and categorical level.

        Combines vehicle count ratio (40%) and physical road occupancy ratio (60%).
        """
        if max_capacity <= 0:
            max_capacity = 25

        count_ratio = min(1.0, vehicle_count / float(max_capacity))
        occupancy_ratio = 0.0
        if roi_total_area > 0:
            occupancy_ratio = min(1.0, max(0.0, occupied_area / roi_total_area))

        # Weighted combination: physical occupancy is very reliable in CCTV, supplemented by count
        if occupied_area > 0 and roi_total_area > 1.0:
            combined_density = 0.60 * occupancy_ratio + 0.40 * count_ratio
        else:
            combined_density = count_ratio

        combined_density = round(float(np.clip(combined_density, 0.0, 1.0)), 3)

        # Classify based on configured thresholds
        if combined_density < self.thresholds["LOW"]:
            level = "LOW"
        elif combined_density < self.thresholds["MODERATE"]:
            level = "MODERATE"
        elif combined_density < self.thresholds["HIGH"]:
            level = "HIGH"
        else:
            level = "SEVERE"

        return combined_density, level

    def recommend_green_time(
        self,
        density: float,
        vehicle_count: int,
        heavy_vehicle_count: int = 0,
        min_limit: Optional[int] = None,
        max_limit: Optional[int] = None
    ) -> int:
        """Calculate adaptive green time in seconds for a specific direction.

        Formula scales non-linearly with density, adding extra clearance time
        for heavy vehicles (buses/trucks) which accelerate slower.
        """
        min_g = min_limit or self.min_green
        max_g = max_limit or self.max_green

        # Base non-linear scaling
        scaled_seconds = min_g + (max_g - min_g) * (density ** 1.15)

        # Slower clearance buffer for heavy vehicles
        heavy_vehicle_buffer = heavy_vehicle_count * 2.0

        recommended = int(round(scaled_seconds + heavy_vehicle_buffer))
        return int(np.clip(recommended, min_g, max_g))

    def analyze_pattern(self, traffic_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate statistical patterns from recent traffic observations."""
        if not traffic_records:
            return {
                "total_vehicles": 0,
                "avg_density": 0.0,
                "busiest_direction": "NONE",
                "peak_density": 0.0,
                "congestion_trend": "STABLE"
            }

        counts = [r.get("vehicle_count", 0) for r in traffic_records]
        densities = [r.get("density", 0.0) for r in traffic_records]
        directions = [r.get("direction", "UNKNOWN") for r in traffic_records]

        # Determine busiest direction
        dir_totals: Dict[str, int] = {}
        for d, c in zip(directions, counts):
            dir_totals[d] = dir_totals.get(d, 0) + c

        busiest_dir = max(dir_totals, key=dir_totals.get) if dir_totals else "NONE"

        # Determine trend (first half vs second half)
        trend = "STABLE"
        if len(densities) >= 6:
            half = len(densities) // 2
            prev_avg = np.mean(densities[:half])
            recent_avg = np.mean(densities[half:])
            diff = recent_avg - prev_avg
            if diff > 0.08:
                trend = "INCREASING"
            elif diff < -0.08:
                trend = "DECREASING"

        return {
            "total_vehicles": int(sum(counts)),
            "avg_density": round(float(np.mean(densities)), 3),
            "peak_density": round(float(np.max(densities)), 3),
            "busiest_direction": busiest_dir,
            "direction_breakdown": dir_totals,
            "congestion_trend": trend
        }
