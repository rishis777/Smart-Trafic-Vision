"""SignalController: Coordinates high-level signal policy decisions,
adaptive green allocation, fixed-time cycling, and emergency preemptions.
"""

from typing import Dict, Any, Optional
import time
from traffic_ai.core.traffic_signal import TrafficSignal, SignalMode, PhaseState
from traffic_ai.core.traffic_analyzer import TrafficAnalyzer
from traffic_ai.config import SIGNAL_TIMINGS, DIRECTIONS


class SignalController:
    """Orchestrates traffic light decisions based on real-time AI density data."""

    def __init__(self, traffic_signal: TrafficSignal, traffic_analyzer: TrafficAnalyzer):
        self.signal = traffic_signal
        self.analyzer = traffic_analyzer
        self.last_decision_time = time.time()
        self.baseline_mode = "ADAPTIVE"  # "ADAPTIVE" or "FIXED"

    def generate_signal_decision(self, current_densities: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate traffic conditions across all directions and make timing decision."""
        current_state = self.signal.get_current_state()

        # If manual override or emergency corridor is active, respect that
        if current_state["is_override_active"]:
            return {
                "action": "HOLD_OVERRIDE",
                "direction": current_state["override_direction"],
                "reason": "Operator manual override in effect"
            }

        if current_state["emergency_corridor_active"]:
            return {
                "action": "HOLD_EMERGENCY",
                "direction": current_state["emergency_direction"],
                "reason": "Emergency vehicle green corridor in effect"
            }

        # If in FIXED-TIME baseline mode
        if self.signal.mode == SignalMode.FIXED:
            fixed_green = SIGNAL_TIMINGS["FIXED_BASELINE_GREEN"]
            return {
                "action": "FIXED_CYCLE",
                "mode": "FIXED",
                "green_time": fixed_green,
                "reason": "Fixed-time baseline cycle"
            }

        # In ADAPTIVE mode:
        active_dir = current_state["active_direction"]
        active_data = current_densities.get(active_dir, {"count": 0, "density": 0.0, "heavy_count": 0})
        active_density = active_data.get("density", 0.0)
        active_count = active_data.get("count", 0)
        active_heavy = active_data.get("heavy_count", 0)

        # Calculate recommended green time for the active direction
        recommended_green = self.analyzer.recommend_green_time(
            density=active_density,
            vehicle_count=active_count,
            heavy_vehicle_count=active_heavy
        )

        # Inspect competing directions
        other_dirs = [d for d in DIRECTIONS if d != active_dir]
        contenders = []
        for d in other_dirs:
            d_info = current_densities.get(d, {"count": 0, "density": 0.0, "heavy_count": 0})
            contenders.append((d, d_info.get("density", 0.0), d_info.get("count", 0)))

        # Sort competing directions by density descending
        contenders.sort(key=lambda x: x[1], reverse=True)
        highest_contender, high_density, high_count = contenders[0] if contenders else ("NONE", 0.0, 0)

        decision = {
            "action": "ADAPTIVE_ALLOCATE",
            "active_direction": active_dir,
            "recommended_green": recommended_green,
            "competing_highest_direction": highest_contender,
            "competing_density": high_density,
            "reason": f"Density {active_density:.2f} vs peak waiting {high_density:.2f}"
        }

        # If current green is still running, check if active green needs extension or if time expired
        if current_state["phase_state"] == PhaseState.GREEN_PHASE.value:
            if current_state["remaining_seconds"] <= 1:
                # Select next direction: either highest waiting density or next in cycle
                target_next = highest_contender if high_density > 0.15 else self._get_next_cyclic(active_dir)
                next_info = current_densities.get(target_next, {"count": 0, "density": 0.0, "heavy_count": 0})
                next_green = self.analyzer.recommend_green_time(
                    density=next_info.get("density", 0.0),
                    vehicle_count=next_info.get("count", 0),
                    heavy_vehicle_count=next_info.get("heavy_count", 0)
                )
                self.signal.change_signal(target_next, next_green, reason=f"High queue in {target_next}")
                decision["action"] = "SCHEDULE_PHASE_CHANGE"
                decision["next_direction"] = target_next
                decision["allocated_green"] = next_green

        return decision

    def _get_next_cyclic(self, current_dir: str) -> str:
        idx = DIRECTIONS.index(current_dir) if current_dir in DIRECTIONS else 0
        return DIRECTIONS[(idx + 1) % len(DIRECTIONS)]

    def send_signal_command(self, target_direction: str, duration: int, reason: str = "") -> bool:
        """Explicitly command a signal transition."""
        return self.signal.change_signal(target_direction, float(duration), reason=reason)

    def handle_emergency_priority(self, corridor_direction: str) -> bool:
        """Initiate immediate emergency green preemption."""
        return self.signal.set_emergency_corridor(corridor_direction)

    def clear_emergency_priority(self) -> None:
        """Clear emergency preemption and return to normal operation."""
        self.signal.clear_emergency_corridor()

    def set_manual_override(self, direction: str, reason: str = "Operator command") -> bool:
        """Trigger manual override."""
        return self.signal.set_manual_override(direction, reason=reason)

    def clear_manual_override(self) -> None:
        """Release manual override."""
        self.signal.clear_manual_override()

    def set_mode(self, mode: str) -> bool:
        """Toggle between ADAPTIVE and FIXED."""
        return self.signal.set_mode(mode)
