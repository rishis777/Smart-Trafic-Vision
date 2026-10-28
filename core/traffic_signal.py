"""Traffic Signal state machine and phase controller for 4-way intersection.

Maintains RED, YELLOW, GREEN states, countdown timers, and supports ADAPTIVE,
FIXED baseline, MANUAL_OVERRIDE, and EMERGENCY corridor modes.
"""

from enum import Enum
import time
from typing import Dict, Any, List, Optional
from traffic_ai.config import DIRECTIONS, SIGNAL_TIMINGS


class SignalState(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class SignalMode(str, Enum):
    ADAPTIVE = "ADAPTIVE"
    FIXED = "FIXED"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    EMERGENCY = "EMERGENCY"


class PhaseState(str, Enum):
    GREEN_PHASE = "GREEN_PHASE"
    YELLOW_PHASE = "YELLOW_PHASE"
    ALL_RED_PHASE = "ALL_RED_PHASE"


class TrafficSignal:
    """State machine governing traffic lights for an intersection."""

    def __init__(self, intersection_id: str = "INT_01", initial_direction: str = "NORTH"):
        self.intersection_id = intersection_id
        self.mode: SignalMode = SignalMode.ADAPTIVE
        self.directions: List[str] = list(DIRECTIONS)

        # Active direction possessing green / transition
        self.active_direction: str = initial_direction if initial_direction in self.directions else "NORTH"
        self.next_direction: Optional[str] = None
        self.phase_state: PhaseState = PhaseState.GREEN_PHASE

        # Timers
        self.allocated_green_time: float = float(SIGNAL_TIMINGS["DEFAULT_GREEN"])
        self.yellow_time: float = float(SIGNAL_TIMINGS["YELLOW"])
        self.all_red_time: float = float(SIGNAL_TIMINGS["ALL_RED"])
        self.time_in_current_state: float = 0.0
        self.phase_remaining_seconds: float = self.allocated_green_time

        # Override & Emergency state
        self.is_override_active: bool = False
        self.override_direction: Optional[str] = None
        self.override_reason: str = ""
        self.emergency_corridor_active: bool = False
        self.emergency_direction: Optional[str] = None

        # Statistics & transition history
        self.last_transition_time = time.time()
        self.cycle_count = 0
        self.last_state_change_reason = "System Initialization"

    def get_current_state(self) -> Dict[str, Any]:
        """Return full snapshot of all directional signal lights and timers."""
        states: Dict[str, str] = {}
        for d in self.directions:
            if self.emergency_corridor_active:
                states[d] = SignalState.GREEN.value if d == self.emergency_direction else SignalState.RED.value
            elif self.is_override_active:
                states[d] = SignalState.GREEN.value if d == self.override_direction else SignalState.RED.value
            else:
                if d == self.active_direction:
                    if self.phase_state == PhaseState.GREEN_PHASE:
                        states[d] = SignalState.GREEN.value
                    elif self.phase_state == PhaseState.YELLOW_PHASE:
                        states[d] = SignalState.YELLOW.value
                    else:  # ALL_RED_PHASE
                        states[d] = SignalState.RED.value
                else:
                    states[d] = SignalState.RED.value

        return {
            "intersection_id": self.intersection_id,
            "mode": self.mode.value,
            "phase_state": self.phase_state.value,
            "active_direction": self.active_direction,
            "next_direction": self.next_direction,
            "signals": states,
            "remaining_seconds": max(0, int(round(self.phase_remaining_seconds))),
            "allocated_green_time": int(self.allocated_green_time),
            "is_override_active": self.is_override_active,
            "override_direction": self.override_direction,
            "emergency_corridor_active": self.emergency_corridor_active,
            "emergency_direction": self.emergency_direction,
            "cycle_count": self.cycle_count,
            "reason": self.last_state_change_reason
        }

    def change_signal(self, target_direction: str, green_duration: float, reason: str = "") -> bool:
        """Schedule transition to a target direction with requested green duration."""
        if target_direction not in self.directions:
            return False

        if self.emergency_corridor_active or self.is_override_active:
            return False

        if target_direction == self.active_direction and self.phase_state == PhaseState.GREEN_PHASE:
            # Extend current green if appropriate
            self.allocated_green_time = max(SIGNAL_TIMINGS["MIN_GREEN"], min(SIGNAL_TIMINGS["MAX_GREEN"], green_duration))
            self.phase_remaining_seconds = max(self.phase_remaining_seconds, self.allocated_green_time - self.time_in_current_state)
            self.last_state_change_reason = f"Extended green: {reason}"
            return True

        # Begin transition through yellow
        self.next_direction = target_direction
        self.allocated_green_time = max(SIGNAL_TIMINGS["MIN_GREEN"], min(SIGNAL_TIMINGS["MAX_GREEN"], green_duration))
        self.last_state_change_reason = reason
        self.phase_state = PhaseState.YELLOW_PHASE
        self.time_in_current_state = 0.0
        self.phase_remaining_seconds = self.yellow_time
        return True

    def update(self, dt: float) -> Optional[Dict[str, Any]]:
        """Advance signal clock by dt seconds. Returns log info if a phase completes."""
        self.time_in_current_state += dt
        self.phase_remaining_seconds = max(0.0, self.phase_remaining_seconds - dt)

        # Manual override / emergency hold indefinitely until cleared
        if self.is_override_active or self.emergency_corridor_active:
            return None

        transition_log = None

        if self.phase_state == PhaseState.GREEN_PHASE:
            if self.phase_remaining_seconds <= 0:
                # Transition to YELLOW
                self.phase_state = PhaseState.YELLOW_PHASE
                self.time_in_current_state = 0.0
                self.phase_remaining_seconds = self.yellow_time
                if not self.next_direction:
                    # Default cyclic progression: NORTH -> EAST -> SOUTH -> WEST
                    idx = self.directions.index(self.active_direction)
                    self.next_direction = self.directions[(idx + 1) % len(self.directions)]

        elif self.phase_state == PhaseState.YELLOW_PHASE:
            if self.phase_remaining_seconds <= 0:
                # Transition to ALL_RED safety clearance
                self.phase_state = PhaseState.ALL_RED_PHASE
                self.time_in_current_state = 0.0
                self.phase_remaining_seconds = self.all_red_time

        elif self.phase_state == PhaseState.ALL_RED_PHASE:
            if self.phase_remaining_seconds <= 0:
                # Transition to GREEN for next direction
                prev_direction = self.active_direction
                self.active_direction = self.next_direction or self.directions[0]
                self.next_direction = None
                self.phase_state = PhaseState.GREEN_PHASE
                self.time_in_current_state = 0.0
                self.phase_remaining_seconds = self.allocated_green_time
                self.cycle_count += 1

                transition_log = {
                    "previous_direction": prev_direction,
                    "active_direction": self.active_direction,
                    "green_time": int(self.allocated_green_time),
                    "mode": self.mode.value,
                    "reason": self.last_state_change_reason
                }

        return transition_log

    def set_manual_override(self, direction: str, reason: str = "Operator override") -> bool:
        """Force signal to green for specified direction, holding red for others."""
        if direction not in self.directions:
            return False
        self.is_override_active = True
        self.override_direction = direction
        self.override_reason = reason
        self.mode = SignalMode.MANUAL_OVERRIDE
        self.last_state_change_reason = f"Manual override to {direction}: {reason}"
        return True

    def clear_manual_override(self) -> None:
        """Release operator manual override and resume automated control."""
        self.is_override_active = False
        self.override_direction = None
        self.mode = SignalMode.ADAPTIVE
        self.phase_state = PhaseState.ALL_RED_PHASE
        self.phase_remaining_seconds = self.all_red_time
        self.time_in_current_state = 0.0
        self.last_state_change_reason = "Manual override cleared"

    def set_emergency_corridor(self, direction: str) -> bool:
        """Establish priority green corridor for emergency vehicle."""
        if direction not in self.directions:
            return False
        self.emergency_corridor_active = True
        self.emergency_direction = direction
        self.mode = SignalMode.EMERGENCY
        self.last_state_change_reason = f"EMERGENCY GREEN CORRIDOR: {direction}"
        return True

    def clear_emergency_corridor(self) -> None:
        """Revert emergency corridor back to normal operation."""
        self.emergency_corridor_active = False
        self.emergency_direction = None
        self.mode = SignalMode.ADAPTIVE
        self.phase_state = PhaseState.ALL_RED_PHASE
        self.phase_remaining_seconds = self.all_red_time
        self.time_in_current_state = 0.0
        self.last_state_change_reason = "Emergency corridor cleared, resumed normal"

    def set_mode(self, mode: str) -> bool:
        """Switch between ADAPTIVE and FIXED baseline modes."""
        try:
            m = SignalMode(mode.upper())
            self.mode = m
            if m == SignalMode.FIXED:
                self.allocated_green_time = float(SIGNAL_TIMINGS["FIXED_BASELINE_GREEN"])
            return True
        except ValueError:
            return False
