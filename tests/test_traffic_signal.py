"""Unit tests for TrafficSignal state machine, mode transitions, and overrides.
"""

import pytest
from traffic_ai.core.traffic_signal import TrafficSignal, SignalState, SignalMode, PhaseState
from traffic_ai.config import SIGNAL_TIMINGS


@pytest.fixture
def signal():
    return TrafficSignal(initial_direction="NORTH")


def test_initial_state(signal):
    """Verify initial signal state is GREEN for initial direction and RED for others."""
    state = signal.get_current_state()
    assert state["active_direction"] == "NORTH"
    assert state["signals"]["NORTH"] == SignalState.GREEN.value
    assert state["signals"]["SOUTH"] == SignalState.RED.value
    assert state["signals"]["EAST"] == SignalState.RED.value
    assert state["signals"]["WEST"] == SignalState.RED.value
    assert state["mode"] == SignalMode.ADAPTIVE.value


def test_state_transitions(signal):
    """Verify transition from GREEN to YELLOW to ALL_RED to next GREEN."""
    signal.phase_remaining_seconds = 1.0

    # Advance 1.5 seconds -> should enter YELLOW
    signal.update(1.5)
    assert signal.phase_state == PhaseState.YELLOW_PHASE
    state = signal.get_current_state()
    assert state["signals"]["NORTH"] == SignalState.YELLOW.value

    # Advance through yellow duration -> should enter ALL_RED
    signal.update(SIGNAL_TIMINGS["YELLOW"] + 0.5)
    assert signal.phase_state == PhaseState.ALL_RED_PHASE
    state = signal.get_current_state()
    assert state["signals"]["NORTH"] == SignalState.RED.value

    # Advance through all-red duration -> should enter next GREEN (e.g. EAST)
    log = signal.update(SIGNAL_TIMINGS["ALL_RED"] + 0.5)
    assert signal.phase_state == PhaseState.GREEN_PHASE
    assert log is not None
    assert log["active_direction"] in ["EAST", "SOUTH", "WEST"]


def test_manual_override(signal):
    """Verify operator manual override forces green on chosen lane."""
    success = signal.set_manual_override("EAST", reason="Operator test command")
    assert success is True
    assert signal.is_override_active is True

    state = signal.get_current_state()
    assert state["signals"]["EAST"] == SignalState.GREEN.value
    assert state["signals"]["NORTH"] == SignalState.RED.value
    assert state["signals"]["SOUTH"] == SignalState.RED.value
    assert state["signals"]["WEST"] == SignalState.RED.value

    # Release override
    signal.clear_manual_override()
    assert signal.is_override_active is False
    assert signal.mode == SignalMode.ADAPTIVE


def test_emergency_corridor(signal):
    """Verify priority emergency corridor forces green for emergency direction."""
    signal.set_emergency_corridor("WEST")
    assert signal.emergency_corridor_active is True

    state = signal.get_current_state()
    assert state["signals"]["WEST"] == SignalState.GREEN.value
    assert state["signals"]["NORTH"] == SignalState.RED.value

    signal.clear_emergency_corridor()
    assert signal.emergency_corridor_active is False
    assert signal.mode == SignalMode.ADAPTIVE
