"""4-Way Intersection Traffic Simulation.

Simulates vehicle arrivals, queues, waiting times, and compares Fixed-Time vs
AI-Adaptive signal control with real statistical metrics output.
Can run in headless benchmark mode or interactive Pygame visualization.
"""

import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add package root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from traffic_ai.core.traffic_signal import TrafficSignal, SignalMode, SignalState
from traffic_ai.core.traffic_analyzer import TrafficAnalyzer
from traffic_ai.core.signal_controller import SignalController
from traffic_ai.config import DIRECTIONS, SIGNAL_TIMINGS


class SimVehicle:
    """Represents a simulated vehicle moving through the intersection."""

    def __init__(self, veh_id: int, direction: str, arrival_time: float, veh_type: str = "car"):
        self.veh_id = veh_id
        self.direction = direction
        self.arrival_time = arrival_time
        self.departure_time: Optional[float] = None
        self.veh_type = veh_type
        self.wait_time: float = 0.0

    def depart(self, current_time: float) -> float:
        self.departure_time = current_time
        self.wait_time = max(0.0, self.departure_time - self.arrival_time)
        return self.wait_time


class IntersectionSimulation:
    """Discrete-event simulation comparing Fixed-Time vs Adaptive control."""

    def __init__(self, mode: str = "ADAPTIVE", rush_hour_multiplier: float = 1.0):
        self.mode = mode.upper()
        self.signal = TrafficSignal(initial_direction="NORTH")
        self.signal.set_mode(self.mode)
        self.analyzer = TrafficAnalyzer()
        self.controller = SignalController(self.signal, self.analyzer)

        self.queues: Dict[str, List[SimVehicle]] = {d: [] for d in DIRECTIONS}
        self.cleared_vehicles: List[SimVehicle] = []
        self.sim_time: float = 0.0
        self.next_vehicle_id = 1
        self.rush_hour = rush_hour_multiplier

        # Arrival rate (vehicles/sec per lane): average 0.25 to 0.55 in rush hour
        self.base_arrival_rates = {
            "NORTH": 0.35 * self.rush_hour,
            "SOUTH": 0.30 * self.rush_hour,
            "EAST": 0.18 * self.rush_hour,
            "WEST": 0.15 * self.rush_hour
        }

        # Discharge rate when green (vehicles per second: ~0.8 to 1.0)
        self.discharge_rate = 0.85

    def step(self, dt: float = 1.0) -> None:
        """Advance simulation clock by dt seconds."""
        self.sim_time += dt

        # 1. Vehicle arrivals (Poisson process approximation)
        for d in DIRECTIONS:
            rate = self.base_arrival_rates[d] * dt
            # Number of arrivals this step
            arrivals = int(random.random() < rate)
            if random.random() < (rate - int(rate)):
                arrivals += 1

            for _ in range(arrivals):
                v_type = "bus" if random.random() < 0.1 else ("motorcycle" if random.random() < 0.2 else "car")
                veh = SimVehicle(self.next_vehicle_id, d, self.sim_time, v_type)
                self.queues[d].append(veh)
                self.next_vehicle_id += 1

        # 2. Advance traffic light state machine
        trans_log = self.signal.update(dt)

        # 3. Check AI Controller logic
        current_densities = {}
        for d in DIRECTIONS:
            q_len = len(self.queues[d])
            dens = min(1.0, round(q_len / 25.0, 3))
            current_densities[d] = {
                "count": q_len,
                "density": dens,
                "heavy_count": sum(1 for v in self.queues[d] if v.veh_type in ["bus", "truck"])
            }

        self.controller.generate_signal_decision(current_densities)

        # 4. Vehicle discharge during GREEN phase
        state = self.signal.get_current_state()
        active_dir = state["active_direction"]
        light_color = state["signals"].get(active_dir, "RED")

        if light_color == SignalState.GREEN.value:
            # Number of vehicles cleared in dt
            discharge_capacity = int(round(self.discharge_rate * dt))
            discharge_capacity = max(1, discharge_capacity)
            cleared_now = 0
            while self.queues[active_dir] and cleared_now < discharge_capacity:
                veh = self.queues[active_dir].pop(0)
                veh.depart(self.sim_time)
                self.cleared_vehicles.append(veh)
                cleared_now += 1

    def run_benchmark(self, duration_sec: int = 1200) -> Dict[str, Any]:
        """Run complete headless simulation and compute performance metrics."""
        step_size = 1.0
        steps = int(duration_sec / step_size)

        queue_history: List[float] = []

        for _ in range(steps):
            self.step(step_size)
            avg_q = sum(len(self.queues[d]) for d in DIRECTIONS) / float(len(DIRECTIONS))
            queue_history.append(avg_q)

        wait_times = [v.wait_time for v in self.cleared_vehicles]
        avg_wait = float(sum(wait_times) / max(1, len(wait_times)))
        max_wait = float(max(wait_times)) if wait_times else 0.0
        throughput_per_min = float(len(self.cleared_vehicles) / (duration_sec / 60.0))

        remaining_unserved = sum(len(self.queues[d]) for d in DIRECTIONS)

        return {
            "mode": self.mode,
            "duration_minutes": duration_sec / 60.0,
            "total_arrived": self.next_vehicle_id - 1,
            "total_cleared": len(self.cleared_vehicles),
            "unserved_vehicles": remaining_unserved,
            "avg_wait_time_sec": round(avg_wait, 2),
            "max_wait_time_sec": round(max_wait, 2),
            "avg_queue_length": round(float(sum(queue_history) / max(1, len(queue_history))), 2),
            "throughput_per_min": round(throughput_per_min, 2),
            "queue_by_direction": {d: len(self.queues[d]) for d in DIRECTIONS}
        }

    def render_pygame(self, screen, width: int = 800, height: int = 800) -> None:
        """Render graphical simulation with Pygame."""
        import pygame
        screen.fill((40, 44, 52))

        cx, cy = width // 2, height // 2
        road_w = 200

        # Roads
        pygame.draw.rect(screen, (70, 75, 80), (cx - road_w // 2, 0, road_w, height))
        pygame.draw.rect(screen, (70, 75, 80), (0, cy - road_w // 2, width, road_w))

        # Lane markings
        pygame.draw.line(screen, (255, 215, 0), (cx, 0), (cx, height), 2)
        pygame.draw.line(screen, (255, 215, 0), (0, cy), (width, cy), 2)

        # Draw signals
        state = self.signal.get_current_state()
        color_map = {
            "RED": (230, 30, 30),
            "YELLOW": (240, 200, 20),
            "GREEN": (30, 220, 50)
        }

        signal_coords = {
            "NORTH": (cx - road_w // 2 - 25, cy - road_w // 2 - 25),
            "SOUTH": (cx + road_w // 2 + 10, cy + road_w // 2 + 10),
            "EAST":  (cx + road_w // 2 + 10, cy - road_w // 2 - 25),
            "WEST":  (cx - road_w // 2 - 25, cy + road_w // 2 + 10)
        }

        for d, pos in signal_coords.items():
            sig_color = state["signals"].get(d, "RED")
            pygame.draw.rect(screen, (20, 20, 20), (pos[0], pos[1], 20, 45), border_radius=4)
            c = color_map.get(sig_color, (230, 30, 30))
            pygame.draw.circle(screen, c, (pos[0] + 10, pos[1] + 22), 8)

        # Draw Queues
        font = pygame.font.SysFont("Arial", 16)
        y_n = cy - road_w // 2 - 15
        for i, _ in enumerate(self.queues["NORTH"][:15]):
            pygame.draw.rect(screen, (220, 100, 50), (cx - 45, y_n - i * 18, 20, 14), border_radius=2)

        y_s = cy + road_w // 2 + 15
        for i, _ in enumerate(self.queues["SOUTH"][:15]):
            pygame.draw.rect(screen, (50, 150, 220), (cx + 25, y_s + i * 18, 20, 14), border_radius=2)

        x_w = cx - road_w // 2 - 15
        for i, _ in enumerate(self.queues["WEST"][:15]):
            pygame.draw.rect(screen, (200, 200, 50), (x_w - i * 18, cy + 25, 14, 20), border_radius=2)

        x_e = cx + road_w // 2 + 15
        for i, _ in enumerate(self.queues["EAST"][:15]):
            pygame.draw.rect(screen, (150, 50, 200), (x_e + i * 18, cy - 45, 14, 20), border_radius=2)

        # HUD Text
        mode_text = font.render(f"MODE: {self.mode} | REMAINING: {state['remaining_seconds']}s", True, (255, 255, 255))
        screen.blit(mode_text, (20, 20))
        stats_text = font.render(f"Cleared: {len(self.cleared_vehicles)} | In Queue: {sum(len(self.queues[d]) for d in DIRECTIONS)}", True, (0, 255, 200))
        screen.blit(stats_text, (20, 45))


def compare_fixed_vs_adaptive(duration_sec: int = 1200) -> Dict[str, Any]:
    """Execute rigorous comparative simulation and return metrics."""
    random.seed(42)  # Fixed seed for fair comparison
    fixed_sim = IntersectionSimulation(mode="FIXED", rush_hour_multiplier=1.2)
    fixed_res = fixed_sim.run_benchmark(duration_sec=duration_sec)

    random.seed(42)  # Identical vehicle arrival schedule for adaptive
    adaptive_sim = IntersectionSimulation(mode="ADAPTIVE", rush_hour_multiplier=1.2)
    adaptive_res = adaptive_sim.run_benchmark(duration_sec=duration_sec)

    # Compute improvements
    wait_reduction_pct = ((fixed_res["avg_wait_time_sec"] - adaptive_res["avg_wait_time_sec"]) / max(0.01, fixed_res["avg_wait_time_sec"])) * 100.0
    throughput_gain_pct = ((adaptive_res["throughput_per_min"] - fixed_res["throughput_per_min"]) / max(0.01, fixed_res["throughput_per_min"])) * 100.0
    queue_reduction_pct = ((fixed_res["avg_queue_length"] - adaptive_res["avg_queue_length"]) / max(0.01, fixed_res["avg_queue_length"])) * 100.0

    return {
        "fixed": fixed_res,
        "adaptive": adaptive_res,
        "wait_time_reduction_percent": round(wait_reduction_pct, 1),
        "throughput_gain_percent": round(throughput_gain_pct, 1),
        "queue_reduction_percent": round(queue_reduction_pct, 1)
    }


def write_metrics_file(results: Dict[str, Any], filepath: Path) -> None:
    """Generate METRICS.md from actual benchmark run."""
    f = results["fixed"]
    a = results["adaptive"]

    content = f"""# Traffic Management System: Simulation Metrics & Performance Analysis

This document contains empirical performance comparison data between **Fixed-Time Baseline** signal control and the **AI-Based Adaptive Signal Control** model implemented in this project.
Both simulations were executed over a simulated duration of **{int(f['duration_minutes'])} minutes** under identical, randomized Poisson vehicle arrival distributions with peak-hour load factors.

---

## 1. Key Performance Comparison Table

| Performance Metric | Fixed-Time Baseline | AI Adaptive Signal | Improvement / Delta |
| :--- | :--- | :--- | :--- |
| **Control Philosophy** | Predetermined 30s Cycles | Dynamic Queue/Density Scaling | Real-time AI Driven |
| **Total Vehicles Cleared** | {f['total_cleared']} | {a['total_cleared']} | **+{a['total_cleared'] - f['total_cleared']} vehicles** |
| **Unserved Queue Remaining** | {f['unserved_vehicles']} | {a['unserved_vehicles']} | **-{f['unserved_vehicles'] - a['unserved_vehicles']} backlog** |
| **Average Waiting Time** | {f['avg_wait_time_sec']} seconds | {a['avg_wait_time_sec']} seconds | **{results['wait_time_reduction_percent']}% Reduction** |
| **Maximum Waiting Time** | {f['max_wait_time_sec']} seconds | {a['max_wait_time_sec']} seconds | **-{round(f['max_wait_time_sec'] - a['max_wait_time_sec'], 1)}s Peak Delay** |
| **Average Queue Length** | {f['avg_queue_length']} vehicles | {a['avg_queue_length']} vehicles | **{results['queue_reduction_percent']}% Shorter Queues** |
| **Throughput Rate** | {f['throughput_per_min']} veh/min | {a['throughput_per_min']} veh/min | **+{results['throughput_gain_percent']}% Higher Capacity** |

---

## 2. Directional Queue Status at Simulation Close

| Direction | Fixed-Time Queue | AI Adaptive Queue | Difference |
| :--- | :--- | :--- | :--- |
| **NORTH (Heavy Flow)** | {f['queue_by_direction']['NORTH']} | {a['queue_by_direction']['NORTH']} | -{f['queue_by_direction']['NORTH'] - a['queue_by_direction']['NORTH']} |
| **SOUTH (Heavy Flow)** | {f['queue_by_direction']['SOUTH']} | {a['queue_by_direction']['SOUTH']} | -{f['queue_by_direction']['SOUTH'] - a['queue_by_direction']['SOUTH']} |
| **EAST (Moderate Flow)** | {f['queue_by_direction']['EAST']} | {a['queue_by_direction']['EAST']} | -{f['queue_by_direction']['EAST'] - a['queue_by_direction']['EAST']} |
| **WEST (Light Flow)** | {f['queue_by_direction']['WEST']} | {a['queue_by_direction']['WEST']} | -{f['queue_by_direction']['WEST'] - a['queue_by_direction']['WEST']} |

---

## 3. Analysis & Viva Talking Points

1. **Why Adaptive Beats Fixed-Time**:
   - In Fixed-Time mode, green time is squandered on empty lanes while congested approaches face spillover delays.
   - The AI-based algorithm monitors vehicle count, queue size, and road occupancy (ROI bounding box intersection), scaling green duration between 10s and 60s.
2. **Heavy Vehicle Compensation**:
   - The system adds an acceleration allowance for buses and trucks, preventing premature phase termination.
3. **Prevention of Starvation**:
   - Minimum green limits (10s) and competing queue urgency checks ensure secondary roads never experience indefinite red lock.
"""
    with open(filepath, "w", encoding="utf-8") as out:
        out.write(content)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run 4-way intersection simulation.")
    parser.add_argument("--benchmark", action="store_true", help="Run automated benchmark comparison")
    parser.add_argument("--gui", action="store_true", help="Launch visual Pygame simulation")
    parser.add_argument("--mode", type=str, default="ADAPTIVE", choices=["ADAPTIVE", "FIXED"], help="Signal mode")
    args = parser.parse_args()

    if args.benchmark:
        print("\n[INFO] Running benchmark comparison (Fixed-Time vs AI-Adaptive)...")
        results = compare_fixed_vs_adaptive(duration_sec=1200)
        metrics_path = Path(__file__).resolve().parent.parent / "METRICS.md"
        write_metrics_file(results, metrics_path)
        print(f"[SUCCESS] Metrics successfully written to: {metrics_path}")
        print(f" -> Wait Time Reduction: {results['wait_time_reduction_percent']}%")
        print(f" -> Throughput Gain:     +{results['throughput_gain_percent']}%")
        print(f" -> Queue Reduction:     {results['queue_reduction_percent']}%\n")
    elif args.gui:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption("AI Traffic Signal Simulation")
        sim = IntersectionSimulation(mode=args.mode)
        clock = pygame.time.Clock()

        running = True
        while running:
            dt = clock.tick(30) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        new_mode = "FIXED" if sim.mode == "ADAPTIVE" else "ADAPTIVE"
                        sim.mode = new_mode
                        sim.signal.set_mode(new_mode)

            sim.step(dt)
            sim.render_pygame(screen)
            pygame.display.flip()

        pygame.quit()
    else:
        # Default: run quick benchmark
        results = compare_fixed_vs_adaptive(duration_sec=600)
        print(f"Simulation completed. Wait time reduction: {results['wait_time_reduction_percent']}%")


if __name__ == "__main__":
    main()
