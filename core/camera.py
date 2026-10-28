"""Camera and Sensor abstractions for video acquisition.

Supports webcam, video files, RTSP/HTTP streams, and a high-fidelity synthetic
traffic video generator with realistic multi-lane 4-way intersection physics,
strict lane-following, car-following queue dynamics, and signal compliance.
"""

from typing import Optional, Tuple, Dict, Any, List
import time
import math
import random
from pathlib import Path
import numpy as np
import cv2


class Sensor:
    """Represents a physical or virtual sensor attached to an intersection lane."""

    def __init__(self, sensor_id: str, direction: str, sensor_type: str = "OPTICAL_CAMERA"):
        self.sensor_id = sensor_id
        self.direction = direction
        self.sensor_type = sensor_type
        self.is_online = False
        self.last_heartbeat = 0.0

    def heartbeat(self) -> None:
        self.is_online = True
        self.last_heartbeat = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "direction": self.direction,
            "sensor_type": self.sensor_type,
            "is_online": self.is_online,
            "last_heartbeat": self.last_heartbeat
        }


class Camera:
    """Camera stream handler with high-precision 4-way intersection synthetic generator."""

    def __init__(
        self,
        source: str | int = 0,
        fps: int = 30,
        width: int = 1280,
        height: int = 720
    ):
        self.source = source
        self.target_fps = fps
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_streaming = False
        self.use_synthetic = False
        self.frame_count = 0
        self.sensor = Sensor(sensor_id=f"CAM_{source}", direction="NORTH")

        # Intersection Geometry constants (1280x720)
        self.cx = self.width // 2      # 640
        self.cy = self.height // 2     # 360
        self.road_w = 260              # Total road width
        self.half_road = self.road_w // 2  # 130

        # Current traffic signal state (synced from SignalController or internal timer)
        self.current_signals: Dict[str, str] = {
            "NORTH": "GREEN",
            "SOUTH": "RED",
            "EAST": "RED",
            "WEST": "RED"
        }
        self.internal_signal_timer = 0.0
        self.internal_phase_idx = 0

        # Background cache for high performance
        self._road_bg: Optional[np.ndarray] = None
        self._synth_vehicles: List[Dict[str, Any]] = []
        self._next_vehicle_id = 1
        self._last_spawn_time = 0.0

        if str(self.source).lower() == "mock":
            self.use_synthetic = True

        self._init_synthetic_scene()

    def set_signals(self, signals: Dict[str, str]) -> None:
        """Update traffic light state for approach queues."""
        if signals:
            self.current_signals.update(signals)

    def _build_road_background(self) -> np.ndarray:
        """Pre-render photorealistic top-down 4-way intersection layout."""
        # 1. Base terrain: Dark urban landscape
        bg = np.full((self.height, self.width, 3), (35, 40, 44), dtype=np.uint8)

        # Draw suburban grass lots in the 4 corners
        corner_color = (42, 62, 50)
        curb_color = (130, 136, 142)
        sidewalk_color = (70, 75, 80)
        asphalt_color = (52, 56, 60)

        # Top-Left corner
        cv2.rectangle(bg, (0, 0), (self.cx - self.half_road, self.cy - self.half_road), corner_color, -1)
        # Top-Right corner
        cv2.rectangle(bg, (self.cx + self.half_road, 0), (self.width, self.cy - self.half_road), corner_color, -1)
        # Bottom-Left corner
        cv2.rectangle(bg, (0, self.cy + self.half_road), (self.cx - self.half_road, self.height), corner_color, -1)
        # Bottom-Right corner
        cv2.rectangle(bg, (self.cx + self.half_road, self.cy + self.half_road), (self.width, self.height), corner_color, -1)

        # 2. Sidewalk borders along roads
        sw_thickness = 16
        # North-South sidewalks
        cv2.rectangle(bg, (self.cx - self.half_road - sw_thickness, 0), (self.cx - self.half_road, self.height), sidewalk_color, -1)
        cv2.rectangle(bg, (self.cx + self.half_road, 0), (self.cx + self.half_road + sw_thickness, self.height), sidewalk_color, -1)
        # East-West sidewalks
        cv2.rectangle(bg, (0, self.cy - self.half_road - sw_thickness), (self.width, self.cy - self.half_road), sidewalk_color, -1)
        cv2.rectangle(bg, (0, self.cy + self.half_road), (self.width, self.cy + self.half_road + sw_thickness), sidewalk_color, -1)

        # Curb highlight lines
        cv2.line(bg, (self.cx - self.half_road, 0), (self.cx - self.half_road, self.height), curb_color, 2)
        cv2.line(bg, (self.cx + self.half_road, 0), (self.cx + self.half_road, self.height), curb_color, 2)
        cv2.line(bg, (0, self.cy - self.half_road), (self.width, self.cy - self.half_road), curb_color, 2)
        cv2.line(bg, (0, self.cy + self.half_road), (self.width, self.cy + self.half_road), curb_color, 2)

        # 3. Main Asphalt Road Surfaces (260px wide)
        # North-South Road: x in [510, 770]
        cv2.rectangle(bg, (self.cx - self.half_road, 0), (self.cx + self.half_road, self.height), asphalt_color, -1)
        # East-West Road: y in [230, 490]
        cv2.rectangle(bg, (0, self.cy - self.half_road), (self.width, self.cy + self.half_road), asphalt_color, -1)
        # Central intersection junction box
        cv2.rectangle(bg, (self.cx - self.half_road, self.cy - self.half_road), (self.cx + self.half_road, self.cy + self.half_road), (56, 61, 65), -1)

        # 4. Bus Stop / No-Parking Alcove on West Road (bottom curb)
        # Designated Bay: x from 120 to 320, y from 490 to 550
        cv2.rectangle(bg, (120, 490), (320, 550), (45, 48, 52), -1)
        # Yellow diagonal hatching for restricted no-parking
        for hx in range(125, 315, 20):
            cv2.line(bg, (hx, 492), (hx + 20, 548), (0, 195, 235), 2)
        cv2.rectangle(bg, (120, 490), (320, 550), (0, 195, 235), 2)
        cv2.putText(bg, "RESTRICTED BUS BAY / NO-PARKING", (125, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1)

        # 5. Lane markings
        # Double Solid Yellow Center Dividers outside intersection box
        yellow = (0, 215, 255)
        # North Center Divider
        cv2.line(bg, (self.cx - 2, 0), (self.cx - 2, self.cy - self.half_road), yellow, 2)
        cv2.line(bg, (self.cx + 2, 0), (self.cx + 2, self.cy - self.half_road), yellow, 2)
        # South Center Divider
        cv2.line(bg, (self.cx - 2, self.cy + self.half_road), (self.cx - 2, self.height), yellow, 2)
        cv2.line(bg, (self.cx + 2, self.cy + self.half_road), (self.cx + 2, self.height), yellow, 2)
        # West Center Divider
        cv2.line(bg, (0, self.cy - 2), (self.cx - self.half_road, self.cy - 2), yellow, 2)
        cv2.line(bg, (0, self.cy + 2), (self.cx - self.half_road, self.cy + 2), yellow, 2)
        # East Center Divider
        cv2.line(bg, (self.cx + self.half_road, self.cy - 2), (self.width, self.cy - 2), yellow, 2)
        cv2.line(bg, (self.cx + self.half_road, self.cy + 2), (self.width, self.cy + 2), yellow, 2)

        # Dashed White Lane Dividers (2 lanes per direction)
        white = (220, 225, 230)
        dash_len = 16
        dash_gap = 16

        # North-South dashed dividers (outer and inner lanes)
        # Incoming Southbound lane divider: x = 575
        for y in range(0, self.cy - self.half_road - 35, dash_len + dash_gap):
            cv2.line(bg, (self.cx - 65, y), (self.cx - 65, y + dash_len), white, 1)
        # Outgoing Northbound lane divider: x = 705
        for y in range(0, self.cy - self.half_road - 35, dash_len + dash_gap):
            cv2.line(bg, (self.cx + 65, y), (self.cx + 65, y + dash_len), white, 1)

        # South incoming Northbound: x = 705
        for y in range(self.cy + self.half_road + 35, self.height, dash_len + dash_gap):
            cv2.line(bg, (self.cx + 65, y), (self.cx + 65, y + dash_len), white, 1)
        # South outgoing Southbound: x = 575
        for y in range(self.cy + self.half_road + 35, self.height, dash_len + dash_gap):
            cv2.line(bg, (self.cx - 65, y), (self.cx - 65, y + dash_len), white, 1)

        # East-West dashed dividers
        # West incoming Eastbound: y = 425
        for x in range(0, self.cx - self.half_road - 35, dash_len + dash_gap):
            cv2.line(bg, (x, self.cy + 65), (x + dash_len, self.cy + 65), white, 1)
        # West outgoing Westbound: y = 295
        for x in range(0, self.cx - self.half_road - 35, dash_len + dash_gap):
            cv2.line(bg, (x, self.cy - 65), (x + dash_len, self.cy - 65), white, 1)

        # East incoming Westbound: y = 295
        for x in range(self.cx + self.half_road + 35, self.width, dash_len + dash_gap):
            cv2.line(bg, (x, self.cy - 65), (x + dash_len, self.cy - 65), white, 1)
        # East outgoing Eastbound: y = 425
        for x in range(self.cx + self.half_road + 35, self.width, dash_len + dash_gap):
            cv2.line(bg, (x, self.cy + 65), (x + dash_len, self.cy + 65), white, 1)

        # 6. Solid White Stop Lines
        # North Stop Line (incoming lane x in [510, 640]): y = 200
        cv2.line(bg, (self.cx - self.half_road, 200), (self.cx, 200), (255, 255, 255), 4)
        # South Stop Line (incoming lane x in [640, 770]): y = 520
        cv2.line(bg, (self.cx, 520), (self.cx + self.half_road, 520), (255, 255, 255), 4)
        # West Stop Line (incoming lane y in [360, 490]): x = 480
        cv2.line(bg, (480, self.cy), (480, self.cy + self.half_road), (255, 255, 255), 4)
        # East Stop Line (incoming lane y in [230, 360]): x = 800
        cv2.line(bg, (800, self.cy - self.half_road), (800, self.cy), (255, 255, 255), 4)

        # 7. Pedestrian Crosswalks (Zebra Stripes)
        zebra_color = (235, 240, 245)
        # North Zebra: y in [206, 226], across full road [510, 770]
        for zx in range(self.cx - self.half_road + 5, self.cx + self.half_road - 10, 18):
            cv2.rectangle(bg, (zx, 206), (zx + 10, 226), zebra_color, -1)

        # South Zebra: y in [494, 514]
        for zx in range(self.cx - self.half_road + 5, self.cx + self.half_road - 10, 18):
            cv2.rectangle(bg, (zx, 494), (zx + 10, 514), zebra_color, -1)

        # West Zebra: x in [484, 504], across road y in [230, 490]
        for zy in range(self.cy - self.half_road + 5, self.cy + self.half_road - 10, 18):
            cv2.rectangle(bg, (484, zy), (504, zy + 10), zebra_color, -1)

        # East Zebra: x in [776, 796]
        for zy in range(self.cy - self.half_road + 5, self.cy + self.half_road - 10, 18):
            cv2.rectangle(bg, (776, zy), (796, zy + 10), zebra_color, -1)

        # 8. Directional Lane Stencil Arrows
        self._draw_road_arrow(bg, self.cx - 65, 120, "DOWN")
        self._draw_road_arrow(bg, self.cx + 65, 600, "UP")
        self._draw_road_arrow(bg, 250, self.cy + 65, "RIGHT")
        self._draw_road_arrow(bg, 1030, self.cy - 65, "LEFT")

        return bg

    def _draw_road_arrow(self, img: np.ndarray, x: int, y: int, direction: str) -> None:
        """Draw clean stenciled road guidance arrow."""
        pts = []
        if direction == "DOWN":
            pts = [(x, y + 16), (x - 7, y), (x - 3, y), (x - 3, y - 16), (x + 3, y - 16), (x + 3, y), (x + 7, y)]
        elif direction == "UP":
            pts = [(x, y - 16), (x - 7, y), (x - 3, y), (x - 3, y + 16), (x + 3, y + 16), (x + 3, y), (x + 7, y)]
        elif direction == "RIGHT":
            pts = [(x + 16, y), (x, y - 7), (x, y - 3), (x - 16, y - 3), (x - 16, y + 3), (x, y + 3), (x, y + 7)]
        elif direction == "LEFT":
            pts = [(x - 16, y), (x, y - 7), (x, y - 3), (x + 16, y - 3), (x + 16, y + 3), (x, y + 3), (x, y + 7)]

        if pts:
            poly = np.array(pts, dtype=np.int32)
            cv2.fillPoly(img, [poly], (180, 185, 190))

    def _init_synthetic_scene(self) -> None:
        """Seed realistic vehicles accurately stationed in their respective road approaches."""
        self._road_bg = self._build_road_background()
        self._synth_vehicles = []
        self._next_vehicle_id = 1

        # Preset approach lanes (Left-Hand Drive, 2 sub-lanes per approach):
        # NORTH (incoming Southbound): center divider at x=640, incoming on left: x in [510, 640]
        # SOUTH (incoming Northbound): center divider at x=640, incoming on left: x in [640, 770]
        # WEST (incoming Eastbound): center divider at y=360, incoming on left: y in [360, 490]
        # EAST (incoming Westbound): center divider at y=360, incoming on left: y in [230, 360]

        initial_specs = [
            # NORTH queue / approaches (moving DOWN)
            {"direction": "NORTH", "sub_lane": 0, "pos": 60, "type": "car", "speed": 4.0},
            {"direction": "NORTH", "sub_lane": 1, "pos": -40, "type": "bus", "speed": 3.6},
            {"direction": "NORTH", "sub_lane": 0, "pos": -140, "type": "car", "speed": 4.2},

            # SOUTH queue / approaches (moving UP)
            {"direction": "SOUTH", "sub_lane": 0, "pos": 660, "type": "car", "speed": 4.0},
            {"direction": "SOUTH", "sub_lane": 1, "pos": 760, "type": "truck", "speed": 3.5},
            {"direction": "SOUTH", "sub_lane": 0, "pos": 860, "type": "motorcycle", "speed": 4.5},

            # WEST queue / approaches (moving RIGHT)
            {"direction": "WEST", "sub_lane": 0, "pos": 380, "type": "car", "speed": 4.2},
            {"direction": "WEST", "sub_lane": 1, "pos": 240, "type": "bus", "speed": 3.8},
            {"direction": "WEST", "sub_lane": 0, "pos": 100, "type": "car", "speed": 4.0},

            # EAST queue / approaches (moving LEFT)
            {"direction": "EAST", "sub_lane": 0, "pos": 900, "type": "car", "speed": 4.0},
            {"direction": "EAST", "sub_lane": 1, "pos": 1040, "type": "car", "speed": 4.2},
            {"direction": "EAST", "sub_lane": 0, "pos": 1180, "type": "ambulance", "speed": 5.2},
        ]

        for s in initial_specs:
            self._create_vehicle(
                direction=s["direction"],
                sub_lane=s["sub_lane"],
                pos=s["pos"],
                veh_type=s["type"],
                target_speed=s["speed"]
            )

    def _create_vehicle(
        self,
        direction: str,
        sub_lane: int = 0,
        pos: float = 0.0,
        veh_type: str = "car",
        target_speed: float = 4.0
    ) -> None:
        """Create and spawn a vehicle strictly constrained to its road lane."""
        veh_id = self._next_vehicle_id
        self._next_vehicle_id += 1

        # Palette: distinct modern vehicle tones
        car_colors = [
            (210, 75, 45),    # Sapphire Blue
            (45, 175, 70),    # Forest Green
            (40, 45, 215),    # Crimson Red
            (225, 225, 230),  # Pearl White
            (70, 75, 80),     # Slate Grey
            (30, 185, 235),   # Taxi Amber
            (145, 40, 160)    # Deep Purple
        ]

        if veh_type == "ambulance":
            color = (250, 250, 250)
            dims = (48, 92)  # (width, length)
        elif veh_type == "bus":
            color = (25, 180, 240) if random.random() < 0.6 else (45, 160, 50)
            dims = (50, 126)
        elif veh_type == "truck":
            color = (80, 95, 105)
            dims = (50, 118)
        elif veh_type == "motorcycle":
            color = (30, 30, 220)
            dims = (24, 46)
        else:
            color = random.choice(car_colors)
            dims = (44, 78)

        # Coordinate positioning strictly inside the chosen lane
        # North-South incoming lanes: x = 550 (lane 0) or 605 (lane 1)
        # South-North incoming lanes: x = 675 (lane 0) or 735 (lane 1)
        # West-East incoming lanes: y = 395 (lane 0) or 455 (lane 1)
        # East-West incoming lanes: y = 265 (lane 0) or 325 (lane 1)

        if direction == "NORTH":
            x = 550 if sub_lane == 0 else 605
            y = float(pos)
            w, h = dims[0], dims[1]
            dx, dy = 0.0, target_speed
        elif direction == "SOUTH":
            x = 675 if sub_lane == 0 else 735
            y = float(pos)
            w, h = dims[0], dims[1]
            dx, dy = 0.0, -target_speed
        elif direction == "WEST":
            x = float(pos)
            y = 395 if sub_lane == 0 else 455
            w, h = dims[1], dims[0]  # Horizontal orientation!
            dx, dy = target_speed, 0.0
        else:  # EAST
            x = float(pos)
            y = 265 if sub_lane == 0 else 325
            w, h = dims[1], dims[0]  # Horizontal orientation!
            dx, dy = -target_speed, 0.0

        self._synth_vehicles.append({
            "id": veh_id,
            "type": veh_type,
            "direction": direction,
            "sub_lane": sub_lane,
            "x": float(x),
            "y": float(y),
            "w": int(w),
            "h": int(h),
            "dx": dx,
            "dy": dy,
            "speed": target_speed,
            "target_speed": target_speed,
            "color": color,
            "crossed_stop_line": False,
            "braking": False,
            "is_stopped": False,
            "strobe_timer": 0
        })

    def start_stream(self) -> bool:
        """Start capturing video from source, fallback to synthetic if mock or invalid."""
        self.frame_count = 0

        can_open = False
        if str(self.source).lower() == "mock":
            self.use_synthetic = True
            self.is_streaming = True
            self.sensor.heartbeat()
            return True

        if isinstance(self.source, int):
            can_open = True
        elif isinstance(self.source, (str, Path)):
            src_str = str(self.source)
            if src_str.isdigit():
                self.source = int(src_str)
                can_open = True
            elif Path(src_str).exists():
                can_open = True
            elif src_str.startswith(("http://", "https://", "rtsp://")):
                can_open = True

        if can_open:
            try:
                self.cap = cv2.VideoCapture(self.source)
                if self.cap.isOpened():
                    self.is_streaming = True
                    self.use_synthetic = False
                    self.sensor.heartbeat()
                    return True
            except Exception:
                pass

        # Fallback to high-quality synthetic traffic scene
        self.use_synthetic = True
        self.is_streaming = True
        self.sensor.heartbeat()
        return True

    def capture_video(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read and return next frame. Gracefully loops video or renders synthetic frame."""
        if not self.is_streaming:
            return False, None

        self.sensor.heartbeat()
        self.frame_count += 1

        if not self.use_synthetic and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                return True, frame
            else:
                # Video ended, loop back to start if it's a file
                if isinstance(self.source, (str, Path)) and not str(self.source).isdigit():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        frame = cv2.resize(frame, (self.width, self.height))
                        return True, frame

        # Render synthetic traffic frame with strict lane-following
        return True, self._generate_synthetic_frame()

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Update physics and render crisp 4-way intersection frame."""
        if self._road_bg is None:
            self._road_bg = self._build_road_background()

        # Internal fallback timer for standalone simulation without external controller
        self.internal_signal_timer += 0.033
        if self.internal_signal_timer > 14.0:
            self.internal_signal_timer = 0.0
            phases = ["NORTH", "EAST", "SOUTH", "WEST"]
            self.internal_phase_idx = (self.internal_phase_idx + 1) % len(phases)
            active = phases[self.internal_phase_idx]
            for p in phases:
                self.current_signals[p] = "GREEN" if p == active else "RED"

        frame = self._road_bg.copy()

        # Update vehicle positions & queues
        self._update_vehicles()

        # Draw vehicles with drop shadows, windows, and light dynamics
        for v in self._synth_vehicles:
            self._draw_vehicle(frame, v)

        # Draw overhead LED signal heads at the 4 approach stops
        self._draw_signal_heads(frame)

        # Telemetry HUD overlay banner
        active_cnt = len(self._synth_vehicles)
        hud_text = f"SIMULATED 4-WAY INTERSECTION | FRAME: {self.frame_count:05d} | ACTIVE VEHICLES: {active_cnt} | LANES: ALIGNED"
        cv2.putText(frame, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)

        return frame

    def _update_vehicles(self) -> None:
        """Simulate intelligent driver behavior, safe car-following buffers, and stop line compliance."""
        dt = 0.033

        # Group vehicles by direction and sub-lane to enforce proper queuing order
        lanes: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for v in self._synth_vehicles:
            key = (v["direction"], v["sub_lane"])
            if key not in lanes:
                lanes[key] = []
            lanes[key].append(v)

        # Sort vehicles in each lane by distance to intersection
        for (direction, sub_lane), v_list in lanes.items():
            if direction == "NORTH":
                # Moving down: higher y is closer to/past intersection
                v_list.sort(key=lambda item: item["y"], reverse=True)
            elif direction == "SOUTH":
                # Moving up: lower y is closer to/past intersection
                v_list.sort(key=lambda item: item["y"])
            elif direction == "WEST":
                # Moving right: higher x is closer to/past intersection
                v_list.sort(key=lambda item: item["x"], reverse=True)
            elif direction == "EAST":
                # Moving left: lower x is closer to/past intersection
                v_list.sort(key=lambda item: item["x"])

            # Stop line coordinates for each approach
            # NORTH: stop line at y = 200 (crossing going DOWN)
            # SOUTH: stop line at y = 520 (crossing going UP)
            # WEST: stop line at x = 480 (crossing going RIGHT)
            # EAST: stop line at x = 800 (crossing going LEFT)

            signal_color = self.current_signals.get(direction, "RED")

            for i, v in enumerate(v_list):
                lead_v = v_list[i - 1] if i > 0 else None

                # Determine if stop line has been passed
                if direction == "NORTH" and v["y"] >= 205:
                    v["crossed_stop_line"] = True
                elif direction == "SOUTH" and v["y"] <= 515:
                    v["crossed_stop_line"] = True
                elif direction == "WEST" and v["x"] >= 485:
                    v["crossed_stop_line"] = True
                elif direction == "EAST" and v["x"] <= 795:
                    v["crossed_stop_line"] = True

                # Determine target stopping distance
                should_stop = False
                stop_pos = 0.0

                if not v["crossed_stop_line"]:
                    # Must stop at stop line if signal is RED or YELLOW (unless emergency vehicle)
                    if signal_color in ["RED", "YELLOW"] and v["type"] != "ambulance":
                        should_stop = True
                        if direction == "NORTH":
                            stop_pos = 195 - v["h"] // 2
                        elif direction == "SOUTH":
                            stop_pos = 525 + v["h"] // 2
                        elif direction == "WEST":
                            stop_pos = 475 - v["w"] // 2
                        elif direction == "EAST":
                            stop_pos = 805 + v["w"] // 2

                # Check car-following safe headway behind lead vehicle in this lane
                if lead_v is not None:
                    safe_gap = 22
                    if direction == "NORTH":
                        max_y = lead_v["y"] - (lead_v["h"] // 2 + v["h"] // 2 + safe_gap)
                        if should_stop:
                            stop_pos = min(stop_pos, max_y)
                        else:
                            should_stop = True
                            stop_pos = max_y
                    elif direction == "SOUTH":
                        min_y = lead_v["y"] + (lead_v["h"] // 2 + v["h"] // 2 + safe_gap)
                        if should_stop:
                            stop_pos = max(stop_pos, min_y)
                        else:
                            should_stop = True
                            stop_pos = min_y
                    elif direction == "WEST":
                        max_x = lead_v["x"] - (lead_v["w"] // 2 + v["w"] // 2 + safe_gap)
                        if should_stop:
                            stop_pos = min(stop_pos, max_x)
                        else:
                            should_stop = True
                            stop_pos = max_x
                    elif direction == "EAST":
                        min_x = lead_v["x"] + (lead_v["w"] // 2 + v["w"] // 2 + safe_gap)
                        if should_stop:
                            stop_pos = max(stop_pos, min_x)
                        else:
                            should_stop = True
                            stop_pos = min_x

                # Smoothly accelerate or brake towards target position
                if should_stop:
                    if direction == "NORTH":
                        dist = stop_pos - v["y"]
                        if dist <= 2:
                            v["speed"] = 0.0
                            v["y"] = min(v["y"], stop_pos)
                            v["is_stopped"] = True
                            v["braking"] = True
                        else:
                            v["speed"] = min(v["target_speed"], max(0.8, dist * 0.15))
                            v["y"] += v["speed"]
                            v["is_stopped"] = False
                            v["braking"] = True
                    elif direction == "SOUTH":
                        dist = v["y"] - stop_pos
                        if dist <= 2:
                            v["speed"] = 0.0
                            v["y"] = max(v["y"], stop_pos)
                            v["is_stopped"] = True
                            v["braking"] = True
                        else:
                            v["speed"] = min(v["target_speed"], max(0.8, dist * 0.15))
                            v["y"] -= v["speed"]
                            v["is_stopped"] = False
                            v["braking"] = True
                    elif direction == "WEST":
                        dist = stop_pos - v["x"]
                        if dist <= 2:
                            v["speed"] = 0.0
                            v["x"] = min(v["x"], stop_pos)
                            v["is_stopped"] = True
                            v["braking"] = True
                        else:
                            v["speed"] = min(v["target_speed"], max(0.8, dist * 0.15))
                            v["x"] += v["speed"]
                            v["is_stopped"] = False
                            v["braking"] = True
                    elif direction == "EAST":
                        dist = v["x"] - stop_pos
                        if dist <= 2:
                            v["speed"] = 0.0
                            v["x"] = max(v["x"], stop_pos)
                            v["is_stopped"] = True
                            v["braking"] = True
                        else:
                            v["speed"] = min(v["target_speed"], max(0.8, dist * 0.15))
                            v["x"] -= v["speed"]
                            v["is_stopped"] = False
                            v["braking"] = True
                else:
                    # Accelerate towards cruise speed
                    v["speed"] = min(v["target_speed"], v["speed"] + 0.25)
                    v["is_stopped"] = False
                    v["braking"] = False

                    if direction == "NORTH":
                        v["y"] += v["speed"]
                    elif direction == "SOUTH":
                        v["y"] -= v["speed"]
                    elif direction == "WEST":
                        v["x"] += v["speed"]
                    elif direction == "EAST":
                        v["x"] -= v["speed"]

        # Remove vehicles that exited the frame bounds
        surviving = []
        for v in self._synth_vehicles:
            if v["direction"] == "NORTH" and v["y"] > self.height + 90:
                continue
            if v["direction"] == "SOUTH" and v["y"] < -90:
                continue
            if v["direction"] == "WEST" and v["x"] > self.width + 90:
                continue
            if v["direction"] == "EAST" and v["x"] < -90:
                continue
            surviving.append(v)
        self._synth_vehicles = surviving

        # Spawn new vehicles if lanes have adequate clearance at boundaries
        now = time.time()
        if len(self._synth_vehicles) < 14 and (now - self._last_spawn_time > 1.2):
            self._last_spawn_time = now
            self._spawn_random_vehicle()

    def _spawn_random_vehicle(self) -> None:
        """Spawn a vehicle at the entrance of a road approach if clear."""
        dirs = ["NORTH", "SOUTH", "EAST", "WEST"]
        random.shuffle(dirs)

        types = ["car", "car", "car", "bus", "truck", "motorcycle"]
        # 4% chance of emergency ambulance
        chosen_type = "ambulance" if random.random() < 0.04 else random.choice(types)

        for d in dirs:
            sub_lane = random.choice([0, 1])

            # Check if entrance is clear
            is_clear = True
            for v in self._synth_vehicles:
                if v["direction"] == d and v["sub_lane"] == sub_lane:
                    if d == "NORTH" and v["y"] < 80:
                        is_clear = False
                    elif d == "SOUTH" and v["y"] > self.height - 80:
                        is_clear = False
                    elif d == "WEST" and v["x"] < 80:
                        is_clear = False
                    elif d == "EAST" and v["x"] > self.width - 80:
                        is_clear = False

            if is_clear:
                pos = -70 if d in ["NORTH", "WEST"] else (self.height + 70 if d == "SOUTH" else self.width + 70)
                speed = 4.8 if chosen_type == "ambulance" else random.uniform(3.6, 4.6)
                self._create_vehicle(
                    direction=d,
                    sub_lane=sub_lane,
                    pos=pos,
                    veh_type=chosen_type,
                    target_speed=speed
                )
                break

    def _draw_vehicle(self, img: np.ndarray, v: Dict[str, Any]) -> None:
        """Render high-fidelity vehicle chassis, windshields, headlights, and taillights."""
        vx, vy = int(v["x"]), int(v["y"])
        vw, vh = int(v["w"]), int(v["h"])
        x1, y1 = vx - vw // 2, vy - vh // 2
        x2, y2 = vx + vw // 2, vy + vh // 2

        # 1. Soft Shadow
        cv2.rectangle(img, (x1 - 3, y1 - 2), (x2 + 3, y2 + 4), (20, 22, 24), -1)

        # 2. Vehicle Main Chassis with subtle border
        body_color = v["color"]
        cv2.rectangle(img, (x1, y1), (x2, y2), body_color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (25, 28, 30), 2)

        # 3. Windshields & Roof Details
        glass_color = (25, 30, 35)

        if v["direction"] in ["NORTH", "SOUTH"]:
            # Vertical vehicle
            # Front windshield
            fw_y1 = y2 - 20 if v["direction"] == "NORTH" else y1 + 6
            fw_y2 = y2 - 6 if v["direction"] == "NORTH" else y1 + 20
            cv2.rectangle(img, (x1 + 4, min(fw_y1, fw_y2)), (x2 - 4, max(fw_y1, fw_y2)), glass_color, -1)

            # Rear windshield
            rw_y1 = y1 + 6 if v["direction"] == "NORTH" else y2 - 20
            rw_y2 = y1 + 18 if v["direction"] == "NORTH" else y2 - 6
            cv2.rectangle(img, (x1 + 5, min(rw_y1, rw_y2)), (x2 - 5, max(rw_y1, rw_y2)), glass_color, -1)

            # Headlights (front corners)
            hl_y = y2 - 2 if v["direction"] == "NORTH" else y1 + 2
            hl_color = (180, 255, 255)
            cv2.circle(img, (x1 + 5, hl_y), 3, hl_color, -1)
            cv2.circle(img, (x2 - 5, hl_y), 3, hl_color, -1)

            # Taillights (rear corners)
            tl_y = y1 + 2 if v["direction"] == "NORTH" else y2 - 2
            tl_color = (0, 0, 255) if v.get("braking", False) else (0, 0, 160)
            tl_radius = 4 if v.get("braking", False) else 3
            cv2.circle(img, (x1 + 5, tl_y), tl_radius, tl_color, -1)
            cv2.circle(img, (x2 - 5, tl_y), tl_radius, tl_color, -1)

        else:
            # Horizontal vehicle (WEST or EAST)
            # Front windshield
            fw_x1 = x2 - 20 if v["direction"] == "WEST" else x1 + 6
            fw_x2 = x2 - 6 if v["direction"] == "WEST" else x1 + 20
            cv2.rectangle(img, (min(fw_x1, fw_x2), y1 + 4), (max(fw_x1, fw_x2), y2 - 4), glass_color, -1)

            # Rear windshield
            rw_x1 = x1 + 6 if v["direction"] == "WEST" else x2 - 20
            rw_x2 = x1 + 18 if v["direction"] == "WEST" else x2 - 6
            cv2.rectangle(img, (min(rw_x1, rw_x2), y1 + 5), (max(rw_x1, rw_x2), y2 - 5), glass_color, -1)

            # Headlights (front corners)
            hl_x = x2 - 2 if v["direction"] == "WEST" else x1 + 2
            hl_color = (180, 255, 255)
            cv2.circle(img, (hl_x, y1 + 5), 3, hl_color, -1)
            cv2.circle(img, (hl_x, y2 - 5), 3, hl_color, -1)

            # Taillights (rear corners)
            tl_x = x1 + 2 if v["direction"] == "WEST" else x2 - 2
            tl_color = (0, 0, 255) if v.get("braking", False) else (0, 0, 160)
            tl_radius = 4 if v.get("braking", False) else 3
            cv2.circle(img, (tl_x, y1 + 5), tl_radius, tl_color, -1)
            cv2.circle(img, (tl_x, y2 - 5), tl_radius, tl_color, -1)

        # 4. Ambulance Emergency Beacon
        if v["type"] == "ambulance":
            v["strobe_timer"] = (v.get("strobe_timer", 0) + 1) % 12
            c_left = (0, 0, 255) if v["strobe_timer"] < 6 else (255, 0, 0)
            c_right = (255, 0, 0) if v["strobe_timer"] < 6 else (0, 0, 255)
            cv2.circle(img, (vx - 6, vy), 4, c_left, -1)
            cv2.circle(img, (vx + 6, vy), 4, c_right, -1)
            # Red cross emblem
            cv2.line(img, (vx - 5, vy), (vx + 5, vy), (0, 0, 220), 2)
            cv2.line(img, (vx, vy - 5), (vx, vy + 5), (0, 0, 220), 2)

    def _draw_signal_heads(self, img: np.ndarray) -> None:
        """Render animated overhead traffic light signals for all 4 approaches."""
        signal_color_map = {
            "RED": (30, 30, 240),
            "YELLOW": (0, 215, 255),
            "GREEN": (40, 230, 70)
        }

        # Positions next to each stop line
        heads = {
            "NORTH": (self.cx - self.half_road - 28, 185),
            "SOUTH": (self.cx + self.half_road + 12, 510),
            "WEST": (465, self.cy + self.half_road + 10),
            "EAST": (785, self.cy - self.half_road - 28)
        }

        for d, (hx, hy) in heads.items():
            state = self.current_signals.get(d, "RED")
            active_c = signal_color_map.get(state, (0, 0, 240))

            # Light housing
            cv2.rectangle(img, (hx, hy), (hx + 18, hy + 32), (20, 24, 28), -1)
            cv2.rectangle(img, (hx, hy), (hx + 18, hy + 32), (90, 95, 100), 1)

            # Active glowing bulb
            bulb_y = hy + 8 if state == "RED" else (hy + 16 if state == "YELLOW" else hy + 24)
            cv2.circle(img, (hx + 9, bulb_y), 5, active_c, -1)
            cv2.circle(img, (hx + 9, bulb_y), 7, active_c, 1)

    def get_active_vehicle_boxes(self) -> List[Dict[str, Any]]:
        """Return ground-truth vehicle detections synchronized with synthetic simulation."""
        results = []
        for v in self._synth_vehicles:
            vx, vy = int(v["x"]), int(v["y"])
            vw, vh = int(v["w"]), int(v["h"])
            x1, y1 = max(0, vx - vw // 2), max(0, vy - vh // 2)
            x2, y2 = min(self.width, vx + vw // 2), min(self.height, vy + vh // 2)

            t_id = v["id"]
            results.append({
                "tracker_id": t_id,
                "box": [x1, y1, x2, y2],
                "confidence": 0.94,
                "vehicle_type": v["type"],
                "centroid": (vx, vy),
                "plate": f"KA-04-TR-{1000 + (t_id * 37) % 9000}"
            })
        return results

    def stop_stream(self) -> None:
        """Release camera device or file reader."""
        self.is_streaming = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
