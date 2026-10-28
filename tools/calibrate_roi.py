"""Interactive ROI, Counting-Line, Stop-Line, and No-Parking Zone Calibration Tool.

Allows clicking points on a video frame or synthetic frame to define coordinates,
saving them to config_data/roi_config.json.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from traffic_ai.config import DEFAULT_ROI_CONFIG_FILE, DEFAULT_ROI_CONFIG, SAMPLE_VIDEOS_DIR


class ROICalibrator:
    """Manages OpenCV window mouse events to define traffic zones."""

    def __init__(self, video_path: str | None = None, output_file: str | None = None):
        self.video_path = video_path
        self.output_file = Path(output_file or DEFAULT_ROI_CONFIG_FILE)
        self.config: Dict[str, Any] = dict(DEFAULT_ROI_CONFIG)

        self.current_direction = "NORTH"
        self.current_mode = "roi_polygon"  # "roi_polygon", "counting_line", "stop_line", "no_parking"
        self.active_points: List[List[int]] = []
        self.frame: np.ndarray | None = None
        self.window_name = "Traffic AI - ROI Calibration Tool"

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for point selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.active_points.append([int(x), int(y)])
            print(f"Point added: ({x}, {y}) for {self.current_direction} - {self.current_mode}")

    def load_initial_frame(self) -> np.ndarray:
        """Capture one frame from video, camera, or generate synthetic overhead frame."""
        if self.video_path and Path(self.video_path).exists():
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return cv2.resize(frame, (1280, 720))

        # Generate reference background
        from traffic_ai.core.camera import Camera
        cam = Camera(source="mock", width=1280, height=720)
        cam.start_stream()
        _, frame = cam.capture_video()
        cam.stop_stream()
        return frame if frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8)

    def save_config(self) -> None:
        """Serialize configuration to JSON."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as f:
            json.dump(self.config, f, indent=4)
        print(f"\n[SUCCESS] Saved ROI configuration to: {self.output_file}\n")

    def run_interactive(self) -> None:
        """Launch interactive GUI session."""
        self.frame = self.load_initial_frame()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("=" * 60)
        print(" Traffic AI - Interactive Calibration Helper")
        print("=" * 60)
        print(" CONTROLS:")
        print("  1-4 : Select direction (1: NORTH, 2: SOUTH, 3: EAST, 4: WEST)")
        print("  'r' : Mode -> Lane ROI Polygon (Click 4 points)")
        print("  'c' : Mode -> Virtual Counting Line (Click 2 points)")
        print("  's' : Mode -> Stop Line (Click 2 points)")
        print("  'p' : Mode -> No Parking Zone (Click 4 points)")
        print("  ENTER: Commit current points to selected direction/mode")
        print("  'w' : Write and Save configuration to JSON")
        print("  'x' : Clear active points")
        print("  'q' : Quit")
        print("=" * 60)

        dir_map = {"1": "NORTH", "2": "SOUTH", "3": "EAST", "4": "WEST"}

        while True:
            display = self.frame.copy()

            # Render existing lanes
            for d, lane in self.config.get("lanes", {}).items():
                poly = np.array(lane.get("roi_polygon", []), dtype=np.int32)
                if len(poly) >= 3:
                    cv2.polylines(display, [poly], True, (0, 255, 0), 2)
                    cv2.putText(display, f"{d} ROI", (poly[0][0], poly[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cnt_line = lane.get("counting_line", [])
                if len(cnt_line) == 2:
                    cv2.line(display, tuple(cnt_line[0]), tuple(cnt_line[1]), (255, 255, 0), 2)

                stp_line = lane.get("stop_line", [])
                if len(stp_line) == 2:
                    cv2.line(display, tuple(stp_line[0]), tuple(stp_line[1]), (0, 0, 255), 2)

            # Render no parking zones
            for zone in self.config.get("no_parking_zones", []):
                poly = np.array(zone.get("polygon", []), dtype=np.int32)
                if len(poly) >= 3:
                    cv2.polylines(display, [poly], True, (0, 165, 255), 2)

            # Render active points currently being clicked
            for pt in self.active_points:
                cv2.circle(display, tuple(pt), 5, (0, 0, 255), -1)
            if len(self.active_points) > 1:
                cv2.polylines(display, [np.array(self.active_points, dtype=np.int32)], False, (0, 0, 255), 2)

            # HUD banner
            hud = f"DIR: {self.current_direction} | MODE: {self.current_mode} | PTS: {len(self.active_points)} | Press 'w' to Save"
            cv2.rectangle(display, (10, 10), (750, 40), (0, 0, 0), -1)
            cv2.putText(display, hud, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif chr(key) in dir_map:
                self.current_direction = dir_map[chr(key)]
                print(f"Switched direction to: {self.current_direction}")
            elif key == ord('r'):
                self.current_mode = "roi_polygon"
                self.active_points = []
            elif key == ord('c'):
                self.current_mode = "counting_line"
                self.active_points = []
            elif key == ord('s'):
                self.current_mode = "stop_line"
                self.active_points = []
            elif key == ord('p'):
                self.current_mode = "no_parking"
                self.active_points = []
            elif key == ord('x'):
                self.active_points = []
                print("Active points cleared.")
            elif key == 13:  # ENTER
                if self.current_mode in ["roi_polygon", "counting_line", "stop_line"]:
                    self.config["lanes"][self.current_direction][self.current_mode] = list(self.active_points)
                    print(f"Committed {len(self.active_points)} points to {self.current_direction}.{self.current_mode}")
                elif self.current_mode == "no_parking":
                    self.config["no_parking_zones"].append({
                        "name": f"Zone_{len(self.config['no_parking_zones']) + 1}",
                        "polygon": list(self.active_points)
                    })
                    print(f"Committed no-parking zone with {len(self.active_points)} points.")
                self.active_points = []
            elif key == ord('w'):
                self.save_config()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Calibrate ROI, counting lines, and stop lines.")
    parser.add_argument("--video", type=str, default=None, help="Path to video file or camera index")
    parser.add_argument("--output", type=str, default=str(DEFAULT_ROI_CONFIG_FILE), help="Output JSON path")
    parser.add_argument("--preset", action="store_true", help="Non-interactively save default calibration JSON")
    args = parser.parse_args()

    calibrator = ROICalibrator(video_path=args.video, output_file=args.output)
    if args.preset:
        calibrator.save_config()
    else:
        try:
            calibrator.run_interactive()
        except Exception as e:
            print(f"GUI not available ({e}), applying preset default calibration...")
            calibrator.save_config()


if __name__ == "__main__":
    main()
