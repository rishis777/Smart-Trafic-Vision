"""Generate a realistic sample traffic video for zero-setup demoing.
"""

import sys
from pathlib import Path
import cv2

# Add package root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from traffic_ai.core.camera import Camera
from traffic_ai.config import SAMPLE_VIDEOS_DIR


def generate_traffic_demo_video(output_path: Path | None = None, num_frames: int = 450, fps: int = 30) -> Path:
    """Render a synthetic 1280x720 4-way intersection MP4 video."""
    target_path = output_path or (SAMPLE_VIDEOS_DIR / "traffic_demo.mp4")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    cam = Camera(source="mock", width=1280, height=720, fps=fps)
    cam.start_stream()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(target_path), fourcc, fps, (1280, 720))

    print(f"Generating {num_frames} frames of simulated traffic to {target_path}...")
    for f in range(num_frames):
        ret, frame = cam.capture_video()
        if ret and frame is not None:
            out.write(frame)

    out.release()
    cam.stop_stream()
    print(f"[SUCCESS] Sample traffic video created at: {target_path} ({target_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return target_path


if __name__ == "__main__":
    generate_traffic_demo_video()
