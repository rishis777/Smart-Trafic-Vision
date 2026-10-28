"""Configuration parameters for the AI-Based Traffic Management System.

Contains thresholds, signal timings, ROI geometry, paths, and security settings.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_VIDEOS_DIR = DATA_DIR / "sample_videos"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
EXPORTS_DIR = DATA_DIR / "exports"
CONFIG_DIR = BASE_DIR / "config_data"

# Ensure runtime directories exist
for p in [DATA_DIR, SAMPLE_VIDEOS_DIR, SNAPSHOTS_DIR, EXPORTS_DIR, CONFIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Database
DB_PATH = DATA_DIR / "traffic_system.db"
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_PATH.as_posix()}"
SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY = os.environ.get("SECRET_KEY", "traffic-ai-super-secret-key-2026")

# Video & AI Engine Settings
DEFAULT_MODEL_NAME = "yolov8n.pt"
# COCO class IDs: 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}
DETECTION_CONFIDENCE = 0.35
TRACKER_TYPE = "bytetrack.yaml"
PROCESS_EVERY_N_FRAMES = 2  # Downsample for smooth CPU/GPU execution

# Traffic Density Thresholds (occupancy ratio or weighted vehicle count)
# Density is calculated as: min(1.0, (occupied_roi_area / total_roi_area) * 0.7 + (vehicle_count / max_capacity) * 0.3)
DENSITY_THRESHOLDS = {
    "LOW": 0.25,
    "MODERATE": 0.55,
    "HIGH": 0.80,
    "SEVERE": 1.00
}

# Signal Timings (seconds)
SIGNAL_TIMINGS = {
    "MIN_GREEN": 10,
    "MAX_GREEN": 60,
    "DEFAULT_GREEN": 25,
    "YELLOW": 4,
    "ALL_RED": 2,
    "FIXED_BASELINE_GREEN": 30  # For fixed-time comparison
}

# Directions and Phase mappings for 4-way intersection
DIRECTIONS = ["NORTH", "SOUTH", "EAST", "WEST"]
# Typically NORTH-SOUTH run together or sequential; we support both 4-phase and paired 2-phase
SIGNAL_PHASES = [
    {"name": "NORTH", "directions": ["NORTH"]},
    {"name": "EAST", "directions": ["EAST"]},
    {"name": "SOUTH", "directions": ["SOUTH"]},
    {"name": "WEST", "directions": ["WEST"]}
]

# Alert Thresholds
CONGESTION_ALERT_DURATION_SEC = 15.0  # Alert if HIGH/SEVERE persists >= 15s
ILLEGAL_PARKING_TIME_SEC = 10.0       # Alert if stationary in no-parking zone >= 10s
ACCIDENT_OVERLAP_IOU = 0.40           # Bounding box IOU overlap threshold
ACCIDENT_SUDDEN_STOP_SPEED = 2.0      # Pixels/sec indicating abrupt stop

# Default Geometry for 640x360 or 1280x720 frame calibration
DEFAULT_ROI_CONFIG_FILE = CONFIG_DIR / "roi_config.json"

DEFAULT_ROI_CONFIG = {
    "frame_width": 1280,
    "frame_height": 720,
    "lanes": {
        "NORTH": {
            "roi_polygon": [[510, 20], [640, 20], [640, 205], [510, 205]],
            "counting_line": [[510, 110], [640, 110]],
            "stop_line": [[510, 200], [640, 200]],
            "allowed_direction": "DOWN"
        },
        "SOUTH": {
            "roi_polygon": [[640, 515], [770, 515], [770, 700], [640, 700]],
            "counting_line": [[640, 610], [770, 610]],
            "stop_line": [[640, 520], [770, 520]],
            "allowed_direction": "UP"
        },
        "EAST": {
            "roi_polygon": [[795, 230], [1260, 230], [1260, 360], [795, 360]],
            "counting_line": [[1050, 230], [1050, 360]],
            "stop_line": [[800, 230], [800, 360]],
            "allowed_direction": "LEFT"
        },
        "WEST": {
            "roi_polygon": [[20, 360], [485, 360], [485, 490], [20, 490]],
            "counting_line": [[250, 360], [250, 490]],
            "stop_line": [[480, 360], [480, 490]],
            "allowed_direction": "RIGHT"
        }
    },
    "no_parking_zones": [
        {
            "name": "Zone_A_Bus_Stop",
            "polygon": [[120, 490], [320, 490], [320, 550], [120, 550]]
        }
    ]
}

# User Roles
ROLES = ["admin", "traffic_police", "control_operator", "authority", "road_user"]
