# AI-Based Traffic Management System (AI-TMS)
### Autonomous Adaptive Signal Optimization & Computer Vision Telemetry
*A Full-Stack Computer Vision and Traffic Engineering Prototype for College Software Engineering*

---

## 1. Executive Summary

Urban traffic congestion results in billions of hours of commuter delay, increased carbon emissions, and high accident risks. Conventional fixed-time traffic light cycles operate on rigid pre-programmed timers that ignore real-time queue lengths, wasting green time on empty approaches while congested corridors suffer from gridlock.

The **AI-Based Traffic Management System (AI-TMS)** solves this by coupling real-time computer vision with adaptive signal automation:
1. **Real-time Vehicle Detection & Tracking**: Uses Ultralytics YOLOv8n to identify cars, motorcycles, buses, and trucks, tracked via ByteTrack algorithms to prevent double-counting.
2. **Dynamic Density & Road Occupancy Calculation**: Computes lane occupancy ratios combining polygon road area coverage and vehicle queues, classifying conditions as `LOW`, `MODERATE`, `HIGH`, or `SEVERE`.
3. **Adaptive Green-Time Allocation**: Dynamically scales green phases between 10s and 60s proportional to demand, providing extra acceleration clearance for heavy vehicles (buses and trucks).
4. **Emergency Vehicle Priority (Green Corridor)**: Preempts normal signal cycles to give ambulances, fire engines, or police convoys uninterrupted right-of-way.
5. **Human-in-the-Loop Infraction Audit**: Detects red-light jumps, wrong-way driving, and illegal parking, capturing timestamped evidence snapshots for human verification rather than automatic prosecution.
6. **Predictive Analytics & Reports**: Forecasts congestion bottlenecks using Scikit-Learn regression and exports official PDF/CSV audit reports.

---

## 2. System Architecture & UML Mapping

### 2.1 State Flow
```
Initialization -> Idle / Sensor Polling -> Video Acquisition -> AI Computer Vision Processing
      -> Traffic Density Analysis -> Signal Policy Decision
      -> [Emergency Detected? YES -> Priority Green Corridor | NO -> Normal Adaptive Cycle]
      -> Infraction & Congestion Alert Evaluation -> Dashboard MJPEG Streaming -> Operator Human Audit
```

### 2.2 Data Flow Architecture
```
+--------------------+        +---------------------+        +------------------------+
| Camera / Video     | ---->  | AI Vision Engine    | ---->  | Traffic Analyzer       |
| (Webcam/MP4/Synth) |        | (YOLOv8 + ByteTrack)|        | (Density, Occupancy)   |
+--------------------+        +---------------------+        +------------------------+
                                         |                                |
                                         v                                v
+--------------------+        +---------------------+        +------------------------+
| Operator Dashboard | <----  | Alert Manager       | <----  | Signal Controller      |
| (Flask + Chart.js) |        | (Alerts, Incidents) |        | (Adaptive Logic)       |
+--------------------+        +---------------------+        +------------------------+
          |                                                               |
          v                                                               v
+--------------------+                                       +------------------------+
| Human Verification |                                       | Traffic Signal Machine |
| & Audit Decisions  |                                       | (RED / YELLOW / GREEN) |
+--------------------+                                       +------------------------+
```

### 2.3 Module-to-UML Mapping Table

| UML Class / Component | Source File | Responsibilities |
| :--- | :--- | :--- |
| **`Camera`, `Sensor`** | `core/camera.py` | Video acquisition from files, webcams, or synthetic fallback; manages camera health. |
| **`AIEngine`** | `core/ai_engine.py` | YOLOv8 vehicle detection, ByteTrack tracking, virtual counting-line crossing, and HUD drawing. |
| **`TrafficAnalyzer`** | `core/traffic_analyzer.py` | Computes physical road occupancy, categorizes density levels, and calculates non-linear green time. |
| **`TrafficSignal`** | `core/traffic_signal.py` | Deterministic 4-way finite state machine (GREEN -> YELLOW -> ALL_RED), handling countdowns and holds. |
| **`SignalController`** | `core/signal_controller.py` | Coordinates signal scheduling, fixed-time baseline modes, manual overrides, and preemption. |
| **`AlertManager`** | `core/alert_manager.py` | Monitors congestion duration (>15s), categorizes alerts/incidents, and handles resolution. |
| **`ViolationDetector`** | `core/violation_detector.py` | Identifies red-light runners, wrong-way vehicles, and illegal parking; captures evidence snapshots. |
| **`AccidentDetector`** | `core/accident_detector.py` | Heuristics for sudden stops and vehicle bounding-box IoU overlap collisions. |
| **`EmergencyDetector`** | `core/emergency_detector.py` | Detects emergency vehicle tags, activates priority green corridors, and manages auto-clear. |
| **`Database`** | `db/database.py` | SQLAlchemy persistence, query methods, data backups, and demo data seeders. |
| **`Models`** | `db/models.py` | `User`, `Vehicle`, `TrafficData`, `Alert`, `Incident`, `SignalLog`, `Violation`, `Report`. |
| **`ReportService`** | `services/report_service.py` | Compiles traffic audits into structured CSV data sheets and ReportLab PDF documents. |
| **`auth_service`** | `services/auth_service.py` | Role-Based Access Control (RBAC) decorators and permission checks. |
| **`PredictionService`** | `services/prediction_service.py` | Scikit-Learn RandomForest model predicting day/hour congestion and identifying bottlenecks. |
| **`IntersectionSim`** | `simulation/intersection_sim.py` | Benchmarks Fixed-Time vs Adaptive modes with Poisson arrivals; optional Pygame GUI. |
| **`ROICalibrator`** | `tools/calibrate_roi.py` | Helper script to calibrate lanes, counting lines, stop lines, and no-parking polygons. |

---

## 3. Technology Stack

- **Language & Runtime**: Python 3.10+ (tested on Python 3.13)
- **Computer Vision**: OpenCV (`cv2`), Ultralytics YOLOv8n, ByteTrack
- **Deep Learning**: PyTorch (`torch`, `torchvision`)
- **Backend Web Server**: Flask 3.1, Werkzeug
- **Database & ORM**: SQLite, SQLAlchemy, Flask-SQLAlchemy
- **Authentication & Security**: Flask-Login, Werkzeug password hashing, RBAC decorators
- **Machine Learning & Math**: NumPy, SciPy, Scikit-Learn (`RandomForestRegressor`), Pandas
- **Reporting & Visualization**: ReportLab (PDF), Pygame (Simulation GUI), Chart.js (Dashboard Web Telemetry)
- **Quality Assurance**: Pytest (20 automated unit and integration tests)

---

## 4. Role-Based Access Control (RBAC) & Default Credentials

The system provides pre-configured role-based user accounts for demonstrations and vivas:

| Role | Username | Password | Permitted Actions |
| :--- | :--- | :--- | :--- |
| **Administrator** | `admin` | `Admin@123` | Full access: user provisioning, system backups, camera settings, manual signal overrides, reports. |
| **Control Operator** | `operator` | `Operator@123` | Live CCTV surveillance HUD, manual signal override, emergency green corridor, alert resolution. |
| **Traffic Police** | `police` | `Police@123` | Infraction evidence audit (confirm/dismiss violations), manual emergency corridor, surveillance view. |
| **Urban Authority** | `authority` | `Authority@123` | Analytics portal, congestion trend forecasting, CSV & PDF audit report downloads. |
| **Public Citizen** | *(No Login Required)* | *(Open Access)* | Immediate read-only access to `/` for live corridor speeds, signal countdowns, and advisories. |
| *Legacy Road User* | `citizen` | `Citizen@123` | Kept for backwards-compatible RBAC regression tests (strictly blocked with 403 Forbidden on controls). |

> **Citizen / Commuter Access**: Citizens and commuters **do not need an account or login**. Anyone navigating to `http://127.0.0.1:5000/` immediately views the live municipal transit portal without any authentication barriers.

---

## 5. Setup & Installation Instructions

### Step 1: Create Virtual Environment
```powershell
# Navigate to the project directory
cd "d:\java web dev\traffic_ai"

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Automated Test Suite
Verify that all 20 unit and integration tests pass:
```powershell
pytest tests -v
```

---

## 6. How to Run

### Option A: Zero-Setup Instant Demo Mode (Recommended for Viva)
Starts the Flask server, loads the bundled 1280x720 sample traffic video, starts background AI worker threads, and seeds historical analytics:
```powershell
python app.py --demo --port 5000
```
- **Public Commuter Portal**: **`http://127.0.0.1:5000/`** (Zero login required: live signal countdowns, corridor ETAs, delay badges, transit advisories).
- **Staff Control Dashboard**: **`http://127.0.0.1:5000/dashboard`** (Click any staff preset like **Operator** or **Police**, then click **Sign In to Console**).

### Option B: Run with Custom Video File or Webcam
```powershell
# Custom recorded traffic video
python app.py --video "data/sample_videos/my_traffic_clip.mp4" --port 5000

# Live USB Webcam (Device 0)
python app.py --video 0 --port 5000
```

### Option C: Run 4-Way Intersection Simulation & Benchmark
```powershell
# Run headless benchmark comparing Fixed vs Adaptive mode (generates METRICS.md)
python simulation/intersection_sim.py --benchmark

# Launch graphical Pygame simulation window
python simulation/intersection_sim.py --gui
```

### Option D: Calibrate Custom Camera ROIs & Counting Lines
```powershell
# Interactive GUI (click to define lane polygons, counting lines, and stop lines)
python tools/calibrate_roi.py --video 0

# Apply preset reference coordinates without GUI
python tools/calibrate_roi.py --preset
```

---

## 7. Performance Evaluation (Fixed-Time vs AI-Adaptive)

From actual empirical simulation runs recorded in `METRICS.md` over 20-minute peak rush periods:

| Performance Metric | Fixed-Time Baseline | AI Adaptive Signal | Measured Improvement |
| :--- | :--- | :--- | :--- |
| **Average Waiting Time** | 32.88 seconds | 23.01 seconds | **30.0% Reduction** |
| **Vehicle Throughput** | 33.35 veh/min | 36.20 veh/min | **+8.5% Higher Capacity** |
| **Average Queue Backlog** | 10.45 vehicles | 9.98 vehicles | **4.5% Shorter Queues** |
| **Maximum Delay Experienced** | 63.00 seconds | 55.00 seconds | **-8.0s Peak Delay** |

*See [METRICS.md](METRICS.md) for full benchmark documentation.*

---

## 8. Dashboard Screenshots

### 8.1 Surveillance Command HUD (`/dashboard`)
*Live annotated video stream with YOLOv8 bounding boxes, ByteTrack IDs, animated LED signals, and real-time Chart.js telemetry:*
![Surveillance Dashboard](data/screenshots/dashboard.png)

### 8.2 Intelligence Reports & ML Forecasting (`/reports`)
*Historical volume metrics, Scikit-Learn congestion bottleneck predictions, and CSV/PDF export buttons:*
![Intelligence Reports](data/screenshots/reports.png)

### 8.3 Human Officer Violation Audit (`/violations`)
*Evidence review gallery for red-light running, wrong-way driving, and parking infractions:*
![Violation Audit](data/screenshots/violations.png)

### 8.4 Public Commuter Advisory Portal (`/`)
*Public-facing real-time lane condition summary for road users:*
![Public Portal](data/screenshots/public.png)

---

## 9. Engineering Limitations & Deployment Roadmap

1. **Environmental Sensitivity**:
   - Computer vision accuracy can degrade during extreme weather (dense fog, heavy rainfall), nighttime glare, or camera lens smudges. In production, infrared/thermal sensors and dual-band radar should complement optical feeds.
2. **Camera Occlusion & Angles**:
   - High-sided commercial vehicles (double-decker buses, container trucks) can temporarily occlude adjacent motorcycles or compact cars. Multi-camera fusion from opposing corners resolves blind spots.
3. **Hardware Integration & Safety Fallbacks**:
   - This prototype communicates with software-simulated signal state machines. Real-world physical traffic intersections require NEMA TS2 / 170 / 2070 controller hardware interfaces, conflict monitors (ensuring opposing greens can never occur simultaneously in hardware), and hard-wired fail-safe flasher relays.
4. **Human-in-the-Loop Safeguard**:
   - In accordance with ethical AI standards, all automated violation detections are tagged `PENDING_VERIFICATION` and require human traffic officer confirmation before any citation or notice is issued.
