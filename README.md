# 🎓 AI Exam Proctoring System

**Team:** Ahmed Mohamed Ezzat · Ahmed Mousa Mousa · Ahmed Ahmed Salah · Ahmed Ehab Kandeel · Ahmed Mohamed El Sayed

---

## 📋 Project Overview

A real-time AI system that monitors online exams using computer vision. It:
- Verifies student identity via **face recognition** (InsightFace)
- Tracks **gaze direction & head pose** (MediaPipe)
- Detects **mobile phones, laptops, and multiple persons** (YOLOv8 / COCO)
- Displays live alerts on a **Streamlit dashboard**
- Logs all violations to JSON and CSV for review

---

## 🏗️ Project Structure

```
exam_proctoring/
├── app.py                  # Streamlit dashboard (main UI)
├── proctoring_engine.py    # Core AI engine (all detection modules)
├── proctor_cli.py          # CLI proctoring (no UI needed)
├── register_student.py     # Student registration script
├── requirements.txt        # Python dependencies
├── README.md
└── data/
    ├── registered_faces/   # Student face images
    │   └── Name_ID/        # Per-student folder
    ├── screenshots/        # Auto-saved violation screenshots
    └── violation_log.json  # Violation history
```

---

## ⚙️ Installation

### 1. Clone / download the project

```bash
cd exam_proctoring
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** InsightFace requires a C++ compiler on Windows. If it fails, the system automatically falls back to OpenCV Haar Cascades for face detection.

### 4. Create data directories

```bash
mkdir -p data/registered_faces data/screenshots
```

---

## 🚀 Usage

### Option A: Streamlit Dashboard (recommended)

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

**Dashboard Tabs:**
| Tab | Description |
|-----|-------------|
| 📹 Live Proctoring | Real-time webcam monitoring with live alerts |
| 👤 Student Registration | Register student faces via webcam or upload |
| 📊 Exam Report | View violation history, charts, export CSV |
| ⚙️ Settings | Configure thresholds, notifications, etc. |

---

### Option B: CLI Mode (lightweight)

```bash
# Basic usage
python proctor_cli.py

# With student name
python proctor_cli.py --student "Ahmed Mohamed"

# Custom sensitivity and disable phone detection
python proctor_cli.py --student "Ahmed" --sensitivity 8 --no-phone

# Record output video
python proctor_cli.py --output recording.mp4 --show-fps
```

---

### Register Students (CLI)

```bash
# Capture from webcam (press SPACE to take each photo)
python register_student.py --name "Ahmed Mohamed" --id "20210001"

# Upload existing photo
python register_student.py --name "Sara Khaled" --id "20210002" --upload photo.jpg

# Capture 10 photos for better accuracy
python register_student.py --name "Omar Hassan" --id "20210003" --photos 10
```

---

## 🤖 AI Models

| Module | Model | Purpose | Fallback |
|--------|-------|---------|----------|
| Face Detection | InsightFace (buffalo_s) | Detect & recognize faces | OpenCV Haar Cascade |
| Gaze Tracking | MediaPipe Face Mesh (468 landmarks) | Head pose + eye direction | OpenCV eye detection |
| Object Detection | YOLOv8n (COCO-trained) | Phone, laptop, multi-person | MobileNet-SSD |

### Model Downloads

Models download automatically on first run:
- **InsightFace:** Downloads `buffalo_s` pack (~30MB) to `~/.insightface/`
- **YOLOv8n:** Downloads `yolov8n.pt` (~6MB) to `models/`

---

## 🚨 Violation Types

| Violation | Severity | Trigger |
|-----------|----------|---------|
| Multiple Persons Detected | CRITICAL | >1 face/person in frame |
| Phone Detected | CRITICAL | Mobile phone in frame |
| Identity Mismatch | CRITICAL | Unregistered person |
| Student Not Visible | HIGH | No face in frame |
| Looking Away (Left/Right) | HIGH | Head yaw > 20° |
| Looking Away (Up/Down) | MEDIUM | Head pitch > 15° |
| Laptop Detected | HIGH | Laptop in frame |

---

## ⚙️ Configuration

Edit settings in the Streamlit sidebar or pass as CLI args:

| Setting | Default | Description |
|---------|---------|-------------|
| `sensitivity` | 7 | Frames before gaze alert triggers (1=instant, 10=slow) |
| `recognition_threshold` | 0.6 | Face match confidence (0–1, lower=strict) |
| `gaze_threshold_h` | 20° | Horizontal head rotation before alert |
| `gaze_threshold_v` | 15° | Vertical head tilt before alert |
| `alert_cooldown` | 5s | Min seconds between same alert type |

---

## 📊 Dataset

- **COCO Dataset** — Used for object detection (person: class 0, cell phone: class 67, laptop: class 63)
- **Custom Face Dataset** — Student faces captured during registration stored in `data/registered_faces/`

---

## 🔧 Troubleshooting

**Camera not detected:**
```bash
# Test camera
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**InsightFace install fails (Windows):**
```bash
pip install insightface --no-build-isolation
# Or use conda: conda install -c conda-forge insightface
```

**MediaPipe fails:**
```bash
pip install mediapipe --upgrade
```

**Low FPS:**
- Reduce camera resolution in Settings → 640×480
- Disable phone detection if not needed (`--no-phone`)
- Object detection runs every 3rd frame by default

---

## 📝 Report Export

After a session, go to **📊 Exam Report** tab and click **⬇️ Download Report (CSV)**.

The CSV includes: time, violation type, severity, student name, confidence score.

Violations are also auto-saved to `data/violation_log.json`.

---

## 👥 Team

| Name | Role |
|------|------|
| Ahmed Mohamed Ezzat | Project Lead, Face Recognition |
| Ahmed Mousa Mousa | Gaze Tracking & MediaPipe |
| Ahmed Ahmed Salah | Object Detection & YOLO |
| Ahmed Ehab Kandeel | Streamlit Dashboard & UI |
| Ahmed Mohamed El Sayed | Data Pipeline & Reporting |
