"""
AI Exam Proctoring System - Main Streamlit Dashboard
Team: Ahmed Mohamed Ezzat, Ahmed Mousa Mousa, Ahmed Ahmed Salah, Ahmed Ehab Kandeel, Ahmed Mohamed El Sayed
"""

import streamlit as st
import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
from pathlib import Path
import threading
from noise_test_page import render_noise_test

# Page configuration
st.set_page_config(
    page_title="AI Exam Proctoring System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: #0f1117; color: #e0e0e0; }
    
    .alert-card {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        color: white;
        box-shadow: 0 4px 15px rgba(255,68,68,0.3);
        animation: pulse 2s infinite;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ff9800, #e65100);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        color: white;
    }
    
    .ok-card {
        background: linear-gradient(135deg, #4caf50, #1b5e20);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        color: white;
    }
    
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: 700;
        color: #5c6bc0;
    }
    
    .metric-label {
        font-size: 0.85em;
        color: #9e9e9e;
        margin-top: 4px;
    }
    
    .student-badge {
        background: #1a237e;
        border: 2px solid #5c6bc0;
        border-radius: 50px;
        padding: 8px 20px;
        color: #c5cae9;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 4px 15px rgba(255,68,68,0.3); }
        50% { box-shadow: 0 4px 30px rgba(255,68,68,0.7); }
        100% { box-shadow: 0 4px 15px rgba(255,68,68,0.3); }
    }
    
    .sidebar .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stProgress > div > div { background: #5c6bc0; }
    
    h1, h2, h3 { color: #c5cae9 !important; }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-active { background: #4caf50; box-shadow: 0 0 8px #4caf50; }
    .status-inactive { background: #f44336; }
    .status-warning { background: #ff9800; box-shadow: 0 0 8px #ff9800; }
</style>
""", unsafe_allow_html=True)


def load_violation_log():
    """Load violation log from file."""
    log_path = Path("data/violation_log.json")
    if log_path.exists():
        with open(log_path, "r") as f:
            return json.load(f)
    return []


def save_violation_log(log):
    """Save violation log to file."""
    Path("data").mkdir(exist_ok=True)
    with open("data/violation_log.json", "w") as f:
        json.dump(log, f, indent=2)


def get_registered_students():
    """Get list of registered students."""
    faces_dir = Path("data/registered_faces")
    if not faces_dir.exists():
        return []
    return [d.name for d in faces_dir.iterdir() if d.is_dir()]


def render_header():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 20px 0;'>
            <h1 style='font-size:2.2em; margin:0;'>🎓 AI Exam Proctoring System</h1>
            <p style='color:#7986cb; margin:5px 0;'>Real-time monitoring powered by Computer Vision & AI</p>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        
        mode = st.selectbox(
            "Mode",
            ["📹 Live Proctoring", "👤 Student Registration", "📊 Exam Report", "🧪 Noise Test", "⚙️ Settings"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 🎓 Session Info")
        
        exam_name = st.text_input("Exam Name", value="Final Exam 2024")
        duration = st.number_input("Duration (min)", min_value=10, max_value=300, value=90)
        
        st.markdown("---")
        st.markdown("### 🔧 Detection Settings")
        
        sensitivity = st.slider("Alert Sensitivity", 1, 10, 7)
        
        st.markdown("**Detection Modules:**")
        detect_gaze = st.checkbox("👁️ Gaze Tracking", value=True)
        detect_face = st.checkbox("🆔 Face Verification", value=True)
        detect_phone = st.checkbox("📱 Phone Detection", value=True)
        detect_multi = st.checkbox("👥 Multiple Persons", value=True)
        
        st.markdown("---")
        
        registered = get_registered_students()
        st.markdown(f"### 👤 Students ({len(registered)})")
        for s in registered[:5]:
            st.markdown(f'<div class="student-badge">👤 {s}</div>', unsafe_allow_html=True)
        if len(registered) > 5:
            st.caption(f"...and {len(registered)-5} more")
        
        settings = {
            "sensitivity": sensitivity,
            "detect_gaze": detect_gaze,
            "detect_face": detect_face,
            "detect_phone": detect_phone,
            "detect_multi": detect_multi,
            "exam_name": exam_name,
            "duration": duration
        }
        
        return mode.split(" ", 1)[1], settings


def render_live_proctoring(settings):
    """Render live proctoring view."""
    from proctoring_engine import ProctoringEngine
    
    col_video, col_alerts = st.columns([3, 2])
    
    with col_video:
        st.markdown("### 📹 Live Feed")
        
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.markdown('<span class="status-indicator status-active"></span>**Camera Active**', unsafe_allow_html=True)
        with status_col2:
            st.markdown('<span class="status-indicator status-active"></span>**AI Running**', unsafe_allow_html=True)
        with status_col3:
            st.markdown(f'<span class="status-indicator status-warning"></span>**Sensitivity: {settings["sensitivity"]}**', unsafe_allow_html=True)
        
        frame_placeholder = st.empty()
        
    with col_alerts:
        st.markdown("### 🚨 Live Alerts")
        alerts_placeholder = st.empty()
        
        st.markdown("### 📊 Session Stats")
        metrics_placeholder = st.empty()
    
    # Initialize engine in session state
    if "engine" not in st.session_state:
        st.session_state.engine = ProctoringEngine(settings)
        st.session_state.violation_count = 0
        st.session_state.violations = []
        st.session_state.session_start = time.time()
    
    # Control buttons
    col_start, col_stop, col_snap = st.columns(3)
    
    with col_start:
        start = st.button("▶️ Start Monitoring", type="primary", use_container_width=True)
    with col_stop:
        stop = st.button("⏹️ Stop", use_container_width=True)
    with col_snap:
        snap = st.button("📸 Snapshot", use_container_width=True)
    
    if stop:
        st.session_state.running = False
        st.info("Monitoring stopped.")
        return
    
    if start:
        st.session_state.running = True
    
    if st.session_state.get("running", False):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot access webcam. Please check your camera connection.")
            # Show demo mode
            render_demo_mode(frame_placeholder, alerts_placeholder, metrics_placeholder, settings)
            return
        
        try:
            while st.session_state.get("running", False):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, violations = st.session_state.engine.process_frame(frame)
                
                # Update violations
                if violations:
                    for v in violations:
                        st.session_state.violations.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "type": v["type"],
                            "severity": v["severity"],
                            "confidence": v.get("confidence", 0.9)
                        })
                        st.session_state.violation_count += 1
                
                # Display frame
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update alerts
                _render_alerts(alerts_placeholder, st.session_state.violations[-5:])
                
                # Update metrics
                elapsed = int(time.time() - st.session_state.session_start)
                _render_metrics(metrics_placeholder, st.session_state.violation_count, elapsed)
                
                time.sleep(0.033)  # ~30 FPS
        finally:
            cap.release()
    else:
        # Show placeholder when not running
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder_img, "Click 'Start Monitoring' to begin", 
                   (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        frame_placeholder.image(placeholder_img, channels="BGR", use_container_width=True)
        
        _render_alerts(alerts_placeholder, [])
        _render_metrics(metrics_placeholder, 0, 0)


def render_demo_mode(frame_placeholder, alerts_placeholder, metrics_placeholder, settings):
    """Demo mode when no webcam is available."""
    st.warning("🎬 Running in Demo Mode (no webcam detected)")
    
    demo_violations = [
        {"time": "10:23:45", "type": "Looking Away", "severity": "HIGH", "confidence": 0.92},
        {"time": "10:24:12", "type": "Phone Detected", "severity": "CRITICAL", "confidence": 0.88},
        {"time": "10:25:01", "type": "Multiple Faces", "severity": "HIGH", "confidence": 0.95},
    ]
    
    # Create demo frame
    demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    demo_frame[:] = (20, 25, 40)
    
    # Draw fake face detection box
    cv2.rectangle(demo_frame, (220, 120), (420, 360), (0, 255, 100), 2)
    cv2.putText(demo_frame, "DEMO MODE", (230, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(demo_frame, "Student: Ahmed M.", (225, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
    cv2.putText(demo_frame, "ID: Verified ✓", (225, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
    
    # Add gaze arrows
    cv2.arrowedLine(demo_frame, (320, 200), (360, 200), (255, 150, 0), 2)
    cv2.putText(demo_frame, "GAZE: FORWARD", (240, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    frame_placeholder.image(demo_frame, channels="BGR", use_container_width=True)
    _render_alerts(alerts_placeholder, demo_violations)
    _render_metrics(metrics_placeholder, 3, 305)


def _render_alerts(placeholder, violations):
    with placeholder.container():
        if not violations:
            st.markdown('<div class="ok-card">✅ No violations detected</div>', unsafe_allow_html=True)
            return
        
        for v in reversed(violations[-5:]):
            severity = v.get("severity", "MEDIUM")
            card_class = "alert-card" if severity in ["HIGH", "CRITICAL"] else "warning-card"
            icon = "🚨" if severity == "CRITICAL" else "⚠️" if severity == "HIGH" else "ℹ️"
            st.markdown(f"""
            <div class="{card_class}">
                {icon} <strong>{v['type']}</strong><br>
                <small>⏰ {v['time']} | Confidence: {v.get('confidence', 0.9):.0%}</small>
            </div>
            """, unsafe_allow_html=True)


def _render_metrics(placeholder, violation_count, elapsed_seconds):
    with placeholder.container():
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{violation_count}</div>
                <div class="metric-label">Violations</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            mins = elapsed_seconds // 60
            secs = elapsed_seconds % 60
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{mins:02d}:{secs:02d}</div>
                <div class="metric-label">Elapsed</div>
            </div>
            """, unsafe_allow_html=True)


def render_registration():
    """Render student registration page."""
    st.markdown("## 👤 Student Registration")
    st.markdown("Register students by capturing their face for identity verification.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Student Details")
        student_name = st.text_input("Full Name", placeholder="e.g. Ahmed Mohamed")
        student_id = st.text_input("Student ID", placeholder="e.g. 20210001")
        
        st.markdown("### 📸 Capture Method")
        method = st.radio("", ["Webcam Capture", "Upload Photo"])
        
        if method == "Upload Photo":
            uploaded = st.file_uploader("Upload Student Photo", type=["jpg", "jpeg", "png"])
            if uploaded and student_name and student_id:
                if st.button("✅ Register Student", type="primary"):
                    _register_student_from_upload(student_name, student_id, uploaded)
        else:
            if student_name and student_id:
                if st.button("📸 Capture & Register", type="primary"):
                    _register_student_from_webcam(student_name, student_id)
    
    with col2:
        st.markdown("### 👥 Registered Students")
        students = get_registered_students()
        
        if students:
            for name in students:
                face_dir = Path(f"data/registered_faces/{name}")
                photos = list(face_dir.glob("*.jpg")) + list(face_dir.glob("*.png"))
                
                with st.expander(f"👤 {name}"):
                    if photos:
                        img = cv2.imread(str(photos[0]))
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, width=150)
                    st.caption(f"Photos: {len(photos)}")
                    if st.button(f"🗑️ Remove", key=f"del_{name}"):
                        import shutil
                        shutil.rmtree(face_dir)
                        st.rerun()
        else:
            st.info("No students registered yet.")


def _register_student_from_upload(name, student_id, uploaded_file):
    face_dir = Path(f"data/registered_faces/{name}_{student_id}")
    face_dir.mkdir(parents=True, exist_ok=True)
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is not None:
        save_path = face_dir / f"photo_0.jpg"
        cv2.imwrite(str(save_path), img)
        st.success(f"✅ Successfully registered {name} (ID: {student_id})")
        st.balloons()
    else:
        st.error("Failed to process image.")


def _register_student_from_webcam(name, student_id):
    face_dir = Path(f"data/registered_faces/{name}_{student_id}")
    face_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access webcam.")
        return
    
    frames_captured = 0
    placeholder = st.empty()
    progress = st.progress(0)
    
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            save_path = face_dir / f"photo_{i}.jpg"
            cv2.imwrite(str(save_path), frame)
            frames_captured += 1
            progress.progress((i+1)/5)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, width=300)
            time.sleep(0.5)
    
    cap.release()
    
    if frames_captured > 0:
        st.success(f"✅ Registered {name} with {frames_captured} photos!")
        st.balloons()
    else:
        st.error("Failed to capture photos.")


def render_report():
    """Render exam report."""
    st.markdown("## 📊 Exam Report")
    
    violations = load_violation_log()
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(violations)
    critical = sum(1 for v in violations if v.get("severity") == "CRITICAL")
    high = sum(1 for v in violations if v.get("severity") == "HIGH")
    students = len(set(v.get("student", "Unknown") for v in violations))
    
    for col, (val, label, color) in zip(
        [col1, col2, col3, col4],
        [(total, "Total Violations", "#5c6bc0"),
         (critical, "Critical", "#f44336"),
         (high, "High Severity", "#ff9800"),
         (students, "Students Flagged", "#4caf50")]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    if violations:
        import pandas as pd
        
        df = pd.DataFrame(violations)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### Violations by Type")
            if "type" in df.columns:
                type_counts = df["type"].value_counts()
                st.bar_chart(type_counts)
        
        with col_chart2:
            st.markdown("### Violations by Severity")
            if "severity" in df.columns:
                sev_counts = df["severity"].value_counts()
                st.bar_chart(sev_counts)
        
        st.markdown("### 📋 Violation Log")
        st.dataframe(df, use_container_width=True)
        
        # Export
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Report (CSV)",
            csv,
            file_name=f"exam_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No violations recorded yet. Start a proctoring session to generate data.")
        
        # Demo data
        st.markdown("### 📈 Sample Report Preview")
        import pandas as pd
        demo_data = {
            "time": ["10:23:45", "10:24:12", "10:25:01", "10:27:33", "10:29:10"],
            "type": ["Looking Away", "Phone Detected", "Multiple Faces", "Looking Away", "Head Turn"],
            "severity": ["HIGH", "CRITICAL", "CRITICAL", "MEDIUM", "HIGH"],
            "student": ["Ahmed M.", "Sara K.", "Ahmed M.", "Omar H.", "Sara K."],
            "confidence": [0.92, 0.88, 0.95, 0.75, 0.83]
        }
        st.dataframe(pd.DataFrame(demo_data), use_container_width=True)


def render_settings():
    """Render settings page."""
    st.markdown("## ⚙️ System Settings")
    
    tab1, tab2, tab3 = st.tabs(["🔧 General", "🎯 Detection", "📧 Notifications"])
    
    with tab1:
        st.markdown("### General Configuration")
        st.text_input("Institution Name", value="Cairo University")
        st.text_input("Department", value="Computer Science")
        st.selectbox("Language", ["English", "Arabic", "French"])
        st.selectbox("Camera Resolution", ["640x480", "1280x720", "1920x1080"])
        st.number_input("Frame Rate (FPS)", min_value=10, max_value=60, value=30)
    
    with tab2:
        st.markdown("### Detection Thresholds")
        st.slider("Face Match Threshold", 0.1, 1.0, 0.6, help="Lower = more strict identity matching")
        st.slider("Gaze Deviation Threshold (degrees)", 5, 45, 20)
        st.slider("Phone Detection Confidence", 0.1, 1.0, 0.7)
        st.slider("Multiple Person Confidence", 0.1, 1.0, 0.8)
        
        st.markdown("### Timing")
        st.number_input("Alert Cooldown (seconds)", min_value=1, max_value=30, value=5)
        st.number_input("Violation Buffer Frames", min_value=1, max_value=30, value=10)
    
    with tab3:
        st.markdown("### Notification Settings")
        st.checkbox("Email alerts to supervisor", value=True)
        st.text_input("Supervisor Email", value="supervisor@university.edu")
        st.checkbox("Log all violations to file", value=True)
        st.checkbox("Auto-screenshot on violation", value=True)
        st.text_input("Screenshots directory", value="data/screenshots")
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Settings saved successfully!")


# ─── Main App ───────────────────────────────────────────────────────────────

def main():
    render_header()
    mode, settings = render_sidebar()
    
    st.markdown("---")
    
    if mode == "Live Proctoring":
        render_live_proctoring(settings)
    elif mode == "Student Registration":
        render_registration()
    elif mode == "Exam Report":
        render_report()
    elif mode == "Noise Test":
        render_noise_test()
    elif mode == "Settings":
        render_settings()


if __name__ == "__main__":
    main()
