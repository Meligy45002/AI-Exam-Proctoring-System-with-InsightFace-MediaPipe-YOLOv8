"""
Proctoring Engine - Core AI Processing Module
Integrates: InsightFace, MediaPipe, COCO Object Detection
"""

import cv2
import numpy as np
import time
import json
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class Violation:
    type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float
    timestamp: float = field(default_factory=time.time)
    description: str = ""
    
    def to_dict(self):
        return {
            "type": self.type,
            "severity": self.severity,
            "confidence": self.confidence,
            "time": datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S"),
            "description": self.description
        }


@dataclass
class DetectionResult:
    faces: List[Dict] = field(default_factory=list)
    gaze_direction: Optional[str] = None
    gaze_angle: Optional[Tuple[float, float]] = None
    objects: List[Dict] = field(default_factory=list)
    identity_match: Optional[bool] = None
    identity_name: Optional[str] = None
    identity_confidence: float = 0.0
    violations: List[Violation] = field(default_factory=list)


# ─── Face Detection Module ────────────────────────────────────────────────────

class FaceDetector:
    """Handles face detection and identity verification using InsightFace."""
    
    def __init__(self, detection_threshold=0.5, recognition_threshold=0.6):
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold
        self.app = None
        self.registered_faces = {}
        self._init_model()
        self._load_registered_faces()
    
    def _init_model(self):
        """Initialize InsightFace model."""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(
                name='buffalo_s',
                allowed_modules=['detection', 'recognition']
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace initialized successfully")
        except ImportError:
            logger.warning("InsightFace not available, using OpenCV fallback")
            self._init_fallback()
        except Exception as e:
            logger.warning(f"InsightFace init failed: {e}, using fallback")
            self._init_fallback()
    
    def _init_fallback(self):
        """Fallback to OpenCV Haar Cascade."""
        self.app = None
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        logger.info("Using OpenCV Haar Cascade fallback")
    
    def _load_registered_faces(self):
        """Load registered face embeddings."""
        faces_dir = Path("data/registered_faces")
        if not faces_dir.exists():
            return
        
        for student_dir in faces_dir.iterdir():
            if not student_dir.is_dir():
                continue
            
            embeddings = []
            for img_path in student_dir.glob("*.jpg"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                if self.app:
                    try:
                        faces = self.app.get(img)
                        if faces:
                            embeddings.append(faces[0].embedding)
                    except Exception:
                        pass
            
            if embeddings:
                self.registered_faces[student_dir.name] = np.mean(embeddings, axis=0)
        
        logger.info(f"Loaded {len(self.registered_faces)} registered students")
    
    def detect_and_verify(self, frame) -> Tuple[List[Dict], Optional[str], float]:
        """
        Detect faces and verify identity.
        Returns: (faces_list, matched_name, confidence)
        """
        if self.app:
            return self._detect_insightface(frame)
        else:
            return self._detect_opencv(frame)
    
    def _detect_insightface(self, frame):
        faces_info = []
        matched_name = None
        best_confidence = 0.0
        
        try:
            faces = self.app.get(frame)
            
            for face in faces:
                bbox = face.bbox.astype(int)
                faces_info.append({
                    "bbox": bbox,
                    "embedding": face.embedding,
                    "det_score": float(face.det_score)
                })
                
                # Identity matching
                if self.registered_faces:
                    name, conf = self._match_face(face.embedding)
                    if conf > best_confidence:
                        best_confidence = conf
                        matched_name = name if conf >= self.recognition_threshold else None
        
        except Exception as e:
            logger.error(f"InsightFace detection error: {e}")
        
        return faces_info, matched_name, best_confidence
    
    def _detect_opencv(self, frame):
        """Fallback OpenCV detection."""
        faces_info = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            faces_info.append({
                "bbox": np.array([x, y, x+w, y+h]),
                "embedding": None,
                "det_score": 0.9
            })
        
        return faces_info, None, 0.0
    
    def _match_face(self, embedding) -> Tuple[Optional[str], float]:
        """Match face embedding against registered faces."""
        if embedding is None or not self.registered_faces:
            return None, 0.0
        
        best_name = None
        best_sim = -1.0
        
        for name, reg_embedding in self.registered_faces.items():
            sim = self._cosine_similarity(embedding, reg_embedding)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        
        return best_name, float(best_sim)
    
    @staticmethod
    def _cosine_similarity(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def reload_faces(self):
        """Reload registered faces (call after registration)."""
        self.registered_faces = {}
        self._load_registered_faces()


# ─── Gaze Tracking Module ─────────────────────────────────────────────────────

class GazeTracker:
    """
    Improved dual-method gaze tracker.

    METHOD 1 — Iris ratio (actual eye gaze):
        Uses MediaPipe iris landmarks (474-477, 469-472).
        Computes where the iris sits within the eye corner bounds → ratio 0-1.
        ratio < 0.35  → looking LEFT
        ratio > 0.65  → looking RIGHT
        vertical ratio < 0.35 → looking UP
        vertical ratio > 0.65 → looking DOWN

    METHOD 2 — Head pose (solvePnP):
        Uses 6 stable 3D face landmarks.
        Extracts yaw/pitch Euler angles.
        Catches big head turns the iris method misses.

    COMBINED: either method firing → alert.
    Instant cheating detection with per-type frame counters.
    Provides draw_overlay() for annotated frame rendering.
    """

    # ── Thresholds ──────────────────────────────────────────────────────────
    # Iris ratio thresholds (0.0 = full left, 1.0 = full right)
    IRIS_LEFT_THRESH   = 0.38   # iris ratio below this → looking left
    IRIS_RIGHT_THRESH  = 0.62   # iris ratio above this → looking right
    IRIS_UP_THRESH     = 0.35   # vertical iris ratio below → looking up
    IRIS_DOWN_THRESH   = 0.68   # vertical iris ratio above → looking down

    # Head pose thresholds (degrees)
    HEAD_YAW_THRESH    = 18     # horizontal head turn
    HEAD_PITCH_THRESH  = 18     # vertical head tilt

    # How many consecutive frames before cheating fires
    CHEAT_FRAMES_NEEDED = 6     # ~0.2s at 30fps — very fast

    # MediaPipe landmark indices
    # Eye corners (for iris ratio calculation)
    L_EYE_LEFT  = 362   # left eye, left corner
    L_EYE_RIGHT = 263   # left eye, right corner
    L_EYE_TOP   = 386
    L_EYE_BOT   = 374
    R_EYE_LEFT  = 133   # right eye, left corner
    R_EYE_RIGHT = 33    # right eye, right corner
    R_EYE_TOP   = 159
    R_EYE_BOT   = 145

    # Iris centers (refine_landmarks=True required)
    L_IRIS_CENTER = 473
    R_IRIS_CENTER = 468

    # Head pose 6-point landmarks
    HEAD_PTS = [1, 33, 61, 199, 263, 291]

    # Corresponding 3-D canonical face model points (mm)
    HEAD_MODEL_3D = np.array([
        [  0.0,    0.0,    0.0  ],   # Nose tip      (1)
        [-83.0,   47.0, -51.0  ],   # Left eye R    (33)
        [-28.0,  -29.0, -57.0  ],   # L mouth       (61)
        [  0.0,  -63.0, -12.0  ],   # Chin          (199)
        [ 83.0,   47.0, -51.0  ],   # Right eye L   (263)
        [ 28.0,  -29.0, -57.0  ],   # R mouth       (291)
    ], dtype=np.float64)

    def __init__(self):
        self.face_mesh = None
        self._init_mediapipe()

        # Per-direction frame counters for fast cheat detection
        self._away_counters: Dict[str, int] = {
            "LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0, "AWAY": 0
        }
        self._forward_counter = 0   # consecutive forward frames (resets away counters)

        # Last raw analysis result (used by noise tester)
        self.last_result: Dict = {}

    # ── Init ────────────────────────────────────────────────────────────────

    def _init_mediapipe(self):
        """Initialize MediaPipe — supports 0.9.x, 0.10.x and handles numpy conflicts."""
        try:
            import mediapipe as mp
            # mp.solutions exists in mediapipe <= 0.10.14 with legacy support
            face_mesh_module = getattr(mp, 'solutions', None)
            if face_mesh_module is None or not hasattr(face_mesh_module, 'face_mesh'):
                raise AttributeError("mp.solutions.face_mesh not available")
            self.mp_face_mesh = face_mesh_module.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("MediaPipe FaceMesh initialized ✓")
        except Exception as e:
            logger.warning(f"MediaPipe not available ({e}) — using OpenCV fallback")
            self.face_mesh = None

    # ── Public API ───────────────────────────────────────────────────────────

    def analyze(self, frame: np.ndarray) -> Dict:
        """
        Full gaze analysis on one BGR frame.
        Returns rich dict with all signals + is_cheating flag.
        """
        result = {
            "gaze_direction":  "FORWARD",
            "is_looking_away": False,
            "is_cheating":     False,         # ← NEW: fires after CHEAT_FRAMES_NEEDED
            "cheat_reason":    "",
            "iris_ratio_h":    0.5,
            "iris_ratio_v":    0.5,
            "head_yaw":        0.0,
            "head_pitch":      0.0,
            "head_roll":       0.0,
            "method":          "none",
            "landmarks_px":    [],            # [(x,y), ...] for drawing
            "iris_pts":        [],            # [(x,y), ...] for drawing
            "eye_boxes":       [],            # [(x1,y1,x2,y2), ...] for drawing
        }

        if self.face_mesh is not None:
            result = self._analyze_mediapipe(frame, result)
        else:
            result = self._analyze_opencv_fallback(frame, result)

        # ── Frame-counter cheat logic ────────────────────────────────────
        direction = result["gaze_direction"]

        if direction == "FORWARD":
            self._forward_counter += 1
            if self._forward_counter >= 3:          # 3 clean frames reset everything
                for k in self._away_counters:
                    self._away_counters[k] = 0
        else:
            self._forward_counter = 0
            if direction in self._away_counters:
                self._away_counters[direction] += 1

            # Fire cheat if any direction exceeds threshold
            for dir_key, count in self._away_counters.items():
                if count >= self.CHEAT_FRAMES_NEEDED:
                    result["is_cheating"]  = True
                    result["is_looking_away"] = True
                    result["cheat_reason"] = (
                        f"Sustained gaze {dir_key} for {count} frames "
                        f"(head yaw={result['head_yaw']:.1f}° "
                        f"iris_h={result['iris_ratio_h']:.2f})"
                    )
                    break

        self.last_result = result
        return result

    def draw_overlay(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw gaze overlay on frame — call this in FrameRenderer.
        Draws: iris circles, eye boxes, gaze arrow, status text.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        direction   = result.get("gaze_direction", "FORWARD")
        is_cheat    = result.get("is_cheating", False)
        iris_h      = result.get("iris_ratio_h", 0.5)
        head_yaw    = result.get("head_yaw", 0.0)
        head_pitch  = result.get("head_pitch", 0.0)

        # ── Colour scheme ─────────────────────────────────────────────────
        if is_cheat:
            main_color = (0, 0, 255)        # red
            bg_color   = (0, 0, 180)
        elif direction != "FORWARD":
            main_color = (0, 165, 255)      # orange
            bg_color   = (0, 100, 180)
        else:
            main_color = (0, 220, 80)       # green
            bg_color   = (0, 130, 50)

        # ── Draw eye bounding boxes ───────────────────────────────────────
        for box in result.get("eye_boxes", []):
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), main_color, 1)

        # ── Draw iris circles ─────────────────────────────────────────────
        for pt in result.get("iris_pts", []):
            cv2.circle(out, pt, 4, (255, 200, 0), -1)   # gold iris dot
            cv2.circle(out, pt, 6, main_color, 1)        # coloured ring

        # ── Direction arrow ───────────────────────────────────────────────
        cx, cy = w // 2, h - 80
        arrow_len = 35
        arrow_map = {
            "LEFT":    (-arrow_len, 0),
            "RIGHT":   ( arrow_len, 0),
            "UP":      (0, -arrow_len),
            "DOWN":    (0,  arrow_len),
            "FORWARD": (0,  0),
            "AWAY":    (-arrow_len, -arrow_len),
        }
        dx, dy = arrow_map.get(direction, (0, 0))
        if dx != 0 or dy != 0:
            cv2.arrowedLine(out, (cx, cy), (cx + dx, cy + dy),
                           main_color, 3, tipLength=0.4)

        # ── Status bar (bottom-left) ──────────────────────────────────────
        lines = [
            f"GAZE: {direction}",
            f"Iris H:{iris_h:.2f}  Yaw:{head_yaw:+.1f}  Pitch:{head_pitch:+.1f}",
        ]
        if is_cheat:
            lines.append("!! CHEATING DETECTED !!")

        for i, line in enumerate(lines):
            y_pos = h - 55 + i * 18
            cv2.rectangle(out, (5, y_pos - 13), (5 + len(line) * 9, y_pos + 4),
                         (10, 10, 10), -1)
            color = (0, 0, 255) if (is_cheat and i == 2) else main_color
            cv2.putText(out, line, (8, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return out

    # ── MediaPipe analysis ───────────────────────────────────────────────────

    def _analyze_mediapipe(self, frame: np.ndarray, result: Dict) -> Dict:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            mp_result = self.face_mesh.process(rgb)
        except Exception as e:
            logger.error(f"MediaPipe process error: {e}")
            return result

        if not mp_result.multi_face_landmarks:
            result["gaze_direction"]  = "NO_FACE"
            result["is_looking_away"] = True
            return result

        lm = mp_result.multi_face_landmarks[0].landmark

        def px(idx):
            return int(lm[idx].x * w), int(lm[idx].y * h)

        def fpt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float64)

        # ── Iris ratio (METHOD 1) ─────────────────────────────────────────
        # Left eye horizontal ratio
        l_left  = fpt(self.L_EYE_LEFT)
        l_right = fpt(self.L_EYE_RIGHT)
        l_iris  = fpt(self.L_IRIS_CENTER)
        l_top   = fpt(self.L_EYE_TOP)
        l_bot   = fpt(self.L_EYE_BOT)

        # Right eye horizontal ratio
        r_left  = fpt(self.R_EYE_LEFT)
        r_right = fpt(self.R_EYE_RIGHT)
        r_iris  = fpt(self.R_IRIS_CENTER)
        r_top   = fpt(self.R_EYE_TOP)
        r_bot   = fpt(self.R_EYE_BOT)

        def safe_ratio(val, lo, hi):
            span = hi - lo
            return float(np.clip((val - lo) / span, 0, 1)) if abs(span) > 1e-6 else 0.5

        l_ratio_h = safe_ratio(l_iris[0], l_right[0], l_left[0])
        r_ratio_h = safe_ratio(r_iris[0], r_right[0], r_left[0])
        iris_h    = (l_ratio_h + r_ratio_h) / 2.0

        l_ratio_v = safe_ratio(l_iris[1], l_top[1], l_bot[1])
        r_ratio_v = safe_ratio(r_iris[1], r_top[1], r_bot[1])
        iris_v    = (l_ratio_v + r_ratio_v) / 2.0

        result["iris_ratio_h"] = round(iris_h, 3)
        result["iris_ratio_v"] = round(iris_v, 3)

        # ── Head pose (METHOD 2) ──────────────────────────────────────────
        img_pts = np.array([fpt(i) for i in self.HEAD_PTS], dtype=np.float64)
        focal   = w
        cam_mat = np.array([[focal, 0, w/2],[0, focal, h/2],[0, 0, 1]], dtype=np.float64)
        dist    = np.zeros((4, 1))

        yaw = pitch = roll = 0.0
        try:
            ok, rvec, tvec = cv2.solvePnP(
                self.HEAD_MODEL_3D, img_pts, cam_mat, dist,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok:
                rmat, _ = cv2.Rodrigues(rvec)
                proj    = cv2.hconcat([rmat, tvec])
                _, _, _, _, _, _, ea = cv2.decomposeProjectionMatrix(
                    cv2.hconcat([proj, np.zeros((3,1))])
                )
                pitch = float(ea[0]); yaw = float(ea[1]); roll = float(ea[2])
        except cv2.error:
            pass

        result["head_yaw"]   = round(yaw, 2)
        result["head_pitch"] = round(pitch, 2)
        result["head_roll"]  = round(roll, 2)

        # ── Drawing data ──────────────────────────────────────────────────
        iris_pts  = [px(self.L_IRIS_CENTER), px(self.R_IRIS_CENTER)]
        eye_boxes = []
        for corners in [
            (self.L_EYE_RIGHT, self.L_EYE_LEFT, self.L_EYE_TOP, self.L_EYE_BOT),
            (self.R_EYE_RIGHT, self.R_EYE_LEFT, self.R_EYE_TOP, self.R_EYE_BOT),
        ]:
            pts = [px(c) for c in corners]
            xs  = [p[0] for p in pts]; ys = [p[1] for p in pts]
            margin = 5
            eye_boxes.append((min(xs)-margin, min(ys)-margin,
                              max(xs)+margin, max(ys)+margin))

        result["iris_pts"]  = iris_pts
        result["eye_boxes"] = eye_boxes

        # ── Combine both methods → single direction ───────────────────────
        #  Priority: if head turns a lot → head pose wins (more reliable for big turns)
        #            if head is mostly forward → iris ratio decides (catches subtle gaze)

        iris_dir = self._iris_direction(iris_h, iris_v)
        head_dir = self._head_direction(yaw, pitch)

        if head_dir != "FORWARD":
            direction = head_dir
            result["method"] = "head_pose"
        elif iris_dir != "FORWARD":
            direction = iris_dir
            result["method"] = "iris"
        else:
            direction = "FORWARD"
            result["method"] = "both_ok"

        result["gaze_direction"]  = direction
        result["is_looking_away"] = (direction != "FORWARD")
        return result

    # ── Direction helpers ────────────────────────────────────────────────────

    def _iris_direction(self, h_ratio: float, v_ratio: float) -> str:
        if h_ratio < self.IRIS_LEFT_THRESH:
            return "LEFT"
        if h_ratio > self.IRIS_RIGHT_THRESH:
            return "RIGHT"
        if v_ratio < self.IRIS_UP_THRESH:
            return "UP"
        if v_ratio > self.IRIS_DOWN_THRESH:
            return "DOWN"
        return "FORWARD"

    def _head_direction(self, yaw: float, pitch: float) -> str:
        if abs(yaw) > self.HEAD_YAW_THRESH:
            return "LEFT" if yaw < 0 else "RIGHT"
        if pitch > self.HEAD_PITCH_THRESH:
            return "DOWN"
        if pitch < -self.HEAD_PITCH_THRESH:
            return "UP"
        return "FORWARD"

    # ── OpenCV fallback (no MediaPipe) ───────────────────────────────────────

    def _analyze_opencv_fallback(self, frame: np.ndarray, result: Dict) -> Dict:
        """
        Fallback when MediaPipe is unavailable.
        Uses pupil localisation inside eye ROI via thresholding.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        eye_cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces) == 0:
            result["gaze_direction"]  = "NO_FACE"
            result["is_looking_away"] = True
            return result

        fx, fy, fw, fh = max(faces, key=lambda r: r[2]*r[3])
        face_roi = gray[fy:fy+fh, fx:fx+fw]

        eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 8, minSize=(20, 20))

        h_ratios = []
        for (ex, ey, ew, eh) in eyes[:2]:
            eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
            eye_roi = cv2.equalizeHist(eye_roi)
            # Threshold → find darkest blob (pupil)
            _, thresh = cv2.threshold(eye_roi, 50, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    h_ratios.append(cx / ew)

        if h_ratios:
            iris_h = float(np.mean(h_ratios))
            result["iris_ratio_h"] = round(iris_h, 3)
            iris_dir = self._iris_direction(iris_h, 0.5)
            result["gaze_direction"]  = iris_dir
            result["is_looking_away"] = (iris_dir != "FORWARD")
            result["method"] = "opencv_pupil"
        else:
            # No eyes found — face visible but eyes closed/not detected
            result["gaze_direction"]  = "AWAY"
            result["is_looking_away"] = True

        return result


# ─── Object Detection Module ──────────────────────────────────────────────────

class ObjectDetector:
    """Detects objects like phones, laptops, and additional persons using COCO model."""
    
    COCO_CLASSES = {
        0: "person", 67: "cell phone", 63: "laptop",
        64: "mouse", 65: "remote", 66: "keyboard",
        24: "backpack", 26: "handbag"
    }
    
    SUSPICIOUS_CLASSES = {67: "Phone", 63: "Laptop", 64: "Mouse"}
    
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.net = None
        self.model_type = None
        self._init_model()
    
    def _init_model(self):
        """Initialize object detection model."""
        # Try YOLOv8 first (best)
        if self._init_yolo():
            return
        # Then try COCO DNN
        if self._init_dnn():
            return
        # Fallback
        logger.warning("No object detection model available, using simple color detection")
        self.model_type = "simple"
    
    def _init_yolo(self) -> bool:
        """Try to initialize YOLOv8."""
        try:
            from ultralytics import YOLO
            model_path = Path("models/yolov8n.pt")
            if not model_path.exists():
                model_path.parent.mkdir(exist_ok=True)
                self.net = YOLO("yolov8n.pt")  # Will download if not present
            else:
                self.net = YOLO(str(model_path))
            self.model_type = "yolo"
            logger.info("YOLOv8 initialized")
            return True
        except Exception as e:
            logger.warning(f"YOLOv8 not available: {e}")
            return False
    
    def _init_dnn(self) -> bool:
        """Try OpenCV DNN with MobileNet-SSD."""
        try:
            prototxt = Path("models/MobileNetSSD_deploy.prototxt")
            caffemodel = Path("models/MobileNetSSD_deploy.caffemodel")
            
            if prototxt.exists() and caffemodel.exists():
                self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
                self.model_type = "dnn"
                logger.info("MobileNet-SSD DNN initialized")
                return True
        except Exception as e:
            logger.warning(f"DNN init failed: {e}")
        return False
    
    def detect(self, frame) -> List[Dict]:
        """Detect objects in frame."""
        if self.model_type == "yolo":
            return self._detect_yolo(frame)
        elif self.model_type == "dnn":
            return self._detect_dnn(frame)
        else:
            return []  # No detection available
    
    def _detect_yolo(self, frame) -> List[Dict]:
        detections = []
        try:
            results = self.net(frame, verbose=False, conf=self.confidence_threshold)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    
                    class_name = r.names.get(cls_id, "unknown")
                    detections.append({
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox": xyxy
                    })
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
        return detections
    
    def _detect_dnn(self, frame) -> List[Dict]:
        detections = []
        h, w = frame.shape[:2]
        
        try:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5
            )
            self.net.setInput(blob)
            output = self.net.forward()
            
            for i in range(output.shape[2]):
                confidence = output[0, 0, i, 2]
                if confidence < self.confidence_threshold:
                    continue
                
                class_id = int(output[0, 0, i, 1])
                bbox = output[0, 0, i, 3:7] * np.array([w, h, w, h])
                
                detections.append({
                    "class_id": class_id,
                    "class_name": self.COCO_CLASSES.get(class_id, "unknown"),
                    "confidence": float(confidence),
                    "bbox": bbox.astype(int)
                })
        except Exception as e:
            logger.error(f"DNN detection error: {e}")
        
        return detections


# ─── Violation Analyzer ───────────────────────────────────────────────────────

class ViolationAnalyzer:
    """Analyzes frames and generates violation reports."""
    
    def __init__(self, settings: Dict):
        self.settings = settings
        self.violation_buffer = {}  # type -> consecutive_count
        self.last_violation_time = {}  # type -> timestamp
        self.cooldown = 5.0  # seconds between same violation
        self.buffer_threshold = settings.get("sensitivity", 7)
    
    def analyze(self, faces, gaze_result, objects, identity_match, identity_name) -> List[Violation]:
        violations = []
        now = time.time()
        
        # 1. Face count check
        if self.settings.get("detect_multi", True):
            n_persons = sum(1 for o in objects if o["class_name"] == "person")
            n_faces = len(faces)
            total_people = max(n_persons, n_faces)
            
            if total_people > 1:
                if self._should_trigger("multiple_persons", now):
                    violations.append(Violation(
                        type="Multiple Persons Detected",
                        severity="CRITICAL",
                        confidence=0.95,
                        description=f"Detected {total_people} people in frame"
                    ))
        
        # 2. No face detected
        if self.settings.get("detect_face", True):
            if len(faces) == 0:
                if self._should_trigger("no_face", now):
                    violations.append(Violation(
                        type="Student Not Visible",
                        severity="HIGH",
                        confidence=0.90,
                        description="No face detected in frame"
                    ))
            
            # Identity mismatch
            elif identity_match is False and identity_name:
                if self._should_trigger("identity_mismatch", now):
                    violations.append(Violation(
                        type="Identity Mismatch",
                        severity="CRITICAL",
                        confidence=0.88,
                        description=f"Unknown person in frame"
                    ))
        
        # 3. Gaze/attention check — uses new is_cheating flag for instant alerts
        if self.settings.get("detect_gaze", True) and gaze_result:
            direction   = gaze_result.get("gaze_direction", "FORWARD")
            is_cheating = gaze_result.get("is_cheating", False)
            is_away     = gaze_result.get("is_looking_away", False)
            iris_h      = gaze_result.get("iris_ratio_h", 0.5)
            yaw         = abs(gaze_result.get("head_yaw", 0.0))
            method      = gaze_result.get("method", "")

            if is_cheating:
                # Instant fire — no extra buffer needed, GazeTracker already counted frames
                if self._should_trigger("gaze_cheat", now):
                    severity = "CRITICAL" if yaw > 30 or iris_h < 0.25 or iris_h > 0.75 else "HIGH"
                    reason = gaze_result.get("cheat_reason", "")
                    violations.append(Violation(
                        type=f"Cheating — Eyes {direction}",
                        severity=severity,
                        confidence=min(0.97, 0.70 + yaw / 100 + abs(iris_h - 0.5)),
                        description=reason
                    ))

            elif is_away and direction not in ("FORWARD", "NO_FACE", ""):
                # Single-frame away — accumulate with sensitivity buffer
                key = f"gaze_{direction}"
                self.violation_buffer[key] = self.violation_buffer.get(key, 0) + 1

                if self.violation_buffer[key] >= self.buffer_threshold:
                    if self._should_trigger("gaze_away", now):
                        violations.append(Violation(
                            type=f"Looking Away ({direction})",
                            severity="MEDIUM",
                            confidence=min(0.90, 0.55 + yaw / 120),
                            description=f"method={method} yaw={gaze_result.get('head_yaw',0):.1f}° iris_h={iris_h:.2f}"
                        ))
                        self.violation_buffer[key] = 0
            else:
                # Reset gaze buffers when looking forward
                for key in list(self.violation_buffer.keys()):
                    if key.startswith("gaze_"):
                        self.violation_buffer[key] = 0
        
        # 4. Phone/laptop detection
        if self.settings.get("detect_phone", True):
            for obj in objects:
                cls = obj["class_name"].lower()
                conf = obj["confidence"]
                
                if "phone" in cls or "cell" in cls:
                    if self._should_trigger("phone", now):
                        violations.append(Violation(
                            type="Phone Detected",
                            severity="CRITICAL",
                            confidence=conf,
                            description="Mobile phone detected in exam area"
                        ))
                elif "laptop" in cls and len(faces) > 0:
                    if self._should_trigger("laptop", now):
                        violations.append(Violation(
                            type="Laptop Detected",
                            severity="HIGH",
                            confidence=conf,
                            description="Unauthorized laptop in exam area"
                        ))
        
        return violations
    
    def _should_trigger(self, violation_type: str, now: float) -> bool:
        """Check if violation should trigger (respects cooldown)."""
        last = self.last_violation_time.get(violation_type, 0)
        if now - last >= self.cooldown:
            self.last_violation_time[violation_type] = now
            return True
        return False


# ─── Frame Renderer ───────────────────────────────────────────────────────────

class FrameRenderer:
    """Renders detection results on frames."""
    
    COLORS = {
        "face_ok": (0, 255, 100),
        "face_unknown": (0, 100, 255),
        "face_mismatch": (0, 0, 255),
        "phone": (0, 0, 255),
        "laptop": (0, 165, 255),
        "person": (255, 100, 0),
        "gaze_forward": (0, 255, 100),
        "gaze_away": (0, 0, 255),
        "violation": (0, 0, 255),
        "info": (200, 200, 200),
        "overlay": (20, 25, 40)
    }
    
    def render(self, frame, detection_result: DetectionResult,
               violations: List[Violation],
               gaze_tracker=None) -> np.ndarray:
        """Render all detections onto frame."""
        output = frame.copy()
        h, w = output.shape[:2]

        # ── Header bar ────────────────────────────────────────────────────
        cv2.rectangle(output, (0, 0), (w, 50), (10, 15, 30), -1)
        cv2.putText(output, "AI PROCTORING SYSTEM", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 150, 255), 1)
        cv2.putText(output, datetime.now().strftime("%H:%M:%S"), (w - 90, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ── Gaze overlay (iris dots, eye boxes, arrow, status bar) ────────
        if gaze_tracker is not None and gaze_tracker.last_result:
            output = gaze_tracker.draw_overlay(output, gaze_tracker.last_result)

        # ── Face bounding boxes ───────────────────────────────────────────
        for face_data in detection_result.faces:
            bbox = face_data["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            if detection_result.identity_match is True:
                color = self.COLORS["face_ok"]
                label = f"OK {detection_result.identity_name or 'Verified'}"
            elif detection_result.identity_match is False:
                color = self.COLORS["face_mismatch"]
                label = "UNKNOWN PERSON"
            else:
                color = self.COLORS["face_unknown"]
                label = "Face"

            self._draw_bbox_corners(output, x1, y1, x2, y2, color)

            label_bg_y = max(0, y1 - 25)
            cv2.rectangle(output, (x1, label_bg_y),
                         (x1 + len(label) * 10 + 10, y1), color, -1)
            cv2.putText(output, label, (x1 + 5, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            if "det_score" in face_data:
                cv2.putText(output, f"{face_data['det_score']:.0%}",
                           (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # ── Detected objects (phone / laptop) ─────────────────────────────
        for obj in detection_result.objects:
            if obj["class_name"] == "person":
                continue
            bbox = obj["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cls   = obj["class_name"].lower()
            color = self.COLORS["phone"] if "phone" in cls else self.COLORS["laptop"]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"{obj['class_name']} {obj['confidence']:.0%}"
            cv2.putText(output, label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # ── Violations border + text ───────────────────────────────────────
        if violations:
            bt = 4
            cv2.rectangle(output, (bt, bt), (w - bt, h - bt), (0, 0, 200), bt)
            for i, v in enumerate(violations[:3]):
                y_pos = h - 40 + i * (-25)
                cv2.rectangle(output, (5, y_pos - 18), (300, y_pos + 5), (0, 0, 150), -1)
                cv2.putText(output, f"! {v.type}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Face count indicator
        face_count = len(detection_result.faces)
        count_color = (0, 255, 100) if face_count == 1 else (0, 0, 255)
        cv2.putText(output, f"Faces: {face_count}", (w - 110, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, count_color, 1)
        
        return output
    
    def _draw_bbox_corners(self, frame, x1, y1, x2, y2, color, thickness=2, corner_len=20):
        """Draw stylish corner-only bounding box."""
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)


# ─── Main Proctoring Engine ───────────────────────────────────────────────────

class ProctoringEngine:
    """
    Main AI Proctoring Engine.
    Orchestrates all detection modules and generates violation reports.
    """
    
    def __init__(self, settings: Dict):
        self.settings = settings
        logger.info("Initializing Proctoring Engine...")
        
        self.face_detector = FaceDetector()
        self.gaze_tracker = GazeTracker()
        self.object_detector = ObjectDetector()
        self.violation_analyzer = ViolationAnalyzer(settings)
        self.renderer = FrameRenderer()
        
        self.frame_count = 0
        self.all_violations = []
        
        # Screenshot settings
        Path("data/screenshots").mkdir(parents=True, exist_ok=True)
        
        logger.info("Proctoring Engine ready ✓")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single video frame.
        
        Args:
            frame: BGR frame from webcam
            
        Returns:
            (annotated_frame, violations_list)
        """
        self.frame_count += 1
        
        detection_result = DetectionResult()
        
        # 1. Face detection & identity verification
        faces, identity_name, identity_conf = self.face_detector.detect_and_verify(frame)
        detection_result.faces = faces
        detection_result.identity_name = identity_name
        detection_result.identity_confidence = identity_conf
        
        if faces and identity_conf > 0:
            detection_result.identity_match = identity_conf >= self.face_detector.recognition_threshold
        elif faces:
            detection_result.identity_match = None  # No registered faces to compare
        
        # 2. Gaze tracking (every frame)
        gaze_result = self.gaze_tracker.analyze(frame)
        detection_result.gaze_direction = gaze_result.get("gaze_direction", "UNKNOWN")
        
        # 3. Object detection (every 3rd frame for performance)
        if self.frame_count % 3 == 0:
            detected_objects = self.object_detector.detect(frame)
            detection_result.objects = detected_objects
        
        # 4. Violation analysis
        violations = self.violation_analyzer.analyze(
            faces=detection_result.faces,
            gaze_result=gaze_result,
            objects=detection_result.objects,
            identity_match=detection_result.identity_match,
            identity_name=detection_result.identity_name
        )
        
        detection_result.violations = violations
        
        # 5. Render annotated frame
        annotated_frame = self.renderer.render(
            frame, detection_result, violations,
            gaze_tracker=self.gaze_tracker
        )
        
        # 6. Save screenshot on critical violation
        if violations and any(v.severity == "CRITICAL" for v in violations):
            self._save_screenshot(annotated_frame, violations[0].type)
        
        # 7. Log violations
        violation_dicts = [v.to_dict() for v in violations]
        if violation_dicts:
            self.all_violations.extend(violation_dicts)
            self._append_to_log(violation_dicts)
        
        return annotated_frame, violation_dicts
    
    def _save_screenshot(self, frame: np.ndarray, violation_type: str):
        """Save screenshot when violation occurs."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_type = violation_type.replace(" ", "_").replace("/", "_")
        path = Path(f"data/screenshots/{safe_type}_{timestamp}.jpg")
        cv2.imwrite(str(path), frame)
    
    def _append_to_log(self, violations: List[Dict]):
        """Append violations to JSON log."""
        log_path = Path("data/violation_log.json")
        existing = []
        
        if log_path.exists():
            with open(log_path, "r") as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = []
        
        existing.extend(violations)
        
        with open(log_path, "w") as f:
            json.dump(existing, f, indent=2)
    
    def get_session_summary(self) -> Dict:
        """Get summary of current proctoring session."""
        return {
            "total_frames": self.frame_count,
            "total_violations": len(self.all_violations),
            "violations_by_type": self._count_by_key(self.all_violations, "type"),
            "violations_by_severity": self._count_by_key(self.all_violations, "severity"),
        }
    
    @staticmethod
    def _count_by_key(items: List[Dict], key: str) -> Dict:
        counts = {}
        for item in items:
            val = item.get(key, "Unknown")
            counts[val] = counts.get(val, 0) + 1
        return counts
