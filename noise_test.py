"""
Noise Robustness Testing Module
Task: Add noisy images → does the model get confused like humans?

Tests how InsightFace, MediaPipe, and YOLOv8 perform under various noise conditions.
Run: python noise_test.py --student "Ahmed Mohamed_20210001" --source 0
"""

import cv2
import numpy as np
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import our proctoring engine modules
from proctoring_engine import FaceDetector, GazeTracker, ObjectDetector


# ─── Noise Functions ──────────────────────────────────────────────────────────

class NoiseGenerator:
    """Generates different types of image noise/corruptions."""

    @staticmethod
    def gaussian_noise(image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """Add Gaussian (random) noise — like a bad camera sensor."""
        std = int(intensity * 255)
        noise = np.random.normal(0, std, image.shape).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy

    @staticmethod
    def salt_pepper_noise(image: np.ndarray, intensity: float = 0.05) -> np.ndarray:
        """Add salt & pepper noise — random black/white pixels."""
        noisy = image.copy()
        total_pixels = image.size // image.shape[2]
        num_salt = int(total_pixels * intensity / 2)
        num_pepper = int(total_pixels * intensity / 2)

        # Salt (white pixels)
        coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 255

        # Pepper (black pixels)
        coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 0

        return noisy

    @staticmethod
    def blur_noise(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Apply Gaussian blur — like an out-of-focus camera."""
        ksize = max(3, int(intensity * 30))
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    @staticmethod
    def motion_blur(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Simulate motion blur — like fast head movement."""
        size = max(3, int(intensity * 40))
        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size)
        kernel /= size
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def brightness_change(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Adjust brightness — dark room or overexposed light."""
        # intensity < 0.5 = darker, > 0.5 = brighter
        factor = intensity * 3.0 - 0.5  # range: -0.5 to 2.5
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + factor), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def occlusion(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """Block part of the face — like hand covering mouth."""
        noisy = image.copy()
        h, w = image.shape[:2]
        block_h = int(h * intensity)
        block_w = int(w * intensity * 1.5)
        y1 = int(h * 0.4)
        x1 = int(w * 0.25)
        noisy[y1:y1 + block_h, x1:x1 + block_w] = 0
        return noisy

    @staticmethod
    def compression_artifact(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """JPEG compression artifacts — like low-bandwidth video."""
        quality = max(1, int((1 - intensity) * 95) + 1)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    @staticmethod
    def pixelation(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Pixelate image — like extreme video compression."""
        h, w = image.shape[:2]
        scale = max(2, int(intensity * 20))
        small = cv2.resize(image, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def color_jitter(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Random color channel shifts — like bad white balance."""
        noisy = image.copy().astype(np.int16)
        for c in range(3):
            shift = int(intensity * 80 * (np.random.rand() - 0.5) * 2)
            noisy[:, :, c] = np.clip(noisy[:, :, c] + shift, 0, 255)
        return noisy.astype(np.uint8)

    @staticmethod
    def rotation(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """Rotate image — like tilted camera."""
        h, w = image.shape[:2]
        angle = intensity * 60 - 30  # -30 to +30 degrees
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    @classmethod
    def get_all(cls):
        """Return dict of all noise functions."""
        return {
            "Gaussian Noise":       cls.gaussian_noise,
            "Salt & Pepper":        cls.salt_pepper_noise,
            "Blur":                 cls.blur_noise,
            "Motion Blur":          cls.motion_blur,
            "Low Brightness":       lambda img, i: cls.brightness_change(img, i * 0.4),
            "High Brightness":      lambda img, i: cls.brightness_change(img, 0.6 + i * 0.4),
            "Face Occlusion":       cls.occlusion,
            "JPEG Compression":     cls.compression_artifact,
            "Pixelation":           cls.pixelation,
            "Color Jitter":         cls.color_jitter,
            "Rotation":             cls.rotation,
        }


# ─── Result Structures ────────────────────────────────────────────────────────

@dataclass
class SingleTestResult:
    noise_type: str
    intensity: float
    intensity_label: str            # "Low" / "Medium" / "High"

    # Face detection
    faces_detected: int = 0
    face_confidence: float = 0.0

    # Identity
    identity_matched: bool = False
    identity_confidence: float = 0.0
    identity_name: Optional[str] = None

    # Gaze
    gaze_direction: str = "UNKNOWN"
    gaze_correct: bool = False      # True if gaze detected as FORWARD on clean img

    # Object detection
    objects_detected: List[str] = field(default_factory=list)

    # Human-like confusion score (0–1, higher = more confused)
    confusion_score: float = 0.0

    # Processing time
    processing_ms: float = 0.0


@dataclass
class NoiseTestReport:
    student_name: str
    timestamp: str
    results: List[SingleTestResult] = field(default_factory=list)
    baseline: Optional[SingleTestResult] = None   # Clean image result

    def summary(self) -> Dict:
        if not self.results:
            return {}

        by_type = {}
        for r in self.results:
            if r.noise_type not in by_type:
                by_type[r.noise_type] = []
            by_type[r.noise_type].append(r.confusion_score)

        return {
            t: {
                "avg_confusion": float(np.mean(scores)),
                "max_confusion": float(np.max(scores)),
                "confused_at_high": scores[-1] > 0.5 if scores else False
            }
            for t, scores in by_type.items()
        }


# ─── Test Engine ─────────────────────────────────────────────────────────────

class NoiseTester:
    """
    Runs noise robustness tests on the proctoring models.
    Compares model performance on clean vs noisy images.
    """

    INTENSITY_LEVELS = [
        (0.1, "Low"),
        (0.4, "Medium"),
        (0.8, "High"),
    ]

    def __init__(self):
        print("  Loading FaceDetector...", end=" ")
        self.face_detector = FaceDetector()
        print("✓")

        print("  Loading GazeTracker...", end=" ")
        self.gaze_tracker = GazeTracker()
        print("✓")

        print("  Loading ObjectDetector...", end=" ")
        self.object_detector = ObjectDetector()
        print("✓")

        self.noise_gen = NoiseGenerator()

    def _run_single(self, image: np.ndarray, noise_fn, intensity: float,
                    noise_name: str, intensity_label: str,
                    baseline_faces: int, baseline_gaze: str) -> SingleTestResult:
        """Run all models on one noisy image and compute confusion score."""

        # Apply noise
        noisy_img = noise_fn(image, intensity)

        t0 = time.time()

        # 1. Face detection
        faces, identity_name, identity_conf = self.face_detector.detect_and_verify(noisy_img)

        # 2. Gaze
        gaze_result = self.gaze_tracker.analyze(noisy_img)
        gaze_dir    = gaze_result.get("gaze_direction", "UNKNOWN")
        is_cheating = gaze_result.get("is_cheating", False)
        iris_h      = gaze_result.get("iris_ratio_h", 0.5)

        # 3. Objects
        objects = self.object_detector.detect(noisy_img)
        object_names = [o["class_name"] for o in objects]

        proc_ms = (time.time() - t0) * 1000

        # ── Compute confusion score ──────────────────────────────────────────
        score = 0.0
        weight_total = 0.0

        # Face count changed?
        face_weight = 0.35
        if baseline_faces > 0 and len(faces) == 0:
            score += face_weight
        elif baseline_faces > 0 and len(faces) != baseline_faces:
            score += face_weight * 0.5
        weight_total += face_weight

        # Identity lost?
        id_weight = 0.25
        if identity_conf > 0:
            id_drop = max(0.0, 1.0 - identity_conf)
            score += id_weight * id_drop
        elif baseline_faces > 0:
            score += id_weight
        weight_total += id_weight

        # Gaze changed incorrectly?
        gaze_weight = 0.25
        if baseline_gaze == "FORWARD" and gaze_dir != "FORWARD":
            score += gaze_weight
        elif baseline_gaze != "FORWARD" and gaze_dir == "FORWARD":
            score += gaze_weight * 0.5
        weight_total += gaze_weight

        # Iris ratio drifted far from centre? (new — catches subtle confusion)
        iris_weight = 0.15
        iris_drift = abs(iris_h - 0.5) * 2   # 0 = centred, 1 = full edge
        score += iris_weight * iris_drift
        weight_total += iris_weight

        # Bonus: if model now thinks it's cheating on a clean baseline
        if is_cheating and baseline_gaze == "FORWARD":
            score = min(1.0, score + 0.15)

        confusion = score / weight_total if weight_total > 0 else 0.0

        return SingleTestResult(
            noise_type=noise_name,
            intensity=intensity,
            intensity_label=intensity_label,
            faces_detected=len(faces),
            face_confidence=float(faces[0]["det_score"]) if faces else 0.0,
            identity_matched=identity_conf >= self.face_detector.recognition_threshold,
            identity_confidence=identity_conf,
            identity_name=identity_name,
            gaze_direction=gaze_dir,
            gaze_correct=(gaze_dir == baseline_gaze),
            objects_detected=object_names,
            confusion_score=round(confusion, 4),
            processing_ms=round(proc_ms, 2),
        )

    def run_on_image(self, image: np.ndarray, student_name: str = "Unknown") -> NoiseTestReport:
        """Run full noise test battery on a single image."""

        report = NoiseTestReport(
            student_name=student_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # ── Baseline (clean image) ───────────────────────────────────────────
        print("\n  📸 Running baseline (clean image)...")
        faces_clean, _, _ = self.face_detector.detect_and_verify(image)
        gaze_clean = self.gaze_tracker.analyze(image)
        baseline_faces = len(faces_clean)
        baseline_gaze = gaze_clean.get("gaze_direction", "FORWARD")

        report.baseline = SingleTestResult(
            noise_type="Clean",
            intensity=0.0,
            intensity_label="None",
            faces_detected=baseline_faces,
            gaze_direction=baseline_gaze,
            confusion_score=0.0,
        )

        print(f"     Faces: {baseline_faces} | Gaze: {baseline_gaze}")

        # ── Noise Tests ──────────────────────────────────────────────────────
        noise_functions = NoiseGenerator.get_all()
        total = len(noise_functions) * len(self.INTENSITY_LEVELS)
        done = 0

        for noise_name, noise_fn in noise_functions.items():
            for intensity, label in self.INTENSITY_LEVELS:
                done += 1
                print(f"  [{done:2d}/{total}] {noise_name:<22} @ {label:<7}", end=" → ")

                result = self._run_single(
                    image, noise_fn, intensity,
                    noise_name, label,
                    baseline_faces, baseline_gaze
                )
                report.results.append(result)

                confusion_bar = "█" * int(result.confusion_score * 10)
                print(f"Confusion: {result.confusion_score:.0%}  {confusion_bar}")

        return report

    def run_from_webcam(self, student_name: str = "Unknown") -> NoiseTestReport:
        """Capture a frame from webcam then run tests."""
        print("\n  Opening webcam — press SPACE to capture, Q to quit...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam.")

        captured_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            cv2.putText(display, "Press SPACE to capture for noise test",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
            cv2.imshow("Noise Test — Capture Frame", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                captured_frame = frame.copy()
                print("  ✓ Frame captured")
                break
            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured_frame is None:
            raise RuntimeError("No frame captured.")

        return self.run_on_image(captured_frame, student_name)

    def run_from_registered(self, student_folder: str) -> NoiseTestReport:
        """Load a registered student photo and run tests on it."""
        face_dir = Path(f"data/registered_faces/{student_folder}")
        photos = list(face_dir.glob("*.jpg")) + list(face_dir.glob("*.png"))

        if not photos:
            raise FileNotFoundError(f"No photos in {face_dir}")

        image = cv2.imread(str(photos[0]))
        if image is None:
            raise ValueError(f"Cannot read {photos[0]}")

        print(f"  Using registered photo: {photos[0].name}")
        return self.run_on_image(image, student_folder)


# ─── Report & Visualization ───────────────────────────────────────────────────

class NoiseReportGenerator:
    """Generates visual reports from noise test results."""

    @staticmethod
    def save_json(report: NoiseTestReport, path: str = None):
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"data/noise_report_{ts}.json"

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "student": report.student_name,
            "timestamp": report.timestamp,
            "baseline": {
                "faces": report.baseline.faces_detected,
                "gaze": report.baseline.gaze_direction,
            } if report.baseline else None,
            "summary": report.summary(),
            "results": [
                {
                    "noise_type": r.noise_type,
                    "intensity": r.intensity_label,
                    "faces_detected": r.faces_detected,
                    "face_confidence": r.face_confidence,
                    "identity_matched": r.identity_matched,
                    "identity_confidence": r.identity_confidence,
                    "gaze_direction": r.gaze_direction,
                    "gaze_correct": r.gaze_correct,
                    "confusion_score": r.confusion_score,
                    "processing_ms": r.processing_ms,
                }
                for r in report.results
            ]
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n  ✓ JSON report saved: {path}")
        return path

    @staticmethod
    def save_chart(report: NoiseTestReport, path: str = None):
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"data/noise_chart_{ts}.png"

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        summary = report.summary()
        noise_types = list(summary.keys())
        avg_scores = [summary[t]["avg_confusion"] for t in noise_types]
        max_scores = [summary[t]["max_confusion"] for t in noise_types]

        # ── Figure layout ────────────────────────────────────────────────────
        fig = plt.figure(figsize=(18, 12), facecolor="#0f1117")
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        bar_color = "#5c6bc0"
        max_color = "#ef5350"
        text_color = "#e0e0e0"
        grid_color = "#2d3250"

        def style_ax(ax):
            ax.set_facecolor("#1e2130")
            ax.tick_params(colors=text_color, labelsize=9)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_edgecolor(grid_color)
            ax.grid(axis="y", color=grid_color, alpha=0.5)

        # ── Chart 1: Average confusion by noise type ─────────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(len(noise_types))
        width = 0.35

        bars1 = ax1.bar(x - width/2, avg_scores, width, label="Avg Confusion",
                        color=bar_color, alpha=0.9)
        bars2 = ax1.bar(x + width/2, max_scores, width, label="Max Confusion (High intensity)",
                        color=max_color, alpha=0.9)

        ax1.set_xticks(x)
        ax1.set_xticklabels(noise_types, rotation=30, ha="right", fontsize=9)
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel("Confusion Score (0–1)")
        ax1.set_title(f"Model Confusion Under Noise — Student: {report.student_name}",
                     fontsize=13, fontweight="bold", pad=12)
        ax1.legend(facecolor="#1e2130", edgecolor=grid_color, labelcolor=text_color)
        ax1.axhline(0.5, color="#ff9800", linestyle="--", alpha=0.6, label="50% confused")

        # Value labels on bars
        for bar in bars1:
            h = bar.get_height()
            if h > 0.05:
                ax1.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                        f"{h:.0%}", ha="center", va="bottom", color=text_color, fontsize=8)

        style_ax(ax1)

        # ── Chart 2: Confusion heatmap per type × intensity ──────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        intensities = ["Low", "Medium", "High"]
        matrix = np.zeros((len(noise_types), 3))

        for r in report.results:
            if r.noise_type in noise_types:
                row = noise_types.index(r.noise_type)
                col = intensities.index(r.intensity_label)
                matrix[row, col] = r.confusion_score

        im = ax2.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(intensities)
        ax2.set_yticks(range(len(noise_types)))
        ax2.set_yticklabels(noise_types, fontsize=8)
        ax2.set_title("Confusion Heatmap\n(Green=OK, Red=Confused)", fontsize=10)
        plt.colorbar(im, ax=ax2)

        for i in range(len(noise_types)):
            for j in range(3):
                ax2.text(j, i, f"{matrix[i,j]:.0%}",
                        ha="center", va="center", fontsize=8,
                        color="white" if matrix[i,j] > 0.5 else "black")

        style_ax(ax2)

        # ── Chart 3: Face detection rate per noise type ───────────────────────
        ax3 = fig.add_subplot(gs[1, 1])
        face_rates = {}
        baseline_faces = report.baseline.faces_detected if report.baseline else 1

        for r in report.results:
            if r.noise_type not in face_rates:
                face_rates[r.noise_type] = []
            detected = min(r.faces_detected, baseline_faces)
            rate = detected / baseline_faces if baseline_faces > 0 else 0
            face_rates[r.noise_type].append(rate)

        avg_face_rates = [np.mean(v) for v in face_rates.values()]
        colors = ["#4caf50" if r >= 0.8 else "#ff9800" if r >= 0.5 else "#f44336"
                 for r in avg_face_rates]

        ax3.barh(list(face_rates.keys()), avg_face_rates, color=colors, alpha=0.9)
        ax3.set_xlim(0, 1.1)
        ax3.axvline(0.8, color="#ff9800", linestyle="--", alpha=0.6)
        ax3.set_xlabel("Face Detection Rate")
        ax3.set_title("Face Detection Rate\nby Noise Type", fontsize=10)
        ax3.set_yticklabels(list(face_rates.keys()), fontsize=8)

        for i, v in enumerate(avg_face_rates):
            ax3.text(v + 0.02, i, f"{v:.0%}", va="center", fontsize=8, color=text_color)

        style_ax(ax3)

        # ── Main title ────────────────────────────────────────────────────────
        fig.suptitle("AI Proctoring System — Noise Robustness Report",
                    color=text_color, fontsize=15, fontweight="bold", y=0.98)

        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
        plt.close()

        print(f"  ✓ Chart saved: {path}")
        return path

    @staticmethod
    def print_summary(report: NoiseTestReport):
        """Print colored terminal summary."""
        summary = report.summary()

        print(f"\n{'='*60}")
        print(f"  NOISE ROBUSTNESS REPORT")
        print(f"  Student : {report.student_name}")
        print(f"  Date    : {report.timestamp}")
        if report.baseline:
            print(f"  Baseline: {report.baseline.faces_detected} face(s), gaze={report.baseline.gaze_direction}")
        print(f"{'='*60}")
        print(f"  {'Noise Type':<24} {'Avg':>6} {'Max':>6}  {'High-Confused?':>14}")
        print(f"  {'-'*54}")

        for noise_type, stats in sorted(summary.items(), key=lambda x: -x[1]["avg_confusion"]):
            avg = stats["avg_confusion"]
            mx = stats["max_confusion"]
            confused = "⚠️  YES" if stats["confused_at_high"] else "✅  no"

            bar = "█" * int(avg * 15) + "░" * (15 - int(avg * 15))
            print(f"  {noise_type:<24} {avg:>5.0%} {mx:>5.0%}  {confused:>14}  {bar}")

        print(f"{'='*60}")

        # Most and least robust
        sorted_by_avg = sorted(summary.items(), key=lambda x: x[1]["avg_confusion"])
        if sorted_by_avg:
            best = sorted_by_avg[0]
            worst = sorted_by_avg[-1]
            print(f"\n  ✅ Most robust against : {best[0]} ({best[1]['avg_confusion']:.0%} avg confusion)")
            print(f"  ⚠️  Most confused by    : {worst[0]} ({worst[1]['avg_confusion']:.0%} avg confusion)")
        print()

    @staticmethod
    def save_noisy_samples(image: np.ndarray, output_dir: str = "data/noise_samples"):
        """Save visual samples of each noise type for the report."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        noise_fns = NoiseGenerator.get_all()

        for name, fn in noise_fns.items():
            for intensity, label in [(0.1, "low"), (0.4, "med"), (0.8, "high")]:
                noisy = fn(image, intensity)
                safe_name = name.replace(" ", "_").replace("&", "and")
                path = f"{output_dir}/{safe_name}_{label}.jpg"
                cv2.imwrite(path, noisy)

        print(f"  ✓ Noise samples saved to: {output_dir}/")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Noise Robustness Test for AI Proctoring")
    parser.add_argument("--source", choices=["webcam", "registered", "image"],
                       default="webcam", help="Image source")
    parser.add_argument("--student", type=str, default="Unknown",
                       help="Student name or registered folder name")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to image file (if source=image)")
    parser.add_argument("--save-samples", action="store_true",
                       help="Save noisy image samples")
    parser.add_argument("--no-chart", action="store_true",
                       help="Skip chart generation")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("  AI PROCTORING — NOISE ROBUSTNESS TEST")
    print(f"{'='*60}")
    print("\n  Initializing models...")

    tester = NoiseTester()

    # ── Get image ─────────────────────────────────────────────────────────────
    if args.source == "webcam":
        report = tester.run_from_webcam(args.student)
        image_for_samples = None

    elif args.source == "registered":
        report = tester.run_from_registered(args.student)
        face_dir = Path(f"data/registered_faces/{args.student}")
        photos = list(face_dir.glob("*.jpg"))
        image_for_samples = cv2.imread(str(photos[0])) if photos else None

    elif args.source == "image":
        if not args.image:
            print("ERROR: --image path required when source=image")
            return
        image = cv2.imread(args.image)
        if image is None:
            print(f"ERROR: Cannot read {args.image}")
            return
        report = tester.run_on_image(image, args.student)
        image_for_samples = image

    # ── Output ────────────────────────────────────────────────────────────────
    NoiseReportGenerator.print_summary(report)

    json_path = NoiseReportGenerator.save_json(report)

    if not args.no_chart:
        chart_path = NoiseReportGenerator.save_chart(report)
        print(f"\n  📊 Open chart: {chart_path}")

    if args.save_samples and image_for_samples is not None:
        NoiseReportGenerator.save_noisy_samples(image_for_samples)

    print(f"  📄 Open report: {json_path}\n")


if __name__ == "__main__":
    main()
