"""
Student Registration Script
Registers student faces for identity verification.
Usage: python register_student.py --name "Ahmed Mohamed" --id "20210001"
"""

import cv2
import argparse
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Register a student for exam proctoring")
    parser.add_argument("--name", required=True, help="Student full name")
    parser.add_argument("--id", required=True, dest="student_id", help="Student ID")
    parser.add_argument("--photos", type=int, default=5, help="Number of photos to capture")
    parser.add_argument("--source", type=int, default=0, help="Camera source")
    parser.add_argument("--upload", type=str, default=None, help="Upload existing photo instead of webcam")
    return parser.parse_args()


def register_from_webcam(name, student_id, num_photos, source):
    """Register student by capturing webcam photos."""
    face_dir = Path(f"data/registered_faces/{name}_{student_id}")
    face_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {source}")
        return False
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    print(f"\nRegistering: {name} (ID: {student_id})")
    print(f"Capturing {num_photos} photos...")
    print("Press SPACE to capture, Q to quit\n")
    
    captured = 0
    
    while captured < num_photos:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        display = frame.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 100), 2)
        
        # UI overlay
        cv2.putText(display, f"Student: {name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Photos: {captured}/{num_photos}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        cv2.putText(display, "SPACE = Capture | Q = Quit", (10, display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Student Registration", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if len(faces) == 0:
                print("  No face detected - please face the camera")
                continue
            
            save_path = face_dir / f"photo_{captured}.jpg"
            cv2.imwrite(str(save_path), frame)
            captured += 1
            print(f"  ✓ Photo {captured} captured")
            time.sleep(0.3)
        
        elif key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if captured > 0:
        print(f"\n✅ Successfully registered {name} with {captured} photos")
        print(f"   Saved to: {face_dir}")
        return True
    else:
        print("\n❌ Registration failed - no photos captured")
        return False


def register_from_photo(name, student_id, photo_path):
    """Register student from existing photo file."""
    img = cv2.imread(photo_path)
    if img is None:
        print(f"ERROR: Cannot read image: {photo_path}")
        return False
    
    face_dir = Path(f"data/registered_faces/{name}_{student_id}")
    face_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = face_dir / "photo_0.jpg"
    cv2.imwrite(str(save_path), img)
    
    print(f"✅ Registered {name} from photo: {photo_path}")
    return True


def list_registered():
    """List all registered students."""
    faces_dir = Path("data/registered_faces")
    if not faces_dir.exists():
        print("No students registered.")
        return
    
    students = [d for d in faces_dir.iterdir() if d.is_dir()]
    if not students:
        print("No students registered.")
        return
    
    print(f"\n{'='*40}")
    print(f"  Registered Students ({len(students)})")
    print(f"{'='*40}")
    for s in students:
        photos = list(s.glob("*.jpg")) + list(s.glob("*.png"))
        print(f"  👤 {s.name} ({len(photos)} photos)")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    args = parse_args()
    
    if args.upload:
        register_from_photo(args.name, args.student_id, args.upload)
    else:
        register_from_webcam(args.name, args.student_id, args.photos, args.source)
    
    list_registered()
