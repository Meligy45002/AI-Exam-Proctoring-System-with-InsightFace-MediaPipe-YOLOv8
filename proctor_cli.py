"""
CLI Proctoring Script - Run proctoring from command line
Usage: python proctor_cli.py [--student NAME] [--source 0]
"""

import cv2
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from proctoring_engine import ProctoringEngine


def parse_args():
    parser = argparse.ArgumentParser(description="AI Exam Proctoring System - CLI")
    parser.add_argument("--student", type=str, default="Unknown", help="Student name")
    parser.add_argument("--source", type=int, default=0, help="Camera source index")
    parser.add_argument("--sensitivity", type=int, default=7, help="Alert sensitivity (1-10)")
    parser.add_argument("--no-phone", action="store_true", help="Disable phone detection")
    parser.add_argument("--no-gaze", action="store_true", help="Disable gaze tracking")
    parser.add_argument("--output", type=str, default=None, help="Save output video path")
    parser.add_argument("--show-fps", action="store_true", help="Show FPS counter")
    return parser.parse_args()


def main():
    args = parse_args()
    
    settings = {
        "sensitivity": args.sensitivity,
        "detect_gaze": not args.no_gaze,
        "detect_face": True,
        "detect_phone": not args.no_phone,
        "detect_multi": True,
        "exam_name": "CLI Session",
        "duration": 90
    }
    
    print(f"\n{'='*50}")
    print("  AI EXAM PROCTORING SYSTEM")
    print(f"  Student: {args.student}")
    print(f"  Camera: {args.source}")
    print(f"  Sensitivity: {args.sensitivity}/10")
    print(f"{'='*50}\n")
    
    # Initialize engine
    print("Initializing AI models...")
    engine = ProctoringEngine(settings)
    print("Ready! Press Q to quit, S to save screenshot.\n")
    
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.source}")
        return
    
    # Optional video writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        print(f"Recording to: {args.output}")
    
    violation_count = 0
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read error. Stopping.")
                break
            
            # Process frame
            annotated, violations = engine.process_frame(frame)
            
            if violations:
                violation_count += len(violations)
                for v in violations:
                    print(f"[{v['time']}] ⚠️  {v['type']} ({v['severity']}) - {v.get('confidence', 0):.0%}")
            
            # FPS overlay
            if args.show_fps:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, annotated.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            # Write to output
            if writer:
                writer.write(annotated)
            
            # Display
            cv2.imshow("AI Proctoring - Press Q to quit", annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                snap_path = f"data/screenshots/manual_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(snap_path, annotated)
                print(f"Screenshot saved: {snap_path}")
            
            frame_count += 1
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Session summary
        elapsed = time.time() - start_time
        summary = engine.get_session_summary()
        
        print(f"\n{'='*50}")
        print("  SESSION SUMMARY")
        print(f"{'='*50}")
        print(f"  Duration: {int(elapsed//60)}m {int(elapsed%60)}s")
        print(f"  Frames processed: {frame_count}")
        print(f"  Total violations: {violation_count}")
        print(f"\n  By type:")
        for t, count in summary.get("violations_by_type", {}).items():
            print(f"    - {t}: {count}")
        print(f"\n  By severity:")
        for s, count in summary.get("violations_by_severity", {}).items():
            print(f"    - {s}: {count}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
