"""
===========================================================
 1_collect_data.py  –  Record Hand-Sign Landmark Data
===========================================================
HOW IT WORKS:
  1. Opens your laptop webcam.
  2. Detects your hand using MediaPipe Hands (21 3D landmarks).
  3. NORMALIZES every landmark by subtracting the wrist (Landmark 0)
     so the data is position-independent.
  4. When you press 's', the normalized landmarks + your chosen
     label are appended to  sign_data.csv .
  5. Press 'q' to quit.

USAGE:
  .venv/bin/python3 1_collect_data.py
  Then follow the on-screen prompts.
===========================================================
"""

import csv
import os
import cv2
import numpy as np

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ─── CONFIG ────────────────────────────────────────────────
CSV_FILE = "dataset/sign_data.csv"        # Where the data goes
MODEL_PATH = "hand_landmarker.task"       # MediaPipe model bundle
NUM_LANDMARKS = 21                # MediaPipe Hands gives 21 points
FEATURES_PER_LANDMARK = 3         # X, Y, Z for each point
TARGET_SAMPLES_PER_LABEL = 250    # Collect balanced data before retraining
MIN_HAND_SPAN_RATIO = 0.18        # Reject distant/low-detail hand detections
LAPTOP_CAPTURE_WIDTH = 1280
LAPTOP_CAPTURE_HEIGHT = 720
# ───────────────────────────────────────────────────────────

# Hand connection pairs for drawing the skeleton
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # Thumb
    (0,5),(5,6),(6,7),(7,8),       # Index
    (0,9),(9,10),(10,11),(11,12),   # Middle
    (0,13),(13,14),(14,15),(15,16), # Ring
    (0,17),(17,18),(18,19),(19,20), # Pinky
    (5,9),(9,13),(13,17),           # Palm connections
]


def main():
    # ─── Step 1: Ask the user which sign they are about to record ──
    label = input("🏷️  Enter the label for the sign you will record (e.g. Hello, Thanks, Yes): ").strip()
    if not label:
        print("❌  Label cannot be empty. Exiting.")
        return

    print(f"\n✅  Recording label: '{label}'")
    print("    Press 's' to SAVE a frame   |   Press 'q' to QUIT\n")

    # ─── Step 2: Build the CSV header (only once) ─────────────
    header = []
    for i in range(NUM_LANDMARKS):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")

    os.makedirs("dataset", exist_ok=True)
    file_exists = os.path.isfile(CSV_FILE)
    if not file_exists:
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"📄  Created new CSV file: {CSV_FILE}")
    else:
        print(f"📄  Appending to existing CSV: {CSV_FILE}")

    # Balanced classes matter much more than many nearly identical frames.
    existing_label_count = 0
    if file_exists:
        with open(CSV_FILE, newline="") as f:
            existing_label_count = sum(
                1 for row in csv.DictReader(f) if row["label"] == label
            )
    print(f"🎯  {label}: {existing_label_count}/{TARGET_SAMPLES_PER_LABEL} samples collected")

    # ─── Step 3: Set up MediaPipe HandLandmarker (Tasks API) ─
    if not os.path.exists(MODEL_PATH):
        print(f"❌  Model file not found: {MODEL_PATH}")
        print("    Make sure 'hand_landmarker.task' is in the same folder.")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=BaseOptions.Delegate.CPU
        ),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # ─── Step 4: Choose Camera Source ─────────────────────────
    print("📷  Choose Camera Source:")
    print("    [1] Laptop Webcam")
    print("    [2] Smart Glasses POV Stream (http://192.168.4.1:81/stream)")
    cam_choice = input("👉  Select (1 or 2, default 1): ").strip()

    is_pov = (cam_choice == "2")
    if is_pov:
        stream_url = "http://192.168.4.1:81/stream"
        print(f"\n📡  Connecting to Smart Glasses POV stream: {stream_url} ...")
        cap = cv2.VideoCapture(stream_url)
    else:
        print("\n💻  Opening Laptop Webcam...")
        cap = cv2.VideoCapture(0)
        # The laptop camera is for high-quality training data. The deployed
        # ESP32 stream can remain fast at QVGA for live recognition.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, LAPTOP_CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LAPTOP_CAPTURE_HEIGHT)

    if not cap.isOpened():
        print(f"❌  Cannot open camera. {'Check connection to SampleESPNetwork' if is_pov else 'Check webcam permissions'}.")
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📐  Capture resolution: {actual_width}x{actual_height}")

    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️  Failed to grab frame. Retrying...")
            cv2.waitKey(100)
            continue

        # Only flip for selfie webcam, keep POV orientation natural for smart glasses
        if not is_pov:
            frame = cv2.flip(frame, 1)

        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image and detect
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect(mp_image)

        # ─── Step 5: If a hand is detected, process it ────────
        normalized = None
        hand_is_large_enough = False
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand_landmarks = results.hand_landmarks[0]

            # Draw the hand skeleton on the frame
            h, w, _ = frame.shape
            pts = []
            for lm in hand_landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                pts.append((px, py))
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

            for c in HAND_CONNECTIONS:
                if c[0] < len(pts) and c[1] < len(pts):
                    cv2.line(frame, pts[c[0]], pts[c[1]], (0, 255, 0), 2)

            xs = [point[0] for point in pts]
            ys = [point[1] for point in pts]
            hand_span_ratio = max(max(xs) - min(xs), max(ys) - min(ys)) / max(w, h)
            hand_is_large_enough = hand_span_ratio >= MIN_HAND_SPAN_RATIO

            # ─── Step 6: NORMALIZE landmarks ──────────────────
            wrist = hand_landmarks[0]
            wx, wy, wz = wrist.x, wrist.y, wrist.z

            normalized = []
            for lm in hand_landmarks:
                normalized.extend([
                    lm.x - wx,
                    lm.y - wy,
                    lm.z - wz,
                ])

            if hand_is_large_enough:
                cv2.putText(frame, "GOOD SAMPLE - press 's' to save",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
            else:
                cv2.putText(frame, "MOVE HAND CLOSER - sample not saved",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 165, 255), 2)
        else:
            cv2.putText(frame, "No hand detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

        # Show the label and count on screen
        cv2.putText(frame,
                    f"Label: {label}  |  Total: {existing_label_count + saved_count}/{TARGET_SAMPLES_PER_LABEL}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)

        cv2.imshow("Collect Sign Data  (s=Save | q=Quit)", frame)

        # ─── Step 7: Handle key presses ───────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if normalized is not None and hand_is_large_enough:
                row = normalized + [label]
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                saved_count += 1
                print(f"   💾  Saved sample #{saved_count} for '{label}'")
            elif normalized is not None:
                print("   ⚠️  Hand is too small in frame — move closer before saving.")
            else:
                print("   ⚠️  No hand on screen — nothing saved.")

        elif key == ord("q"):
            print(f"\n🏁  Done! Saved {saved_count} samples for '{label}'.")
            break

    # ─── Cleanup ──────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"📊  Data is in: {CSV_FILE}")


if __name__ == "__main__":
    # Change to script directory so relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
