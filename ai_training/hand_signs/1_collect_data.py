"""
===========================================================
 1_collect_data.py  –  Record Dual-Hand / Single-Hand Sign Data
===========================================================
HOW IT WORKS:
  1. Opens laptop webcam OR Smart Glasses POV stream.
  2. Detects up to TWO hands using MediaPipe HandLandmarker.
  3. NORMALIZES every landmark by subtracting each hand's wrist (Landmark 0)
     so the data is position-independent.
  4. Produces a 126-element feature vector (Hand 1: 63 + Hand 2: 63).
     - Single-hand signs: Hand 1 has 63 values, Hand 2 is padded with 0s.
     - Dual-hand signs: Both Hand 1 and Hand 2 have 63 values.
  5. Press 's' to save a sample, 'q' to quit.

USAGE:
  .venv/bin/python3 1_collect_data.py
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
CSV_FILE = "dataset/sign_data.csv"
MODEL_PATH = "hand_landmarker.task"
NUM_LANDMARKS = 21                # 21 points per hand
NUM_HANDS = 2                     # Support both single and dual hands
TOTAL_FEATURES = NUM_HANDS * NUM_LANDMARKS * 3  # 2 * 21 * 3 = 126 features
# ───────────────────────────────────────────────────────────

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # Thumb
    (0,5),(5,6),(6,7),(7,8),       # Index
    (0,9),(9,10),(10,11),(11,12),   # Middle
    (0,13),(13,14),(14,15),(15,16), # Ring
    (0,17),(17,18),(18,19),(19,20), # Pinky
    (5,9),(9,13),(13,17),           # Palm connections
]

HAND_COLORS = [(0, 255, 0), (255, 128, 0)]  # Hand 1: Green, Hand 2: Orange


def extract_dual_hand_features(hand_landmarks_list):
    """
    Extracts a 126-element feature vector from detected hands.
    Sorts hands from left-to-right on screen for deterministic feature ordering.
    """
    if not hand_landmarks_list:
        return None

    # Sort hands by wrist X coordinate (left hand first, right hand second)
    sorted_hands = sorted(hand_landmarks_list, key=lambda hand: hand[0].x)

    features = []

    for hand_idx in range(NUM_HANDS):
        if hand_idx < len(sorted_hands):
            hand = sorted_hands[hand_idx]
            wrist = hand[0]
            wx, wy, wz = wrist.x, wrist.y, wrist.z

            for lm in hand:
                features.extend([
                    lm.x - wx,
                    lm.y - wy,
                    lm.z - wz,
                ])
        else:
            # Pad second hand with zeros if only 1 hand is present
            features.extend([0.0] * (NUM_LANDMARKS * 3))

    return features


def main():
    label = input("🏷️  Enter the label for the sign (e.g. Hello, Thanks, Namaste, Help): ").strip()
    if not label:
        print("❌  Label cannot be empty. Exiting.")
        return

    print(f"\n✅  Recording label: '{label}' (Dual-Hand & Single-Hand Supported)")
    print("    Press 's' to SAVE a frame   |   Press 'q' to QUIT\n")

    # ─── Build the 126-column CSV header ─────────────
    header = []
    for h in range(1, NUM_HANDS + 1):
        for i in range(NUM_LANDMARKS):
            header.extend([f"h{h}_x{i}", f"h{h}_y{i}", f"h{h}_z{i}"])
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

    # ─── Set up MediaPipe HandLandmarker for 2 Hands ─
    if not os.path.exists(MODEL_PATH):
        print(f"❌  Model file not found: {MODEL_PATH}")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=BaseOptions.Delegate.CPU
        ),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # ─── Choose Camera Source ─────────────────────────
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

    if not cap.isOpened():
        print(f"❌  Cannot open camera. {'Check connection to SampleESPNetwork' if is_pov else 'Check webcam permissions'}.")
        return

    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️  Failed to grab frame. Retrying...")
            cv2.waitKey(100)
            continue

        if not is_pov:
            frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect(mp_image)

        normalized_features = None
        num_detected = 0

        if results.hand_landmarks:
            num_detected = len(results.hand_landmarks)
            h, w, _ = frame.shape

            # Draw skeletons for each detected hand
            for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
                color = HAND_COLORS[hand_idx % len(HAND_COLORS)]
                pts = []
                for lm in hand_landmarks:
                    px, py = int(lm.x * w), int(lm.y * h)
                    pts.append((px, py))
                    cv2.circle(frame, (px, py), 5, color, -1)

                for c in HAND_CONNECTIONS:
                    if c[0] < len(pts) and c[1] < len(pts):
                        cv2.line(frame, pts[c[0]], pts[c[1]], color, 2)

            normalized_features = extract_dual_hand_features(results.hand_landmarks)

            hand_text = f"{num_detected} HAND{'S' if num_detected > 1 else ''} DETECTED - press 's' to save"
            cv2.putText(frame, hand_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hands detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"Label: {label}  |  Saved: {saved_count}  |  Hands: {num_detected}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Dual-Hand Sign Collector  (s=Save | q=Quit)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if normalized_features is not None:
                row = normalized_features + [label]
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                saved_count += 1
                print(f"   💾  Saved sample #{saved_count} for '{label}' ({num_detected} hand{'s' if num_detected > 1 else ''})")
            else:
                print("   ⚠️  No hand on screen — nothing saved.")

        elif key == ord("q"):
            print(f"\n🏁  Done! Saved {saved_count} samples for '{label}'.")
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"📊  Data saved to: {CSV_FILE}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
