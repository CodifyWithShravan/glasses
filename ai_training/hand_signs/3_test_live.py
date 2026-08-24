"""
===========================================================
 3_test_live.py  –  Live Sign Language Prediction
===========================================================
HOW IT WORKS:
  1. Loads the trained model from  sign_model.pkl .
  2. Opens your webcam and runs MediaPipe Hands.
  3. Normalizes landmarks the SAME way as in data collection
     (subtract Landmark 0 / wrist from all points).
  4. Feeds the normalized landmarks into the model.
  5. Displays the predicted sign word on the video feed.

USAGE:
  python 3_test_live.py
  Press 'q' to quit.
===========================================================
"""

import cv2
import mediapipe as mp
import joblib
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────
MODEL_FILE = "sign_model.pkl"
NUM_LANDMARKS = 21
# ───────────────────────────────────────────────────────────

# ─── Step 1: Load the trained model ──────────────────────
print("📦  Loading model from", MODEL_FILE, "...")
model = joblib.load(MODEL_FILE)
print("   ✅  Model loaded!\n")

# ─── Step 2: Set up MediaPipe Hands ──────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ─── Step 3: Open the webcam ─────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌  Cannot open webcam.")
    exit()

print("🎥  Webcam is live!  Press 'q' to quit.\n")

# We'll keep track of the current prediction to display it smoothly
current_prediction = "..."
confidence = 0.0

while True:
    success, frame = cap.read()
    if not success:
        continue

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw the skeleton
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
        )

        # ─── Step 4: NORMALIZE (same as data collection!) ─
        wrist = hand_landmarks.landmark[0]
        wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

        normalized = []
        for lm in hand_landmarks.landmark:
            normalized.extend([
                lm.x - wrist_x,
                lm.y - wrist_y,
                lm.z - wrist_z,
            ])

        # ─── Step 5: Predict ─────────────────────────────
        # Reshape into a 2D array (1 sample, 63 features)
        features = np.array(normalized).reshape(1, -1)

        # Get the predicted label
        current_prediction = model.predict(features)[0]

        # Get confidence (probability of the top prediction)
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities) * 100  # Convert to %

    # ─── Step 6: Display the prediction on screen ─────────
    # Big green text showing the predicted sign
    cv2.putText(
        frame,
        f"Sign: {current_prediction}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,           # Font scale (big!)
        (0, 255, 0),   # Green color
        3,             # Thickness
    )

    # Smaller text showing confidence percentage
    cv2.putText(
        frame,
        f"Confidence: {confidence:.0f}%",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),  # Yellow
        2,
    )

    # Show the frame
    cv2.imshow("Sign Language Tester  (q=Quit)", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ─── Cleanup ──────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
hands.close()
print("👋  Done!")
