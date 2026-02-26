import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- CONFIG ---
LIBRARY_FILE = 'asl_motion_library.npy'
RECORD_DURATION = 4 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

if os.path.exists(LIBRARY_FILE):
    motion_library = np.load(LIBRARY_FILE, allow_pickle=True).item()
else:
    motion_library = {}

cap = cv2.VideoCapture(0)

while True:
    print("\n" + "="*30)
    letter = input("Enter Letter (e.g., 'j' or 'z'): ").strip().lower()
    if letter == 'exit': break
    
    hand_side = input("Enter Side (l for Left / r for Right): ").strip().lower()
    side_label = "left" if hand_side == 'l' else "right"
    
    # This creates the unique key: e.g., 'j_left'
    label = f"{letter}_{side_label}"

    print(f"PREPARING TO RECORD: {label.upper()}")
    print(f"Ensure your {side_label.upper()} hand is in view. Press 'R' to start.")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Ready: {label}", (10, 50), 1, 2, (255, 255, 255), 2)
        cv2.imshow('Recorder', frame)
        if cv2.waitKey(1) & 0xFF == ord('r'): break

    sequence = []
    start_time = time.time()
    while time.time() - start_time < RECORD_DURATION:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            # Check if MediaPipe agrees with your side choice
            detected_side = results.multi_handedness[0].classification[0].label.lower()
            
            pts = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
            normalized = (pts - pts[0]).flatten()
            sequence.append(normalized)
            
            # Visual feedback: Green if correct hand, Red if wrong hand
            color = (0, 255, 0) if detected_side == side_label else (0, 0, 255)
            mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"REC: {detected_side.upper()}", (10, 50), 1, 2, color, 2)

        cv2.imshow('Recorder', frame)
        cv2.waitKey(1)

    if len(sequence) > 15:
        motion_library[label] = np.array(sequence)
        np.save(LIBRARY_FILE, motion_library)
        print(f"✅ Saved as '{label}'")

cap.release()
cv2.destroyAllWindows()