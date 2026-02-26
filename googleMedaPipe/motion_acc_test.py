import cv2
import mediapipe as mp
import numpy as np
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# --- CONFIG ---
MOTION_LIB_PATH = 'asl_motion_library.npy'
BUFFER_SIZE = 40
THRESHOLD = 1500 

if not os.path.exists(MOTION_LIB_PATH):
    print("❌ Library not found!")
    exit()

motion_lib = np.load(MOTION_LIB_PATH, allow_pickle=True).item()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.8, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
motion_buffer = []

def is_finger_up(hand_landmarks, finger_tip_id, finger_pip_id):
    """Returns True if the finger tip is above the PIP joint (finger is extended)"""
    return hand_landmarks.landmark[finger_tip_id].y < hand_landmarks.landmark[finger_pip_id].y

while True:
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    best_match = "None"
    lowest_dist = float('inf')
    status_msg = "Searching..."

    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        current_side = results.multi_handedness[0].classification[0].label.lower()
        
        # --- FINGER GATE LOGIC ---
        # Index Tip = 8, Index PIP = 6
        index_up = is_finger_up(hand_lms, 8, 6)
        # Pinky Tip = 20, Pinky PIP = 18
        pinky_up = is_finger_up(hand_lms, 20, 18)

        # Normalize and Buffer
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
        wrist = pts[0]
        scale = np.linalg.norm(pts[0] - pts[5])
        normalized = ((pts - wrist) / scale).flatten()
        
        motion_buffer.append(normalized)
        if len(motion_buffer) > BUFFER_SIZE: motion_buffer.pop(0)

        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # Only compare if we have enough data
        if len(motion_buffer) >= 20:
            current_seq = np.array(motion_buffer)[::2]
            
            for name, saved_seq in motion_lib.items():
                # GATE 1: Must match Left/Right side
                if current_side in name.lower():
                    
                    # GATE 2: If Index is up, ONLY check for 'Z'
                    if index_up and 'z' in name.lower():
                        dist, _ = fastdtw(current_seq, saved_seq[::2], radius=1, dist=euclidean)
                        if dist < lowest_dist:
                            lowest_dist, best_match = dist, name
                    
                    # GATE 3: If Pinky is up, ONLY check for 'J'
                    elif pinky_up and 'j' in name.lower():
                        dist, _ = fastdtw(current_seq, saved_seq[::2], radius=1, dist=euclidean)
                        if dist < lowest_dist:
                            lowest_dist, best_match = dist, name

    # --- UI Logic ---
    if lowest_dist < THRESHOLD:
        color = (0, 255, 0)
        display_text = best_match.upper()
    else:
        color = (0, 0, 255)
        display_text = "SIGNING..."

    cv2.putText(frame, f"DETECTED: {display_text}", (10, 50), 1, 2, color, 2)
    cv2.imshow('ASL Finger-Gated Test', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()