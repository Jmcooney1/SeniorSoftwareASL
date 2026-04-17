import cv2
import mediapipe as mp
import numpy as np
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

MOTION_LIB_PATH = 'asl_motion_library.npy'
BUFFER_SIZE = 40
THRESHOLD = 1500
BRIDGE_FILE = '/tmp/asl_detected_letter.txt'

if not os.path.exists(MOTION_LIB_PATH):
    print("Library not found!")
    exit()

motion_lib = np.load(MOTION_LIB_PATH, allow_pickle=True).item()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.8, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

motion_buffer = []

def is_finger_up(hand_landmarks, finger_tip_id, finger_pip_id):
    return hand_landmarks.landmark[finger_tip_id].y < hand_landmarks.landmark[finger_pip_id].y

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    best_match = "None"
    lowest_dist = float('inf')

    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        current_side = results.multi_handedness[0].classification[0].label.lower()

        index_up = is_finger_up(hand_lms, 8, 6)
        pinky_up = is_finger_up(hand_lms, 20, 18)

        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
        wrist = pts[0]
        scale = np.linalg.norm(pts[0] - pts[5])
        if scale == 0: scale = 1
        normalized = ((pts - wrist) / scale).flatten()

        motion_buffer.append(normalized)
        if len(motion_buffer) > BUFFER_SIZE:
            motion_buffer.pop(0)

        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        if len(motion_buffer) >= 20:
            current_seq = np.array(motion_buffer)[::2]
            
            for name, saved_seq in motion_lib.items():
                if current_side in name.lower():
                    saved_downsampled = saved_seq[::2]

                    # ← Skip if shapes don't match
                    if current_seq.shape[1] != saved_downsampled.shape[1]:
                        continue
                    
                    if index_up and 'z' in name.lower():
                        dist, _ = fastdtw(current_seq, saved_downsampled, radius=1, dist=euclidean)
                        if dist < lowest_dist:
                            lowest_dist, best_match = dist, name
                    
                    elif pinky_up and 'j' in name.lower():
                        dist, _ = fastdtw(current_seq, saved_downsampled, radius=1, dist=euclidean)
                        if dist < lowest_dist:
                            lowest_dist, best_match = dist, name

        # ← Write to bridge file when confident match found
        if lowest_dist < THRESHOLD and best_match != "None":
            display_letter = best_match.split('_')[0].upper()
            with open(BRIDGE_FILE, 'w') as f:
                f.write(display_letter)

        color = (0, 255, 0) if lowest_dist < THRESHOLD else (0, 0, 255)
        display_text = best_match.upper() if lowest_dist < THRESHOLD else "SIGNING..."
        cv2.putText(frame, f"DETECTED: {display_text}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('ASL Finger-Gated Test', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()