import cv2
import mediapipe as mp
import numpy as np
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# --- CONFIG ---
MOTION_LIB_PATH = 'asl_motion_library.npy'

# Load the library
if not os.path.exists(MOTION_LIB_PATH):
    print("❌ Motion library not found!")
    exit()
motion_lib = np.load(MOTION_LIB_PATH, allow_pickle=True).item()

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
buffer_size = 60 # We look at the last ~3 seconds of movement
motion_buffer = []

print("--- MOTION ACCURACY TESTER ---")
print("Perform a motion (J or Z) and the AI will check the similarity score.")

while True:
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # 1. Capture current frame data
        pts = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
        normalized = (pts - pts[0]).flatten()
        
        # 2. Add to a rolling "buffer" 
        motion_buffer.append(normalized)
        if len(motion_buffer) > buffer_size:
            motion_buffer.pop(0)
        
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    # 3. Compare buffer against every saved motion in the library
    best_match = "None"
    lowest_distance = float('inf')

    if len(motion_buffer) > 20: 
        current_sequence = np.array(motion_buffer)
        
        for name, saved_sequence in motion_lib.items():
            # Use DTW to find the distance
            distance, _ = fastdtw(current_sequence, saved_sequence, dist=euclidean)
            
            if distance < lowest_distance:
                lowest_distance = distance
                best_match = name

    # 4. Display Results (FIXED INDENTATION AND OVERFLOW)
    accuracy_text = f"Best Match: {best_match.upper()}"
    
    if lowest_distance == float('inf'):
        score_text = "Distance Score: N/A"
    else:
        score_text = f"Distance Score: {int(lowest_distance)}"
    
    # Text displays
    cv2.putText(frame, accuracy_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, score_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Accuracy Test', frame)
    
    # Use 'q' to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()