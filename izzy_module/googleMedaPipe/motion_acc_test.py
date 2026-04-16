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

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.8, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# --- DOCKER-OPTIMIZED CAMERA SETUP ---
cap = cv2.VideoCapture(0)

# Force MJPEG to prevent 'select() timeout' in Docker/WSL
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ ERROR: Could not open video device. Ensure usbipd is attached to WSL.")
    exit()

motion_buffer = []

def is_finger_up(hand_landmarks, finger_tip_id, finger_pip_id):
    """Returns True if the finger tip is above the PIP joint (finger is extended)"""
    return hand_landmarks.landmark[finger_tip_id].y < hand_landmarks.landmark[finger_pip_id].y

print("✅ Camera started. Press 'ESC' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("⚠️ Failed to grab frame. Checking connection...")
        break
        
    frame = cv2.flip(frame, 1)
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    best_match = "None"
    lowest_dist = float('inf')
    display_text = "SEARCHING..."
    color = (0, 0, 255) # Default Red

    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        current_side = results.multi_handedness[0].classification[0].label.lower()
        
        # --- FINGER GATE LOGIC ---
        index_up = is_finger_up(hand_lms, 8, 6)   # Index Tip (8) vs PIP (6)
        pinky_up = is_finger_up(hand_lms, 20, 18) # Pinky Tip (20) vs PIP (18)

        # Normalize and Buffer
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
        wrist = pts[0]
        # Scale based on palm size (Wrist to Index MCP)
        scale = np.linalg.norm(pts[0] - pts[5])
        if scale == 0: scale = 1 # Prevent division by zero
        
        normalized = ((pts - wrist) / scale).flatten()
        
        motion_buffer.append(normalized)
        if len(motion_buffer) > BUFFER_SIZE: 
            motion_buffer.pop(0)

        # Draw the hand skeleton
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # Only compare if we have enough temporal data
        if len(motion_buffer) >= 20:
            current_seq = np.array(motion_buffer)[::2] # Downsample for speed
            
            for name, saved_seq in motion_lib.items():
                # GATE 1: Side Match
                if current_side in name.lower():
                    
                    # GATE 2: If Index is up, check for 'Z'
                    if index_up and 'z' in name.lower():
                        dist, _ = fastdtw(current_seq, saved_seq[::2], radius=1, dist=euclidean)
                        if dist < lowest_dist:
                            lowest_dist, best_match = dist, name
                    
                    # GATE 3: If Pinky is up, check for 'J'
                    elif pinky_up and 'j' in name.lower():
                        dist, _ = fastdtw(current_seq, saved_seq[::2], radius=1, dist=euclidean)
                        if dist < lowest_dist:
                            lowest_dist, best_match = dist, name

        # --- UI Logic ---
        if lowest_dist < THRESHOLD:
            color = (0, 255, 0) # Green for match
            display_text = best_match.upper()
        else:
            display_text = "SIGNING..."

    # Overlay text on the frame
    cv2.putText(frame, f"DETECTED: {display_text}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Show the result via X11/VcXsrv
    cv2.imshow('ASL Finger-Gated Test', frame)
    
    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()