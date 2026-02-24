import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- CONFIGURATION ---
DOWNLOAD_DIR = r"C:\Users\izzyd\OneDrive\Documents\ASL senior software\SeniorSoftwareASL\download_imges"
LIBRARY_FILE = 'asl_motion_library.npy'
FPS = 20.0  
RECORD_DURATION = 5  # 5 seconds of recording

if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

if os.path.exists(LIBRARY_FILE):
    motion_library = np.load(LIBRARY_FILE, allow_pickle=True).item()
else:
    motion_library = {}

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

print("--- CAMERA-CONTROLLED ASL RECORDER ---")

while True:
    label = input("\nEnter Motion Name (e.g., 'j_left_side') or 'exit': ").lower().strip()
    if label == 'exit':
        break

    print(f"Waiting for trigger... Press 'R' on the CAMERA WINDOW to begin.")

    # --- IDLE PHASE: Wait for 'R' key on the camera window ---
    while True:
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, f"TARGET: {label.upper()}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'R' to Start Recording", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Motion Capture', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('r'):
            break

    # --- 1. GET READY PHASE (3 Seconds) ---
    start_ready = time.time()
    while time.time() - start_ready < 3:
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        
        countdown = 3 - int(time.time() - start_ready)
        cv2.putText(frame, f"STARTING IN: {countdown}", (frame_width//4, frame_height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        cv2.imshow('Motion Capture', frame)
        cv2.waitKey(1)

    # --- 2. RECORDING PHASE (5 Seconds) ---
    sequence = []
    video_path = os.path.join(DOWNLOAD_DIR, f"{label}_motion.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, FPS, (frame_width, frame_height))

    start_record = time.time()
    while time.time() - start_record < RECORD_DURATION:
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        
        out.write(frame)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        display_frame = frame.copy()
        if results.multi_hand_landmarks:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
            normalized = (pts - pts[0]).flatten()
            sequence.append(normalized)
            mp_draw.draw_landmarks(display_frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        
        elapsed = time.time() - start_record
        remaining = max(0, RECORD_DURATION - elapsed)
        
        # UI Overlay
        cv2.putText(display_frame, "● RECORDING", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display_frame, f"TIME LEFT: {remaining:.1f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Motion Capture', display_frame)
        cv2.waitKey(1)

    out.release()

    if len(sequence) > 0:
        motion_library[label] = np.array(sequence)
        np.save(LIBRARY_FILE, motion_library)
        print(f"✅ Saved data and video for '{label}'")
    else:
        print("⚠️ Recording finished but no hand was detected.")

cap.release()
cv2.destroyAllWindows()