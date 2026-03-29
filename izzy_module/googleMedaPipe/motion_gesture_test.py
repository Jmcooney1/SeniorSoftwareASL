import cv2
import mediapipe as mp
import numpy as np
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# --- CONFIG ---
MOTION_LIB_PATH = 'asl_motion_library.npy'
BUFFER_SIZE = 45 
THRESHOLD_SINGLE = 1600 
THRESHOLD_DUAL = 2500 
ALPHA = 0.3 # Smoothing for better "Proof"

if not os.path.exists(MOTION_LIB_PATH):
    print(f"❌ Error: {MOTION_LIB_PATH} not found!")
    exit()

motion_lib = np.load(MOTION_LIB_PATH, allow_pickle=True).item()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.7, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
sync_buffer = []
history = {"left": None, "right": None}

def get_smoothed_norm(hand_lms, side):
    pts = np.array([[lm.x, lm.y, lm.z * 2.0] for lm in hand_lms.landmark])
    wrist = pts[0]
    scale = np.linalg.norm(pts[0] - pts[5]) or 1.0
    new_norm = ((pts - wrist) / scale).flatten()
    if history[side] is None:
        history[side] = new_norm
    else:
        history[side] = (ALPHA * new_norm) + ((1 - ALPHA) * history[side])
    return history[side]

while True:
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    final_gesture = "SEARCHING..."
    display_color = (0, 0, 255)
    
    frame_data = {"left": None, "right": None}
    pointing_at_me = {"left": False, "right": False}

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            side = "left" if hand_lms.landmark[0].x < 0.5 else "right"
            
            # Depth check for 'Me'
            z_depth = hand_lms.landmark[8].z - hand_lms.landmark[5].z
            pointing_at_me[side] = z_depth > 0.05
            
            frame_data[side] = get_smoothed_norm(hand_lms, side)
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    sync_buffer.append(frame_data)
    if len(sync_buffer) > BUFFER_SIZE: sync_buffer.pop(0)

    best_dist = float('inf')
    best_name = None

    if len(sync_buffer) >= 20:
        # Check active hands
        hands_active = {"left": frame_data["left"] is not None, "right": frame_data["right"] is not None}

        # 1. DUAL-HAND CHECK (New: _dual_1 | Old: _dual)
        if hands_active["left"] and hands_active["right"]:
            for name, saved_seq in motion_lib.items():
                if "_dual" in name.lower():
                    live_slice = sync_buffer[::2]
                    saved_slice = saved_seq[::2]
                    
                    # Process saved sequence (Handles both list of dicts and raw arrays)
                    # This try/except block ensures old file formats don't crash the script
                    try:
                        l_save = [f['left'] if (isinstance(f, dict) and f['left'] is not None) else np.zeros(63) for f in saved_slice]
                        r_save = [f['right'] if (isinstance(f, dict) and f['right'] is not None) else np.zeros(63) for f in saved_slice]
                        
                        l_live = [f['left'] if f['left'] is not None else np.zeros(63) for f in live_slice]
                        r_live = [f['right'] if f['right'] is not None else np.zeros(63) for f in live_slice]

                        d_l, _ = fastdtw(l_live, l_save, radius=1, dist=euclidean)
                        d_r, _ = fastdtw(r_live, r_save, radius=1, dist=euclidean)
                        
                        dist = (d_l + d_r) / 2
                        if dist < best_dist and dist < THRESHOLD_DUAL:
                            best_dist = dist
                            best_name = name
                    except: continue

        # 2. SINGLE-HAND CHECK (New: _left_1 | Old: _left)
        if best_name is None:
            for side in ["left", "right"]:
                if not hands_active[side]: continue
                live_seq = [f[side] for f in sync_buffer if f[side] is not None]
                
                for name, saved_seq in motion_lib.items():
                    # Flexible naming check: matches 'silly_left_1' AND 'hello_left'
                    if f"_{side}" in name.lower() and "_dual" not in name.lower():
                        try:
                            dist, _ = fastdtw(live_seq[::2], saved_seq[::2], radius=1, dist=euclidean)
                            if 'me' in name.lower():
                                dist = dist * 0.4 if pointing_at_me[side] else dist * 3.0
                            
                            if dist < best_dist and dist < THRESHOLD_SINGLE:
                                best_dist = dist
                                best_name = name
                        except: continue

    if best_name:
        # Display cleanup: "silly_left_1" -> "SILLY", "COOK_dual" -> "COOK"
        final_gesture = best_name.split('_')[0].upper()
        display_color = (0, 255, 0)

    cv2.putText(frame, f"GESTURE: {final_gesture}", (10, 50), 1, 2, display_color, 2)
    cv2.imshow('Universal Test Runner', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()