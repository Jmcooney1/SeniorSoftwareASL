import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- CONFIG ---
LIBRARY_FILE = 'asl_motion_library.npy'
RECORD_DURATION = 2.5 
Z_SCALE = 2.0  
ALPHA = 0.3  

# --- INITIALIZE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

def load_library():
    if os.path.exists(LIBRARY_FILE):
        return np.load(LIBRARY_FILE, allow_pickle=True).item()
    return {}

def save_to_library(label, sequence):
    lib = load_library()
    lib[label] = np.array(sequence, dtype=object)
    np.save(LIBRARY_FILE, lib)
    print(f"\n✅ SUCCESSFULLY SAVED: {label} ({len(sequence)} frames)")

# --- START DYNAMIC SESSION ---
print("="*40)
print("ASL MOTION RECORDER V2.0")
print("="*40)
user_name = input("Who is recording today? ").strip().lower() or "user"
cap = cv2.VideoCapture(0)
history = {"left": None, "right": None}

while True:
    print(f"\n--- New Recording Session: {user_name.upper()} ---")
    mode = input("Select Mode: [1] Single Hand | [2] Dual Hand | [exit]: ").strip().lower()
    if mode == 'exit': break
    
    gesture_name = input("Enter Gesture Name (e.g. hello, apple): ").strip().lower()
    
    target_side = "dual"
    if mode == "1":
        side_in = input("Which hand? [l/r]: ").lower()
        target_side = "left" if side_in == 'l' else "right"

    # Dynamic Label Creation
    # Format: username_side_gesture_instance
    prefix = f"{user_name}_{target_side}_{gesture_name}"
    
    # Auto-increment logic
    lib = load_library()
    existing = [int(k.split('_')[-1]) for k in lib.keys() if k.startswith(prefix + "_")]
    instance = max(existing) + 1 if existing else 1
    final_label = f"{prefix}_{instance}"

    print(f"\nREADY TO RECORD: {final_label}")
    print("Action: Focus on the camera window and hold 'R' to start.")

    sequence = []
    recording = False
    start_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        curr_data = {"left": None, "right": None}
        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                side = "left" if lms.landmark[0].x < 0.5 else "right"
                # Normalization logic
                pts = np.array([[lm.x, lm.y, lm.z * Z_SCALE] for lm in lms.landmark])
                wrist = pts[0]
                scale = np.linalg.norm(pts[0] - pts[5]) or 1.0
                norm = ((pts - wrist) / scale).flatten()
                
                # Simple smoothing
                if history[side] is None: history[side] = norm
                else: history[side] = (ALPHA * norm) + ((1 - ALPHA) * history[side])
                
                curr_data[side] = history[side]
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

        if not recording:
            cv2.putText(frame, "HOLD 'R' TO START", (50, 50), 1, 2, (0, 255, 0), 2)
            if cv2.waitKey(1) & 0xFF == ord('r'):
                recording = True
                start_time = time.time()
        else:
            elapsed = time.time() - start_time
            # Capture logic
            if mode == "1":
                val = curr_data[target_side] if curr_data[target_side] is not None else np.zeros(63)
                sequence.append(val)
            else:
                sequence.append(curr_data)
            
            cv2.putText(frame, f"RECORDING: {elapsed:.1f}s", (50, 50), 1, 2, (0, 0, 255), 2)
            if elapsed >= RECORD_DURATION: break

        cv2.imshow('Trainer Window', frame)
        if cv2.waitKey(1) & 0xFF == 27: break # ESC to abort

    # --- THE GATE ---
    print(f"\nRECORDING COMPLETE: {len(sequence)} frames captured.")
    choice = input(f"Save as '{final_label}'? [y] Save | [r] Retry | [n] Discard: ").lower().strip()
    
    if choice == 'y':
        save_to_library(final_label, sequence)
    elif choice == 'r':
        print("🔄 Restarting recording for the same index...")
        # (Logic will loop back and use the same instance number)
    else:
        print("🗑️ Discarded.")

cap.release()
cv2.destroyAllWindows()