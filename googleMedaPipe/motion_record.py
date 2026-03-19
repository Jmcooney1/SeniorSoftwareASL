import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- CONFIG ---
LIBRARY_FILE = 'asl_motion_library.npy'
RECORD_DURATION = 2.5 
Z_SCALE = 2.0  
ALPHA = 0.3  # Smoothing factor

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=1, 
    min_detection_confidence=0.8, 
    min_tracking_confidence=0.8, 
    max_num_hands=2
)
mp_draw = mp.solutions.drawing_utils

def load_library():
    if os.path.exists(LIBRARY_FILE):
        return np.load(LIBRARY_FILE, allow_pickle=True).item()
    return {}

motion_library = load_library()
history = {"left": None, "right": None}

def get_smoothed_norm(hand_lms, side):
    pts = np.array([[lm.x, lm.y, lm.z * Z_SCALE] for lm in hand_lms.landmark])
    wrist = pts[0]
    scale = np.linalg.norm(pts[0] - pts[5]) or 1.0
    new_norm = ((pts - wrist) / scale).flatten()
    
    if history[side] is None:
        history[side] = new_norm
    else:
        history[side] = (ALPHA * new_norm) + ((1 - ALPHA) * history[side])
    return history[side]

cap = cv2.VideoCapture(0)

while True:
    print("\n" + "="*40)
    print("TRAINER: 1 (Single Hand) | 2 (Dual Hand) | exit")
    mode = input("Select Mode: ").strip()
    if mode.lower() == 'exit': break
    
    base_name = input("Enter Gesture Name (e.g., 'silly'): ").strip().lower()
    
    # --- AUTO-INCREMENT LOGIC ---
    if mode == "1":
        side_input = input("Target Side (l/r): ").lower()
        target_side = "left" if side_input == 'l' else "right"
        prefix = f"{base_name}_{target_side}_"
    else:
        prefix = f"{base_name}_dual_"

    # Find next available number (e.g., silly_left_3)
    existing = [int(k.split('_')[-1]) for k in motion_library.keys() if k.startswith(prefix)]
    instance_num = max(existing) + 1 if existing else 1
    label = f"{prefix}{instance_num}"

    print(f"\n>>> PREPARING: {label.upper()}")
    print("Hold 'R' in the window to start recording.")

    sequence = []
    recording = False
    start_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        current_frame_data = {"left": None, "right": None}
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                side = "left" if hand_lms.landmark[0].x < 0.5 else "right"
                current_frame_data[side] = get_smoothed_norm(hand_lms, side)
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        if not recording:
            cv2.putText(frame, f"READY: {label}", (10, 50), 1, 1.5, (0, 255, 0), 2)
            if cv2.waitKey(1) & 0xFF == ord('r'):
                recording = True
                start_time = time.time()
        else:
            elapsed = time.time() - start_time
            if mode == "1":
                val = current_frame_data[target_side] if current_frame_data[target_side] is not None else np.zeros(63)
                sequence.append(val)
            else:
                sequence.append(current_frame_data)
            
            cv2.putText(frame, f"RECORDING... {elapsed:.1f}s", (10, 50), 1, 2, (0, 0, 255), 2)
            if elapsed >= RECORD_DURATION: break

        cv2.imshow('Universal Trainer', frame)
        cv2.waitKey(1)

    # --- TERMINAL DECISION GATE ---
    print(f"\n--- REVIEW RECORDING ---")
    print(f"Label: {label}")
    print(f"Frames: {len(sequence)}")
    
    # The script stops here and waits for your terminal input
    choice = input("SAVE this recording? (y = Yes / n = Discard / r = Retry same index): ").strip().lower()

    if choice == 'y':
        motion_library[label] = np.array(sequence, dtype=object)
        np.save(LIBRARY_FILE, motion_library)
        print(f"✅ SAVED TO LIBRARY: {label}")
    elif choice == 'r':
        print(f"🔄 Retrying {label}... (Index not incremented)")
        # This will loop back and the next iteration will use the same index because it wasn't saved
        continue 
    else:
        print(f"🗑️ DISCARDED: {label} was not saved.")

cap.release()
cv2.destroyAllWindows()