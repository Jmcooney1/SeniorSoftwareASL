import cv2
import mediapipe as mp
import numpy as np
import os

# --- SETUP ---
if not os.path.exists('asl_library.npy'):
    print("❌ Error: asl_library.npy not found!")
    exit()

asl_library = np.load('asl_library.npy', allow_pickle=True).item()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- DOCKER & WSL OPTIMIZED CAMERA SETUP ---
cap = cv2.VideoCapture(0)

# 1. Force MJPEG to prevent 'select() timeout'
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 2. Initialize X11 Window for Docker visibility
cv2.namedWindow('ASL Real-Time Recognition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ASL Real-Time Recognition', 960, 720)

print("✅ Accuracy Test started. Looking for matches...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⚠️ Failed to grab frame. Check camera connection.")
        break
        
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    best_match = "None"
    highest_accuracy = 0
    
    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        
        # 1. Capture and Normalize Live Geometry
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
        live_geo = (pts - pts[0]).flatten() 
        norm_live = np.linalg.norm(live_geo)
        
        # 2. SCAN THE ENTIRE LIBRARY
        for letter, saved_geo in asl_library.items():
            # Standard Cosine Similarity calculation
            dot_product = np.dot(live_geo, saved_geo)
            norm_saved = np.linalg.norm(saved_geo)
            
            # Prevent division by zero
            if norm_live == 0 or norm_saved == 0: continue
            
            similarity = dot_product / (norm_live * norm_saved)
            accuracy = similarity * 100
            
            # Keep track of the highest score
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_match = letter

        # 3. VISUAL FEEDBACK
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
        
        # Set color based on confidence (Green > 90%, Yellow otherwise)
        color = (0, 255, 0) if highest_accuracy > 90 else (0, 255, 255)
        
        # Extract base letter name (e.g., 'a_front' becomes 'A')
        display_name = best_match.split('_')[0].upper()
        
        # Visual text overlays
        cv2.putText(frame, f"PREDICTION: {display_name}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Match: {highest_accuracy:.1f}%", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the result in the resized window
    cv2.imshow('ASL Real-Time Recognition', frame)
    
# FIX: Check if the user pressed ESC OR clicked the 'X' on the window
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or cv2.getWindowProperty('ASL Real-Time Recognition', cv2.WND_PROP_VISIBLE) < 1: 
        break

# Proper Cleanup
print("Closing camera and cleaning up...")
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1) # Extra kick to help X11 close the window