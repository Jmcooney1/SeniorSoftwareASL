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

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
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
        # Draw the skeleton
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
        
        # Display the result
        color = (0, 255, 0) if highest_accuracy > 90 else (0, 255, 255)
        
        # Extract base letter name (e.g., 'a_front' becomes 'A')
        display_name = best_match.split('_')[0].upper()
        
        cv2.putText(frame, f"PREDICTION: {display_name}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Match: {highest_accuracy:.1f}%", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('ASL Real-Time Recognition', frame)
    
    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()