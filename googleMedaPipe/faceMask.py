import cv2
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Load your ripped library
# Using try/except in case the file isn't created yet
try:
    asl_library = np.load('asl_library.npy', allow_pickle=True).item()
except FileNotFoundError:
    asl_library = {}
    print("Warning: asl_library.npy not found. Run create_library.py first.")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    model_complexity=1, # Increased for better accuracy
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks (Standard)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- 2. PREPARE LIVE DATA ---
            # Extract and normalize (Wrist subtraction)
            live_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            live_norm = (live_pts - live_pts[0]).flatten().reshape(1, -1)

            # --- 3. COMPARE TO LIBRARY ---
            best_match = "Scanning..."
            highest_score = 0
            
            for letter, ref_norm in asl_library.items():
                # Compare live hand to each letter in dictionary
                score = cosine_similarity(live_norm, ref_norm.reshape(1, -1))[0][0]
                
                if score > highest_score:
                    highest_score = score
                    best_match = letter

            # --- 4. DISPLAY RESULTS ---
            # If the score is high enough (e.g., > 90%), show the letter
            if highest_score > 0.92:
                display_text = f"Letter: {best_match.upper()} ({highest_score:.2f})"
                color = (0, 255, 0) # Green for match
            else:
                display_text = "Aligning..."
                color = (0, 0, 255) # Red for no confident match

            cv2.putText(image, display_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('ASL Real-Time Recognition', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()