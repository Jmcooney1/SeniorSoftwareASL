import cv2
import mediapipe as mp
import numpy as np
import os

# --- SETTINGS ---
# Use the same path you used before
SAVE_PATH = r"C:\Users\izzyd\OneDrive\Documents\ASL senior software\SeniorSoftwareASL\maual_downloads"
os.makedirs(SAVE_PATH, exist_ok=True)

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("--- ASL MANUAL RECORDER ---")
print("1. Make a hand sign in front of the camera.")
print("2. Type the LETTER on your keyboard to save it (e.g., press 'e' for E).")
print("3. Press 'ESC' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1) # Mirror view
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw landmarks so you know if MediaPipe can 'see' you
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, "READY TO SAVE", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Recorder', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27: # ESC to quit
        break
    elif 97 <= key <= 122: # If any letter a-z is pressed
        letter = chr(key)
        filename = f"{letter}.jpg"
        full_path = os.path.join(SAVE_PATH, filename)
        
        # Save the actual image to your manual_downloads folder
        cv2.imwrite(full_path, frame)
        print(f"✅ Saved {letter.upper()} to {full_path}")

cap.release()
cv2.destroyAllWindows()