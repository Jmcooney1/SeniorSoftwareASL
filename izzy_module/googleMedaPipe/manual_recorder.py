import cv2
import mediapipe as mp
import numpy as np
import os

# --- PATH CONFIGURATION ---
DOWNLOAD_DIR = r"C:\Users\izzyd\OneDrive\Documents\ASL senior software\SeniorSoftwareASL\download_imges"
LIBRARY_FILE = 'asl_library.npy'

if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

if os.path.exists(LIBRARY_FILE):
    asl_library = np.load(LIBRARY_FILE, allow_pickle=True).item()
else:
    asl_library = {}

cap = cv2.VideoCapture(0)

print(f"--- ASL RECORDER (FIXED FOR 'Q') ---")
print(f"Type 'exit' to close the program.")

while True:
    # CHANGED: Now 'exit' quits, so 'q' is free for the letter Q
    letter_input = input("\nWhich Letter? (Type 'exit' to quit): ").lower().strip()
    
    if letter_input == 'exit':
        break
    
    # Validation to ensure it's a valid letter (a-z)
    if len(letter_input) > 1 and letter_input not in ['exit']:
         print("⚠️ Please enter a single letter or 'exit'.")
         continue

    hand_side = input("Hand? (l/r): ").lower().strip()
    hand_label = "left" if hand_side == 'l' else "right"
    
    view = input("View? (front/side/top): ").lower().strip()
    target_name = f"{letter_input}_{hand_label}_{view}"

    print(f">> Capturing for {target_name.upper()}. Press 'S' on the camera window.")

    while True:
        success, image = cap.read()
        if not success: break
        
        image = cv2.flip(image, 1)
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        display_img = image.copy()
        if results.multi_hand_landmarks:
            mp_draw.draw_landmarks(display_img, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.putText(display_img, f"Target: {target_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Manual Record Mode', display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if results.multi_hand_landmarks:
                pts = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
                normalized = (pts - pts[0]).flatten()
                
                filename = f"{target_name}.jpg"
                final_path = os.path.join(DOWNLOAD_DIR, filename)
                
                counter = 1
                while os.path.exists(final_path):
                    final_path = os.path.join(DOWNLOAD_DIR, f"{target_name}({counter}).jpg")
                    counter += 1

                cv2.imwrite(final_path, image)
                file_key = os.path.basename(final_path).replace('.jpg', '')
                asl_library[file_key] = normalized
                np.save(LIBRARY_FILE, asl_library)
                
                print(f"✅ Success! Saved {file_key}")
                break 
            else:
                print("⚠️ Hand not detected!")
        
        elif key == 27: # ESC
            break

cap.release()
cv2.destroyAllWindows()