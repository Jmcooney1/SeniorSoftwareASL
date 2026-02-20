import cv2
import mediapipe as mp
import numpy as np
import os

# --- SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

library_path = 'asl_library.npy'
download_folder = 'download_imges'

if not os.path.exists(download_folder):
    os.makedirs(download_folder)

if os.path.exists(library_path):
    asl_library = np.load(library_path, allow_pickle=True).item()
else:
    asl_library = {}

cap = cv2.VideoCapture(0)

print(f"--- ASL DETAILED RECORDER ---")

while True:
    # 1. Ask for the Letter
    letter = input("\nWhich Letter? (or 'q' to quit): ").lower().strip()
    if letter == 'q': break
    
    # 2. Ask for the Hand
    hand_side = input("Hand? (l/r): ").lower().strip()
    hand_label = "left" if hand_side == 'l' else "right"
    
    # 3. Ask for the View
    view = input("View? (front/side/top): ").lower().strip()
    
    # Create the unique base name
    target_name = f"{letter}_{hand_label}_{view}"

    print(f">> Ready to record: {target_name.upper()}")
    print(">> Press 'S' on the camera window to capture.")

    while True:
        success, image = cap.read()
        if not success: break
        
        image = cv2.flip(image, 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Visual Feedback
        display_img = image.copy()
        if results.multi_hand_landmarks:
            mp_draw.draw_landmarks(display_img, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.putText(display_img, f"Target: {target_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Manual Record Mode', display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if results.multi_hand_landmarks:
                # Get landmarks
                pts = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
                normalized = (pts - pts[0]).flatten()
                
                # Overwrite protection with auto-numbering
                final_path = os.path.join(download_folder, f"{target_name}.jpg")
                counter = 1
                while os.path.exists(final_path):
                    final_path = os.path.join(download_folder, f"{target_name}({counter}).jpg")
                    counter += 1

                # Save Image and Data
                cv2.imwrite(final_path, image)
                file_key = os.path.basename(final_path).replace('.jpg', '')
                asl_library[file_key] = normalized
                np.save(library_path, asl_library)
                
                print(f"✅ Saved: {final_path}")
                break # Go back to ask for the next letter/view
            else:
                print("⚠️ Hand not detected! Try adjusting your angle.")
        
        elif key == 27: # ESC to skip
            break

cap.release()
cv2.destroyAllWindows()