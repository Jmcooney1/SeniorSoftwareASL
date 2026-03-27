import cv2
import mediapipe as mp
import numpy as np
import os

# --- SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, model_complexity=1, min_detection_confidence=0.3)

# 1. Path Setup
library_path = 'asl_library.npy'
# Using your specific absolute path
manual_folder = r"C:\Users\izzyd\OneDrive\Documents\ASL senior software\SeniorSoftwareASL\download_imges"

if not os.path.exists(manual_folder):
    print(f"❌ Error: Path not found! Check the folder name: {manual_folder}")
    # Optional: fallback to local folder if absolute path fails
    manual_folder = 'download_imges'

if os.path.exists(library_path):
    asl_library = np.load(library_path, allow_pickle=True).item()
    print(f"Loaded existing library with {len(asl_library)} letters.")
else:
    asl_library = {}
    print("No existing library found. Starting fresh.")

print(f"--- Processing images in: {manual_folder} ---")

# 2. AUTOMATICALLY LOOP THROUGH THE FOLDER
if not os.path.exists(manual_folder):
    print(f"❌ Error: The folder '{manual_folder}' does not exist!")
else:
    for filename in os.listdir(manual_folder):
        # Only process .jpg or .png files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(manual_folder, filename)
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Could not read {filename}")
                continue

            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                # Extract the 21 landmarks
                pts = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
                
                # Normalize by subtracting the wrist (landmark 0)
                # This assumes your filename is 'a.jpg', 'b.jpg', etc.
                letter = os.path.splitext(filename)[0].lower()
                asl_library[letter] = (pts - pts[0]).flatten()
                print(f"✅ Successfully added/updated letter: {letter.upper()}")
            else:
                print(f"⚠️ MediaPipe failed to find a hand in {filename}.")

# 3. SAVE THE UPDATED LIBRARY
np.save(library_path, asl_library)
print(f"\nLibrary saved! Total letters now: {len(asl_library)}")