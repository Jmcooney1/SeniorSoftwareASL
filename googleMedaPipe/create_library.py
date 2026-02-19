import cv2
import mediapipe as mp
import numpy as np
import os

# --- SETUP ---
mp_hands = mp.solutions.hands
# Keep confidence low (0.3) to help with the grainy Lifeprint photos
hands = mp_hands.Hands(static_image_mode=True, model_complexity=1, min_detection_confidence=0.3)

# 1. Load your existing library so we don't overwrite the good data
library_path = 'asl_library.npy'
if os.path.exists(library_path):
    asl_library = np.load(library_path, allow_pickle=True).item()
    print(f"Loaded existing library with {len(asl_library)} letters.")
else:
    asl_library = {}
    print("No existing library found. Starting fresh.")

# 2. DEFINE YOUR TARGETS HERE
# Put the exact filenames you want to process in this list
target_files = ['q.jpg'] 
manual_folder = r"C:\Users\izzyd\OneDrive\Documents\ASL senior software\SeniorSoftwareASL\maual_downloads"

print(f"--- Processing Targeted Files in: {manual_folder} ---")

for filename in target_files:
    img_path = os.path.join(manual_folder, filename)
    
    if not os.path.exists(img_path):
        print(f"❌ File not found: {filename}")
        continue

    img = cv2.imread(img_path)
    # MediaPipe works best when the hand is clear and centered
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        # Extract the 21 landmarks
        pts = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
        # Normalize by subtracting the wrist (landmark 0)
        letter = filename.split('.')[0].lower()
        asl_library[letter] = (pts - pts[0]).flatten()
        print(f"✅ Successfully added/updated letter: {letter.upper()}")
    else:
        print(f"⚠️ MediaPipe still failed on {filename}. Check contrast or hand visibility.")

# 3. SAVE THE UPDATED LIBRARY
np.save(library_path, asl_library)
print(f"\nLibrary saved! Total letters now: {len(asl_library)}")