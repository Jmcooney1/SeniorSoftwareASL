import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- CONFIG ---
MOTION_LIB_PATH = 'asl_motion_library.npy'
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

if not os.path.exists(MOTION_LIB_PATH):
    print(f"❌ Error: {MOTION_LIB_PATH} not found!")
    exit()

# Load the library
motion_lib = np.load(MOTION_LIB_PATH, allow_pickle=True).item()

print("--- MOTION VERIFICATION TOOL ---")
print("Available motions:", list(motion_lib.keys()))

while True:
    choice = input("\nEnter the name of the motion to playback (or 'exit'): ").lower().strip()
    
    if choice == 'exit':
        break
    
    if choice not in motion_lib:
        print(f"⚠️ '{choice}' not found in library.")
        continue

    # Get the sequence of frames
    sequence = motion_lib[choice]
    print(f"Playing back '{choice}' ({len(sequence)} frames)...")

    # Create a blank black canvas for playback
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)

    # Loop through the recorded frames
    for frame_data in sequence:
        canvas.fill(0) # Clear the screen for next frame
        
        # Reshape the flat 63 points back into (21, 3) landmarks
        landmarks_rescaled = frame_data.reshape(21, 3)
        
        # Draw the points on our canvas
        # We multiply by 300 and add 250 to center it on the black window
        for i, pt in enumerate(landmarks_rescaled):
            x = int(pt[0] * 300 + 250)
            y = int(pt[1] * 300 + 250)
            cv2.circle(canvas, (x, y), 3, (0, 255, 0), -1)

        cv2.putText(canvas, f"PLAYBACK: {choice}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.imshow('Motion Verification', canvas)
        
        # Play at a speed that looks natural (roughly 30fps)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
            
    print("Playback finished.")

cv2.destroyAllWindows()