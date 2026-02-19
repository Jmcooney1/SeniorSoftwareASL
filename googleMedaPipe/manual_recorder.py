import cv2
import mediapipe as mp
import numpy as np

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.7)
asl_library = {}
alphabet = "abcdefghijklmnopqrstuvwxyz"
index = 0

cap = cv2.VideoCapture(0)

print(f"--- ASL LIBRARY GENERATOR ---")
print(f"Instructions: Make the sign for the letter shown, then press 'S' to save it.")

while index < len(alphabet):
    success, image = cap.read()
    if not success: break
    
    image = cv2.flip(image, 1)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    current_letter = alphabet[index].upper()

    if results.multi_hand_landmarks:
        # Draw for feedback
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        
    # UI Overlay
    cv2.putText(image, f"SIGN THIS: {current_letter}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(image, "'S' to Save | 'Q' to Quit", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow('Manual Library Builder', image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and results.multi_hand_landmarks:
        # Extract and Normalize (Wrist Subtraction)
        pts = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
        normalized = (pts - pts[0]).flatten() # Landmark 0 is the wrist
        
        asl_library[alphabet[index]] = normalized
        print(f"✅ Saved letter: {current_letter}")
        index += 1
    elif key == ord('q'):
        break

# FINAL STEP: Save the dictionary
if asl_library:
    np.save('asl_library.npy', asl_library)
    print("\n🎉 SUCCESS! 'asl_library.npy' has been created.")
else:
    print("\n❌ Nothing was saved.")

cap.release()
cv2.destroyAllWindows()