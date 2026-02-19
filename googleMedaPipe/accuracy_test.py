import cv2
import mediapipe as mp
import numpy as np

# --- UTILITY: Calculate angle between three points ---
def calculate_angle(a, b, c):
    a = np.array(a) # First point (e.g., Tip)
    b = np.array(b) # Mid point (e.g., Knuckle)
    c = np.array(c) # End point (e.g., Wrist)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

# --- SETUP ---
asl_library = np.load('asl_library.npy', allow_pickle=True).item()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
target_letter = 'a' 

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        
        # 1. EXTRACT JOINT POSITIONS
        # We focus on the "bend" of each finger (Landmarks 5-8, 9-12, etc.)
        points = [[lm.x, lm.y] for lm in hand_lms.landmark]
        
        # 2. CALCULATE LIVE FINGER BENDS
        # Example: Index finger bend at the middle knuckle
        index_bend = calculate_angle(points[8], points[6], points[5])
        thumb_bend = calculate_angle(points[4], points[2], points[1])
        middle_bend = calculate_angle(points[12], points[10], points[9])
        
        # 3. COMPARE TO LIBRARY GEOMETRY
        # (This assumes your library has the same landmark structure)
        # For simplicity, we compare the flattened normalized vector 
        # which already ignores screen position (since we subtracted the wrist)
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
        live_geo = (pts - pts[0]).flatten() 
        
        if target_letter in asl_library:
            saved_geo = asl_library[target_letter]
            
            # Use Cosine Similarity: This measures the ANGLE between vectors
            # It is famous for ignoring "size" and "position" and only looking at "shape"
            dot_product = np.dot(live_geo, saved_geo)
            norm_live = np.linalg.norm(live_geo)
            norm_saved = np.linalg.norm(saved_geo)
            similarity = dot_product / (norm_live * norm_saved)
            
            accuracy = max(0, similarity * 100)
            
            # --- DISPLAY FINGER DATA ---
            cv2.putText(frame, f"Finger Geometry Match: {accuracy:.1f}%", (10, 50), 1, 2, (0, 255, 0), 2)
            cv2.putText(frame, f"Index Bend: {int(index_bend)}deg", (10, 100), 1, 1, (255, 255, 255), 1)
            cv2.putText(frame, f"Target: {target_letter.upper()}", (10, 130), 1, 1, (255, 255, 0), 1)

    cv2.imshow('Anatomical Accuracy', frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        target_letter = input("New Target: ").lower()
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()