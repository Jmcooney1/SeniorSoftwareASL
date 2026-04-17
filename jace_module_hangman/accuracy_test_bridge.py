import cv2
import mediapipe as mp
import numpy as np
import os

asl_library = np.load('asl_library.npy', allow_pickle=True).item()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow('ASL Real-Time Recognition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ASL Real-Time Recognition', 960, 720)

BRIDGE_FILE = '/tmp/asl_detected_letter.txt'  # shared communication file

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    best_match = "None"
    highest_accuracy = 0

    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
        live_geo = (pts - pts[0]).flatten()
        norm_live = np.linalg.norm(live_geo)

        for letter, saved_geo in asl_library.items():
            dot_product = np.dot(live_geo, saved_geo)
            norm_saved = np.linalg.norm(saved_geo)
            if norm_live == 0 or norm_saved == 0:
                continue
            similarity = dot_product / (norm_live * norm_saved)
            accuracy = similarity * 100
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_match = letter

        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
        color = (0, 255, 0) if highest_accuracy > 90 else (0, 255, 255)
        display_name = best_match.split('_')[0].upper()

        # ← Write detected letter to bridge file
        if highest_accuracy > 90:
            with open(BRIDGE_FILE, 'w') as f:
                f.write(display_name)

        cv2.putText(frame, f"PREDICTION: {display_name}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Match: {highest_accuracy:.1f}%", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('ASL Real-Time Recognition', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or cv2.getWindowProperty('ASL Real-Time Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()