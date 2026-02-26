import cv2
import csv
import numpy as np
from collections import defaultdict
import mediapipe as mp

# ========== SETTINGS ==========
pose_csv_path = "./savedVideoPoints/pose/pose_output.csv"
canvas_width = 1280
canvas_height = 720
fps = 30
# ==============================

mp_pose = mp.solutions.pose

# Group landmarks by frame
frames = defaultdict(dict)

with open(pose_csv_path, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        frame_idx = int(row[0])
        landmark_idx = int(row[1])
        x = float(row[2])
        y = float(row[3])
        z = float(row[4])
        frames[frame_idx][landmark_idx] = (x, y, z)

# Animate
sorted_frames = sorted(frames.keys())

for frame_idx in sorted_frames:

    # Create black canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    landmarks = frames[frame_idx]

    # Draw connections
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx in landmarks and end_idx in landmarks:

            x1 = int(landmarks[start_idx][0] * canvas_width)
            y1 = int(landmarks[start_idx][1] * canvas_height)

            x2 = int(landmarks[end_idx][0] * canvas_width)
            y2 = int(landmarks[end_idx][1] * canvas_height)

            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw points
    for lm_idx in landmarks:
        x = int(landmarks[lm_idx][0] * canvas_width)
        y = int(landmarks[lm_idx][1] * canvas_height)
        cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Projected Skeleton", canvas)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
        break

cv2.destroyAllWindows()