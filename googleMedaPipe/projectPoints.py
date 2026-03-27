# projectPoints.py

def run(root_directory):
    import os
    import cv2
    import csv
    import numpy as np
    from collections import defaultdict
    import mediapipe as mp

    canvas_width = 1280
    canvas_height = 720
    fps = 30

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    # ---------- Loader functions (reuse your existing functions) ----------
    def load_combined_csv(csv_path):
        pose_frames = defaultdict(dict)
        hands_frames = defaultdict(lambda: defaultdict(dict))
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                frame_idx = int(row[0])
                hand_index = row[1]
                hand_lm_idx = row[2]
                hand_x = row[3]
                hand_y = row[4]
                hand_z = row[5]
                pose_lm_idx = row[6]
                pose_x = row[7]
                pose_y = row[8]
                pose_z = row[9]

                if hand_index not in ("", "None"):
                    hand_index = int(hand_index)
                    hand_lm_idx = int(hand_lm_idx)
                    hands_frames[frame_idx][hand_index][hand_lm_idx] = (
                        float(hand_x), float(hand_y), float(hand_z)
                    )
                if pose_lm_idx not in ("", "None"):
                    pose_lm_idx = int(pose_lm_idx)
                    pose_frames[frame_idx][pose_lm_idx] = (
                        float(pose_x), float(pose_y), float(pose_z)
                    )
        return dict(sorted(pose_frames.items())), dict(sorted(hands_frames.items()))

    def draw_pose(canvas, landmarks):
        for s, e in mp_pose.POSE_CONNECTIONS:
            if s in landmarks and e in landmarks:
                x1 = int(landmarks[s][0] * canvas_width)
                y1 = int(landmarks[s][1] * canvas_height)
                x2 = int(landmarks[e][0] * canvas_width)
                y2 = int(landmarks[e][1] * canvas_height)
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for lm in landmarks:
            x = int(landmarks[lm][0] * canvas_width)
            y = int(landmarks[lm][1] * canvas_height)
            cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)

    def draw_hands(canvas, hands_frame_data):
        for hand_index in hands_frame_data:
            landmarks = hands_frame_data[hand_index]
            for s, e in mp_hands.HAND_CONNECTIONS:
                if s in landmarks and e in landmarks:
                    x1 = int(landmarks[s][0] * canvas_width)
                    y1 = int(landmarks[s][1] * canvas_height)
                    x2 = int(landmarks[e][0] * canvas_width)
                    y2 = int(landmarks[e][1] * canvas_height)
                    cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)
            for lm in landmarks:
                x = int(landmarks[lm][0] * canvas_width)
                y = int(landmarks[lm][1] * canvas_height)
                cv2.circle(canvas, (x, y), 4, (255, 255, 0), -1)

    # ---------- Main Loop ----------
    for folder_name in sorted(os.listdir(root_directory)):
        
        folder_path = os.path.join(root_directory, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        
        print(f"\nPlaying set: {folder_name}")
        pose_data, hands_data = {}, {}
        
        
        combined_path = os.path.join(folder_path, "combined_output.csv")
        if os.path.exists(combined_path):
            pose_data, hands_data = load_combined_csv(combined_path)
            
        
        pose_frames = list(pose_data.values())
        hands_frames = list(hands_data.values())
        max_length = max(len(pose_frames), len(hands_frames))
        if max_length == 0:
            print("  No data found.")
            continue

        for i in range(max_length):
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            pose_frame = pose_frames[i] if i < len(pose_frames) else None
            hands_frame = hands_frames[i] if i < len(hands_frames) else None

            if pose_frame:
                draw_pose(canvas, pose_frame)
            if hands_frame:
                draw_hands(canvas, hands_frame)

            cv2.imshow("Skeleton Animation", canvas)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
                cv2.destroyAllWindows()
                return

        cv2.waitKey(500)

    cv2.destroyAllWindows()