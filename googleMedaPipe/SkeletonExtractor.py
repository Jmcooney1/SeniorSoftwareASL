import os
import cv2
import mediapipe as mp
import csv
import numpy as np

class SkeletonExtractor:
    def __init__(self, video_source=0, run_face=True, run_hands=True, run_pose=False,
                 save_hand_coords=True, save_pose_coords=True,
                 draw_skeleton=False, show_video=True, no_play=False, skeleton_only=False,
                 base_dir="./savedVideoPoints"):

        self.video_source = video_source
        self.run_face = run_face
        self.run_hands = run_hands
        self.run_pose = run_pose
        self.save_hand_coords = save_hand_coords
        self.save_pose_coords = save_pose_coords
        self.draw_skeleton = draw_skeleton
        self.show_video = show_video
        self.no_play = no_play
        self.skeleton_only = skeleton_only

        # Directories
        self.base_dir = base_dir
        self.hands_dir = os.path.join(base_dir, "hands")
        self.pose_dir = os.path.join(base_dir, "pose")
        self.combined_dir = os.path.join(base_dir, "combined")
        os.makedirs(self.hands_dir, exist_ok=True)
        os.makedirs(self.pose_dir, exist_ok=True)
        os.makedirs(self.combined_dir, exist_ok=True)

        self.hand_file_path = os.path.join(self.hands_dir, "hands_output.csv")
        self.pose_file_path = os.path.join(self.pose_dir, "pose_output.csv")
        self.combined_file_path = os.path.join(self.combined_dir, "combined_output.csv")

        # MediaPipe Setup
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

        self.POSE_LANDMARKS_TO_USE = list(range(0, 17)) + [23, 24]

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source '{self.video_source}'")

        # ==============================
        # Open CSV files
        # ==============================
        if self.save_hand_coords:
            self.hand_csv = open(self.hand_file_path, "w", newline="")
            self.hand_writer = csv.writer(self.hand_csv)
            self.hand_writer.writerow(
                ["frame", "hand_index", "landmark_index", "x", "y", "z"]
            )

        if self.save_pose_coords:
            self.pose_csv = open(self.pose_file_path, "w", newline="")
            self.pose_writer = csv.writer(self.pose_csv)
            self.pose_writer.writerow(
                ["frame", "landmark_index", "x", "y", "z"]
            )

        if self.save_hand_coords and self.save_pose_coords:
            self.combined_csv = open(self.combined_file_path, "w", newline="")
            self.combined_writer = csv.writer(self.combined_csv)
            self.combined_writer.writerow([
                "frame",
                "hand_index", "hand_landmark_index", "hand_x", "hand_y", "hand_z",
                "pose_landmark_index", "pose_x", "pose_y", "pose_z"
            ])

        # Initialize MediaPipe models
        face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) if self.run_face else None

        hands = self.mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) if self.run_hands else None

        pose = self.mp_pose.Pose(
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) if self.run_pose else None

        frame_idx = 0

        # ==============================
        # Main Loop
        # ==============================
        while cap.isOpened():
            

            success, image = cap.read()
            if not success:
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_results = face_mesh.process(image_rgb) if face_mesh else None
            hand_results = hands.process(image_rgb) if hands else None
            pose_results = pose.process(image_rgb) if pose else None

            display_image = np.zeros_like(image) if self.skeleton_only else image.copy()

            # Draw skeleton
            if self.draw_skeleton or self.skeleton_only:

                if pose_results and pose_results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        display_image,
                        pose_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                if hand_results and hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            display_image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
                        )

            # ==============================
            # Save Hand Data
            # ==============================
            if self.save_hand_coords and hand_results and hand_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    for lm_idx, lm in enumerate(hand_landmarks.landmark):

                        if self.save_hand_coords:
                            self.hand_writer.writerow(
                                [frame_idx, hand_idx, lm_idx, lm.x, lm.y, lm.z]
                            )

            # ==============================
            # Save Pose Data
            # ==============================
            if self.save_pose_coords and pose_results and pose_results.pose_landmarks:
                for lm_idx in self.POSE_LANDMARKS_TO_USE:
                    lm = pose_results.pose_landmarks.landmark[lm_idx]

                    if self.save_pose_coords:
                        self.pose_writer.writerow(
                            [frame_idx, lm_idx, lm.x, lm.y, lm.z]
                        )

            # ==============================
            # Save Combined Data
            # ==============================
            if (self.save_hand_coords and self.save_pose_coords and
                hand_results and hand_results.multi_hand_landmarks and
                pose_results and pose_results.pose_landmarks):

                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    for lm_idx, lm in enumerate(hand_landmarks.landmark):
                        self.combined_writer.writerow([
                            frame_idx,
                            hand_idx, lm_idx, lm.x, lm.y, lm.z,
                            None, None, None, None
                        ])

                for lm_idx in self.POSE_LANDMARKS_TO_USE:
                    lm = pose_results.pose_landmarks.landmark[lm_idx]
                    self.combined_writer.writerow([
                        frame_idx,
                        None, None, None, None, None,
                        lm_idx, lm.x, lm.y, lm.z
                    ])

            # Show video
            if self.show_video and not self.no_play:
                cv2.imshow("Skeleton", cv2.flip(display_image, 1))
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1

        # Cleanup
        cap.release()
        if self.show_video and not self.no_play:
            cv2.destroyAllWindows()

        if self.save_hand_coords:
            self.hand_csv.close()

        if self.save_pose_coords:
            self.pose_csv.close()

        if self.save_hand_coords and self.save_pose_coords:
            self.combined_csv.close()