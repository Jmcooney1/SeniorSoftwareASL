import cv2
import mediapipe as mp
import argparse
import csv
import numpy as np

# Drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Models
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Pose landmarks to track
POSE_LANDMARKS_TO_USE = list(range(0, 17)) + [23, 24]


def run(video_source=0, run_face=True, run_hands=True, run_pose=False,
        save_hand_coords=False, save_pose_coords=False):
    """
    video_source: 0 for webcam, or video file path
    run_face: bool, run face mesh
    run_hands: bool, run hand tracking
    run_pose: bool, run pose tracking
    save_hand_coords: bool, save hand landmarks to CSV
    save_pose_coords: bool, save pose landmarks to CSV
    """

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Prepare CSV files
    if save_hand_coords:
        hand_csv_file = "hand_landmarks.csv"
        hand_csv = open(hand_csv_file, mode="w", newline="")
        hand_writer = csv.writer(hand_csv)
        hand_writer.writerow(["frame", "hand_index", "landmark_index", "x", "y", "z"])
        print(f"Saving hand coordinates to {hand_csv_file}")

    if save_pose_coords:
        pose_csv_file = "pose_landmarks.csv"
        pose_csv = open(pose_csv_file, mode="w", newline="")
        pose_writer = csv.writer(pose_csv)
        pose_writer.writerow(["frame", "landmark_index", "x", "y", "z"])
        print(f"Saving pose coordinates to {pose_csv_file}")

    # Initialize models
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) if run_face else None

    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) if run_hands else None

    pose = mp_pose.Pose(
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) if run_pose else None

    frame_idx = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process models
        face_results = face_mesh.process(image_rgb) if face_mesh else None
        hand_results = hands.process(image_rgb) if hands else None
        pose_results = pose.process(image_rgb) if pose else None

        # Back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw Face
        if face_results and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )

        # Draw Hands
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )
                # Save hand coordinates
                if save_hand_coords:
                    for lm_idx, lm in enumerate(hand_landmarks.landmark):
                        hand_writer.writerow([frame_idx, hand_idx, lm_idx, lm.x, lm.y, lm.z])

        # Draw Pose (subset landmarks)
        if pose_results and pose_results.pose_landmarks:
            h, w, _ = image.shape
            for lm_idx in POSE_LANDMARKS_TO_USE:
                lm = pose_results.pose_landmarks.landmark[lm_idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
                if save_pose_coords:
                    pose_writer.writerow([frame_idx, lm_idx, lm.x, lm.y, lm.z])

        # Show
        cv2.imshow("MediaPipe: Face + Hands + Pose", cv2.flip(image, 1))
        frame_idx += 1

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    if save_hand_coords:
        hand_csv.close()
    if save_pose_coords:
        pose_csv.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe Face, Hands, Pose tracking")
    parser.add_argument("--video", type=str, default=None, help="Video file path. Leave empty for webcam.")
    parser.add_argument("--face", action="store_true", help="Run face mesh")
    parser.add_argument("--hands", action="store_true", help="Run hand tracking")
    parser.add_argument("--pose", action="store_true", help="Run pose tracking")
    parser.add_argument("--save_hands", action="store_true", help="Save hand coordinates")
    parser.add_argument("--save_pose", action="store_true", help="Save pose coordinates")
    args = parser.parse_args()

    # Default to running all if nothing specified
    run_face = args.face or (not args.face and not args.hands and not args.pose)
    run_hands = args.hands or (not args.face and not args.hands and not args.pose)
    run_pose = args.pose or (not args.face and not args.hands and not args.pose)

    video_source = args.video if args.video else 0
    run(video_source=video_source, run_face=run_face, run_hands=run_hands,
        run_pose=run_pose, save_hand_coords=args.save_hands, save_pose_coords=args.save_pose)
