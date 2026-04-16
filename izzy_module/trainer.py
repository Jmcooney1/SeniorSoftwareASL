import os
import time
import cv2
import numpy as np
import mediapipe as mp
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFrame, QMessageBox
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

# --- DYNAMIC PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Styles ────────────────────────────────────────────────────────────────────
_STYLE_MAIN     = "background: #1e293b; color: white; font-weight: bold; border-radius: 8px; padding: 12px;"
_STYLE_SAVE     = "background: #16a34a; color: white; font-weight: bold; border-radius: 8px; padding: 15px; font-size: 16px;"
_STYLE_DISCARD  = "background: #ef4444; color: white; font-weight: bold; border-radius: 8px; padding: 15px; font-size: 16px;"
_STYLE_INACTIVE = "background: #475569; color: #cbd5e1; border-radius: 8px; padding: 12px;"

class TrainerWidget(QWidget):
    def __init__(self):
        super().__init__()

        # --- PATH SETUP ---
        self.lib_path = os.path.join(SCRIPT_DIR, "googleMedaPipe", "asl_motion_library.npy")
        os.makedirs(os.path.dirname(self.lib_path), exist_ok=True)

        # --- CONFIG ---
        self.RECORD_DURATION = 2.5
        self.ALPHA           = 0.3 # Smoothing factor for the 69-pt EMA

        # --- STATE ---
        self.history        = {"left": None, "right": None}
        self.recording      = False
        self.sequence       = []
        self.cap            = None
        self.last_saved_key = None
        self.start_time     = None

        # --- MEDIAPIPE ---
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            model_complexity=1, max_num_hands=2,
            min_detection_confidence=0.8, min_tracking_confidence=0.8
        )
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self._init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

    def _init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(12)
        self.main_layout.setContentsMargins(15, 15, 15, 15)

        # --- HEADER / CONFIG ---
        config_panel = QFrame()
        config_panel.setStyleSheet("background: #f1f5f9; border-radius: 12px; border: 1px solid #cbd5e1;")
        config_layout = QVBoxLayout(config_panel)

        mode_row = QHBoxLayout()
        self.mode_btn = QPushButton("MODE: DUAL HAND 👐")
        self.mode_btn.setCheckable(True)
        self.mode_btn.setStyleSheet(_STYLE_MAIN)
        self.mode_btn.clicked.connect(self._toggle_mode)

        self.side_btn = QPushButton("SIDE: RIGHT ➡️")
        self.side_btn.setStyleSheet("background: #3b82f6; color: white; border-radius: 8px; padding: 12px;")
        self.side_btn.setVisible(False)
        self.side_btn.clicked.connect(self._toggle_side)

        mode_row.addWidget(QLabel("<b>Input:</b>"))
        mode_row.addWidget(self.mode_btn)
        mode_row.addWidget(self.side_btn)
        mode_row.addStretch()
        config_layout.addLayout(mode_row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("<b style='color: #1e293b;'>Gesture:</b>"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. apple, hello, thank_you")
        self.name_input.setStyleSheet("padding: 10px; background: white; color: black; border-radius: 6px; border: 2px solid #94a3b8;")
        name_row.addWidget(self.name_input)
        config_layout.addLayout(name_row)
        self.main_layout.addWidget(config_panel)

        # --- CAMERA FEED ---
        self.feed = QLabel("Camera Offline")
        self.feed.setMinimumSize(640, 440)
        self.feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed.setStyleSheet("background: black; border-radius: 15px; border: 4px solid #1e293b;")
        self.main_layout.addWidget(self.feed, stretch=10)

        # --- REVIEW CONTROLS (Green/Red) ---
        self.decision_widget = QWidget()
        d_layout = QHBoxLayout(self.decision_widget)
        self.save_btn = QPushButton("✔   SAVE GESTURE")
        self.save_btn.setStyleSheet(_STYLE_SAVE)
        self.save_btn.clicked.connect(self._save_data)
        self.discard_btn = QPushButton("✖   DISCARD")
        self.discard_btn.setStyleSheet(_STYLE_DISCARD)
        self.discard_btn.clicked.connect(self._discard_data)
        d_layout.addWidget(self.save_btn)
        d_layout.addWidget(self.discard_btn)
        self.decision_widget.hide()
        self.main_layout.addWidget(self.decision_widget)

        # --- MAIN CONTROLS (Blue/Red) ---
        self.control_widget = QWidget()
        c_layout = QHBoxLayout(self.control_widget)
        self.cam_btn = QPushButton("START CAMERA")
        self.cam_btn.setStyleSheet(_STYLE_MAIN)
        self.cam_btn.clicked.connect(self._toggle_camera)
        self.rec_btn = QPushButton("🔴   RECORD 2.5s")
        self.rec_btn.setEnabled(False)
        self.rec_btn.setStyleSheet(_STYLE_INACTIVE)
        self.rec_btn.clicked.connect(self._start_recording)
        c_layout.addWidget(self.cam_btn)
        c_layout.addWidget(self.rec_btn)
        self.main_layout.addWidget(self.control_widget)

        # --- FOOTER ---
        self.footer = QFrame()
        self.footer.setStyleSheet("background: #0f172a; border-radius: 8px; padding: 5px;")
        f_layout = QHBoxLayout(self.footer)
        self.history_label = QLabel("Ready for capture...")
        self.history_label.setStyleSheet("color: #94a3b8; font-size: 13px; font-family: monospace;")
        self.remove_btn = QPushButton("DELETE LAST TAKE")
        self.remove_btn.setEnabled(False)
        self.remove_btn.setStyleSheet("background: #7f1d1d; color: #fecaca; border-radius: 4px; padding: 4px 10px;")
        self.remove_btn.clicked.connect(self._remove_last)
        f_layout.addWidget(self.history_label)
        f_layout.addStretch()
        f_layout.addWidget(self.remove_btn)
        self.main_layout.addWidget(self.footer)

    def _toggle_mode(self):
        is_single = self.mode_btn.isChecked()
        self.mode_btn.setText("MODE: SINGLE HAND ☝" if is_single else "MODE: DUAL HAND 👐")
        self.side_btn.setVisible(is_single)

    def _toggle_side(self):
        is_left = "RIGHT" in self.side_btn.text()
        self.side_btn.setText("SIDE: LEFT ⬅️" if is_left else "SIDE: RIGHT ➡️")
        self.side_btn.setStyleSheet(f"background: {'#8b5cf6' if is_left else '#3b82f6'}; color: white; border-radius: 8px; padding: 12px;")

    def _toggle_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.timer.start(30)
                self.cam_btn.setText("STOP CAMERA")
                self.rec_btn.setEnabled(True)
                self.rec_btn.setStyleSheet("background: #dc2626; color: white; font-weight: bold; border-radius: 8px; padding: 12px;")
        else:
            self._stop_camera()

    def _stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cam_btn.setText("START CAMERA")
        self.rec_btn.setEnabled(False)
        self.rec_btn.setStyleSheet(_STYLE_INACTIVE)
        self.feed.setText("Camera Offline")

    def _start_recording(self):
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Missing Name", "Please enter a gesture name first.")
            return
        
        # --- RESET STATE FOR NEW TAKE ---
        self.history   = {"left": None, "right": None}
        self.sequence  = []
        self.recording = True
        self.start_time = time.time()
        self.control_widget.hide()
        self.name_input.setEnabled(False)

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Face Anchor (Nose Tip)
        face_res = self.face_detector.process(rgb)
        nose_tip = None
        if face_res.detections:
            nose_tip = mp.solutions.face_detection.get_key_point(
                face_res.detections[0], mp.solutions.face_detection.FaceKeyPoint.NOSE_TIP
            )

        # 2. Hand Extraction
        hand_res = self.hands.process(rgb)
        curr_frame_features = {"left": None, "right": None}

        if hand_res.multi_hand_landmarks:
            for i, lms in enumerate(hand_res.multi_hand_landmarks):
                side = hand_res.multi_handedness[i].classification[0].label.lower()
                curr_frame_features[side] = self._get_69_pt_vector(lms, nose_tip, side)
                self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)

        # 3. Recording Logic
        if self.recording:
            elapsed = time.time() - self.start_time
            # Draw blue progress bar
            bar_w = int(frame.shape[1] * (elapsed / self.RECORD_DURATION))
            cv2.rectangle(frame, (0, 0), (bar_w, 15), (59, 130, 246), -1)

            # Package 69-pt features
            if not self.mode_btn.isChecked():
                # DUAL MODE
                self.sequence.append({
                    "left": curr_frame_features["left"] if curr_frame_features["left"] is not None else np.zeros(69),
                    "right": curr_frame_features["right"] if curr_frame_features["right"] is not None else np.zeros(69)
                })
            else:
                # SINGLE MODE
                target = "left" if "LEFT" in self.side_btn.text() else "right"
                val = curr_frame_features[target] if curr_frame_features[target] is not None else np.zeros(69)
                self.sequence.append(val)

            if elapsed >= self.RECORD_DURATION:
                self.recording = False
                self._pause_for_review()

        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self.feed.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.feed.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def _get_69_pt_vector(self, lms, nose, side):
        """Constructs the Shape(63) + Spatial(3) + Velocity(3) signature."""
        pts = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark])
        wrist = pts[0]
        
        # 1. Shape (Local joints)
        hand_shape = ((pts - wrist) * 10).flatten() 
        # 2. Spatial (Relative to face)
        spatial = np.array([wrist[0]-nose.x, wrist[1]-nose.y, 0]) * 10 if nose else np.zeros(3)
        # 3. Velocity (Movement delta)
        if self.history[side] is not None:
            prev_spatial = self.history[side][63:66]
            velocity = (spatial - prev_spatial) * 5
        else:
            velocity = np.zeros(3)

        combined = np.concatenate([hand_shape, spatial, velocity])

        # EMA Smoothing to match Predictor math
        if self.history[side] is None:
            self.history[side] = combined
        else:
            self.history[side] = (self.ALPHA * combined) + ((1 - self.ALPHA) * self.history[side])
        return self.history[side]

    def _pause_for_review(self):
        self.timer.stop()
        self.control_widget.hide()
        self.decision_widget.show()

    def _save_data(self):
        name_token = self.name_input.text().lower().strip().replace(" ", "_")
        lib = np.load(self.lib_path, allow_pickle=True).item() if os.path.exists(self.lib_path) else {}

        # Auto-indexing
        mode_tag = "dual" if not self.mode_btn.isChecked() else ("left" if "LEFT" in self.side_btn.text() else "right")
        prefix = f"{name_token}_{mode_tag}_"
        existing = [int(k.split("_")[-1]) for k in lib if k.startswith(prefix)]
        idx = max(existing) + 1 if existing else 1
        label = f"{prefix}{idx}"

        lib[label] = np.array(self.sequence, dtype=object)
        np.save(self.lib_path, lib)

        self.last_saved_key = label
        self.history_label.setText(f"SAVED: {label}")
        self.remove_btn.setEnabled(True)
        self._reset_ui()

    def _remove_last(self):
        if not self.last_saved_key: return
        ans = QMessageBox.question(self, "Confirm", f"Delete '{self.last_saved_key}'?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ans == QMessageBox.StandardButton.Yes:
            lib = np.load(self.lib_path, allow_pickle=True).item()
            if self.last_saved_key in lib:
                del lib[self.last_saved_key]
                np.save(self.lib_path, lib)
                self.history_label.setText(f"REMOVED: {self.last_saved_key}")
                self.last_saved_key = None
                self.remove_btn.setEnabled(False)

    def _discard_data(self):
        self._reset_ui()

    def _reset_ui(self):
        self.decision_widget.hide()
        self.control_widget.show()
        self.name_input.setEnabled(True)
        self.history = {"left": None, "right": None}
        self.sequence = []
        if self.cap: self.timer.start(30)

    def hideEvent(self, event):
        self._stop_camera()
        super().hideEvent(event)

def get_tab():
    return TrainerWidget()