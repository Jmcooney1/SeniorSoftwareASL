import os
import time
import cv2
import json
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFrame, QMessageBox
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

class TrainerWidget(QWidget):
    def __init__(self):
        super().__init__()

        # --- DYNAMIC PATH SETUP ---
        self.lib_path = self._get_library_path_from_config()
        if self.lib_path:
            os.makedirs(os.path.dirname(self.lib_path), exist_ok=True)

        # --- CONFIG ---
        self.RECORD_DURATION = 2.5
        self.ALPHA = 0.3 

        # --- STATE ---
        self.history = {"left": None, "right": None}
        self.recording = False
        self.sequence = [] 
        self.cap = None
        self.start_time = None

        # --- MEDIAPIPE ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
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

    def _get_library_path_from_config(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            config_file = os.path.join(current_dir, "config.json")
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    return os.path.abspath(cfg.get("library_path", ""))
            current_dir = os.path.dirname(current_dir)
        return ""

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # --- USER IDENTIFICATION (Normalized Input) ---
        user_panel = QFrame()
        user_panel.setStyleSheet("background: #1e293b; border-radius: 10px; border: 1px solid #334155;")
        user_layout = QHBoxLayout(user_panel)
        
        user_label = QLabel("👤 CURRENT USER:")
        user_label.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter Name (e.g., Izzy or Jace)")
        self.user_input.setText("Default")
        self.user_input.setStyleSheet("padding: 8px; background: #0f172a; color: white; border: 1px solid #38bdf8; border-radius: 4px;")
        
        user_layout.addWidget(user_label)
        user_layout.addWidget(self.user_input)
        layout.addWidget(user_panel)

        # --- GESTURE CONFIG ---
        config_panel = QFrame()
        config_panel.setStyleSheet("background: #f1f5f9; border-radius: 12px; border: 1px solid #cbd5e1;")
        config_layout = QVBoxLayout(config_panel)

        mode_row = QHBoxLayout()
        self.mode_btn = QPushButton("MODE: DUAL HAND 👐")
        self.mode_btn.setCheckable(True)
        self.mode_btn.setStyleSheet("background: #1e293b; color: white; border-radius: 8px; padding: 12px; font-weight: bold;")
        self.mode_btn.clicked.connect(self._toggle_mode)

        self.side_btn = QPushButton("SIDE: RIGHT ➡️")
        self.side_btn.setStyleSheet("background: #3b82f6; color: white; border-radius: 8px; padding: 12px; font-weight: bold;")
        self.side_btn.setVisible(False)
        self.side_btn.clicked.connect(self._toggle_side)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Gesture Name (e.g. apple)")
        self.name_input.setStyleSheet("padding: 10px; background: white; color: black; border-radius: 6px; border: 2px solid #94a3b8;")

        mode_row.addWidget(self.mode_btn)
        mode_row.addWidget(self.side_btn)
        mode_row.addWidget(self.name_input)
        config_layout.addLayout(mode_row)
        layout.addWidget(config_panel)

        self.feed = QLabel("Camera Offline")
        self.feed.setMinimumSize(640, 480)
        self.feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed.setStyleSheet("background: #0f172a; border-radius: 15px; border: 5px solid #334155;")
        layout.addWidget(self.feed, stretch=5)

        self.decision_widget = QWidget()
        d_layout = QHBoxLayout(self.decision_widget)
        self.save_btn = QPushButton("✔ SAVE GESTURE")
        self.save_btn.setStyleSheet("background: #16a34a; color: white; font-weight: bold; padding: 15px; border-radius: 8px;")
        self.save_btn.clicked.connect(self._save_data)
        self.discard_btn = QPushButton("✖ DISCARD")
        self.discard_btn.setStyleSheet("background: #ef4444; color: white; font-weight: bold; padding: 15px; border-radius: 8px;")
        self.discard_btn.clicked.connect(self._discard_data)
        d_layout.addWidget(self.save_btn)
        d_layout.addWidget(self.discard_btn)
        self.decision_widget.hide()
        layout.addWidget(self.decision_widget)

        self.control_widget = QWidget()
        c_layout = QHBoxLayout(self.control_widget)
        self.cam_btn = QPushButton("START CAMERA")
        self.cam_btn.clicked.connect(self._toggle_camera)
        self.cam_btn.setStyleSheet("background: #1e293b; color: white; padding: 12px; border-radius: 8px;")
        self.rec_btn = QPushButton("🔴 RECORD")
        self.rec_btn.setEnabled(False)
        self.rec_btn.clicked.connect(self._start_recording)
        self.rec_btn.setStyleSheet("background: #475569; color: #cbd5e1; padding: 12px; border-radius: 8px;")
        c_layout.addWidget(self.cam_btn)
        c_layout.addWidget(self.rec_btn)
        layout.addWidget(self.control_widget)

        self.history_label = QLabel("Ready...")
        layout.addWidget(self.history_label)

    def _toggle_mode(self):
        is_single = self.mode_btn.isChecked()
        self.mode_btn.setText("MODE: SINGLE ☝" if is_single else "MODE: DUAL 👐")
        self.side_btn.setVisible(is_single)

    def _toggle_side(self):
        is_left = "RIGHT" in self.side_btn.text()
        self.side_btn.setText("SIDE: LEFT ⬅️" if is_left else "SIDE: RIGHT ➡️")

    def _toggle_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.cam_btn.setText("STOP CAMERA")
            self.rec_btn.setEnabled(True)
            self.rec_btn.setStyleSheet("background: #dc2626; color: white; padding: 12px; border-radius: 8px; font-weight: bold;")
        else:
            self._stop_camera()

    def _stop_camera(self):
        self.timer.stop()
        if self.cap: self.cap.release()
        self.cap = None
        self.cam_btn.setText("START CAMERA")
        self.rec_btn.setEnabled(False)
        self.rec_btn.setStyleSheet("background: #475569; color: #cbd5e1; padding: 12px; border-radius: 8px;")
        self.feed.setText("Camera Offline")

    def _start_recording(self):
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Error", "Please name the gesture.")
            return
        if not self.user_input.text().strip():
            QMessageBox.warning(self, "Error", "Please identify the user.")
            return
        self.sequence = []
        self.recording = True
        self.start_time = time.time()
        self.control_widget.hide()
        self.name_input.setEnabled(False)
        self.user_input.setEnabled(False)

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_res = self.face_detector.process(rgb)
        nose_lm = None
        if face_res.detections:
            nose_lm = mp.solutions.face_detection.get_key_point(
                face_res.detections[0], mp.solutions.face_detection.FaceKeyPoint.NOSE_TIP
            )

        hand_res = self.hands.process(rgb)
        curr_hands = {"left": None, "right": None}

        if hand_res.multi_hand_landmarks:
            for i, lms in enumerate(hand_res.multi_hand_landmarks):
                side = hand_res.multi_handedness[i].classification[0].label.lower()
                curr_hands[side] = self._get_complex_norm(lms, nose_lm, side)
                self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)

        if self.recording:
            elapsed = time.time() - self.start_time
            bar_w = int(frame.shape[1] * (elapsed / self.RECORD_DURATION))
            cv2.rectangle(frame, (0,0), (bar_w, 20), (59, 130, 246), -1)

            if not self.mode_btn.isChecked():
                self.sequence.append({
                    "left": curr_hands["left"] if curr_hands["left"] is not None else np.zeros(66),
                    "right": curr_hands["right"] if curr_hands["right"] is not None else np.zeros(66)
                })
            else:
                target = "left" if "LEFT" in self.side_btn.text() else "right"
                self.sequence.append(curr_hands[target] if curr_hands[target] is not None else np.zeros(66))

            if elapsed >= self.RECORD_DURATION:
                self.recording = False
                self._stop_camera()
                self.decision_widget.show()

        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self.feed.setPixmap(QPixmap.fromImage(qimg).scaled(self.feed.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def _get_complex_norm(self, hand_lms, nose_lm, side):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
        wrist = pts[0]
        hand_shape = ((pts - wrist) * 10).flatten() 
        spatial_pos = np.array([wrist[0] - nose_lm.x, wrist[1] - nose_lm.y, 0]) * 10 if nose_lm else np.zeros(3)
        combined = np.concatenate([hand_shape, spatial_pos])

        if self.history[side] is None:
            self.history[side] = combined
        else:
            self.history[side] = (self.ALPHA * combined) + ((1 - self.ALPHA) * self.history[side])
        return self.history[side]

    def _save_data(self):
        # --- NORMALIZATION ---
        user = self.user_input.text().strip().lower().replace(" ", "_")
        name = self.name_input.text().strip().lower().replace(" ", "_")
        
        lib = np.load(self.lib_path, allow_pickle=True).item() if os.path.exists(self.lib_path) else {}
        
        side_tag = "dual" if not self.mode_btn.isChecked() else ("left" if "LEFT" in self.side_btn.text() else "right")
        prefix = f"{user}_{name}_{side_tag}_"
        
        existing = [int(k.split("_")[-1]) for k in lib if k.startswith(prefix)]
        idx = max(existing) + 1 if existing else 1
        label = f"{prefix}{idx}"

        lib[label] = np.array(self.sequence, dtype=object)
        np.save(self.lib_path, lib)
        
        self.history_label.setText(f"✅ Saved for '{user}': {label}")
        self._reset_ui()

    def _discard_data(self):
        self._reset_ui()

    def _reset_ui(self):
        self.decision_widget.hide()
        self.control_widget.show()
        self.name_input.setEnabled(True)
        self.user_input.setEnabled(True)
        self._toggle_camera()

    def hideEvent(self, event):
        self._stop_camera()
        super().hideEvent(event)

def get_tab():
    return TrainerWidget()