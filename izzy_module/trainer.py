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
        self.ALPHA = 0.3 # Smoothing for the preview

        # --- STATE ---
        self.history = {"left": None, "right": None}
        self.recording = False
        self.sequence = [] 
        self.cap = None
        self.last_saved_key = None
        self.start_time = None

        # --- MEDIAPIPE ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1, max_num_hands=2,
            min_detection_confidence=0.8, min_tracking_confidence=0.8
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
        
        # Config Panel
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

        # Camera
        self.feed = QLabel("Camera Offline")
        self.feed.setMinimumSize(640, 480)
        self.feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed.setStyleSheet("background: #0f172a; border-radius: 15px; border: 5px solid #334155;")
        layout.addWidget(self.feed, stretch=5)

        # Decision Row
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

        # Controls
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
        self.sequence = []
        self.recording = True
        self.start_time = time.time()
        self.control_widget.hide()
        self.name_input.setEnabled(False)

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        curr_hands = {"left": None, "right": None}

        if results.multi_hand_landmarks:
            for i, lms in enumerate(results.multi_hand_landmarks):
                side = results.multi_handedness[i].classification[0].label.lower()
                
                # NORMALIZE (Wrist center + Unit Scale)
                pts = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark])
                wrist = pts[0]
                scale = np.linalg.norm(pts[0] - pts[9]) or 1.0
                norm_pts = ((pts - wrist) / scale).flatten()
                
                curr_hands[side] = norm_pts
                self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)

        if self.recording:
            elapsed = time.time() - self.start_time
            bar_w = int(frame.shape[1] * (elapsed / self.RECORD_DURATION))
            cv2.rectangle(frame, (0,0), (bar_w, 20), (59, 130, 246), -1)

            if not self.mode_btn.isChecked():
                # DUAL MODE: Save both hands
                self.sequence.append({
                    "left": curr_hands["left"] if curr_hands["left"] is not None else np.zeros(63),
                    "right": curr_hands["right"] if curr_hands["right"] is not None else np.zeros(63)
                })
            else:
                # SINGLE MODE: Save target hand as a direct array
                target = "left" if "LEFT" in self.side_btn.text() else "right"
                self.sequence.append(curr_hands[target] if curr_hands[target] is not None else np.zeros(63))

            if elapsed >= self.RECORD_DURATION:
                self.recording = False
                self._stop_camera()
                self.decision_widget.show()

        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self.feed.setPixmap(QPixmap.fromImage(qimg).scaled(self.feed.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def _save_data(self):
        name = self.name_input.text().strip().lower()
        is_single = self.mode_btn.isChecked()
        
        lib = np.load(self.lib_path, allow_pickle=True).item() if os.path.exists(self.lib_path) else {}
        
        prefix = f"{name}_dual_" if not is_single else f"{name}_{'left' if 'LEFT' in self.side_btn.text() else 'right'}_"
        existing = [int(k.split("_")[-1]) for k in lib if k.startswith(prefix)]
        idx = max(existing) + 1 if existing else 1
        label = f"{prefix}{idx}"

        lib[label] = np.array(self.sequence, dtype=object)
        np.save(self.lib_path, lib)
        
        self.history_label.setText(f"✅ Saved: {label}")
        self._reset_ui()

    def _discard_data(self):
        self._reset_ui()

    def _reset_ui(self):
        self.decision_widget.hide()
        self.control_widget.show()
        self.name_input.setEnabled(True)
        self._toggle_camera()

def get_tab():
    return TrainerWidget()