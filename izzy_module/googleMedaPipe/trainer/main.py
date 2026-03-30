import sys
import os
import cv2
import numpy as np
import mediapipe as mp
import time
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, 
    QPushButton, QHBoxLayout, QLineEdit, QFrame, QMessageBox
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

# --- DYNAMIC PATH SETUP ---
# This ensures the module can find the launcher's config loader if run from the shell
current_file_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_path, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from launcher import load_config
except ImportError:
    # Fallback for standalone testing outside the shell
    def load_config(): return {"library_path": "asl_motion_library.npy"}

# --- STYLES ---
STYLE_MAIN = "background: #1e293b; color: white; font-weight: bold; border-radius: 8px; padding: 12px;"
STYLE_SAVE = "background: #16a34a; color: white; font-weight: bold; border-radius: 8px; padding: 15px; font-size: 16px;"
STYLE_DISCARD = "background: #ef4444; color: white; font-weight: bold; border-radius: 8px; padding: 15px; font-size: 16px;"
STYLE_INACTIVE = "background: #475569; color: #cbd5e1; border-radius: 8px; padding: 12px;"

class ASLTrainerTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # 1. Config from Shell
        cfg = load_config()
        self.lib_path = cfg.get("library_path", "asl_motion_library.npy")
        
        # 2. State
        self.RECORD_DURATION = 2.5
        self.Z_SCALE = 2.0
        self.ALPHA = 0.3 
        self.history = {"left": None, "right": None}
        self.recording = False
        self.sequence = []
        self.cap = None
        self.last_saved_key = None

        # 3. MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1, max_num_hands=2, 
            min_detection_confidence=0.8, min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils

        self._init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def _init_ui(self):
        self.main_layout = QVBoxLayout(self) 
        self.main_layout.setSpacing(15)

        # --- TOP CONFIG PANEL ---
        config_panel = QFrame()
        config_panel.setStyleSheet("background: #f1f5f9; border-radius: 12px; border: 1px solid #cbd5e1;")
        config_layout = QVBoxLayout(config_panel)
        
        mode_row = QHBoxLayout()
        self.mode_btn = QPushButton("MODE: DUAL HAND 👐")
        self.mode_btn.setCheckable(True)
        self.mode_btn.setStyleSheet(STYLE_MAIN)
        self.mode_btn.clicked.connect(self.toggle_mode)
        
        self.side_btn = QPushButton("SIDE: RIGHT ➡️")
        self.side_btn.setStyleSheet("background: #3b82f6; color: white; border-radius: 8px; padding: 12px;")
        self.side_btn.setVisible(False) 
        self.side_btn.clicked.connect(self.toggle_side)
        
        mode_row.addWidget(QLabel("<b>Configuration:</b>"))
        mode_row.addWidget(self.mode_btn)
        mode_row.addWidget(self.side_btn)
        mode_row.addStretch()
        config_layout.addLayout(mode_row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("<b style='color: #1e293b;'>Gesture Name:</b>"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter sign name (e.g. apple)")
        self.name_input.setStyleSheet("padding: 10px; background-color: white; color: black; border-radius: 6px; border: 2px solid #94a3b8;")
        name_row.addWidget(self.name_input)
        config_layout.addLayout(name_row)
        self.main_layout.addWidget(config_panel)

        # --- CAMERA FEED ---
        self.feed = QLabel("Camera Offline")
        self.feed.setMinimumSize(640, 480)
        self.feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed.setStyleSheet("background: #0f172a; border-radius: 15px; border: 5px solid #334155;")
        self.main_layout.addWidget(self.feed, stretch=5)

        # --- DECISION OVERLAY ---
        self.decision_widget = QWidget()
        d_layout = QHBoxLayout(self.decision_widget)
        self.save_btn = QPushButton("✔ SAVE GESTURE")
        self.save_btn.setStyleSheet(STYLE_SAVE)
        self.save_btn.clicked.connect(self.save_data)
        self.discard_btn = QPushButton("✖ DISCARD & RETRY")
        self.discard_btn.setStyleSheet(STYLE_DISCARD)
        self.discard_btn.clicked.connect(self.discard_data)
        d_layout.addWidget(self.save_btn)
        d_layout.addWidget(self.discard_btn)
        self.decision_widget.hide()
        self.main_layout.addWidget(self.decision_widget)

        # --- MAIN CONTROLS ---
        self.control_widget = QWidget()
        c_layout = QHBoxLayout(self.control_widget)
        self.cam_btn = QPushButton("START CAMERA")
        self.cam_btn.setStyleSheet(STYLE_MAIN)
        self.cam_btn.clicked.connect(self.toggle_camera)
        self.rec_btn = QPushButton("🔴 RECORD 2.5s")
        self.rec_btn.setEnabled(False)
        self.rec_btn.setStyleSheet(STYLE_INACTIVE)
        self.rec_btn.clicked.connect(self.start_recording)
        c_layout.addWidget(self.cam_btn)
        c_layout.addWidget(self.rec_btn)
        self.main_layout.addWidget(self.control_widget)

        # --- FOOTER ---
        self.footer = QFrame()
        self.footer.setStyleSheet("background: #f8fafc; border-top: 1px solid #e2e8f0; padding: 10px;")
        f_layout = QHBoxLayout(self.footer)
        self.history_label = QLabel("Waiting for first recording...")
        self.remove_btn = QPushButton("REMOVE LAST TAKE")
        self.remove_btn.setEnabled(False)
        self.remove_btn.setStyleSheet("background: #450a0a; color: #fecaca; padding: 5px 15px; border-radius: 5px;")
        self.remove_btn.clicked.connect(self.remove_last)
        f_layout.addWidget(self.history_label)
        f_layout.addStretch()
        f_layout.addWidget(self.remove_btn)
        self.main_layout.addWidget(self.footer)

    def toggle_mode(self):
        is_single = self.mode_btn.isChecked()
        self.mode_btn.setText("MODE: SINGLE HAND ☝" if is_single else "MODE: DUAL HAND 👐")
        self.side_btn.setVisible(is_single)

    def toggle_side(self):
        is_left = "RIGHT" in self.side_btn.text()
        self.side_btn.setText("SIDE: LEFT ⬅️" if is_left else "SIDE: RIGHT ➡️")
        self.side_btn.setStyleSheet(f"background: {'#8b5cf6' if is_left else '#3b82f6'}; color: white; border-radius: 8px; padding: 12px;")

    def toggle_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.cam_btn.setText("STOP CAMERA")
            self.rec_btn.setEnabled(True)
            self.rec_btn.setStyleSheet("background: #dc2626; color: white; font-weight: bold; border-radius: 8px; padding: 12px;")
        else:
            self.stop_camera()

    def stop_camera(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None
        self.cam_btn.setText("START CAMERA")
        self.rec_btn.setEnabled(False)
        self.rec_btn.setStyleSheet(STYLE_INACTIVE)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        curr_frame_data = {"left": None, "right": None}
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                side = "left" if hand_lms.landmark[0].x < 0.5 else "right"
                curr_frame_data[side] = self.get_smoothed_norm(hand_lms, side)
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        if self.recording:
            elapsed = time.time() - self.start_time
            cv2.rectangle(frame, (0, 0), (int(frame.shape[1] * (elapsed/self.RECORD_DURATION)), 15), (59, 130, 246), -1)
            
            if not self.mode_btn.isChecked():
                self.sequence.append(curr_frame_data)
            else:
                target = "left" if "LEFT" in self.side_btn.text() else "right"
                val = curr_frame_data[target] if curr_frame_data[target] is not None else np.zeros(63)
                self.sequence.append(val)
            
            if elapsed >= self.RECORD_DURATION:
                self.recording = False
                self.show_review_screen()

        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch*w, QImage.Format.Format_BGR888)
        self.feed.setPixmap(QPixmap.fromImage(qimg).scaled(self.feed.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def get_smoothed_norm(self, hand_lms, side):
        pts = np.array([[lm.x, lm.y, lm.z * self.Z_SCALE] for lm in hand_lms.landmark])
        wrist = pts[0]
        scale = np.linalg.norm(pts[0] - pts[5]) or 1.0
        new_norm = ((pts - wrist) / scale).flatten()
        if self.history[side] is None:
            self.history[side] = new_norm
        else:
            self.history[side] = (self.ALPHA * new_norm) + ((1 - self.ALPHA) * self.history[side])
        return self.history[side]

    def start_recording(self):
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Missing Name", "Please name the gesture before recording.")
            return
        self.sequence = []
        self.recording = True
        self.start_time = time.time()
        self.control_widget.hide()
        self.name_input.setEnabled(False)

    def show_review_screen(self):
        self.stop_camera()
        self.decision_widget.show()

    def save_data(self):
        base = self.name_input.text().lower().strip()
        # Ensure we load the library file correctly from the config path
        lib = np.load(self.lib_path, allow_pickle=True).item() if os.path.exists(self.lib_path) else {}
        prefix = f"{base}_dual_" if not self.mode_btn.isChecked() else f"{base}_{('left' if 'LEFT' in self.side_btn.text() else 'right')}_"
        existing = [int(k.split('_')[-1]) for k in lib.keys() if k.startswith(prefix)]
        idx = max(existing) + 1 if existing else 1
        label = f"{prefix}{idx}"
        lib[label] = np.array(self.sequence, dtype=object)
        np.save(self.lib_path, lib)
        self.last_saved_key = label
        self.history_label.setText(f"Last Saved: <b>{label}</b>")
        self.remove_btn.setEnabled(True)
        self.reset_ui()

    def remove_last(self):
        if not self.last_saved_key: return
        ans = QMessageBox.warning(self, "Confirm Delete", f"Delete {self.last_saved_key}?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ans == QMessageBox.StandardButton.Yes:
            lib = np.load(self.lib_path, allow_pickle=True).item()
            if self.last_saved_key in lib:
                del lib[self.last_saved_key]
                np.save(self.lib_path, lib)
                self.history_label.setText(f"Deleted: {self.last_saved_key}")
                self.last_saved_key = None
                self.remove_btn.setEnabled(False)

    def discard_data(self):
        self.reset_ui()

    def reset_ui(self):
        self.decision_widget.hide()
        self.control_widget.show()
        self.name_input.setEnabled(True)
        self.toggle_camera()

    def hideEvent(self, event):
        """CRITICAL: Free the camera resource when the user switches tabs"""
        self.stop_camera()
        super().hideEvent(event)

# --- LAUNCHER ENTRY POINT ---
def get_tab():
    """This function is called by the Shell launcher to load the tab"""
    return ASLTrainerTab()

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    # Allows testing this file directly without the launcher
    w = ASLTrainerTab()
    w.show()
    sys.exit(app.exec())