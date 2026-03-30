import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

# --- DYNAMIC PATH SETUP ---
current_file_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_path, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from launcher import load_config

class PredictorTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # Load library from config
        cfg = load_config()
        self.lib_path = cfg.get("library_path", "asl_motion_library.npy")
        self.library = self._load_library()

        # State
        self.cap = None
        self.live_buffer = [] # Holds the last ~75 frames (2.5s at 30fps)
        self.BUFFER_SIZE = 75 
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        self._init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def _load_library(self):
        if os.path.exists(self.lib_path):
            return np.load(self.lib_path, allow_pickle=True).item()
        return {}

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Prediction Display
        self.result_label = QLabel("Waiting for gesture...")
        self.result_label.setStyleSheet("""
            font-size: 32px; font-weight: bold; color: #3b82f6; 
            background: #1e293b; padding: 20px; border-radius: 10px;
        """)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        # Camera Feed
        self.feed = QLabel()
        self.feed.setStyleSheet("background: black; border-radius: 10px;")
        layout.addWidget(self.feed, stretch=1)

    def update_frame(self):
        if self.cap is None: return
        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        curr_frame = np.zeros(63) # Default empty hand
        if results.multi_hand_landmarks:
            # For simplicity, we'll just track the first hand detected
            lm = results.multi_hand_landmarks[0]
            curr_frame = np.array([[l.x, l.y, l.z] for l in lm.landmark]).flatten()
            self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

        # Update Buffer
        self.live_buffer.append(curr_frame)
        if len(self.live_buffer) > self.BUFFER_SIZE:
            self.live_buffer.pop(0)

        # Perform Recognition every 5 frames to save CPU
        if len(self.live_buffer) == self.BUFFER_SIZE:
            self.recognize_gesture()

        # Show Frame
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch*w, QImage.Format.Format_BGR888)
        self.feed.setPixmap(QPixmap.fromImage(qimg).scaled(self.feed.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def recognize_gesture(self):
        best_match = "None"
        min_dist = float('inf')

        # Compare buffer against every saved sequence in library
        for name, saved_sequence in self.library.items():
            # Basic distance check (Euclidean distance across the sequence)
            # Note: In a real app, you'd use Dynamic Time Warping (DTW)
            try:
                dist = np.linalg.norm(np.array(self.live_buffer) - np.array(saved_sequence))
                if dist < min_dist:
                    min_dist = dist
                    best_match = name.split('_')[0] # Remove the timestamp
            except:
                continue

        # Confidence Threshold (Tweak this based on your data)
        if min_dist < 15.0:
            self.result_label.setText(f"GESTURE: {best_match.upper()}")
        else:
            self.result_label.setText("Scanning...")

    def showEvent(self, event):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        # Reload library in case the user added new signs in the Trainer tab
        self.library = self._load_library()

    def hideEvent(self, event):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()

def get_tab():
    return PredictorTab()