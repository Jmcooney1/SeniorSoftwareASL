import os
import cv2
import json
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QFrame
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

# Importing the engine from the root folder
from googleMedaPipe.predictions import MotionPredictor

class PredictorWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        # --- PATH & ENGINE SETUP ---
        self.lib_path = self._get_library_path_from_config()
        self.engine = MotionPredictor(library_file=self.lib_path)
        
        self.cap = None
        
        # --- MEDIAPIPE SOLUTIONS ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7
        )
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self._init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

    def _get_library_path_from_config(self):
        curr = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            cfg_p = os.path.join(curr, "config.json")
            if os.path.exists(cfg_p):
                with open(cfg_p, "r") as f:
                    return json.load(f).get("library_path", "")
            curr = os.path.dirname(curr)
        return ""

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # --- NEW: USER IDENTIFICATION PANEL ---
        # This matches the Trainer so the engine knows who is signing
        user_panel = QFrame()
        user_panel.setStyleSheet("background: #1e293b; border-radius: 10px; border: 1px solid #334155;")
        user_layout = QHBoxLayout(user_panel)
        
        user_label = QLabel("👤 IDENTIFY USER:")
        user_label.setStyleSheet("color: #38bdf8; font-weight: bold;")
        
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter Name (e.g., Izzy)")
        self.user_input.setText("Default")
        self.user_input.setStyleSheet("padding: 8px; background: #0f172a; color: white; border: 1px solid #38bdf8; border-radius: 4px;")
        # When text changes, tell the engine who the user is
        self.user_input.textChanged.connect(self._update_engine_user)
        
        user_layout.addWidget(user_label)
        user_layout.addWidget(self.user_input)
        layout.addWidget(user_panel)

        self.status = QLabel(f"Status: {len(self.engine.library)} Signs Loaded")
        self.status.setStyleSheet("color: #38bdf8; font-weight: bold; background: #0f172a; padding: 10px; border-radius: 5px;")
        layout.addWidget(self.status)

        self.conf_meter = QLabel("Match Confidence: 0%")
        self.conf_meter.setFixedHeight(45)
        self.conf_meter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.conf_meter.setStyleSheet("background: #1e293b; color: #94a3b8; font-size: 16px; font-weight: bold; border-radius: 8px; border: 2px solid #334155;")
        layout.addWidget(self.conf_meter)

        self.feed = QLabel("Camera Off")
        self.feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed.setStyleSheet("background: black; border-radius: 10px; border: 3px solid #1e293b;")
        layout.addWidget(self.feed, stretch=5)

        self.result_label = QLabel("READY")
        self.result_label.setStyleSheet("font-size: 80px; color: #f8fafc; font-weight: 900;")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        self.btn = QPushButton("START LIVE PREDICTOR")
        self.btn.clicked.connect(self._toggle_cam)
        self.btn.setStyleSheet("padding: 20px; background: #2563eb; color: white; font-weight: bold; border-radius: 10px;")
        layout.addWidget(self.btn)

    def _update_engine_user(self):
        """Update the engine's current user whenever the text box changes."""
        new_user = self.user_input.text()
        self.engine.current_user = new_user

    def showEvent(self, event):
        if hasattr(self, 'engine'):
            self.engine.library = self.engine.load_library(self.lib_path)
            self.status.setText(f"Status: {len(self.engine.library)} Signs Loaded")
            # Sync user on show
            self._update_engine_user()
        super().showEvent(event)

    def _toggle_cam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.timer.start(30)
                self.btn.setText("STOP PREDICTOR")
                self.btn.setStyleSheet("padding: 20px; background: #dc2626; color: white; font-weight: bold; border-radius: 10px;")
        else:
            self._stop_cam()

    def _stop_cam(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.feed.setText("Camera Off")
        self.btn.setText("START LIVE PREDICTOR")
        self.btn.setStyleSheet("padding: 20px; background: #2563eb; color: white; font-weight: bold; border-radius: 10px;")

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_results = self.face_detector.process(rgb)
        nose_point = None
        if face_results.detections:
            nose_point = mp.solutions.face_detection.get_key_point(
                face_results.detections[0], 
                mp.solutions.face_detection.FaceKeyPoint.NOSE_TIP
            )

        results = self.hands.process(rgb)
        best_word = "..."
        best_conf = 0
        current_visible_sides = []

        if results.multi_hand_landmarks:
            for i, lms in enumerate(results.multi_hand_landmarks):
                side = results.multi_handedness[i].classification[0].label
                current_visible_sides.append(side.lower())
                
                # Use current nose_point
                word, conf = self.engine.process_frame(lms, side, nose_point)
                
                if conf > best_conf:
                    best_conf = conf
                    best_word = word
                
                self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)

            # --- UI UPDATES ---
            if best_conf >= 95:
                color = "#22d3ee"
                text = f"🌟 PERFECT: {best_conf}%"
            elif best_conf >= 80:
                color = "#4ade80"
                text = f"Match Confidence: {best_conf}%"
            elif best_conf >= 60:
                color = "#fbbf24"
                text = f"Getting Closer... {best_conf}%"
            else:
                color = "#f87171"
                text = f"Searching... ({best_conf}%)"

            self.conf_meter.setText(text)
            self.conf_meter.setStyleSheet(f"background: #0f172a; color: {color}; font-size: 16px; font-weight: 900; border-radius: 8px; border: 2px solid {color};")

            if best_word != "..." and best_conf > 55: # Matches engine trigger
                self.result_label.setText(best_word.upper())
                self.result_label.setStyleSheet(f"font-size: 90px; color: {color}; font-weight: 900;")
            elif best_conf < 30:
                self.result_label.setText("SEARCHING...")
                self.result_label.setStyleSheet("font-size: 80px; color: #475569; font-weight: 900;")

        else:
            self.engine.reset_hand("left")
            self.engine.reset_hand("right")
            self.conf_meter.setText("Match Confidence: 0%")
            self.conf_meter.setStyleSheet("background: #1e293b; color: #94a3b8; border-radius: 8px; border: 2px solid #334155;")
            self.result_label.setText("READY")
            self.result_label.setStyleSheet("font-size: 80px; color: #f8fafc; font-weight: 900;")

        if "left" not in current_visible_sides: self.engine.reset_hand("left")
        if "right" not in current_visible_sides: self.engine.reset_hand("right")

        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self.feed.setPixmap(QPixmap.fromImage(qimg).scaled(self.feed.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def hideEvent(self, event):
        self._stop_cam()
        super().hideEvent(event)

def get_tab():
    return PredictorWidget()