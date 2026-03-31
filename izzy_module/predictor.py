import os
import cv2
import json
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
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
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        self._init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

    def _get_library_path_from_config(self):
        """Finds config.json in the root to get the central .npy path."""
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
        
        # Status Label
        self.status = QLabel(f"Status: {len(self.engine.library)} Signs Loaded")
        self.status.setStyleSheet("color: #38bdf8; font-weight: bold; background: #0f172a; padding: 10px; border-radius: 5px;")
        layout.addWidget(self.status)

        # Confidence Meter
        self.conf_meter = QLabel("Match Confidence: 0%")
        self.conf_meter.setFixedHeight(40)
        self.conf_meter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.conf_meter.setStyleSheet("""
            background: #1e293b; 
            color: #94a3b8; 
            font-size: 16px; 
            font-weight: bold; 
            border-radius: 8px;
            border: 2px solid #334155;
        """)
        layout.addWidget(self.conf_meter)

        # Camera Feed
        self.feed = QLabel("Camera Off")
        self.feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed.setStyleSheet("background: black; border-radius: 10px; border: 3px solid #1e293b;")
        layout.addWidget(self.feed, stretch=5)

        # Prediction Result
        self.result_label = QLabel("READY")
        self.result_label.setStyleSheet("font-size: 80px; color: #f8fafc; font-weight: 900;")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        # Controls
        self.btn = QPushButton("START LIVE PREDICTOR")
        self.btn.clicked.connect(self._toggle_cam)
        self.btn.setStyleSheet("padding: 20px; background: #2563eb; color: white; font-weight: bold; border-radius: 10px;")
        layout.addWidget(self.btn)

    def showEvent(self, event):
        """Automatically reloads library when switching to this tab."""
        if hasattr(self, 'engine'):
            self.engine.library = self.engine.load_library(self.lib_path)
            self.status.setText(f"Status: {len(self.engine.library)} Signs Loaded")
        super().showEvent(event)

    def _toggle_cam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
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
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for i, lms in enumerate(results.multi_hand_landmarks):
                side = results.multi_handedness[i].classification[0].label
                word, conf = self.engine.process_frame(lms, side)
                
                self.conf_meter.setText(f"Match Confidence: {conf}%")
                
                # --- TOUGHER COLOR GRADIENTS ---
                if conf >= 95:
                    # ELITE MATCH: Glowing Cyan/Green
                    meter_color = "#22d3ee" 
                    status_text = f"🌟 PERFECT: {conf}%"
                elif conf >= 80:
                    # GREAT MATCH: Solid Green
                    meter_color = "#4ade80"
                    status_text = f"Match Confidence: {conf}%"
                elif conf >= 60:
                    # OKAY: Yellow
                    meter_color = "#fbbf24"
                    status_text = f"Getting Closer... {conf}%"
                else:
                    # POOR: Red
                    meter_color = "#f87171"
                    status_text = "Searching..."

                self.conf_meter.setText(status_text)
                self.conf_meter.setStyleSheet(f"""
                    background: #0f172a; color: {meter_color}; 
                    font-size: 16px; font-weight: 900; 
                    border-radius: 8px; border: 2px solid {meter_color};
                """)

                if word != "..." and conf > 60:
                    self.result_label.setText(word)
                    self.result_label.setStyleSheet(f"font-size: 90px; color: {meter_color}; font-weight: 900;")
                elif conf < 30:
                    self.result_label.setText("SEARCHING...")
                    self.result_label.setStyleSheet("font-size: 80px; color: #475569; font-weight: 900;")

                # --- SMOOTH WORD UPDATE ---
                # Only change the label text if a real word is detected.
                # This prevents it from flickering to "READY" if conf drops slightly.
                if word != "...":
                    self.result_label.setText(word)
                    self.result_label.setStyleSheet(f"font-size: 80px; color: {meter_color}; font-weight: 900;")
                elif conf < 20:
                    # Only reset text if confidence is extremely low (meaning user moved away)
                    self.result_label.setText("SEARCHING...")
                    self.result_label.setStyleSheet("font-size: 80px; color: #475569; font-weight: 900;")
                
                self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)
        else:
            # RESET when hands leave the camera entirely
            self.engine.reset_hand("left")
            self.engine.reset_hand("right")
            self.conf_meter.setText("Match Confidence: 0%")
            self.conf_meter.setStyleSheet("background: #1e293b; color: #94a3b8; border-radius: 8px; border: 2px solid #334155;")
            self.result_label.setText("READY")
            self.result_label.setStyleSheet("font-size: 80px; color: #f8fafc; font-weight: 900;")

        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self.feed.setPixmap(QPixmap.fromImage(qimg).scaled(self.feed.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def hideEvent(self, event):
        self._stop_cam()
        super().hideEvent(event)

def get_tab():
    return PredictorWidget()