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

        # Fixed window constraints
        self.setMinimumSize(800, 700) 
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
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(8)
        
        # --- TOP PANEL ---
        top_row = QHBoxLayout()
        user_panel = QFrame()
        user_panel.setStyleSheet("background: #1e293b; border-radius: 8px; border: 1px solid #334155;")
        user_layout = QHBoxLayout(user_panel)
        user_label = QLabel("👤 USER:")
        user_label.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 13px;")
        self.user_input = QLineEdit()
        self.user_input.setText("Default")
        self.user_input.setStyleSheet("padding: 4px; background: #0f172a; color: white; border: 1px solid #38bdf8; border-radius: 4px;")
        self.user_input.textChanged.connect(self._update_engine_user)
        user_layout.addWidget(user_label)
        user_layout.addWidget(self.user_input)
        
        self.status = QLabel(f"Signs: {len(self.engine.library)}")
        self.status.setStyleSheet("color: #38bdf8; font-weight: bold; background: #0f172a; padding: 6px; border-radius: 8px; border: 1px solid #334155; font-size: 13px;")
        
        top_row.addWidget(user_panel, stretch=2)
        top_row.addWidget(self.status, stretch=1)
        layout.addLayout(top_row)

        # --- CONFIDENCE METER ---
        self.conf_meter = QLabel("Match Confidence: 0%")
        self.conf_meter.setFixedHeight(35)
        self.conf_meter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.conf_meter.setStyleSheet("background: #0f172a; color: #94a3b8; font-size: 14px; font-weight: bold; border-radius: 6px; border: 1px solid #1e293b;")
        layout.addWidget(self.conf_meter)

        # --- CAMERA FEED (STRETCHED TO FILL) ---
        self.feed = QLabel("Camera Off")
        self.feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed.setStyleSheet("background: black; border-radius: 12px; border: 3px solid #1e293b;")
        layout.addWidget(self.feed, stretch=20)

        # --- RESULT TEXT (CAPPED SIZES) ---
        self.result_label = QLabel("READY")
        # Final capped font size: 45px
        self.result_label.setStyleSheet("font-size: 45px; color: #f8fafc; font-weight: 900; margin: 2px;")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        # --- CONTROL BUTTON ---
        self.btn = QPushButton("START LIVE PREDICTOR")
        self.btn.clicked.connect(self._toggle_cam)
        self.btn.setStyleSheet("padding: 12px; background: #2563eb; color: white; font-size: 16px; font-weight: bold; border-radius: 10px;")
        layout.addWidget(self.btn)

    def _update_engine_user(self):
        self.engine.current_user = self.user_input.text()

    def showEvent(self, event):
        if hasattr(self, 'engine'):
            self.engine.library = self.engine.load_library(self.lib_path)
            self.status.setText(f"Signs: {len(self.engine.library)}")
            self._update_engine_user()
        super().showEvent(event)

    def _toggle_cam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.timer.start(30)
                self.btn.setText("STOP PREDICTOR")
                self.btn.setStyleSheet("padding: 12px; background: #dc2626; color: white; font-size: 16px; font-weight: bold; border-radius: 10px;")
        else:
            self._stop_cam()

    def _stop_cam(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.feed.setText("Camera Off")
        self.btn.setText("START LIVE PREDICTOR")
        self.btn.setStyleSheet("padding: 12px; background: #2563eb; color: white; font-size: 16px; font-weight: bold; border-radius: 10px;")

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
        best_word, best_conf = "...", 0
        current_visible_sides = []

        if results.multi_hand_landmarks:
            for i, lms in enumerate(results.multi_hand_landmarks):
                side = results.multi_handedness[i].classification[0].label
                current_visible_sides.append(side.lower())
                
                word, conf = self.engine.process_frame(lms, side, nose_point)
                
                if conf > best_conf:
                    best_conf, best_word = conf, word
                
                self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)

            if best_conf >= 55: color = "#4ade80"
            else: color = "#f87171"

            self.conf_meter.setText(f"Confidence: {best_conf}%")
            self.conf_meter.setStyleSheet(f"background: #0f172a; color: {color}; font-size: 14px; font-weight: 900; border-radius: 8px; border: 1px solid {color};")

            if best_word != "..." and best_conf > 55:
                self.result_label.setText(best_word.upper())
                # Capped at 50px for active results
                self.result_label.setStyleSheet(f"font-size: 50px; color: {color}; font-weight: 900;")
            else:
                self.result_label.setText("READY")
                # Capped at 45px for idle
                self.result_label.setStyleSheet("font-size: 45px; color: #475569; font-weight: 900;")

        else:
            self.engine.reset_hand("left")
            self.engine.reset_hand("right")
            self.result_label.setText("READY")
            self.result_label.setStyleSheet("font-size: 45px; color: #f8fafc; font-weight: 900;")

        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        # Scale to fit WITHOUT zooming in or expanding the container
        self.feed.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.feed.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))

    def hideEvent(self, event):
        self._stop_cam()
        super().hideEvent(event)

# --- CRITICAL EXPORT FOR LAUNCHER ---
def get_tab():
    return PredictorWidget()