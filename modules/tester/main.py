import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QApplication, QHBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Predictor - Hybrid Precision")
        self.setMinimumSize(1000, 850)

        # --- PATH SETUP (googleMedaPipe) ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        self.lib_path = os.path.join(project_root, "googleMedaPipe", "asl_motion_library.npy")
        
        # --- CONFIG ---
        self.library = self.load_and_verify_library()
        self.Z_SCALE = 2.0
        self.ALPHA = 0.25      # Smoothing factor for stability
        self.THRESHOLD = 0.65  # The "Sweet Spot" for the hybrid math

        self.history = {"left": None, "right": None}
        self.cap = None

        # --- MEDIAPIPE ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.mp_draw = mp.solutions.drawing_utils

        self._init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def load_and_verify_library(self):
        print(f"🔍 Looking for data in: {self.lib_path}")
        if not os.path.exists(self.lib_path):
            print("❌ DATABASE NOT FOUND")
            return {}
        try:
            data = np.load(self.lib_path, allow_pickle=True).item()
            print(f"✅ SUCCESS: Loaded {list(data.keys())}")
            return data
        except Exception as e:
            print(f"❌ LOAD ERROR: {e}")
            return {}

    def _init_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        
        self.status_bar = QLabel(f"Library: {len(self.library)} gestures loaded")
        self.status_bar.setStyleSheet("background: #0f172a; color: #38bdf8; padding: 10px; font-weight: bold;")
        layout.addWidget(self.status_bar)

        self.feed = QLabel("Camera Off")
        self.feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed.setStyleSheet("background: black; border: 4px solid #1e293b; border-radius: 10px;")
        layout.addWidget(self.feed, stretch=5)

        self.score_label = QLabel("Match Confidence: 0%")
        self.score_label.setStyleSheet("color: #94a3b8; font-size: 18px; font-family: monospace;")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.score_label)

        self.result_label = QLabel("READY")
        self.result_label.setStyleSheet("font-size: 60px; color: #f8fafc; font-weight: 900;")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        self.btn = QPushButton("START LIVE PREDICTOR")
        self.btn.clicked.connect(self.toggle_cam)
        self.btn.setStyleSheet("padding: 20px; background: #2563eb; color: white; font-weight: bold; border-radius: 10px;")
        layout.addWidget(self.btn)
        self.setCentralWidget(central)

    def toggle_cam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.btn.setText("STOP PREDICTOR")
            self.btn.setStyleSheet("padding: 20px; background: #dc2626; color: white; font-weight: bold; border-radius: 10px;")
        else:
            self.cap.release()
            self.cap = None
            self.timer.stop()
            self.btn.setText("START LIVE PREDICTOR")
            self.btn.setStyleSheet("padding: 20px; background: #2563eb; color: white; font-weight: bold; border-radius: 10px;")

    def compare_hybrid(self, v1, v2):
        """Combines Direction (Cosine) and Position (Euclidean) for better accuracy"""
        if v1 is None or v2 is None: return 0
        v1, v2 = np.array(v1).flatten(), np.array(v2).flatten()
        if v1.shape != v2.shape: return 0
        
        # 1. Directional Similarity (Stops random 95% spikes)
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_score = np.dot(v1, v2) / norm if norm > 0 else 0
        
        # 2. Positional Accuracy (Ensures fingers are in the right spots)
        dist = np.linalg.norm(v1 - v2)
        euc_score = np.exp(-0.85 * dist) # Adjusted decay for better tolerance
        
        # 3. Final Weighting
        return (cos_score * 0.4) + (euc_score * 0.6)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        live_hands = {"left": None, "right": None}
        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                side = "left" if lms.landmark[0].x < 0.5 else "right"
                pts = np.array([[lm.x, lm.y, lm.z * self.Z_SCALE] for lm in lms.landmark])
                wrist = pts[0]
                scale = np.linalg.norm(pts[0] - pts[5]) or 1.0
                norm = ((pts - wrist) / scale).flatten()
                
                if self.history[side] is None: self.history[side] = norm
                else: self.history[side] = (self.ALPHA * norm) + ((1 - self.ALPHA) * self.history[side])
                live_hands[side] = self.history[side]
                self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)

        top_name, top_score = "---", 0
        for label, saved_seq in self.library.items():
            # Check a few sample frames from the saved motion
            for i in [0, len(saved_seq)//2, -1]:
                ref = saved_seq[i]
                if isinstance(ref, dict): # Dual Hand Support
                    s_l, s_r = ref.get("left"), ref.get("right")
                    c_l, c_r = live_hands["left"], live_hands["right"]
                    score_l = self.compare_hybrid(s_l, c_l) if s_l is not None else (1.0 if c_l is None else 0.0)
                    score_r = self.compare_hybrid(s_r, c_r) if s_r is not None else (1.0 if c_r is None else 0.0)
                    score = (score_l + score_r) / 2
                else: # Single Hand Support
                    side = "left" if "_left_" in label else "right"
                    score = self.compare_hybrid(ref, live_hands[side])
                
                if score > top_score:
                    top_score, top_name = score, label.split('_')[0].upper()

        # Update UI
        self.score_label.setText(f"Match Confidence: {int(top_score*100)}% ({top_name})")
        
        if top_score > self.THRESHOLD:
            self.result_label.setText(top_name)
            self.result_label.setStyleSheet("font-size: 60px; color: #4ade80; font-weight: 900;")
        else:
            self.result_label.setText("SEARCHING...")
            self.result_label.setStyleSheet("font-size: 60px; color: #475569; font-weight: 900;")

        # Convert image for PyQt
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch*w, QImage.Format.Format_BGR888)
        self.feed.setPixmap(QPixmap.fromImage(qimg).scaled(self.feed.size(), Qt.AspectRatioMode.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())