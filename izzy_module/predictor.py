import os
import cv2
import numpy as np
import mediapipe as mp
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class PredictorWidget(QWidget):
    def __init__(self):
        super().__init__()

        # --- PATH SETUP ---
        self.lib_path = os.path.join(SCRIPT_DIR, "googleMedaPipe", "asl_motion_library.npy")

        # --- CONFIG ---
        self.library   = self._load_library()
        self.Z_SCALE   = 2.0
        self.ALPHA     = 0.25
        self.THRESHOLD = 0.65

        self.history = {"left": None, "right": None}
        self.cap     = None

        # --- MEDIAPIPE ---
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.mp_draw = mp.solutions.drawing_utils

        self._init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

    # ── Library ──────────────────────────────────────────────────────────────
    def _load_library(self):
        print(f"🔍 Predictor looking for library: {self.lib_path}")
        if not os.path.exists(self.lib_path):
            print("❌ Library not found")
            return {}
        try:
            data = np.load(self.lib_path, allow_pickle=True).item()
            print(f"✅ Loaded {len(data)} gestures: {list(data.keys())}")
            return data
        except Exception as e:
            print(f"❌ Load error: {e}")
            return {}

    # ── UI ───────────────────────────────────────────────────────────────────
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        self.status_bar = QLabel(f"Library: {len(self.library)} gestures loaded")
        self.status_bar.setStyleSheet(
            "background: #0f172a; color: #38bdf8; padding: 10px; "
            "font-weight: bold; border-radius: 6px;"
        )
        layout.addWidget(self.status_bar)

        self.feed = QLabel("Camera Off")
        self.feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed.setStyleSheet(
            "background: black; border: 4px solid #1e293b; border-radius: 10px;"
        )
        layout.addWidget(self.feed, stretch=5)

        self.score_label = QLabel("Match Confidence: 0%")
        self.score_label.setStyleSheet(
            "color: #94a3b8; font-size: 18px; font-family: monospace;"
        )
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.score_label)

        self.result_label = QLabel("READY")
        self.result_label.setStyleSheet(
            "font-size: 60px; color: #f8fafc; font-weight: 900;"
        )
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        self.btn = QPushButton("START LIVE PREDICTOR")
        self.btn.clicked.connect(self._toggle_cam)
        self.btn.setStyleSheet(
            "padding: 20px; background: #2563eb; color: white; "
            "font-weight: bold; border-radius: 10px;"
        )
        layout.addWidget(self.btn)

    # ── Camera ───────────────────────────────────────────────────────────────
    def _toggle_cam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.btn.setText("STOP PREDICTOR")
            self.btn.setStyleSheet(
                "padding: 20px; background: #dc2626; color: white; "
                "font-weight: bold; border-radius: 10px;"
            )
        else:
            self._stop_cam()

    def _stop_cam(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn.setText("START LIVE PREDICTOR")
        self.btn.setStyleSheet(
            "padding: 20px; background: #2563eb; color: white; "
            "font-weight: bold; border-radius: 10px;"
        )
        self.feed.setText("Camera Off")

    # ── Core logic ───────────────────────────────────────────────────────────
    def _compare_hybrid(self, v1, v2):
        """Combines cosine similarity and euclidean distance for accuracy."""
        if v1 is None or v2 is None:
            return 0
        v1, v2 = np.array(v1).flatten(), np.array(v2).flatten()
        if v1.shape != v2.shape:
            return 0

        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_score = np.dot(v1, v2) / norm if norm > 0 else 0

        dist      = np.linalg.norm(v1 - v2)
        euc_score = np.exp(-0.85 * dist)

        return (cos_score * 0.4) + (euc_score * 0.6)

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        live_hands = {"left": None, "right": None}
        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                side = "left" if lms.landmark[0].x < 0.5 else "right"
                pts   = np.array([[lm.x, lm.y, lm.z * self.Z_SCALE] for lm in lms.landmark])
                wrist = pts[0]
                scale = np.linalg.norm(pts[0] - pts[5]) or 1.0
                norm  = ((pts - wrist) / scale).flatten()

                if self.history[side] is None:
                    self.history[side] = norm
                else:
                    self.history[side] = (
                        (self.ALPHA * norm) + ((1 - self.ALPHA) * self.history[side])
                    )
                live_hands[side] = self.history[side]
                self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)

        top_name, top_score = "---", 0
        for label, saved_seq in self.library.items():
            for i in [0, len(saved_seq) // 2, -1]:
                ref = saved_seq[i]
                if isinstance(ref, dict):
                    s_l, s_r = ref.get("left"), ref.get("right")
                    c_l, c_r = live_hands["left"], live_hands["right"]
                    score_l = self._compare_hybrid(s_l, c_l) if s_l is not None else (1.0 if c_l is None else 0.0)
                    score_r = self._compare_hybrid(s_r, c_r) if s_r is not None else (1.0 if c_r is None else 0.0)
                    score   = (score_l + score_r) / 2
                else:
                    side  = "left" if "_left_" in label else "right"
                    score = self._compare_hybrid(ref, live_hands[side])

                if score > top_score:
                    top_score = score
                    top_name  = label.split("_")[0].upper()

        self.score_label.setText(f"Match Confidence: {int(top_score * 100)}%  ({top_name})")

        if top_score > self.THRESHOLD:
            self.result_label.setText(top_name)
            self.result_label.setStyleSheet(
                "font-size: 60px; color: #4ade80; font-weight: 900;"
            )
        else:
            self.result_label.setText("SEARCHING...")
            self.result_label.setStyleSheet(
                "font-size: 60px; color: #475569; font-weight: 900;"
            )

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        self.feed.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.feed.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation  # ← required in PySide6
            )
        )

    # ── Cleanup ──────────────────────────────────────────────────────────────
    def hideEvent(self, event):
        """Stop camera when tab is switched away."""
        self._stop_cam()
        super().hideEvent(event)