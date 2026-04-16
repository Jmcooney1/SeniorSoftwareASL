import os
import cv2
import numpy as np
import mediapipe as mp
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

# --- PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Points to the engine folder inside izzy_module
LIB_PATH = os.path.join(SCRIPT_DIR, "googleMedaPipe", "asl_motion_library.npy")

class PredictorWidget(QWidget):
    def __init__(self):
        super().__init__()

        # --- CONFIG ---
        self.library   = self._load_library()
        self.ALPHA     = 0.3   # Smoothing factor for 69-pt vector
        self.THRESHOLD = 0.70  # Hybrid confidence threshold

        self.history = {"left": None, "right": None}
        self.cap     = None

        # --- MEDIAPIPE ---
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self._init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

    def _load_library(self):
        if not os.path.exists(LIB_PATH):
            print(f"❌ Library not found at: {LIB_PATH}")
            return {}
        try:
            return np.load(LIB_PATH, allow_pickle=True).item()
        except Exception as e:
            print(f"❌ Load error: {e}")
            return {}

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        self.status_bar = QLabel(f"Signs Loaded: {len(self.library)}")
        self.status_bar.setStyleSheet("background: #0f172a; color: #38bdf8; padding: 10px; border-radius: 6px; font-weight: bold;")
        layout.addWidget(self.status_bar)

        self.feed = QLabel("Camera Off")
        self.feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed.setStyleSheet("background: black; border: 4px solid #1e293b; border-radius: 12px;")
        layout.addWidget(self.feed, stretch=10)

        self.score_label = QLabel("Match Confidence: 0%")
        self.score_label.setStyleSheet("color: #94a3b8; font-size: 16px; font-family: monospace;")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.score_label)

        self.result_label = QLabel("READY")
        self.result_label.setStyleSheet("font-size: 55px; color: #f8fafc; font-weight: 900;")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        self.btn = QPushButton("START LIVE PREDICTOR")
        self.btn.clicked.connect(self._toggle_cam)
        self.btn.setStyleSheet("padding: 18px; background: #2563eb; color: white; font-weight: bold; border-radius: 10px;")
        layout.addWidget(self.btn)

    def _compare_hybrid(self, v1, v2):
        """Standardizes vectors to 69pts and applies Hybrid Score."""
        if v1 is None or v2 is None: return 0
        v1, v2 = np.array(v1).flatten(), np.array(v2).flatten()
        
        # Backward compatibility padding
        if v1.shape != v2.shape:
            mlen = max(len(v1), len(v2))
            v1 = np.pad(v1, (0, mlen - len(v1)))
            v2 = np.pad(v2, (0, mlen - len(v2)))

        # 1. Cosine (Angular Accuracy)
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_score = np.dot(v1, v2) / norm if norm > 0 else 0

        # 2. Euclidean (Position/Velocity Accuracy)
        dist = np.linalg.norm(v1 - v2)
        euc_score = np.exp(-0.1 * dist) # Decay factor for velocity-inclusive vectors

        return (cos_score * 0.45) + (euc_score * 0.55)

    def _toggle_cam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.btn.setText("STOP PREDICTOR")
            self.btn.setStyleSheet("padding: 18px; background: #dc2626; color: white; border-radius: 10px;")
        else:
            self._stop_cam()

    def _stop_cam(self):
        self.timer.stop()
        if self.cap: self.cap.release()
        self.cap = None
        self.feed.setText("Camera Off")
        self.btn.setText("START LIVE PREDICTOR")
        self.btn.setStyleSheet("padding: 18px; background: #2563eb; color: white; border-radius: 10px;")

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection for Nose reference
        face_res = self.face_detector.process(rgb)
        nose_pt = None
        if face_res.detections:
            nose_pt = mp.solutions.face_detection.get_key_point(
                face_res.detections[0], mp.solutions.face_detection.FaceKeyPoint.NOSE_TIP
            )

        hand_res = self.hands.process(rgb)
        live_hands = {"left": None, "right": None}

        if hand_res.multi_hand_landmarks:
            for i, lms in enumerate(hand_res.multi_hand_landmarks):
                side = hand_res.multi_handedness[i].classification[0].label.lower()
                
                # 1. Hand Shape (63 pts)
                pts = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark])
                wrist = pts[0]
                hand_shape = ((pts - wrist) * 10).flatten()
                
                # 2. Spatial Position (3 pts)
                spatial = np.array([wrist[0]-nose_pt.x, wrist[1]-nose_pt.y, 0]) * 10 if nose_pt else np.zeros(3)
                
                # 3. Velocity / Path (3 pts)
                if self.history[side] is not None:
                    prev_spatial = self.history[side][63:66] # index of spatial in 69-pt vector
                    velocity = (spatial - prev_spatial) * 5
                else:
                    velocity = np.zeros(3)

                combined = np.concatenate([hand_shape, spatial, velocity])

                # Smooth results
                if self.history[side] is None:
                    self.history[side] = combined
                else:
                    self.history[side] = (self.ALPHA * combined) + ((1 - self.ALPHA) * self.history[side])
                
                live_hands[side] = self.history[side]
                self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)

        top_word, top_score = "READY", 0
        for label, saved_seq in self.library.items():
            ref = saved_seq[-1] # Compare to last frame of movement
            if isinstance(ref, dict):
                s_l, s_r = ref.get("left"), ref.get("right")
                sc_l = self._compare_hybrid(s_l, live_hands["left"])
                sc_r = self._compare_hybrid(s_r, live_hands["right"])
                score = (sc_l + sc_r) / 2
            else:
                s_type = "left" if "_left_" in label.lower() else "right"
                score = self._compare_hybrid(ref, live_hands[s_type])

            if score > top_score:
                top_score = score
                top_word = label.split("_")[1].upper() if "_" in label else label.upper()

        # UI Updates
        self.score_label.setText(f"Hybrid Match: {int(top_score * 100)}% ({top_word})")
        
        if top_score > self.THRESHOLD:
            self.result_label.setText(top_word)
            self.result_label.setStyleSheet("font-size: 55px; color: #4ade80; font-weight: 900;")
        else:
            self.result_label.setText("READY")
            self.result_label.setStyleSheet("font-size: 50px; color: #475569; font-weight: 900;")

        # Render Feed
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self.feed.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.feed.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def hideEvent(self, event):
        self._stop_cam()
        super().hideEvent(event)

def get_tab():
    return PredictorWidget()