import os
import time
import cv2
import numpy as np
import mediapipe as mp
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFrame, QMessageBox
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Styles ────────────────────────────────────────────────────────────────────
_STYLE_MAIN    = "background: #1e293b; color: white; font-weight: bold; border-radius: 8px; padding: 12px;"
_STYLE_SAVE    = "background: #16a34a; color: white; font-weight: bold; border-radius: 8px; padding: 15px; font-size: 16px;"
_STYLE_DISCARD = "background: #ef4444; color: white; font-weight: bold; border-radius: 8px; padding: 15px; font-size: 16px;"
_STYLE_INACTIVE = "background: #475569; color: #cbd5e1; border-radius: 8px; padding: 12px;"


class TrainerWidget(QWidget):
    def __init__(self):
        super().__init__()

        # --- PATH SETUP ---
        self.lib_path = os.path.join(SCRIPT_DIR, "googleMedaPipe", "asl_motion_library.npy")
        os.makedirs(os.path.dirname(self.lib_path), exist_ok=True)

        # --- CONFIG ---
        self.RECORD_DURATION = 2.5
        self.Z_SCALE         = 2.0
        self.ALPHA           = 0.3

        # --- STATE ---
        self.history        = {"left": None, "right": None}
        self.recording      = False
        self.sequence       = []
        self.cap            = None
        self.last_saved_key = None
        self.start_time     = None

        # --- MEDIAPIPE ---
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            model_complexity=1, max_num_hands=2,
            min_detection_confidence=0.8, min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils

        self._init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

    # ── UI ───────────────────────────────────────────────────────────────────
    def _init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(14, 14, 14, 14)

        # --- CONFIG PANEL ---
        config_panel = QFrame()
        config_panel.setStyleSheet(
            "background: #f1f5f9; border-radius: 12px; border: 1px solid #cbd5e1;"
        )
        config_layout = QVBoxLayout(config_panel)

        mode_row = QHBoxLayout()
        self.mode_btn = QPushButton("MODE: DUAL HAND 👐")
        self.mode_btn.setCheckable(True)
        self.mode_btn.setStyleSheet(_STYLE_MAIN)
        self.mode_btn.clicked.connect(self._toggle_mode)

        self.side_btn = QPushButton("SIDE: RIGHT ➡️")
        self.side_btn.setStyleSheet(
            "background: #3b82f6; color: white; border-radius: 8px; padding: 12px;"
        )
        self.side_btn.setVisible(False)
        self.side_btn.clicked.connect(self._toggle_side)

        mode_row.addWidget(QLabel("<b>Configuration:</b>"))
        mode_row.addWidget(self.mode_btn)
        mode_row.addWidget(self.side_btn)
        mode_row.addStretch()
        config_layout.addLayout(mode_row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("<b style='color: #1e293b;'>Gesture Name:</b>"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter sign name (e.g. apple)")
        self.name_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                background-color: white;
                color: black;
                border-radius: 6px;
                border: 2px solid #94a3b8;
                font-size: 14px;
                font-weight: normal;
            }
        """)
        name_row.addWidget(self.name_input)
        config_layout.addLayout(name_row)
        self.main_layout.addWidget(config_panel)

        # --- CAMERA FEED ---
        self.feed = QLabel("Camera Offline")
        self.feed.setMinimumSize(640, 480)
        self.feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed.setStyleSheet(
            "background: #0f172a; border-radius: 15px; border: 5px solid #334155;"
        )
        self.main_layout.addWidget(self.feed, stretch=5)

        # --- SAVE / DISCARD row (hidden until recording finishes) ---
        self.decision_widget = QWidget()
        d_layout = QHBoxLayout(self.decision_widget)
        self.save_btn = QPushButton("✔  SAVE GESTURE")
        self.save_btn.setStyleSheet(_STYLE_SAVE)
        self.save_btn.clicked.connect(self._save_data)
        self.discard_btn = QPushButton("✖  DISCARD & RETRY")
        self.discard_btn.setStyleSheet(_STYLE_DISCARD)
        self.discard_btn.clicked.connect(self._discard_data)
        d_layout.addWidget(self.save_btn)
        d_layout.addWidget(self.discard_btn)
        self.decision_widget.hide()
        self.main_layout.addWidget(self.decision_widget)

        # --- MAIN CONTROLS ---
        self.control_widget = QWidget()
        c_layout = QHBoxLayout(self.control_widget)
        self.cam_btn = QPushButton("START CAMERA")
        self.cam_btn.setStyleSheet(_STYLE_MAIN)
        self.cam_btn.clicked.connect(self._toggle_camera)
        self.rec_btn = QPushButton("🔴  RECORD 2.5s")
        self.rec_btn.setEnabled(False)
        self.rec_btn.setStyleSheet(_STYLE_INACTIVE)
        self.rec_btn.clicked.connect(self._start_recording)
        c_layout.addWidget(self.cam_btn)
        c_layout.addWidget(self.rec_btn)
        self.main_layout.addWidget(self.control_widget)

        # --- FOOTER ---
        self.footer = QFrame()
        self.footer.setStyleSheet(
            "background: #f8fafc; border-top: 1px solid #e2e8f0; padding: 10px;"
        )
        f_layout = QHBoxLayout(self.footer)
        self.history_label = QLabel("Waiting for first recording...")
        self.remove_btn = QPushButton("REMOVE LAST TAKE")
        self.remove_btn.setEnabled(False)
        self.remove_btn.setStyleSheet(
            "background: #450a0a; color: #fecaca; padding: 5px 15px; border-radius: 5px;"
        )
        self.remove_btn.clicked.connect(self._remove_last)
        f_layout.addWidget(self.history_label)
        f_layout.addStretch()
        f_layout.addWidget(self.remove_btn)
        self.main_layout.addWidget(self.footer)

    # ── Mode toggles ─────────────────────────────────────────────────────────
    def _toggle_mode(self):
        is_single = self.mode_btn.isChecked()
        self.mode_btn.setText("MODE: SINGLE HAND ☝" if is_single else "MODE: DUAL HAND 👐")
        self.side_btn.setVisible(is_single)

    def _toggle_side(self):
        is_left = "RIGHT" in self.side_btn.text()
        self.side_btn.setText("SIDE: LEFT ⬅️" if is_left else "SIDE: RIGHT ➡️")
        self.side_btn.setStyleSheet(
            f"background: {'#8b5cf6' if is_left else '#3b82f6'}; "
            "color: white; border-radius: 8px; padding: 12px;"
        )

    # ── Camera ───────────────────────────────────────────────────────────────
    def _toggle_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.cam_btn.setText("STOP CAMERA")
            self.rec_btn.setEnabled(True)
            self.rec_btn.setStyleSheet(
                "background: #dc2626; color: white; font-weight: bold; "
                "border-radius: 8px; padding: 12px;"
            )
        else:
            self._stop_camera()

    def _stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cam_btn.setText("START CAMERA")
        self.rec_btn.setEnabled(False)
        self.rec_btn.setStyleSheet(_STYLE_INACTIVE)
        self.feed.setText("Camera Offline")

    # ── Recording ────────────────────────────────────────────────────────────
    def _start_recording(self):
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Missing Name", "Please name the gesture before recording.")
            return
        self.sequence   = []
        self.recording  = True
        self.start_time = time.time()
        self.control_widget.hide()
        self.name_input.setEnabled(False)

    # ── Frame processing ─────────────────────────────────────────────────────
    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        curr_frame_data = {"left": None, "right": None}
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                side = "left" if hand_lms.landmark[0].x < 0.5 else "right"
                curr_frame_data[side] = self._get_smoothed_norm(hand_lms, side)
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        if self.recording:
            elapsed = time.time() - self.start_time
            bar_w   = int(frame.shape[1] * (elapsed / self.RECORD_DURATION))
            cv2.rectangle(frame, (0, 0), (bar_w, 15), (59, 130, 246), -1)

            if not self.mode_btn.isChecked():
                self.sequence.append(curr_frame_data)
            else:
                target = "left" if "LEFT" in self.side_btn.text() else "right"
                val    = curr_frame_data[target] if curr_frame_data[target] is not None else np.zeros(63)
                self.sequence.append(val)

            if elapsed >= self.RECORD_DURATION:
                self.recording = False
                self._show_review_screen()

        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self.feed.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.feed.size(),
                Qt.AspectRatioMode.KeepAspectRatio
            )
        )

    def _get_smoothed_norm(self, hand_lms, side):
        pts      = np.array([[lm.x, lm.y, lm.z * self.Z_SCALE] for lm in hand_lms.landmark])
        wrist    = pts[0]
        scale    = np.linalg.norm(pts[0] - pts[5]) or 1.0
        new_norm = ((pts - wrist) / scale).flatten()

        if self.history[side] is None:
            self.history[side] = new_norm
        else:
            self.history[side] = (
                (self.ALPHA * new_norm) + ((1 - self.ALPHA) * self.history[side])
            )
        return self.history[side]

    # ── Review / save / discard ───────────────────────────────────────────────
    def _show_review_screen(self):
        self._stop_camera()
        self.decision_widget.show()

    def _save_data(self):
        base = self.name_input.text().lower().strip()
        lib  = (
            np.load(self.lib_path, allow_pickle=True).item()
            if os.path.exists(self.lib_path) else {}
        )

        prefix   = (
            f"{base}_dual_" if not self.mode_btn.isChecked()
            else f"{base}_{'left' if 'LEFT' in self.side_btn.text() else 'right'}_"
        )
        existing = [int(k.split("_")[-1]) for k in lib if k.startswith(prefix)]
        idx      = max(existing) + 1 if existing else 1
        label    = f"{prefix}{idx}"

        lib[label] = np.array(self.sequence, dtype=object)
        np.save(self.lib_path, lib)

        self.last_saved_key = label
        self.history_label.setText(f"Last Saved: <b>{label}</b>")
        self.remove_btn.setEnabled(True)
        self._reset_ui()

    def _remove_last(self):
        if not self.last_saved_key:
            return
        ans = QMessageBox.warning(
            self, "Confirm Delete",
            f"Delete '{self.last_saved_key}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if ans == QMessageBox.StandardButton.Yes:
            lib = np.load(self.lib_path, allow_pickle=True).item()
            if self.last_saved_key in lib:
                del lib[self.last_saved_key]
                np.save(self.lib_path, lib)
                self.history_label.setText(f"Deleted: {self.last_saved_key}")
                self.last_saved_key = None
                self.remove_btn.setEnabled(False)

    def _discard_data(self):
        self._reset_ui()

    def _reset_ui(self):
        self.decision_widget.hide()
        self.control_widget.show()
        self.name_input.setEnabled(True)
        self._toggle_camera()

    # ── Cleanup ──────────────────────────────────────────────────────────────
    def hideEvent(self, event):
        """Stop camera when tab is switched away."""
        self._stop_camera()
        super().hideEvent(event)