"""
test_sign_player_resize.py
──────────────────────────
Standalone resize-test harness for SignPlayerWidget.

Run with:
    python test_sign_player_resize.py [optional/path/to/sign.csv]

The window gives you:
  • A freely resizable main window (drag any edge/corner)
  • A live W × H readout that updates as you resize
  • Width / Height sliders (64 – 1920 px) for precise control
  • Preset buttons: 16:9, 4:3, 1:1, portrait
  • Start / Stop / Swap controls for the player itself
"""

import os
import sys


# ── locate the david_module package ───────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
# Adjust this if your package lives somewhere else relative to this script
_PACKAGE_ROOT = _HERE
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)


from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QSizePolicy,
    QFileDialog, QGroupBox, QGridLayout,
    QSplitter,
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QFont

from david_module.sign_player import SignPlayerWidget


# ── helpers ───────────────────────────────────────────────────────────────────

def _available_signs():
    try:
        from david_module.panda_port.animation import list_csv_signs
        return list_csv_signs()
    except Exception:
        return []


# ── main window ───────────────────────────────────────────────────────────────

class ResizeTestWindow(QMainWindow):
    """Full-featured resize-test harness."""

    _PRESETS = [
        ("16:9  (1280×720)",  1280, 720),
        ("4:3   (800×600)",    800, 600),
        ("1:1   (600×600)",    600, 600),
        ("Portrait (480×854)", 480, 854),
        ("Small  (320×240)",   320, 240),
    ]

    def __init__(self, initial_csv: str | None = None):
        super().__init__()
        self.setWindowTitle("SignPlayerWidget — Resize Test Harness")
        self.resize(1100, 780)
        self.setMinimumSize(480, 400)

        self._csv_path: str | None = initial_csv
        self._signs = _available_signs()

        self._build_ui()

        # Keep size readout current even when user drags the window edge
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._sync_readout)
        self._poll_timer.start(100)

        if self._csv_path:
            self.player.play(self._csv_path)
            self._update_status(f"Playing: {os.path.basename(self._csv_path)}")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # ── LEFT: control panel ───────────────────────────────────────────────
        ctrl_panel = QWidget()
        ctrl_panel.setFixedWidth(260)
        ctrl_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)
        ctrl_layout.setSpacing(10)

        # Title
        title = QLabel("Resize Tester")
        title.setFont(QFont("monospace", 14, QFont.Bold))
        title.setStyleSheet("color: #1e293b;")
        ctrl_layout.addWidget(title)

        # Live size readout
        self.size_label = QLabel("W: —   H: —")
        self.size_label.setFont(QFont("monospace", 12))
        self.size_label.setStyleSheet(
            "background:#0f172a; color:#38bdf8; padding:6px 10px;"
            "border-radius:6px; letter-spacing:1px;"
        )
        ctrl_layout.addWidget(self.size_label)

        # Player size readout
        self.player_size_label = QLabel("Player  W: —   H: —")
        self.player_size_label.setFont(QFont("monospace", 10))
        self.player_size_label.setStyleSheet(
            "background:#1e293b; color:#94a3b8; padding:4px 10px;"
            "border-radius:6px;"
        )
        ctrl_layout.addWidget(self.player_size_label)

        ctrl_layout.addSpacing(4)

        # ── Slider group ──────────────────────────────────────────────────────
        slider_box = QGroupBox("Manual Resize (player container)")
        slider_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        slider_grid = QGridLayout(slider_box)

        slider_grid.addWidget(QLabel("Width"), 0, 0)
        self.w_slider = QSlider(Qt.Horizontal)
        self.w_slider.setRange(64, 1920)
        self.w_slider.setValue(700)
        self.w_value = QLabel("700")
        self.w_value.setFixedWidth(40)
        self.w_slider.valueChanged.connect(lambda v: self._apply_size(w=v))
        slider_grid.addWidget(self.w_slider, 0, 1)
        slider_grid.addWidget(self.w_value, 0, 2)

        slider_grid.addWidget(QLabel("Height"), 1, 0)
        self.h_slider = QSlider(Qt.Horizontal)
        self.h_slider.setRange(64, 1080)
        self.h_slider.setValue(500)
        self.h_value = QLabel("500")
        self.h_value.setFixedWidth(40)
        self.h_slider.valueChanged.connect(lambda v: self._apply_size(h=v))
        slider_grid.addWidget(self.h_slider, 1, 1)
        slider_grid.addWidget(self.h_value, 1, 2)

        ctrl_layout.addWidget(slider_box)

        # ── Preset buttons ────────────────────────────────────────────────────
        preset_box = QGroupBox("Presets")
        preset_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        preset_layout = QVBoxLayout(preset_box)
        for label, w, h in self._PRESETS:
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda _, _w=w, _h=h: self._apply_size(w=_w, h=_h, fixed=True))
            preset_layout.addWidget(btn)
        ctrl_layout.addWidget(preset_box)

        # ── Player controls ───────────────────────────────────────────────────
        play_box = QGroupBox("Player Controls")
        play_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        play_layout = QVBoxLayout(play_box)

        browse_btn = QPushButton("📂  Browse CSV…")
        browse_btn.clicked.connect(self._browse_csv)
        play_layout.addWidget(browse_btn)

        self.play_btn = QPushButton("▶  Start / Swap")
        self.play_btn.setStyleSheet("background:#16a34a; color:white; font-weight:bold;")
        self.play_btn.clicked.connect(self._play)
        play_layout.addWidget(self.play_btn)

        stop_btn = QPushButton("■  Stop")
        stop_btn.setStyleSheet("background:#dc2626; color:white; font-weight:bold;")
        stop_btn.clicked.connect(self._stop)
        play_layout.addWidget(stop_btn)

        ctrl_layout.addWidget(play_box)

        # ── Status ────────────────────────────────────────────────────────────
        self.status_label = QLabel("No sign loaded.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color:#64748b; font-size:11px;")
        ctrl_layout.addWidget(self.status_label)

        ctrl_layout.addStretch(1)

        # ── RIGHT: player area ────────────────────────────────────────────────
        player_container = QWidget()
        player_container.setObjectName("playerContainer")
        player_container.setStyleSheet(
            "#playerContainer { background:#0a0c10; border:2px dashed #334155; border-radius:8px; }"
        )
        player_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        player_container_layout = QVBoxLayout(player_container)
        player_container_layout.setContentsMargins(0, 0, 0, 0)

        self.player = SignPlayerWidget()
        self.player.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        player_container_layout.addWidget(self.player)

        self.player_container = player_container

        # ── Splitter so user can drag left panel wider ─────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(ctrl_panel)
        splitter.addWidget(player_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 800])

        root_layout.addWidget(splitter)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _sync_readout(self):
        w, h = self.width(), self.height()
        pw, ph = self.player.width(), self.player.height()
        self.size_label.setText(f"Window   {w} × {h}")
        self.player_size_label.setText(f"Player    {pw} × {ph}")

    def _apply_size(self, w: int | None = None, h: int | None = None, fixed: bool = False):
        if fixed:
            # Resize the whole window to match the preset (plus control panel)
            ctrl_w = 260 + 16
            total_w = ctrl_w + w + 8
            total_h = h + 16
            self.resize(total_w, total_h)
            self.w_slider.setValue(w)
            self.h_slider.setValue(h)
        else:
            if w is not None:
                self.w_value.setText(str(w))
                self.player_container.setFixedWidth(w)
            if h is not None:
                self.h_value.setText(str(h))
                self.player_container.setFixedHeight(h)

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Sign CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self._csv_path = path
            self._update_status(f"Selected: {os.path.basename(path)}")

    def _play(self):
        if self._csv_path is None:
            # Try first available sign
            if self._signs:
                self._csv_path = str(self._signs[0][1])
            else:
                self._update_status("No CSV selected — use Browse.")
                return
        self.player.play(self._csv_path)
        self._update_status(f"Playing: {os.path.basename(self._csv_path)}")

    def _stop(self):
        self.player.stop()
        self._update_status("Stopped.")

    def _update_status(self, msg: str):
        self.status_label.setText(msg)

    def closeEvent(self, event):
        self._poll_timer.stop()
        self.player.stop()
        super().closeEvent(event)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    win = ResizeTestWindow(initial_csv=csv_arg)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()